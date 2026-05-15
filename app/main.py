import os
import uuid
import shutil
import logging
import asyncio
import subprocess
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

import cv2
import psycopg2
import psycopg2.pool
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------------------------
load_dotenv()

DATABASE_URL: str = os.getenv("DATABASE_URL", "")
BASE_URL: str = os.getenv("BASE_URL", "http://localhost:8000")  # used to build absolute video URLs
MAX_STORED_VIDEOS: int = int(os.getenv("MAX_STORED_VIDEOS", 3))
MAX_STORED_INPUTS: int = int(os.getenv("MAX_STORED_INPUTS", 10))
MAX_UPLOAD_MB: int = int(os.getenv("MAX_UPLOAD_MB", 200))
ALLOWED_EXTENSIONS: set[str] = {".mp4", ".avi", ".mov", ".mkv"}
DETECTION_CONFIDENCE: float = float(os.getenv("DETECTION_CONF", 0.45))
VEHICLE_CLASSES: list[int] = [2, 5, 7]  # car, bus, truck

DATA_DIR = Path("data")
STATIC_DIR = Path("static")
DATA_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 2. LOGGING  (replaces bare print statements)
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("traffic_ai")

# ---------------------------------------------------------------------------
# 3. DATABASE — connection pool instead of a new connection per request
# ---------------------------------------------------------------------------
_db_pool: Optional[psycopg2.pool.SimpleConnectionPool] = None


def get_db_pool() -> Optional[psycopg2.pool.SimpleConnectionPool]:
    global _db_pool
    if _db_pool is None and DATABASE_URL:
        try:
            _db_pool = psycopg2.pool.SimpleConnectionPool(1, 5, DATABASE_URL)
            logger.info("Database connection pool created.")
        except Exception as exc:
            logger.error("Failed to create DB pool: %s", exc)
    return _db_pool


def save_to_db(job_id: str, filename: str, total_cars: int, video_url: str) -> None:
    pool = get_db_pool()
    if pool is None:
        logger.warning("DB pool unavailable – skipping persistence for job %s.", job_id)
        return

    conn = pool.getconn()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO traffic_logs (job_id, filename, total_cars, video_url)
                    VALUES (%s, %s, %s, %s)
                    """,
                    (job_id, filename, total_cars, video_url),
                )
                # Keep only the N most-recent rows
                cur.execute(
                    """
                    DELETE FROM traffic_logs
                    WHERE id NOT IN (
                        SELECT id FROM traffic_logs ORDER BY created_at DESC LIMIT %s
                    )
                    """,
                    (MAX_STORED_VIDEOS,),
                )
        logger.info("Job %s persisted to DB.", job_id)
    except Exception as exc:
        logger.error("DB write failed for job %s: %s", job_id, exc)
    finally:
        pool.putconn(conn)


# ---------------------------------------------------------------------------
# 4. APP LIFECYCLE — load model once, close pool on shutdown
# ---------------------------------------------------------------------------
_model: Optional[YOLO] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model
    logger.info("Loading YOLO model…")
    _model = YOLO("yolov8m.pt")
    logger.info("Model ready.")
    yield
    # Shutdown
    if _db_pool:
        _db_pool.closeall()
        logger.info("DB pool closed.")


app = FastAPI(title="Traffic AI Engine", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://traffic-ai-dashboard.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# 5. HELPERS
# ---------------------------------------------------------------------------

def validate_upload(file: UploadFile) -> None:
    """Raise HTTPException for invalid uploads before touching disk."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {ALLOWED_EXTENSIONS}",
        )
    # Content-length header check (not always present, but catches obvious abuse)
    if file.size and file.size > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_MB} MB limit.",
        )


def evict_old_files(directory: Path, pattern: str, keep: int) -> None:
    """Delete oldest files matching *pattern* until only *keep* remain."""
    files = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    for stale in files[: max(0, len(files) - keep + 1)]:
        try:
            stale.unlink()
            logger.info("Evicted old file: %s", stale)
        except OSError as exc:
            logger.warning("Could not delete %s: %s", stale, exc)


def remux_to_h264(src: Path, dst: Path) -> bool:
    """
    Re-encode with FFmpeg to H.264 so browsers can play it natively.
    Returns True on success, False if FFmpeg is unavailable.
    """
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", str(src),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-movflags", "+faststart",
                "-an",          # strip audio (traffic feed rarely has meaningful audio)
                str(dst),
            ],
            capture_output=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.error("FFmpeg stderr: %s", result.stderr.decode())
            return False
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("FFmpeg not available or timed out: %s", exc)
        return False


# ---------------------------------------------------------------------------
# 6. CORE PROCESSING — runs in a thread-pool so it doesn't block the loop
# ---------------------------------------------------------------------------

def _process_video(input_path: Path, output_path: Path) -> int:
    """
    Detect & count vehicles using YOLO + ByteTrack.
    Returns total vehicles that crossed the midline.
    Raises RuntimeError on unrecoverable failures.
    """
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open '{input_path}'.")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width == 0 or height == 0:
        cap.release()
        raise RuntimeError("Video has zero-dimension frames – file may be corrupt.")

    # Write to a temp path first; rename on success (atomic-ish)
    tmp_output = output_path.with_suffix(".tmp.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(tmp_output), fourcc, fps, (width, height))

    middle_line = height // 2
    zone_top = middle_line - 80
    zone_bottom = middle_line + 80

    counted_ids: set[int] = set()
    car_sequence_numbers: dict[int, int] = {}
    vehicle_positions: dict[int, int] = {}
    active_ids: set[int] = set()   # IDs seen in the current frame
    total_cars_passed = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detection zone overlay
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, zone_top), (width, zone_bottom), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
            cv2.line(frame, (0, zone_top), (width, zone_top), (0, 0, 255), 2)
            cv2.line(frame, (0, zone_bottom), (width, zone_bottom), (0, 0, 255), 2)

            results = _model.track(
                frame,
                classes=VEHICLE_CLASSES,
                persist=True,
                tracker="bytetrack.yaml",
                conf=DETECTION_CONFIDENCE,
                verbose=False,
            )

            current_frame_ids: set[int] = set()

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.int().cpu().tolist()
                track_ids = results[0].boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    current_frame_ids.add(track_id)
                    center_x = (box[0] + box[2]) // 2
                    center_y = (box[1] + box[3]) // 2

                    prev_y = vehicle_positions.get(track_id, center_y)

                    # Tripwire: must cross the absolute midline in motion
                    if track_id not in counted_ids:
                        crossed_downward = prev_y < middle_line <= center_y
                        crossed_upward = prev_y > middle_line >= center_y
                        if crossed_downward or crossed_upward:
                            counted_ids.add(track_id)
                            total_cars_passed += 1
                            car_sequence_numbers[track_id] = total_cars_passed

                    vehicle_positions[track_id] = center_y

                    if track_id in counted_ids:
                        box_color = (0, 255, 0)
                        label = f"Counted: #{car_sequence_numbers[track_id]}"
                    else:
                        box_color = (0, 0, 255)
                        label = "Uncounted"

                    cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), box_color, 2)
                    cv2.circle(frame, (center_x, center_y), 6, box_color, -1)
                    cv2.putText(
                        frame, label,
                        (box[0], max(box[1] - 10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2,
                    )

            # -- Prune stale track state to prevent unbounded memory growth --
            stale_ids = active_ids - current_frame_ids
            for sid in stale_ids:
                vehicle_positions.pop(sid, None)
                # Keep counted_ids & car_sequence_numbers; they're small and needed for labels
            active_ids = current_frame_ids

            # Counter HUD
            text = f"TOTAL PASSED: {total_cars_passed}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.rectangle(frame, (20, 20), (40 + tw, 50 + th), (0, 255, 0), -1)
            cv2.putText(frame, text, (30, 40 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

            out.write(frame)

    finally:
        cap.release()
        out.release()

    # Rename temp file only if processing completed cleanly
    tmp_output.rename(output_path)
    return total_cars_passed


# ---------------------------------------------------------------------------
# 7. ENDPOINT
# ---------------------------------------------------------------------------

@app.post("/api/v1/analyze")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    # --- Validate before touching disk ---
    validate_upload(file)

    job_id = uuid.uuid4().hex[:8]
    safe_stem = Path(file.filename or "upload").stem[:64]  # cap filename length
    ext = Path(file.filename or "upload").suffix.lower()
    input_path = DATA_DIR / f"{job_id}_{safe_stem}{ext}"
    raw_output_path = STATIC_DIR / f"output_{job_id}_raw.mp4"
    final_output_path = STATIC_DIR / f"output_{job_id}.mp4"

    # --- Save upload to disk with a size guard ---
    max_bytes = MAX_UPLOAD_MB * 1024 * 1024
    bytes_written = 0
    try:
        with input_path.open("wb") as buf:
            while chunk := await file.read(1024 * 256):  # 256 KB chunks
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Upload exceeds the {MAX_UPLOAD_MB} MB limit.",
                    )
                buf.write(chunk)
    except HTTPException:
        input_path.unlink(missing_ok=True)
        raise

    # --- Evict old files before adding new ones ---
    evict_old_files(STATIC_DIR, "output_*.mp4", keep=MAX_STORED_VIDEOS)
    evict_old_files(DATA_DIR, "*", keep=MAX_STORED_INPUTS)

    # --- Run blocking CV work off the async event loop ---
    try:
        total_cars = await asyncio.get_event_loop().run_in_executor(
            None, _process_video, input_path, raw_output_path
        )
    except Exception as exc:
        logger.exception("Video processing failed for job %s: %s", job_id, exc)
        # Clean up partial artefacts
        background_tasks.add_task(_cleanup, input_path, raw_output_path)
        raise HTTPException(status_code=500, detail=f"Processing failed: {exc}")

    # --- Re-mux to H.264 for browser compatibility ---
    remuxed = await asyncio.get_event_loop().run_in_executor(
        None, remux_to_h264, raw_output_path, final_output_path
    )
    if remuxed:
        background_tasks.add_task(_cleanup, raw_output_path)  # delete intermediate
        served_path = final_output_path
    else:
        # FFmpeg unavailable – serve the raw mp4v file (may not play in all browsers)
        logger.warning("Serving raw mp4v for job %s (FFmpeg unavailable).", job_id)
        served_path = raw_output_path

    # --- Remove input after processing ---
    background_tasks.add_task(_cleanup, input_path)

    video_url = f"/static/{served_path.name}"
    absolute_video_url = f"{BASE_URL}{video_url}"

    # --- Persist in background so the response isn't blocked ---
    background_tasks.add_task(
        save_to_db, job_id, file.filename or "unknown", total_cars, absolute_video_url
    )

    logger.info("Job %s complete – %d vehicles counted.", job_id, total_cars)

    return JSONResponse(
        status_code=200,
        content={
            "status": "success",
            "job_id": job_id,
            "original_filename": file.filename,
            "total_cars_passed": total_cars,
            "video_url": video_url,
            "video_url_absolute": absolute_video_url,
        },
    )


# ---------------------------------------------------------------------------
# 8. HEALTH CHECK
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    pool_ok = _db_pool is not None and not _db_pool.closed
    return {"status": "ok", "model_loaded": _model is not None, "db_pool": pool_ok}


# ---------------------------------------------------------------------------
# 9. UTILITIES
# ---------------------------------------------------------------------------

def _cleanup(*paths: Path) -> None:
    """Best-effort file removal; logs but never raises."""
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not delete temp file %s: %s", p, exc)