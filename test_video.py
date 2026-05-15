import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
from ultralytics import YOLO

print("[*] Booting Enterprise ROI Zone Tracker (Export Edition)...")
model = YOLO('yolov8m.pt') 

video_path = "data/test.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"CRITICAL ERROR: Could not open {video_path}.")
    exit()

# --- NEW: VIDEO EXPORT SETUP ---
# We need to grab the original video's exact size and speed to make a perfect copy
fps = int(cap.get(cv2.CAP_PROP_FPS))
if fps == 0: fps = 30
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create the Video Writer (Saves as 'output_tracked.mp4' in your folder)
output_path = "output_tracked.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # The MP4 codec
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
print(f"[*] Recording output to: {output_path}")
# -------------------------------

counted_ids = set()
clean_id_map = {}
total_cars_passed = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    LINE_TOP = int(height * 0.50)
    LINE_BOTTOM = int(height * 0.85)

    cv2.line(frame, (0, LINE_TOP), (width, LINE_TOP), (0, 0, 255), 2)
    cv2.line(frame, (0, LINE_BOTTOM), (width, LINE_BOTTOM), (0, 0, 255), 2)
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, LINE_TOP), (width, LINE_BOTTOM), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    results = model.track(frame, classes=[2, 5, 7], persist=True, tracker="bytetrack.yaml", conf=0.25, verbose=False)

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            center_y = int((y1 + y2) / 2)
            center_x = int((x1 + x2) / 2)

            box_color = (0, 0, 255) 
            display_text = "Uncounted"

            if LINE_TOP < center_y < LINE_BOTTOM:
                if track_id not in counted_ids:
                    counted_ids.add(track_id)
                    total_cars_passed += 1
                    clean_id_map[track_id] = total_cars_passed

            if track_id in clean_id_map:
                box_color = (0, 255, 0)
                clean_id = clean_id_map[track_id]
                display_text = f"Counted: #{clean_id}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(frame, (center_x, center_y), 5, box_color, -1)
            cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    cv2.putText(frame, f"TOTAL PASSED: {total_cars_passed}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

    # --- NEW: SAVE THE FRAME TO THE MP4 ---
    out.write(frame)

    cv2.imshow("Traffic AI - ROI Zone Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up all memory and close the file
cap.release()
out.release() # This tells Windows the video file is finished and safe to open!
cv2.destroyAllWindows()

print(f"[*] Done! Your processed video is saved as: {output_path}")