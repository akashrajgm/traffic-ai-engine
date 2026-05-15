# Lock to a stable, specific version of Linux (bookworm) so it never secretly breaks
FROM python:3.10-slim-bookworm

# Install the modern C++ system drivers (libgl1 replaces the old mesa package)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy your requirements file first
COPY requirements.txt .

# Install all your Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your API code into the container
COPY . .

# Tell Docker which port the API will run on
EXPOSE 8000

# The command to boot up the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]