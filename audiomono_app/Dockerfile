FROM python:3.11-slim

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Set working directory
WORKDIR /app

# Copy only the relevant app folder
COPY ./audiomono_app /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Start FastAPI via uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
