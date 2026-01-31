# Stage 1: Use an official Python runtime as a parent image
FROM python:3.11-slim

# Stage 2: Set up the environment
WORKDIR /app

# Set environment variables to prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Stage 3: Install dependencies
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# Stage 4: Copy the application code, models, and data
COPY ./app /app/app
COPY ./models /app/models
COPY ./data /app/data

# Stage 5: Expose port and run the application
# Render provides the PORT environment variable
EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]