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

# Stage 4: Copy the application code and models 
# Copy the application code, trained models, and templates into the container
COPY ./app /app/app
COPY ./models /app/models

# Stage 5: Expose port and run the application
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]