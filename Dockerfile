# Use the official Python image as a base
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container
COPY . .

# Install required dependencies
RUN pip install --no-cache-dir fastapi \
    uvicorn \
    python-multipart \
    python-dotenv \
    requests \
    pydantic \
    sqlalchemy \
    scikit-learn \
    numpy \
    pillow \
    bs4 \
    docstring_parser \
    python-dateutil \
    uv \
    duckdb \
    Markdown \
    pytesseract \
    whisper

# If you need Prettier, install Node.js before using npm
RUN apt-get update && apt-get install -y nodejs npm && npm install -g prettier

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]