# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# install basic dependencies
RUN apt-get update && \
    apt-get install --fix-broken --no-install-recommends -y \
    git && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt ./  
RUN pip install --upgrade pip && \  
    pip install --no-cache-dir --no-deps -r /app/requirements.txt

# Copy the rest of the application's code
COPY . .

# Expose port 8000 to the outside once the container has launched
EXPOSE 8000

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Run the application
CMD ["chainlit", "run", "-h", "--port=8000", "/app/sage/chat.py"]
