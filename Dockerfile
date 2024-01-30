# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# install basic dependencies
RUN apt-get update && \
    apt-get install --fix-broken --no-install-recommends -y \
    git ffmpeg libsm6 libxext6 poppler-utils libmagic-mgc libmagic1 && \
    apt-get autoremove -y && apt-get clean && \
    rm -rf /var/lib/apt/lists/*
# Create a non-root user
RUN adduser --disabled-password --gecos '' appuser

# Set work directory
WORKDIR /home/appuser

# Install Python dependencies
COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --no-deps -r requirements.txt

# Copy the rest of the application's code
COPY --chmod=777 entrypoint /

COPY . .

# Expose port 8000 to the outside once the container has launched
EXPOSE 8000

STOPSIGNAL SIGQUIT

RUN chown -R appuser:appuser /home/appuser

# Run the application
ENTRYPOINT ["/bin/bash", "/entrypoint"]
CMD ["chainlit", "run", "-h", "--port=8000", "/home/appuser/sage/chat.py"]
