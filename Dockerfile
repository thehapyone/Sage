# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV POETRY_VERSION 1.7.1

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
ENV PYTHONPATH /home/appuser

# Install Poetry
RUN pip install --upgrade pip && \
    pip install "poetry==$POETRY_VERSION"

# Copy only the files needed for installing dependencies
COPY pyproject.toml poetry.lock ./

# Install project dependencies
RUN poetry config virtualenvs.create false && \
    poetry install --no-cache --no-plugins --no-interaction --no-ansi

# Copy the rest of the application's code
COPY --chmod=777 entrypoint /entrypoint
COPY . .

# Expose port 8000 to the outside once the container has launched
EXPOSE 8000

STOPSIGNAL SIGQUIT

RUN chown -R appuser:appuser /home/appuser

# Run the application
ENTRYPOINT ["/bin/bash", "/entrypoint"]
CMD ["chainlit", "run", "-h", "--port=8000", "/home/appuser/sage/chat.py"]
