# Use the official Python image with the specified version
FROM python:3.11.4-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    ghostscript \
    libc6 \
    && apt-get clean

# Copy the application files
COPY . /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

# Expose the port that Streamlit uses
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "Home.py"]