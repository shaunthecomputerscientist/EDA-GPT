# Use the official Python image with the specified version
FROM python:3.11.4-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project folder to the container
COPY . /app/

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt || pip install -r /app/packages.txt

# Expose the port that Streamlit uses
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the Streamlit app
CMD ["streamlit", "run", "/app/Home.py"]