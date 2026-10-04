# Use a minimal base image with Python 3.12 installed
FROM python:3.12-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy all files from the current directory on the host to the container's /app directory
COPY . .

# Install Python dependencies listed in requirements.txt without using cache to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Define the command to run the Streamlit app on port 8501 and make it accessible externally
# NOTE: XSRF protection is disabled so file uploads for batch prediction work through the forwarded URL
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
