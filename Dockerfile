# ======================================================
# DTRA - Dynamic Threat Response Agent
# DOCKERFILE: Creates a portable container for the API
# ======================================================

# Step 1: Use the official Python slim image (smaller, faster)
FROM python:3.10-slim

# Step 2: Set working directory INSIDE the container
WORKDIR /app

# Step 3: Copy requirements first (Docker caches this layer)
COPY requirements.txt .

# Step 4: Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy all project files into /app inside container
COPY server/ ./server/
COPY ui/ ./ui/
COPY models/ ./models/

# Step 6: Create uploads directory
RUN mkdir -p uploads

# Step 7: Expose port 5000 (Flask default)
EXPOSE 5000

# Step 8: Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=server/api.py

# Step 9: Run the Flask server when container starts
CMD ["python", "server/api.py"]
