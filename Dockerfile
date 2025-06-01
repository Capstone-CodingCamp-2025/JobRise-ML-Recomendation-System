FROM python:3.9-slim

# Install OS dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirement list dan install
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy semua source code ke dalam container
COPY . .

# Expose port
EXPOSE 8000

# Jalankan server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
