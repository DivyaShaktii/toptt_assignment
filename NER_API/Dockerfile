FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Copy files
COPY ./app ./app
COPY ./Best_Model ./Best_Model
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    python -c "import nltk; nltk.download('punkt')"

# Expose port
EXPOSE 8000

# Run API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]