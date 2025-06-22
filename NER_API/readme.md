Build & Run Docker Container
# Build
docker build -t NER_API .

# Run
docker run -p 8000:8000 ner-api

curl -X POST "http://localhost:8000/predict/" \
  -F "file=@test_input.csv" --output predictions.csv

