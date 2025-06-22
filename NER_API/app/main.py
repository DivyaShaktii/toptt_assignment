from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import pandas as pd
import os
from app.model_utils import NERPipeline

app = FastAPI()
ner_pipeline = NERPipeline(model_dir="bert_ner/final_model")

@app.post("/predict/")
async def predict_from_csv(file: UploadFile = File(...)):
    # Read the input CSV
    df = pd.read_csv(file.file)
    if "text" not in df.columns:
        return {"error": "CSV must contain 'text' column"}

    predictions = []
    for _, row in df.iterrows():
        pred = ner_pipeline.predict(row['text'])
        predictions.append(pred)

    # Prepare results
    results_df = pd.DataFrame(predictions)
    results_df = results_df[['persons', 'organizations', 'locations']]
    results_df.fillna("", inplace=True)

    output_path = "output/predictions.csv"
    os.makedirs("output", exist_ok=True)
    results_df.to_csv(output_path, index=False)

    return FileResponse(output_path, media_type='text/csv', filename='predictions.csv')