from app.model_utils import NERPipeline
import pandas as pd
from typing import List

class InferenceEngine:
    def __init__(self, model_dir: str = "bert_ner/final_model"):
        self.pipeline = NERPipeline(model_dir)

    def predict_from_texts(self, texts: List[str]) -> pd.DataFrame:
        results = []
        for text in texts:
            pred = self.pipeline.predict(text)
            results.append(pred)

        df = pd.DataFrame(results)
        df = df[['persons', 'organizations', 'locations']]
        df.fillna("", inplace=True)
        return df

    def predict_from_csv(self, input_csv_path: str, output_csv_path: str) -> None:
        input_df = pd.read_csv(input_csv_path)
        if "text" not in input_df.columns:
            raise ValueError("CSV must contain a 'text' column")

        results_df = self.predict_from_texts(input_df['text'].tolist())
        results_df.to_csv(output_csv_path, index=False)