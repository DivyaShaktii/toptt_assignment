import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

LABEL_MAP = {'B-PER': 'persons', 'I-PER': 'persons',
             'B-ORG': 'organizations', 'I-ORG': 'organizations',
             'B-LOC': 'locations', 'I-LOC': 'locations'}

class NERPipeline:
    def __init__(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.eval()
        self.label_map = self.model.config.id2label

    def predict(self, text: str):
        tokens = word_tokenize(text)
        inputs = self.tokenizer(tokens, is_split_into_words=True, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs).logits
        preds = torch.argmax(outputs, dim=-1).squeeze().tolist()

        word_ids = inputs.word_ids()
        grouped_preds = {}
        current_entity = ""
        current_type = ""
        entities = {'persons': [], 'organizations': [], 'locations': []}

        for idx, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            label = self.label_map[preds[idx]]
            if label == "O":
                if current_entity and current_type:
                    entities[current_type].append(current_entity.strip())
                current_entity = ""
                current_type = ""
            else:
                entity_type = LABEL_MAP.get(label, "")
                if label.startswith("B-") or entity_type != current_type:
                    if current_entity and current_type:
                        entities[current_type].append(current_entity.strip())
                    current_entity = tokens[word_id]
                    current_type = entity_type
                else:
                    current_entity += " " + tokens[word_id]

        if current_entity and current_type:
            entities[current_type].append(current_entity.strip())

        return {k: ";".join(list(dict.fromkeys(v))) for k, v in entities.items()}