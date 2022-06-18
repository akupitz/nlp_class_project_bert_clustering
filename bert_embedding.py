from sentence_transformers import SentenceTransformer
from config import MPNET_MODEL_NAME, DISTIL_ROBERTA_MODEL_NAME, LEGAL_BERT_MODEL_NAME


class BERTEmbedder:
    def __init__(self):
        self.bert_models = {}

    def embed(self, text: str, model_name: str = DISTIL_ROBERTA_MODEL_NAME):
        if model_name not in [MPNET_MODEL_NAME, DISTIL_ROBERTA_MODEL_NAME, LEGAL_BERT_MODEL_NAME]:
            raise ValueError(f"Unrecognized sentence transformer model: {model_name} was used")
        if model_name not in self.bert_models:
            self.bert_models[model_name] = SentenceTransformer(model_name)
        return self.bert_models[model_name].encode(text, show_progress_bar=True)


# example_texts = ["today is gonna be a great day", "i dont feel so good today",
#                  "you know it rains when there are such dark clouds in the sky",
#                  "i hate this weather", "i love the sun", "there is nothing better than being here with you",
#                  "i love you",
#                  "i hate you", "have a nice day", "what a lovely day", "lets have fun in the sun",
#                  "dont be so rude", "i cant help you right now", "do you need any help"]
#
