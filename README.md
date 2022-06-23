# nlp_class_project_bert_topic_modeling

* Run data_preparation.py after changing config.py to match your needs

* Run a notebook for your topic modeling using your dataset (or a modification you want on it):
import pandas as pd
from config import OUTPUT_WITH_EMBEDDING_PICKLE_PATH, WHY_NOT_ETHICAL_CLEAN_TEXT_COLUMN, RISK_1_COLUMN, EMBEDDING_COLUMN, MIN_REASON_COUNT_TO_KEEP
with open(OUTPUT_WITH_EMBEDDING_PICKLE_PATH, "rb") as f:
    df = pickle.load(f)
df = df[[WHY_NOT_ETHICAL_CLEAN_TEXT_COLUMN, RISK_1_COLUMN, EMBEDDING_COLUMN]]

* Choose your algorithms (you should take a look at the notebooks, they will help you)
* Evaluate using evaluation utils 
