import os

IS_ETHICAL_COLUMN = "is.ethical"
WHY_NOT_ETHICAL_TEXT_COLUMN = "why.not.ethical"
WHY_NOT_ETHICAL_CLEAN_TEXT_COLUMN = "why.not.ethical_clean"
WHY_NOT_ETHICAL_CLEAN_WORD_COUNT_COLUMN = "why.not.ethical_clean_word_count"
RISK_1_COLUMN = "Risk 1"

DISTIL_ROBERTA_MODEL_NAME = "all-distilroberta-v1"
DISTIL_ROBERTA_EMBEDDING_COLUMN = "distilroberta_embedding"
MPNET_MODEL_NAME = "all-mpnet-base-v2"
MPNET_EMBEDDING_COLUMN = "mpnet_embedding"
LEGAL_BERT_MODEL_NAME = "nlpaueb/legal-bert-base-uncased"
LEGAL_BERT_EMBEDDING_COLUMN = "legalbert_embedding"

MIN_WORDS_TO_KEEP = 5
MAX_WORDS_TO_KEEP = 50

MIN_REASON_COUNT_TO_KEEP = 300  # 30

HOME_DIR_PATH = "/home/amit/Downloads"
ORIGINAL_EXCEL_PATH = os.path.join(HOME_DIR_PATH, "nlp_project_topic_modeling.xlsx")
OUTPUT_WITH_EMBEDDING_PICKLE_PATH = os.path.join(HOME_DIR_PATH, "nlp_project_topic_modeling.pkl")
