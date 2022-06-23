from collections import Counter
from typing import Union
import pickle
import pandas as pd
from config import IS_ETHICAL_COLUMN, WHY_NOT_ETHICAL_TEXT_COLUMN, WHY_NOT_ETHICAL_CLEAN_TEXT_COLUMN, \
    WHY_NOT_ETHICAL_CLEAN_WORD_COUNT_COLUMN, RISK_1_COLUMN, MIN_WORDS_TO_KEEP, MAX_WORDS_TO_KEEP, \
    DISTIL_ROBERTA_EMBEDDING_COLUMN, \
    ORIGINAL_EXCEL_PATH, OUTPUT_WITH_EMBEDDING_PICKLE_PATH, MIN_REASON_COUNT_TO_KEEP, MPNET_EMBEDDING_COLUMN, \
    LEGAL_BERT_EMBEDDING_COLUMN, DISTIL_ROBERTA_MODEL_NAME, MPNET_MODEL_NAME, LEGAL_BERT_MODEL_NAME
from bert_embedding import BERTEmbedder

pd.options.mode.chained_assignment = None


def fix_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    not_ethical_with_word_count_in_a_normal_range
    :param df:
    :return: Part of the dataframe that is not ethical, with a normal word count (not too many, and not too little)
    """
    df = df[~df[IS_ETHICAL_COLUMN]]
    df[WHY_NOT_ETHICAL_CLEAN_TEXT_COLUMN] = df.apply(lambda row: clean_text(row[WHY_NOT_ETHICAL_TEXT_COLUMN]), axis=1)
    df[WHY_NOT_ETHICAL_CLEAN_WORD_COUNT_COLUMN] = df.apply(
        lambda row: len(row[WHY_NOT_ETHICAL_CLEAN_TEXT_COLUMN].split()),
        axis=1)
    df = df[df[WHY_NOT_ETHICAL_CLEAN_WORD_COUNT_COLUMN].between(MIN_WORDS_TO_KEEP, MAX_WORDS_TO_KEEP)]
    df = df[~df[RISK_1_COLUMN].isna()]
    df[RISK_1_COLUMN] = df.apply(lambda row: clean_text(row[RISK_1_COLUMN], to_lower=True), axis=1)
    return df


def _leave_only_main_risks(df):
    risk_reason_to_count_mapping = dict(Counter(df[RISK_1_COLUMN]))
    relevant_risk_reasons = [risk_reason for risk_reason, count in risk_reason_to_count_mapping.items() if
                             count >= MIN_REASON_COUNT_TO_KEEP]
    relevant_df = df[df[RISK_1_COLUMN].isin(relevant_risk_reasons)]
    return relevant_df


def read_excel_df(df_path: str) -> pd.DataFrame:
    df = pd.read_excel(df_path)
    # fix excel format
    df = df.replace(r'\n', ' ', regex=True)
    df = df.replace(r'[^\x00-\x7F]+', '', regex=True)
    df = df.replace({r"_x([0-9a-fA-F]{4})_": ""}, regex=True)
    return df


def clean_text(dirty_text: Union[str, int], to_lower: bool = False) -> str:
    dirty_text = str(dirty_text)
    # clean_text = re.sub(r'[^a-zA-Z0-9./ ]', r'', dirty_text)
    text_after_cleaning = " ".join(dirty_text.strip().split())
    if to_lower:
        text_after_cleaning = text_after_cleaning.lower()
    return text_after_cleaning


if __name__ == "__main__":
    topic_modeling_df = read_excel_df(ORIGINAL_EXCEL_PATH)
    topic_modeling_df = fix_df(topic_modeling_df)

    bert_embedder = BERTEmbedder()
    topic_modeling_df[DISTIL_ROBERTA_EMBEDDING_COLUMN] = topic_modeling_df.apply(
        lambda row: bert_embedder.embed(row[WHY_NOT_ETHICAL_CLEAN_TEXT_COLUMN], DISTIL_ROBERTA_MODEL_NAME), axis=1)
    topic_modeling_df[LEGAL_BERT_EMBEDDING_COLUMN] = topic_modeling_df.apply(
        lambda row: bert_embedder.embed(row[WHY_NOT_ETHICAL_CLEAN_TEXT_COLUMN], LEGAL_BERT_MODEL_NAME), axis=1)
    topic_modeling_df[MPNET_EMBEDDING_COLUMN] = topic_modeling_df.apply(
        lambda row: bert_embedder.embed(row[WHY_NOT_ETHICAL_CLEAN_TEXT_COLUMN], MPNET_MODEL_NAME), axis=1)

    topic_modeling_df = topic_modeling_df.reset_index(drop=True)
    with open(OUTPUT_WITH_EMBEDDING_PICKLE_PATH, "wb") as f:
        pickle.dump(topic_modeling_df, f)
