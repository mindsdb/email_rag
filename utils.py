from datetime import timedelta
from typing import List
import time

import pandas as pd
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

# gpt-3.5-turbo
_DEFAULT_TPM_LIMIT = 60000
_DEFAULT_RATE_LIMIT_INTERVAL = timedelta(seconds=10)


def documents_to_df(content_column_name: str,
                    documents: List[Document],
                    embeddings_model: Embeddings = None,
                    with_embeddings: bool = False) -> pd.DataFrame:
    """
    Given a list of documents, convert it to a dataframe.

    :param content_column_name: str
    :param documents: List[Document]
    :param embeddings_model: Embeddings
    :param with_embeddings: bool

    :return: pd.DataFrame
    """
    df = pd.DataFrame([doc.metadata for doc in documents])

    df[content_column_name] = [doc.page_content for doc in documents]

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Reordering the columns to have the content column first.
    df = df[[content_column_name] + [col for col in df.columns if col != content_column_name]]

    if with_embeddings:
        df["embeddings"] = embeddings_model.embed_documents(df[content_column_name].tolist())

    return df


def vector_store_from_documents(
        vector_store: VectorStore,
        documents: List[Document],
        embeddings_model: Embeddings,
        tpm_limit: int = _DEFAULT_TPM_LIMIT,
        rate_limit_interval: timedelta = _DEFAULT_RATE_LIMIT_INTERVAL) -> VectorStore:
    first_document = documents[0]
    # Underlying rate limit mechanism behind this isn't sufficient for a bulk call.
    # We can easily go above our tokens per minute quota (60 000 for gpt-3.5-turbo).
    # To get around this, we initialize the store with the first document, then use
    # our own super simple rate limiting to handle many large embedding calls.
    store = vector_store.from_documents(
        documents=[first_document],
        embedding=embeddings_model,
    )
    # Approximate token usage by length of document content.
    current_token_usage = len(first_document.page_content)
    for doc in documents[1:]:
        if current_token_usage >= tpm_limit:
            # We do what we can since we don't have access to rate limit headers
            # https://platform.openai.com/docs/guides/rate-limits/usage-tiers?context=tier-one.
            time.sleep(rate_limit_interval.total_seconds())
            current_token_usage = 0
        store.add_documents([doc])
        current_token_usage += len(doc.page_content)
    return store
