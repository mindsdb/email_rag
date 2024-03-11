from datetime import timedelta
from typing import List, Tuple
import time

import pandas as pd
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore

# gpt-3.5-turbo
_DEFAULT_TPM_LIMIT = 60000
_DEFAULT_RATE_LIMIT_INTERVAL = timedelta(seconds=10)
_INITIAL_TOKEN_USAGE = 0



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


class VectorStoreOperator:
    """
    Encapsulates the logic for adding documents to a vector store with rate limiting.
    """

    def __init__(self,
                 vector_store: VectorStore,
                 documents: List[Document],
                 embeddings_model: Embeddings,
                 token_per_minute_limit: int = _DEFAULT_TPM_LIMIT,
                 rate_limit_interval: timedelta = _DEFAULT_RATE_LIMIT_INTERVAL):

        self.documents = documents
        self.embeddings_model = embeddings_model
        self.token_per_minute_limit = token_per_minute_limit
        self.rate_limit_interval = rate_limit_interval
        self.current_token_usage = _INITIAL_TOKEN_USAGE
        self._add_documents_to_store(documents, vector_store)

    @property
    def vector_store(self):
        return self._vector_store

    @staticmethod
    def _calculate_token_usage(document):
        return len(document.page_content)

    def _rate_limit(self):
        if self.current_token_usage >= self.token_per_minute_limit:
            time.sleep(self.rate_limit_interval.total_seconds())
            self.current_token_usage = _INITIAL_TOKEN_USAGE

    def _update_token_usage(self, document: Document):
        self._rate_limit()
        self.current_token_usage += self._calculate_token_usage(document)

    def _add_document(self, document: Document):
        self._update_token_usage(document)
        self.vector_store.add_documents([document])

    def _add_documents_to_store(self, documents: List[Document], vector_store: VectorStore):
        for i, document in enumerate(documents):
            if i == 0:
                self._vector_store = vector_store.from_documents(
                    documents=[document], embedding=self.embeddings_model
                )

            self._add_document(document)

    def add_documents(self, documents: List[Document]):
        for document in documents:
            self._add_document(document)
