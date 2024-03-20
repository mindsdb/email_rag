from typing import List
import json

from langchain.docstore.document import Document
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableSerializable
from langchain_core.vectorstores import VectorStore
import pandas as pd

from retrievers.base import BaseRetriever
from settings import (DEFAULT_LLM,
                      DEFAULT_EMBEDDINGS,
                      DEFAULT_AUTO_META_PROMPT_TEMPLATE,
                      DEFAULT_CARDINALITY_THRESHOLD,
                      DEFAUlT_VECTOR_STORE, DEFAULT_CONTENT_COLUMN_NAME
                      )
from utils import documents_to_df, VectorStoreOperator


class AutoRetriever(BaseRetriever):
    """
    AutoRetrieval is a class that uses langchain to extract metadata from a dataframe and query it using self retrievers.

    """

    def __init__(
            self,
            data: List[Document],
            content_column_name: str = DEFAULT_CONTENT_COLUMN_NAME,
            vectorstore: VectorStore = DEFAUlT_VECTOR_STORE,
            embeddings_model: Embeddings = DEFAULT_EMBEDDINGS,
            llm: BaseChatModel = DEFAULT_LLM,
            filter_columns: List[str] = None,
            document_description: str = "",
            prompt_template: str = DEFAULT_AUTO_META_PROMPT_TEMPLATE,
            cardinality_threshold: int = DEFAULT_CARDINALITY_THRESHOLD,
            vector_store_operator: VectorStoreOperator = None
    ):
        """
        Given a dataframe, use llm to extract metadata from it.
        :param data: List[Document]
        :param content_column_name: str
        :param vectorstore: VectorStore
        :param filter_columns: List[str]
        :param document_description: str
        :param embeddings_model: Embeddings
        :param llm: BaseChatModel
        :param prompt_template: str
        :param cardinality_threshold: int

        """

        self.data = data
        self.content_column_name = content_column_name
        self.vectorstore = vectorstore
        self.filter_columns = filter_columns
        self.document_description = document_description
        self.llm = llm
        self.embeddings_model = embeddings_model
        self.prompt_template = prompt_template
        self.cardinality_threshold = cardinality_threshold
        self.vector_store_operator = vector_store_operator

    def _get_low_cardinality_columns(self, data: pd.DataFrame):
        """
        Given a dataframe, return a list of columns with low cardinality if datatype is not bool.
        :return:
        """
        low_cardinality_columns = []
        columns = data.columns if self.filter_columns is None else self.filter_columns
        for column in columns:
            if data[column].dtype != "bool":
                if data[column].nunique() < self.cardinality_threshold:
                    low_cardinality_columns.append(column)
        return low_cardinality_columns

    def get_metadata_field_info(self):
        """
        Given a dataframe, use llm to extract metadata from it.
        :return:
        """

        def _alter_description(data: pd.DataFrame,
                               low_cardinality_columns: list,
                               result: List[dict]):
            """
            For low cardinality columns, alter the description to include the sorted valid values.
            :param data: pd.DataFrame
            :param low_cardinality_columns: list
            :param result: List[dict]
            """
            for column_name in low_cardinality_columns:
                valid_values = sorted(data[column_name].unique())
                for entry in result:
                    if entry["name"] == column_name:
                        entry["description"] += f". Valid values: {valid_values}"

        data = documents_to_df(
            self.content_column_name,
            self.data
        )

        prompt = self.prompt_template.format(dataframe=data.head().to_json(),
                                             description=self.document_description)
        result: List[dict] = json.loads(self.llm.invoke(input=prompt).content)

        _alter_description(
            data,
            self._get_low_cardinality_columns(data),
            result
        )

        return result

    def df_to_documents(self):
        """
        Given a dataframe, convert it to a list of documents.
        :return:
        """
        docs = []
        for _, row in self.data.iterrows():
            metadata_dict = row.drop(
                self.content_column_name).dropna().to_dict()
            docs.append(
                Document(page_content=row[self.content_column_name], metadata=metadata_dict))

        return docs

    def get_vectorstore(self):
        """
        Given data either List[Documents] pd.Dataframe,  use it to create a vectorstore.
        :return:
        """
        if self.vector_store_operator:
            return self.vector_store_operator.vector_store
        documents = self.df_to_documents() if isinstance(
            self.data, pd.DataFrame) else self.data
        return VectorStoreOperator(vector_store=self.vectorstore,
                                   documents=documents,
                                   embeddings_model=self.embeddings_model).vector_store

    def as_runnable(self) -> RunnableSerializable:
        """
        return the self-query retriever
        :return:
        """
        vectorstore = self.get_vectorstore()

        return SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=vectorstore,
            document_contents=self.document_description,
            metadata_field_info=self.get_metadata_field_info(),
            verbose=True
        )

    def query(self, question: str) -> List[Document]:
        """
        Given a question, use llm to query the dataframe.
        :param question: str
        :return: List[Document]
        """
        retriever_runnable = self.as_runnable()

        return retriever_runnable.get_relevant_documents(question)
