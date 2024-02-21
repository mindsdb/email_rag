from typing import List

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore

from langchain.sql_database import SQLDatabase

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.docstore.document import Document

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import re
from langchain_core.runnables import RunnableLambda

from settings import (DEFAULT_LLM,
                      DEFAULT_EMBEDDINGS,
                      DEFAULT_AUTO_META_PROMPT_TEMPLATE,
                      DEFAULT_SQL_RETRIEVAL_PROMPT_TEMPLATE,
                      DEFAULT_CARDINALITY_THRESHOLD,
                      DEFAUlT_VECTOR_STORE, DEFAULT_CONTENT_COLUMN_NAME, documents_to_df
                      )
from utils import documents_to_df
import pandas as pd
import json


class AutoRetriever:
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
            cardinality_threshold: int = DEFAULT_CARDINALITY_THRESHOLD
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
            metadata_dict = row.drop(self.content_column_name).dropna().to_dict()
            docs.append(Document(page_content=row[self.content_column_name], metadata=metadata_dict))

        return docs

    def get_vectorstore(self):
        """
        Given data either List[Documents] pd.Dataframe,  use it to create a vectorstore.
        :return:
        """
        documents = self.df_to_documents() if isinstance(self.data, pd.DataFrame) else self.data
        return self.vectorstore.from_documents(documents, self.embeddings_model)

    def as_retriever(self):
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
        retriever = self.as_retriever()

        return retriever.get_relevant_documents(question)


class SQLRetriever:
    """
    a retriever used to connect to a postgres DB with pgvector extension
    """

    def __init__(self,
                 connection_string: str,
                 llm: BaseChatModel = DEFAULT_LLM,
                 embeddings_model: Embeddings = DEFAULT_EMBEDDINGS,
                 prompt_template: dict = DEFAULT_SQL_RETRIEVAL_PROMPT_TEMPLATE
                 ):
        self.prompt_template = prompt_template

        self.db = SQLDatabase.from_uri(connection_string)
        self.llm = llm
        self.embeddings_model = embeddings_model

    @staticmethod
    def format_prompt(prompt_template: str):
        """
        format prompt template

        :return:
        """
        return ChatPromptTemplate.from_messages(
            [("system", prompt_template), ("human", "{question}")]
        )

    def get_schema(self, _):
        """
        Get DB schema
        :return:
        """
        return self.db.get_table_info()

    def replace_brackets(self, match):
        words_inside_brackets = match.group(1).split(", ")
        embedded_words = [
            str(self.embeddings_model.embed_query(word)) for word in words_inside_brackets
        ]
        return "', '".join(embedded_words)

    def get_query(self, query):
        sql_query = re.sub(r"\[([\w\s,]+)\]", self.replace_brackets, query)
        return sql_query

    @property
    def sql_query_chain(self):
        return (
                RunnablePassthrough.assign(schema=self.get_schema)
                | self.format_prompt(self.prompt_template["sql_query"])
                | self.llm.bind(stop=["\nSQLResult:"])
                | StrOutputParser()
        )

    def as_retriever(self):
        return (
                RunnablePassthrough.assign(query=self.sql_query_chain)
                | RunnablePassthrough.assign(
            schema=self.get_schema,
            response=RunnableLambda(lambda x: self.db.run(self.get_query(x["query"]))),
        )
                | self.format_prompt(self.prompt_template["sql_result"])
                | self.llm
                | StrOutputParser()
        )
