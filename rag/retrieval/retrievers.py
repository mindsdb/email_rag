from typing import List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.sql_database import SQLDatabase
from langchain_community.vectorstores import Chroma

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.docstore.document import Document

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import re
from langchain_core.runnables import RunnableLambda

from rag.settings import (DEFAULT_LLM_MODEL,
                          DEFAULT_AUTO_META_PROMPT_TEMPLATE,
                          DEFAULT_SQL_RESULT_PROMPT,
                          DEFAULT_TEXT_2_PGVECTOR_PROMPT, DEFAULT_CARDINALITY_THRESHOLD
                          )
import pandas as pd
import json


class AutoRetriever:
    """
    AutoRetrieval is a class that uses langchain to extract metadata from a dataframe and query it using self retrieval.

    pass in a dataframe and a content column name, and it will use langchain to extract metadata from the dataframe.
    if you pass in a vectorstore, it will use it in the self retrieval.
    """

    def __init__(
            self,
            df: pd.DataFrame,
            content_column_name: str,
            vectorstore: VectorStore = None,
            filter_columns: List[str] = None,
            document_description: str = "",
            model: str = DEFAULT_LLM_MODEL,
            prompt_template: str = DEFAULT_AUTO_META_PROMPT_TEMPLATE,
            cardinality_threshold: int = DEFAULT_CARDINALITY_THRESHOLD
    ):
        """
        Given a dataframe, use llm to extract metadata from it.
        :param df: pd.DataFrame
        :param content_column_name: str
        :param vectorstore: VectorStore
        :param filter_columns: List[str]
        :document_description: str
        :param model: str
        :param prompt_template: str
        :param cardinality_threshold: int

        """

        self.df = df
        self.content_column_name = content_column_name
        self.vectorstore = vectorstore
        self.filter_columns = filter_columns
        self.document_description = document_description
        self.llm = ChatOpenAI(model=model)
        self.embeddings_model = OpenAIEmbeddings()
        self.prompt_template = prompt_template
        self.cardinality_threshold = cardinality_threshold

    def _get_low_cardinality_columns(self):
        """
        Given a dataframe, return a list of columns with low cardinality if datatype is not bool.
        :return:
        """
        low_cardinality_columns = []
        columns = self.df.columns if self.filter_columns is None else self.filter_columns
        for column in columns:
            if self.df[column].dtype != "bool":
                if self.df[column].nunique() < self.cardinality_threshold:
                    low_cardinality_columns.append(column)
        return low_cardinality_columns

    def _alter_description(self, result: List[dict]):
        """
        For low cardinality columns, alter the description to include the sorted valid values.
        :param result: List[dict]
        """

        low_cardinality_columns = self._get_low_cardinality_columns()
        for column_name in low_cardinality_columns:
            valid_values = sorted(self.df[column_name].unique())
            for entry in result:
                if entry["name"] == column_name:
                    entry["description"] += f"{entry['description']}. Valid values: {valid_values}"

    def get_metadata_field_info(self):
        """
        Given a dataframe, use llm to extract metadata from it.
        :return:
        """
        prompt = self.prompt_template.format(dataframe=self.df.head().to_json(), description=self.document_description)
        result: List[dict] = json.loads(self.llm.invoke(input=prompt).content)
        self._alter_description(result)

        return result

    def df_to_documents(self):
        """
        Given a dataframe, convert it to a list of documents.
        :return:
        """
        docs = []
        for _, row in self.df.iterrows():
            metadata_dict = row.drop(self.content_column_name).dropna().to_dict()
            docs.append(Document(page_content=row[self.content_column_name], metadata=metadata_dict))

        return docs

    def get_vectorstore(self):
        """
        Given a dataframe, convert it to a list of documents and use it to create a vectorstore.
        :return:
        """
        self.vectorstore = Chroma.from_documents(self.df_to_documents(), self.embeddings_model)

    def get_retriever(self):
        """
        return the self-query retriever
        :return:
        """
        if self.vectorstore is None:
            # if no vector stor is passed on init, create one from input data.

            self.get_vectorstore()

        return SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=self.vectorstore,
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
        retriever = self.get_retriever()

        return retriever.get_relevant_documents(question)


class SQLRetriever:
    """
    a retriever used to connect to a postgres DB with pgvector extension
    """

    def __init__(self,
                 user: str,
                 host: str,
                 port: int,
                 db_name: str,
                 sql_query_prompt_template: str = DEFAULT_TEXT_2_PGVECTOR_PROMPT,
                 sql_result_prompt_template: str = DEFAULT_SQL_RESULT_PROMPT
                 ):
        connection_string = f"postgresql+psycopg2://postgres:{user}@{host}:{port}/{db_name}"
        self.db = SQLDatabase.from_uri(connection_string)
        self.llm = ChatOpenAI(model_name="gpt-4", temperature=0)
        self.sql_query_prompt_template = sql_query_prompt_template
        self.sql_result_prompt_template = sql_result_prompt_template
        self.embeddings_model = OpenAIEmbeddings()

    @staticmethod
    def format_prompt(prompt_template: str):
        """
        format prompt template

        :return:
        """
        return ChatPromptTemplate.from_messages(
            [("system", prompt_template), ("human", "{question}")]
        )

    def get_schema(self):
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
                | self.format_prompt(self.sql_query_prompt_template)
                | self.llm.bind(stop=["\nSQLResult:"])
                | StrOutputParser()
        )

    @property
    def full_chain(self):
        return (
                RunnablePassthrough.assign(query=self.sql_query_chain)
                | RunnablePassthrough.assign(
            schema=self.get_schema,
            response=RunnableLambda(lambda x: self.db.run(self.get_query(x["query"]))),
        )
                | self.format_prompt(self.sql_result_prompt_template)
                | self.llm
        )
