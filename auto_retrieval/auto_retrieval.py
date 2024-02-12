from typing import List

from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import Chroma

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.docstore.document import Document

from settings import DEFAULT_LLM_MODEL, DEFAULT_PROMPT_TEMPLATE
import pandas as pd
import json


class AutoRetrieval:
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
            prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
            cardinality_threshold: int = 40
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
        embedder = OpenAIEmbeddings()
        self.vectorstore = Chroma.from_documents(self.df_to_documents(), embedder)

    def query(self, question: str) -> List[Document]:
        """
        Given a question, use llm to query the dataframe.
        :param question: str
        :return: List[Document]
        """

        if self.vectorstore is None:
            self.get_vectorstore()

        retriever = SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=self.vectorstore,
            document_contents=self.document_description,
            metadata_field_info=self.get_metadata_field_info(),
            verbose=True
        )
        return retriever.get_relevant_documents(question)


if __name__ == "__main__":
    df = pd.read_csv("./data/test/movie_test.csv")
    df.fillna("unknown", inplace=True)
    ar = AutoRetrieval(df, "content", document_description="Brief summary of a movie")
    print(ar.query("I want to watch a movie rated higher than 8.5"))



