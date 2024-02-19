from typing import Union, List

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from retrievers.retrievers import SQLRetriever, AutoRetriever
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSerializable
from langchain.docstore.document import Document

from settings import DEFAULT_LLM, DEFAULT_SQL_RETRIEVAL_PROMPT_TEMPLATE, DEFAULT_AUTO_META_PROMPT_TEMPLATE


class LangChainRAGPipeline:
    """
    Builds a RAG pipeline using langchain LCEL components
    """

    def __init__(self, retriever, prompt_template, llm = DEFAULT_LLM):

        self.retriever = retriever
        self.prompt_template = prompt_template
        self.llm = llm

    def rag_with_returned_sources(self) -> RunnableSerializable:
        """
        Builds a RAG pipeline with returned sources
        :return:
        """

        def format_docs(docs):
            if isinstance(docs, str):
                # this is to handle the case where the retriever returns a string
                # instead of a list of documents e.g. SQLRetriever
                return docs
            return "\n\n".join(doc.page_content for doc in docs)

        prompt = ChatPromptTemplate.from_template(self.prompt_template)

        rag_chain_from_docs = (
                RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
                | prompt
                | self.llm
                | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": self.retriever, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        return rag_chain_with_source

    @classmethod
    def from_retriever(cls, retriever: BaseRetriever, prompt_template: str, llm: BaseChatModel):
        """
        Builds a RAG pipeline with returned sources using a BaseRetriever
        :param retriever: BaseRetriever
        :param prompt_template: str
        :param llm: BaseChatModel

        :return:
        """
        return cls(retriever, prompt_template, llm)

    @classmethod
    def from_sql_retriever(cls,
                           connection_string,
                           retriever_prompt_template: dict,
                           rag_prompt_template,
                           llm: BaseChatModel = None
                           ):
        """
        Builds a RAG pipeline with returned sources using a SQLRetriever

        :param connection_string: str
        :param retriever_prompt_template: dict
        :param rag_prompt_template: str
        :param llm: BaseChatModel

        :return:
        """
        retriever_prompt_template = retriever_prompt_template or DEFAULT_SQL_RETRIEVAL_PROMPT_TEMPLATE

        retriever = SQLRetriever(
            connection_string=connection_string,
            prompt_template=retriever_prompt_template
        ).as_retriever()

        return cls(retriever, rag_prompt_template, llm)

    @classmethod
    def from_auto_retriever(cls,
                            retriever_prompt_template: str,
                            rag_prompt_template: str,
                            data_description: str,
                            content_column_name: str,
                            data: Union[pd.DataFrame, List[Document]],
                            vectorstore: VectorStore = None,
                            llm: BaseChatModel = None
                            ):
        """
        Builds a RAG pipeline with returned sources using a AutoRetriever

        NB specify either data or vectorstore, not both

        if data is specified, it should be a pd.DataFrame or a List[Document],
        by default Chroma will be used to create a vectorstore

        :param retriever_prompt_template: str
        :param rag_prompt_template: str
        :param data_description: str
        :param content_column_name: str
        :param data: Union[pd.DataFrame, List[Document]]
        :param vectorstore: VectorStore
        :param llm: BaseChatModel

        :return:
        """
        retriever_prompt_template = retriever_prompt_template or DEFAULT_AUTO_META_PROMPT_TEMPLATE

        retriever = AutoRetriever(data=data, content_column_name=content_column_name, vectorstore=vectorstore,
                                  document_description=data_description,
                                  prompt_template=retriever_prompt_template).as_retriever()
        return cls(retriever, rag_prompt_template, llm)
