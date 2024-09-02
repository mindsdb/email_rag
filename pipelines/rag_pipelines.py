from copy import deepcopy, copy
from typing import List, Dict, Optional
import re

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker, CohereRerank
from langchain.text_splitter import TextSplitter
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore

from rerankers.settings import get_reranker, ReRankerConfig
from retrievers.auto_retriever import AutoRetriever
from retrievers.ensemble_retriever import EnsembleRetriever
from retrievers.multi_vector_retriever import MultiVectorRetriever, MultiVectorRetrieverMode
from retrievers.sql_retriever import SQLRetriever
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableSerializable
from langchain.docstore.document import Document

from settings import DEFAULT_LLM, DEFAULT_SQL_RETRIEVAL_PROMPT_TEMPLATE, DEFAULT_AUTO_META_PROMPT_TEMPLATE, \
    DEFAULT_RERANKING_PROMPT_TEMPLATE
from utils import VectorStoreOperator


class LangChainRAGPipeline:

    """
    Builds a RAG pipeline with returned sources
    """

    def __init__(self, retriever_runnable, prompt_template, llm=DEFAULT_LLM, reranker_config: Optional[ReRankerConfig] = None):
        self.retriever_runnable = retriever_runnable
        self.prompt_template = prompt_template
        self.llm = llm
        self.reranker_config = reranker_config


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

        # Function to format documents with labels
        def format_docs_with_labels(docs):
            if isinstance(docs, str):
                # this is to handle the case where the retriever returns a string
                # instead of a list of documents e.g. SQLRetriever
                return docs
            formatted_docs = ""
            for i, doc in enumerate(docs, start=1):
                formatted_docs += f"Document {i}:\n{doc}\n\n"
            return formatted_docs

        # Function to extract document IDs from reranking output
        def extract_document_ids(reranking_output):
            doc_ids = re.findall(r"Doc: (\d+), Relevance:", reranking_output)
            return [int(doc_id) for doc_id in doc_ids]

        # Function to map document IDs back to the original documents
        def get_reranked_docs(doc_ids, original_docs):
            max_index = len(original_docs)
            reranked_docs = []
            for doc_id in doc_ids:
                if 1 <= doc_id <= max_index:
                    reranked_docs.append(original_docs[doc_id - 1])
            return format_docs(reranked_docs)

        prompt = ChatPromptTemplate.from_template(self.prompt_template)

        if self.reranker_config:
            # create reranker from config
            reranker = get_reranker(self.reranker_config)
            if reranker:
                retriever = copy(self.retriever_runnable)
                self.retriever_runnable = ContextualCompressionRetriever(
                    base_compressor=reranker, base_retriever=retriever
                )


        rag_chain_from_docs = (
                RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
                | prompt
                | self.llm
                | StrOutputParser()
        )

        rag_chain_with_source = RunnableParallel(
            {"context": self.retriever_runnable, "question": RunnablePassthrough()}
        ).assign(answer=rag_chain_from_docs)

        return rag_chain_with_source

    @classmethod
    def from_retriever(cls, retriever, prompt_template, llm = DEFAULT_LLM,
                       reranker_config: Optional[ReRankerConfig] = None):
        """
        Builds a RAG pipeline with returned sources using a BaseRetriever
        :param retriever: BaseRetriever
        :param prompt_template: str
        :param llm: BaseChatModel
        :param reranker_config: Optional[ReRankerConfig]

        :return: LangChainRAGPipeline
        """
        return cls(retriever, prompt_template, llm, reranker_config)

    @classmethod
    def from_sql_retriever(cls,
                           connection_string,
                           retriever_prompt_template: dict,
                           rag_prompt_template,
                           llm: BaseChatModel = None,
                           reranker_config: Optional[ReRankerConfig] = None
                           ):
        """
        Builds a RAG pipeline with returned sources using a SQLRetriever

        :param connection_string: str
        :param retriever_prompt_template: dict
        :param rag_prompt_template: str
        :param llm: BaseChatModel
        :param reranker_config: Optional[ReRankerConfig]

        :return: LangChainRAGPipeline
        """
        retriever_prompt_template = retriever_prompt_template or DEFAULT_SQL_RETRIEVAL_PROMPT_TEMPLATE

        retriever_runnable = SQLRetriever(
            connection_string=connection_string,
            prompt_template=retriever_prompt_template
        ).as_runnable()

        return cls(retriever_runnable, rag_prompt_template, llm, reranker_config)

    @classmethod
    def from_ensemble_retriever(cls,
                                rag_prompt_template: str,
                                runnable_retrievers: List[Dict],
                                llm: BaseChatModel = None,
                                reranker_config: Optional[ReRankerConfig] = None
                                ):
        """
        Builds a RAG pipeline with returned sources using an EnsembleRetriever

        :param rag_prompt_template: str
        :param runnable_retrievers: list[dict]
        :param llm: BaseChatModel
        :param reranker_config: Optional[ReRankerConfig]

        :return: LangChainRAGPipeline
        """

        retriever_runnable = EnsembleRetriever(
            runnable_retrievers,
            llm=llm
        ).as_runnable()

        return cls(retriever_runnable, rag_prompt_template, llm, reranker_config)

    @classmethod
    def from_auto_retriever(cls,
                            retriever_prompt_template: str,
                            rag_prompt_template: str,
                            data_description: str,
                            content_column_name: str,
                            data: List[Document],
                            vectorstore: VectorStore = None,
                            llm: BaseChatModel = None,
                            vector_store_operator: VectorStoreOperator = None,
                            reranker_config: Optional[ReRankerConfig] = None,
                            ):
        """
        Builds a RAG pipeline with returned sources using an AutoRetriever

        NB specify either data or vectorstore, not both

        :param retriever_prompt_template: str
        :param rag_prompt_template: str
        :param data_description: str
        :param content_column_name: str
        :param data: List[Document]
        :param vectorstore: VectorStore
        :param llm: BaseChatModel
        :param vector_store_operator: VectorStoreOperator
        :param reranker_config: Optional[ReRankerConfig]

        :return: LangChainRAGPipeline
        """
        retriever_prompt_template = retriever_prompt_template or DEFAULT_AUTO_META_PROMPT_TEMPLATE

        retriever_runnable = AutoRetriever(data=data, content_column_name=content_column_name, vectorstore=vectorstore,
                                           document_description=data_description,
                                           prompt_template=retriever_prompt_template,
                                           vector_store_operator=vector_store_operator).as_runnable()
        return cls(retriever_runnable, rag_prompt_template, llm, reranker_config)

    @classmethod
    def from_multi_vector_retriever(
            cls,
            documents: List[Document],
            doc_ids: list[str],
            rag_prompt_template: str,
            vectorstore: VectorStore = None,
            text_splitter: TextSplitter = None,
            llm: BaseChatModel = None,
            mode: MultiVectorRetrieverMode = MultiVectorRetrieverMode.BOTH,
            vector_store_operator: VectorStoreOperator = None,
            reranker_config: Optional[ReRankerConfig] = None
    ):
        """
        Builds a RAG pipeline with returned sources using a MultiVectorRetriever

        :param documents: List[Document]
        :param doc_ids: list[str]
        :param rag_prompt_template: str
        :param vectorstore: VectorStore
        :param text_splitter: TextSplitter
        :param llm: BaseChatModel
        :param mode: MultiVectorRetrieverMode
        :param vector_store_operator: VectorStoreOperator
        :param reranker_config: Optional[ReRankerConfig]

        :return: LangChainRAGPipeline
        """
        retriever_runnable = MultiVectorRetriever(
            documents=documents, doc_ids=doc_ids, vectorstore=vectorstore, text_splitter=text_splitter, mode=mode,
            vector_store_operator=vector_store_operator).as_runnable()
        return cls(retriever_runnable, rag_prompt_template, llm, reranker_config)
