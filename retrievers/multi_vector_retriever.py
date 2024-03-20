from enum import Enum
from typing import List
import uuid

from langchain.retrievers.multi_vector import MultiVectorRetriever as LangChainMultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain.text_splitter import TextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_core.stores import BaseStore
from langchain_core.vectorstores import VectorStore
from langchain_openai import ChatOpenAI

from retrievers.base import BaseRetriever
from settings import DEFAULT_EMBEDDINGS, DEFAUlT_VECTOR_STORE, DEFAULT_LLM_MODEL, get_llm
from utils import VectorStoreOperator

_DEFAULT_ID_KEY = "doc_id"
_MAX_CONCURRENCY = 5


class MultiVectorRetrieverMode(Enum):
    """
    Enum for MultiVectorRetriever types.
    """
    SPLIT = "split"
    SUMMARIZE = "summarize"
    BOTH = "both"


class MultiVectorRetriever(BaseRetriever):
    """
    MultiVectorRetriever stores multiple vectors per document.
    """

    def __init__(
            self,
            documents: List[Document],
            doc_ids: list[str] = None,
            id_key: str = _DEFAULT_ID_KEY,
            vectorstore: VectorStore = DEFAUlT_VECTOR_STORE,
            parentstore: BaseStore = None,
            text_splitter: TextSplitter = None,
            embeddings_model: Embeddings = DEFAULT_EMBEDDINGS,
            mode: MultiVectorRetrieverMode = MultiVectorRetrieverMode.BOTH,
            vector_store_operator: VectorStoreOperator = None
    ):
        self.vectorstore = vectorstore
        self.parentstore = parentstore if parentstore is not None else InMemoryByteStore()
        self.id_key = id_key
        self.documents = documents
        self.text_splitter = text_splitter
        self.embeddings_model = embeddings_model
        self.mode = mode
        self.doc_ids = doc_ids
        if vector_store_operator:
            self.vector_store_operator = vector_store_operator
        else:
            self.vector_store_operator = self._create_vector_store_operator(documents=documents)


    def _generate_id_and_split_document(self, doc: Document) -> tuple[str, list[Document]]:
        """
        Generate a unique id for the document and split it into sub-documents.
        :param doc:
        :return:
        """
        doc_id = str(uuid.uuid4())
        sub_docs = self.text_splitter.split_documents([doc])
        for sub_doc in sub_docs:
            sub_doc.metadata[self.id_key] = doc_id
        return doc_id, sub_docs

    def _split_documents(self) -> tuple[list[Document], list[str]]:
        """
        Split the documents into sub-documents and generate unique ids for each document.
        :return:
        """
        split_info = list(map(self._generate_id_and_split_document, self.documents))
        doc_ids, split_docs_lists = zip(*split_info)
        split_docs = [doc for sublist in split_docs_lists for doc in sublist]
        return split_docs, list(doc_ids)

    def _create_retriever_and_vs_operator(self, docs: List[Document]) \
            -> tuple[LangChainMultiVectorRetriever, VectorStoreOperator]:

        retriever = LangChainMultiVectorRetriever(
            vectorstore=self.vector_store_operator.vector_store,
            byte_store=self.parentstore,
            id_key=self.id_key
        )
        return retriever, self.vector_store_operator

    def _get_document_summaries(self) -> List[str]:
        chain = (
                {"doc":lambda x:x.page_content}
                | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
                | get_llm(max_retries=0)
                | StrOutputParser()
        )
        return chain.batch(self.documents, {"max_concurrency": _MAX_CONCURRENCY})

    def as_runnable(self) -> RunnableSerializable:
        if not self.documents or not self.doc_ids:
            split_docs, doc_ids = self._split_documents()
        else:
            split_docs = self.documents
            doc_ids = self.doc_ids
        if self.mode in {MultiVectorRetrieverMode.SPLIT, MultiVectorRetrieverMode.BOTH}:
            retriever, vstore_operator = self._create_retriever_and_vs_operator(split_docs)
            summaries = self._get_document_summaries()
            summary_docs = [
                Document(page_content=s, metadata={self.id_key: doc_ids[i]})
                for i, s in enumerate(summaries)
            ]
            vstore_operator.add_documents(summary_docs)
            retriever.docstore.mset(list(zip(doc_ids, self.documents)))
            return retriever

        elif self.mode == MultiVectorRetrieverMode.SUMMARIZE:
            summaries = self._get_document_summaries()
            summary_docs = [
                Document(page_content=s, metadata={self.id_key: doc_ids[i]})
                for i, s in enumerate(summaries)
            ]
            retriever, vstore_operator = self._create_retriever_and_vs_operator(summary_docs)
            retriever.docstore.mset(list(zip(doc_ids, self.documents)))
            return retriever

        else:
            raise ValueError(f"Invalid mode: {self.mode}")
