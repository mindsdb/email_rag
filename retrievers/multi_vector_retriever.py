from typing import List
import uuid

from langchain.docstore.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever as LangChainMultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain.text_splitter import TextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableSerializable
from langchain_core.stores import BaseStore
from langchain_core.vectorstores import VectorStore

from retrievers.base import BaseRetriever
from settings import DEFAULT_EMBEDDINGS, DEFAUlT_VECTOR_STORE
from utils import vector_store_from_documents

_DEFAULT_ID_KEY = "doc_id"


class MultiVectorRetriever(BaseRetriever):

    """
    MultiVectorRetriever stores multiple vectors per document.
    """

    def __init__(
            self,
            documents: List[Document],
            id_key: str = _DEFAULT_ID_KEY,
            vectorstore: VectorStore = DEFAUlT_VECTOR_STORE,
            parentstore: BaseStore = None,
            text_splitter: TextSplitter = None,
            embeddings_model: Embeddings = DEFAULT_EMBEDDINGS
    ):
        self.vectorstore = vectorstore
        self.parentstore = parentstore
        if self.parentstore is None:
            self.parentstore = InMemoryByteStore()
        self.id_key = id_key
        self.documents = documents
        self.text_splitter = text_splitter
        self.embeddings_model = embeddings_model

    def as_runnable(self) -> RunnableSerializable:
        doc_ids = []
        split_docs = []
        for doc in self.documents:
            doc_id = str(uuid.uuid4())
            sub_docs = self.text_splitter.split_documents([doc])
            for sub_doc in sub_docs:
                sub_doc.metadata[self.id_key] = doc_id
            doc_ids.append(doc_id)
            split_docs.extend(sub_docs)

        store = vector_store_from_documents(
            self.vectorstore, split_docs, self.embeddings_model)
        retriever = LangChainMultiVectorRetriever(
            vectorstore=store,
            byte_store=self.parentstore,
            id_key=self.id_key
        )
        retriever.docstore.mset(list(zip(doc_ids, self.documents)))

        return retriever
