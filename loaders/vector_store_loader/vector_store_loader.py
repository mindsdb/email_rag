from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma, PGVector
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel
from settings import VectorStoreType, VectorStoreConfig


class VectorStoreFactory:
    @staticmethod
    def create(embeddings_model: Embeddings):
        settings = VectorStoreConfig()

        if settings.type == VectorStoreType.CHROMA:
            return Chroma(
                persist_directory=settings.persist_directory,
                collection_name=settings.collection_name,
                embedding_function=embeddings_model,
            )
        elif settings.type == VectorStoreType.PGVECTOR:
            return PGVector.from_existing_index(
                embedding=embeddings_model,
                **settings.dict()
            )
        else:
            raise ValueError(f"Invalid vector store type, must be one either {VectorStoreType.__members__.keys()}")


class VectorStoreLoader(BaseModel):
    embeddings_model: Embeddings
    vector_store: VectorStore = None

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"
        validate_assignment = True

    def load(self) -> VectorStore:
        self.vector_store = VectorStoreFactory.create(self.embeddings_model)
        return self.vector_store


def load_vector_store(embeddings_model: Embeddings) -> VectorStore:
    loader = VectorStoreLoader(embeddings_model=embeddings_model)
    return loader.load()
