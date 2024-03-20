from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma, PGVector
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, parse_obj_as
from settings import VectorStoreType, VectorStoreConfig


class VectorStoreFactory:
    @staticmethod
    def create(embeddings_model: Embeddings, config: dict = None):
        if config:
            settings = parse_obj_as(VectorStoreConfig, config)
        else:
            settings = VectorStoreConfig()
        if settings.type == VectorStoreType.CHROMA:
            return VectorStoreFactory._create_chroma_vectorstore(embeddings_model, settings)
        elif settings.type == VectorStoreType.PGVECTOR:
            return VectorStoreFactory._create_pgvector_vectorstore(embeddings_model, settings)
        else:
            raise ValueError(f"Invalid vector store type, must be one either {VectorStoreType.__members__.keys()}")

    @staticmethod
    def _create_chroma_vectorstore(embeddings_model: Embeddings, settings):
        return Chroma(
            persist_directory=settings.persist_directory,
            collection_name=settings.collection_name,
            embedding_function=embeddings_model,
        )

    @staticmethod
    def _create_pgvector_vectorstore(embeddings_model: Embeddings, settings):
        return PGVector.from_existing_index(
            embedding=embeddings_model,
            **settings.dict()
        )


class VectorStoreLoader(BaseModel):
    embeddings_model: Embeddings
    vector_store: VectorStore = None
    config: dict = None

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"
        validate_assignment = True

    def load(self) -> VectorStore:
        self.vector_store = VectorStoreFactory.create(self.embeddings_model, self.config)
        return self.vector_store


def load_vector_store(embeddings_model: Embeddings, config: dict={}) -> VectorStore:
    loader = VectorStoreLoader(embeddings_model=embeddings_model, config=config)
    return loader.load()
