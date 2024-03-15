from langchain_core.embeddings import Embeddings

from langchain_community.vectorstores import Chroma, PGVector
from pydantic_settings import BaseSettings

from settings import VectorStoreType


class BaseVectorStoreConfig(BaseSettings):
    pass

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"


class ChromaConfig(BaseVectorStoreConfig):
    persist_directory: str
    collection_name: str


class PgVectorConfig(BaseVectorStoreConfig):
    connection_string: str
    collection_name: str


class VectorDBLoader:
    """
    Encapsulates the logic for loading a vector store.
    """
    def __init__(
            self,
            config: dict,
            embeddings_model: Embeddings,
            vector_store_type: VectorStoreType = VectorStoreType.CHROMA
    ):
        self.config = config
        self.embeddings_model = embeddings_model
        self.vector_store_type = vector_store_type
        self.vector_storage = None

        self.load()

    def load(self):
        """
        Load the vector store based on the vector store type.
        :return:
        """
        if self.vector_store_type == VectorStoreType.CHROMA:
            config = ChromaConfig(**self.config)
            self.vector_storage = Chroma(
                persist_directory=config.persist_directory,
                collection_name=config.collection_name,
                embeddings_model=self.embeddings_model
            )
        elif self.vector_store_type == VectorStoreType.PGVECTOR:
            config = PgVectorConfig(**self.config)
            self.vector_storage = PGVector.from_existing_index(
                embedding=self.embeddings_model, **config.dict()
            )
        else:
            raise ValueError(f'Invalid vector store type, must be one either {VectorStoreType.__members__.keys()}')
