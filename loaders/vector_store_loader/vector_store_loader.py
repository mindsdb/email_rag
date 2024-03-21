import ast
import uuid

from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma, PGVector
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, parse_obj_as
from settings import VectorStoreType, VectorStoreConfig
import pandas as pd

from sqlalchemy import create_engine
from sqlalchemy.orm import scoped_session, sessionmaker
from sqlalchemy.exc import DisconnectionError

COL_ID = "id"
COL_EMBEDDINGS = "embeddings"
COL_METADATA = "metadata"
COL_CONTENT = "content"


class VectorStoreFactory:
    @staticmethod
    def create(embeddings_model: Embeddings, config: dict = None):
        if config:
            settings = parse_obj_as(VectorStoreConfig, config)
        else:
            settings = VectorStoreConfig()
        if settings.type == VectorStoreType.CHROMA:
            return VectorStoreFactory._load_chromadb_store(embeddings_model, settings)
        elif settings.type == VectorStoreType.PGVECTOR:
            return VectorStoreFactory._load_pgvector_store(embeddings_model, settings)
        else:
            raise ValueError(f"Invalid vector store type, must be one either {VectorStoreType.__members__.keys()}")

    @staticmethod
    def _load_chromadb_store(embeddings_model: Embeddings, settings) -> Chroma:
        return Chroma(
            persist_directory=settings.persist_directory,
            collection_name=settings.collection_name,
            embedding_function=embeddings_model,
        )

    @staticmethod
    def _load_pgvector_store(embeddings_model: Embeddings, settings) -> PGVector:
        empty_store = PGVector(
            connection_string=settings.connection_string,
            collection_name=settings.collection_name,
            embedding_function=embeddings_model,
            pre_delete_collection=True,
        )
        return VectorStoreFactory._load_data_into_langchain_pgvector(settings, empty_store)

    @staticmethod
    def _load_data_into_langchain_pgvector(settings, vectorstore: PGVector) -> PGVector:
        df = VectorStoreFactory._fetch_data_from_db(settings)

        df[COL_EMBEDDINGS] = df[COL_EMBEDDINGS].apply(ast.literal_eval)
        df[COL_METADATA] = df[COL_METADATA].apply(ast.literal_eval)

        metadata = df[COL_METADATA].tolist()
        embeddings = df[COL_EMBEDDINGS].tolist()
        texts = df[COL_CONTENT].tolist()
        ids = [str(uuid.uuid1()) for _ in range(len(df))] if COL_ID not in df.columns else df[COL_ID].tolist()

        vectorstore.add_embeddings(
            texts=texts,
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids
        )
        return vectorstore

    @staticmethod
    def _fetch_data_from_db(settings) -> pd.DataFrame:
        try:
            engine = create_engine(settings.connection_string)
            db = scoped_session(sessionmaker(bind=engine))

            df = pd.read_sql(f"SELECT * FROM {settings.collection_name}", engine)

            return df
        except DisconnectionError as e:
            print("Unable to connect to the database. Please check your connection string and try again.")
            raise e
        finally:
            db.close()


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
