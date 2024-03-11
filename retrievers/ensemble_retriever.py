from typing import List

from langchain_core.runnables import RunnableSerializable
from langchain.retrievers.ensemble import EnsembleRetriever as LangChainEnsembleRetriever
from pydantic import BaseModel


from retrievers.base import BaseRetriever


class EnsembleRetrieverConfig(BaseModel):
    retrievers: List[BaseRetriever]
    weights: List[float] = None  # weighting for each retriever as equal if not provided

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"


class EnsembleRetriever(BaseRetriever):

    """
    EnsembleRetriever creates an ensemble of multiple retrievers.
    """

    def __init__(
            self,
            ensemble_config: EnsembleRetrieverConfig

    ):
        self.ensemble_config = ensemble_config

    def as_runnable(self) -> RunnableSerializable:
        return LangChainEnsembleRetriever(**self.ensemble_config.dict())
