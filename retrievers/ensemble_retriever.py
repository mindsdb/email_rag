import re
from typing import Dict

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableSerializable
from langchain.retrievers import EnsembleRetriever as LangChainEnsembleRetriever
from retrievers.base import BaseRetriever
from settings import (DEFAULT_LLM,
                      DEFAULT_EMBEDDINGS
                      )


class EnsembleRetriever(BaseRetriever):
    """
    A retriever that combines multiple retrievers together with weighted results
    """

    def __init__(self,
                 runnable_retrievers,
                 llm: BaseChatModel = DEFAULT_LLM,
                 embeddings_model: Embeddings = DEFAULT_EMBEDDINGS
                 ):
        self.llm = llm
        self.embeddings_model = embeddings_model
        self.retrievers = []
        self.weights = []
        self.set_retrievers_weights(runnable_retrievers)

    def set_retrievers_weights(self, runnable_retrievers):
        self.retrievers = []
        self.weights = []
        for config in runnable_retrievers:
            self.retrievers.append(config['runnable'])
            self.weights.append(config['weight'])

    def as_runnable(self) -> RunnableSerializable:

        return (
            LangChainEnsembleRetriever(
                retrievers=self.retrievers, weights=self.weights)
        )
