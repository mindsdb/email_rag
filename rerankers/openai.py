from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pydantic import BaseModel
from langchain.schema import Document
from langchain_core.callbacks import Callbacks
from langchain.retrievers.document_compressors.base import BaseDocumentCompressor
import logging
import openai

from settings import DEFAULT_LLM_MODEL

log = logging.getLogger(__name__)


class Ranking(BaseModel):
    index: int
    relevance_score: float
    is_relevant: bool


class OpenAIReranker(BaseDocumentCompressor):
    _default_model: str = DEFAULT_LLM_MODEL

    top_n: int = 5  # Default number of documents to return
    model: str = DEFAULT_LLM_MODEL  # Model to use for reranking
    temperature: float = 0.0  # Temperature for the model
    openai_api_key: Optional[str] = None
    remove_irrelevant: bool = True  # New flag to control removal of irrelevant documents,
    # by default it will remove irrelevant documents

    _api_key_var: str = "OPENAI_API_KEY"
    client: Optional[Any] = None

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        """Initialize the OpenAI client after the model is fully initialized."""
        super().__init__()
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the OpenAI client if not already initialized."""
        if not self.client:
            api_key = self.openai_api_key or os.getenv(self._api_key_var)
            if not api_key:
                raise ValueError(
                    f"OpenAI API key must be provided either through the 'openai_api_key' parameter or the {self._api_key_var} environment variable."
                )
            openai.api_key = api_key
            self.client = openai

    def _get_client(self) -> Any:
        """Ensure client is initialized and return it."""
        if not self.client:
            self._initialize_client()
        return self.client

    def _rank_single_document(self, document: str, query: str) -> Tuple[bool, float]:
        client = self._get_client()

        prompt_template = f"""
Query: {query}

Document:
{document}

Is this document relevant to the query? Respond with either 'Relevant' or 'Not Relevant'.
"""

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role":"system", "content":"You are a document reranker."},
                    {"role":"user", "content":prompt_template}
                ],
                max_tokens=1,
                temperature=self.temperature,
                logprobs=True
            )

            # Extract the logprob for the last token (relevant or not relevant)
            last_token = response.choices[0].logprobs.content[0].token.strip().lower()

            is_relevant = last_token == 'relevant'
            relevance_score = float(response.choices[0].logprobs.content[0].logprob)

            log.debug(f"Document relevance: is_relevant={is_relevant}, score={relevance_score:.4f}")

            return is_relevant, relevance_score

        except Exception as e:
            log.error(f"Error in OpenAI API call: {e}")
            return False, 0.0  # Default to not relevant if API call fails

    def _rank(self, documents: List[str], query: str) -> List[Ranking]:
        log.info(f"Starting reranking process for {len(documents)} documents")
        rankings = []
        for idx, doc in enumerate(documents):
            is_relevant, relevance_score = self._rank_single_document(doc, query)
            rankings.append(Ranking(index=idx, relevance_score=relevance_score, is_relevant=is_relevant))

        log.info(f"Reranking complete. {len(rankings)} documents processed")

        if self.remove_irrelevant:
            rankings = [r for r in rankings if r.is_relevant]
            log.info(f"{len(rankings)} relevant documents found after filtering")

        if not rankings:
            log.warning("No relevant documents found after reranking and filtering")
            return []

        sorted_rankings = sorted(rankings, key=lambda x:x.relevance_score, reverse=True)[:self.top_n]
        log.info(f"Returning top {len(sorted_rankings)} documents")
        return sorted_rankings

    def compress_documents(
            self,
            documents: Sequence[Document],
            query: str,
            callbacks: Optional[Callbacks] = None,
    ) -> Sequence[Document]:
        """Compress documents using OpenAI's rerank capability with individual document assessment."""
        log.info(f"Compressing documents. Initial count: {len(documents)}")
        if len(documents) == 0 or self.top_n < 1:
            log.warning("No documents to compress or top_n < 1. Returning empty list.")
            return []

        doc_contents = [doc.page_content for doc in documents]
        rankings = self._rank(documents=doc_contents, query=query)

        compressed = []
        for ranking in rankings:
            doc = documents[ranking.index]
            doc.metadata["relevance_score"] = ranking.relevance_score
            doc.metadata["is_relevant"] = ranking.is_relevant
            compressed.append(doc)

        log.info(f"Compression complete. {len(compressed)} documents returned")
        if not compressed:
            log.warning("No documents found after compression")

        return compressed

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model":self.model,
            "top_n":self.top_n,
            "temperature":self.temperature,
            "remove_irrelevant":self.remove_irrelevant,
        }
