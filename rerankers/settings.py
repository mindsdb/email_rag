import logging
from langchain.retrievers.document_compressors import CohereRerank, CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_nvidia_ai_endpoints import NVIDIARerank
from pydantic import BaseModel, Field
from typing import Union, Any
from enum import Enum

from rerankers.openai import OpenAIReranker

logger = logging.getLogger(__name__)


class ReRankerType(str, Enum):
    DISABLED = "disabled"
    OPENAI_LOGPROBS = "openai_logprobs"
    NVIDIA = "nvidia"
    COHERE = "cohere"
    CROSS_ENCODER = "cross_encoder"


class BaseReRankerConfig(BaseModel):
    type: ReRankerType = ReRankerType.DISABLED
    top_n: int = 3


class OpenAILogProbsConfig(BaseReRankerConfig):
    type: ReRankerType = ReRankerType.OPENAI_LOGPROBS
    model: str = Field('gpt-4o', description="OpenAI model to use for reranking")
    remove_irrelevant: bool = Field(True, description="Remove irrelevant documents")



class NvidiaConfig(BaseReRankerConfig):
    type: ReRankerType = ReRankerType.NVIDIA
    model: str = Field(..., description="NVIDIA model to use for reranking")
    base_url: str = Field(..., description="Base URL for NVIDIA model listing and invocation")


class CohereConfig(BaseReRankerConfig):
    type: ReRankerType = ReRankerType.COHERE
    model: str = Field(..., description="Cohere model to use for reranking")


class CrossEncoderConfig(BaseReRankerConfig):
    type: ReRankerType = ReRankerType.CROSS_ENCODER
    model_name: str = Field(None, description="HuggingFace model to use for cross-encoding")


class ReRankerConfig(BaseModel):
    config: Union[BaseReRankerConfig, OpenAILogProbsConfig, NvidiaConfig, CohereConfig, CrossEncoderConfig]

    @classmethod
    def create(cls, reranker_type: Union[ReRankerType, str], **kwargs) -> 'ReRankerConfig':
        config_map = {
            ReRankerType.DISABLED:BaseReRankerConfig,
            ReRankerType.OPENAI_LOGPROBS:OpenAILogProbsConfig,
            ReRankerType.NVIDIA:NvidiaConfig,
            ReRankerType.COHERE:CohereConfig,
            ReRankerType.CROSS_ENCODER:CrossEncoderConfig,
        }

        logger.debug(f"Creating ReRankerConfig with type: {reranker_type}, kwargs: {kwargs}")
        return cls(config=config_map[reranker_type.value](**kwargs))

def get_reranker(config: ReRankerConfig) -> Any:
    logger.debug(f"Getting reranker for config type: {config.config.type}")

    if config.config.type == ReRankerType.DISABLED:
        return None
    elif config.config.type == ReRankerType.OPENAI_LOGPROBS:
        return OpenAIReranker(model=config.config.model, top_n=config.config.top_n)
    elif config.config.type == ReRankerType.NVIDIA:
        return NVIDIARerank(model=config.config.model, top_n=config.config.top_n, base_url=config.config.base_url)
    elif config.config.type == ReRankerType.COHERE:
        return CohereRerank(model=config.config.model, top_n=config.config.top_n)
    elif config.config.type == ReRankerType.CROSS_ENCODER:
        model = HuggingFaceCrossEncoder(model_name=config.config.model_name)
        return CrossEncoderReranker(model=model, top_n=config.config.top_n)
    else:
        logger.error(f"Unsupported reranker type: {config.config.type}")
        raise ValueError(f"Unsupported reranker type: {config.config.type}")
