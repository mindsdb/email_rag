from retrievers.auto_retriever import AutoRetriever
from retrievers.multi_vector_retriever import MultiVectorRetriever
from retrievers.sql_retriever import SQLRetriever


def get_retriever_instance(retriever_name, **kwargs):
    retrievers = {
        'multi_vector': MultiVectorRetriever,
        'auto': AutoRetriever,
        'sql': SQLRetriever,
    }

    if retriever_name in retrievers:
        return retrievers[retriever_name](**kwargs)
    else:
        raise ValueError(f'Invalid retriever name: {retriever_name}')
