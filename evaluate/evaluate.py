import json
import logging

from datasets import Dataset
from langchain_core.runnables.base import RunnableSerializable
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas.run_config import RunConfig
from pandas import DataFrame

from callback_handlers.log_callback_handler import LogCallbackHandler
from settings import DEFAULT_LLM, DEFAULT_EMBEDDINGS

_DEFAULT_MAX_RAGAS_RETRIES = 20
_DEFAULT_MAX_RAGAS_WAIT_TIME_SECS = 500


def evaluate(
        pipeline: RunnableSerializable,
        qa_dataset: str,
        output_path: str) -> DataFrame:
    '''
    Evaluates a RAG pipeline against the given Q&A dataset using the RAGAs library.

    Parameters:
        pipeline (RunnableSerializable): The RAG pipeline to run
        qa_dataset_path (str): Path to the JSON file containing questions & answers.
            MUST follow format: {
                'examples': [{'query': '...', 'reference_answer': '...'}]
            }
        output_path (str): Path to output evaluation results
    Returns:
        df (DataFrame): DataFrame representing evaluation results
    '''
    examples = qa_dataset['examples']
    questions = []
    ground_truths = []
    for example in examples:
        questions.append(example['query'])
        ground_truths.append([example['reference_answer']])

    answers = []
    contexts = []
    reranked_docs = []
    num_questions = len(questions)
    for i, question in enumerate(questions):
        logging.info('Answering question {} of {}'.format(
            i + 1, num_questions))
        result = pipeline.invoke(question)
        answers.append(result['answer'])
        contexts.append(
            [docs.page_content for docs in result['context']])
        if 'reranked_docs' in result:
            reranked_docs.append(result['reranked_docs'])

    data = {
        'question': questions,
        'answer': answers,
        'contexts': contexts,
        'ground_truths': ground_truths
    }
    if len(reranked_docs) != 0:
        data['reranked_docs'] = reranked_docs
    dataset = Dataset.from_dict(data)

    result = ragas_evaluate(
        dataset=dataset,
        llm=DEFAULT_LLM,
        embeddings=DEFAULT_EMBEDDINGS,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
        callbacks=[LogCallbackHandler('evaluate')],
        # More generous RunConfig since we could be sending many requests.
        run_config=RunConfig(timeout=None, max_retries=_DEFAULT_MAX_RAGAS_RETRIES,
                             max_wait=_DEFAULT_MAX_RAGAS_WAIT_TIME_SECS),
        raise_exceptions=False
    )

    df = result.to_pandas()
    df.fillna(0.0)
    df.to_csv(output_path)
    return df
