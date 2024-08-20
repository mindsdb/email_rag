import json
import logging
from typing import Dict, Tuple
from datetime import datetime

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
_DEFAULT_METRICS = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']

def evaluate(
        pipeline: RunnableSerializable,
        qa_dataset: Dict,
        output_path: str) -> Tuple[DataFrame, DataFrame]:
    '''
    Evaluates a RAG pipeline against the given Q&A dataset using the RAGAs library.

    Parameters:
        pipeline (RunnableSerializable): The RAG pipeline to run
        qa_dataset (Dict): Dictionary containing questions & answers.
            MUST follow format: {
                'examples': [{'query': '...', 'reference_answer': '...'}]
            }
        output_path (str): Path to output evaluation results
    Returns:
        Tuple[DataFrame, DataFrame]: Two DataFrames representing evaluation results:
            1. Scores for each individual example
            2. Summary statistics for all examples
    '''
    examples = qa_dataset['examples']
    questions = []
    ground_truths = []
    for example in examples:
        questions.append(example['query'])
        ground_truths.append(example['reference_answer'])

    answers = []
    contexts = []
    reranked_docs = []
    num_questions = len(questions)
    for i, question in enumerate(questions):
        logging.info(f'Answering question {i + 1} of {num_questions}')
        result = pipeline.invoke(question)
        answers.append(result['answer'])
        contexts.append([doc.page_content for doc in result['context']])
        if 'reranked_docs' in result:
            reranked_docs.append(result['reranked_docs'])

    data = {
        'question': questions,
        'answer': answers,
        'contexts': contexts,
        'ground_truth': ground_truths  # Changed from 'ground_truths' to 'ground_truth'
    }
    if reranked_docs:
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
        run_config=RunConfig(timeout=None, max_retries=_DEFAULT_MAX_RAGAS_RETRIES,
                             max_wait=_DEFAULT_MAX_RAGAS_WAIT_TIME_SECS),
        raise_exceptions=False
    )

    # Individual scores DataFrame
    individual_scores_df = result.to_pandas()
    individual_scores_df.fillna(0.0, inplace=True)

    # Summary statistics DataFrame
    summary_df = DataFrame({
        'metric': _DEFAULT_METRICS,
        'mean': individual_scores_df[_DEFAULT_METRICS].mean(),
        'median': individual_scores_df[_DEFAULT_METRICS].median(),
        'std': individual_scores_df[_DEFAULT_METRICS].std(),
        'min': individual_scores_df[_DEFAULT_METRICS].min(),
        'max': individual_scores_df[_DEFAULT_METRICS].max(),
    })

    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save results to CSV with timestamp
    individual_scores_path = f"{output_path}_individual_scores_{timestamp}.csv"
    summary_path = f"{output_path}_summary_{timestamp}.csv"

    individual_scores_df.to_csv(individual_scores_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    logging.info(f"Individual scores saved to: {individual_scores_path}")
    logging.info(f"Summary statistics saved to: {summary_path}")

    return individual_scores_df, summary_df
