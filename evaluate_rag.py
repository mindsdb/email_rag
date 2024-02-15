from datetime import datetime
from pathlib import Path
import argparse
import logging
import os
import platform
from enum import Enum
from typing import Union

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore

from evaluate import evaluate
from ingestors.email_ingestor.email_client import EmailClient
from ingestors.email_ingestor.email_ingestor import EmailIngestor
from ingestors.email_ingestor.email_search_options import EmailSearchOptions
from ingestors.file_ingestor import FileIngestor
from pipelines.rag_pipelines import LangChainRAGPipeline
from settings import DEFAULT_LLM, DEFAUlT_VECTOR_STORE, DEFAULT_EVALUATION_PROMPT_TEMPLATE, DEFAULT_EMBEDDINGS, \
    DEFAULT_AUTO_META_PROMPT_TEMPLATE, DEFAULT_SQL_RETRIEVAL_PROMPT_TEMPLATE


# Define the type of retriever to use.
class RetrieverType(Enum):
    VECTOR_STORE = 'vector_store'
    AUTO = 'auto'
    SQL = 'sql'


class InputDataType(Enum):
    EMAIL = 'email'
    FILE = 'file'


def ingest_files(dataset: str):
    # Define the path to the source files directory
    source_files_path = Path('./data') / dataset / 'source_files'

    source_files = []

    for f in source_files_path.iterdir():

        if f.is_file():
            source_files.append(str(f))

    ingestor = FileIngestor(source_files)
    logging.info('Loading documents from {}'.format(source_files_path))
    all_documents = ingestor.ingest()
    logging.info('Documents loaded')

    return all_documents


def ingest_emails():
    username = os.getenv('EMAIL_USERNAME')
    password = os.getenv('EMAIL_PASSWORD')
    email_client = EmailClient(username, password)

    # Use default search options to get emails from the last 10 days.
    search_options = EmailSearchOptions(
        mailbox='INBOX',
        subject=None,
        to_email=None,
        from_email=None,
        since_date=None,
        until_date=None,
        since_email_id=None
    )
    email_ingestor = EmailIngestor(email_client, search_options)
    logging.info('Ingesting emails')
    all_documents = email_ingestor.ingest()
    logging.info('Ingested')

    return all_documents


def evaluate_rag(dataset: str,
                 db_connection_dict: dict = None,
                 vector_store: VectorStore = DEFAUlT_VECTOR_STORE,
                 retriever: BaseRetriever = None,
                 llm: BaseChatModel = DEFAULT_LLM,
                 embeddings_model: Embeddings = DEFAULT_EMBEDDINGS,
                 rag_prompt_template: str = DEFAULT_EVALUATION_PROMPT_TEMPLATE,
                 retriever_prompt_template: Union[str, dict] = None,
                 retriever_type: str = RetrieverType.VECTOR_STORE,
                 input_data_type: str = InputDataType.EMAIL
                 ):
    """
    Evaluates a RAG pipeline that answers questions from a dataset
    about various emails, depending on the dataset.

    :param dataset: str
    :param db_connection_dict: dict
    :param vector_store: VectorStore
    :param retriever: BaseRetriever
    :param llm: BaseChatModel
    :param embeddings_model: Embeddings
    :param rag_prompt_template: str
    :param retriever_prompt_template: Union[str, dict]
    :param retriever_type: RetrieverType
    :param input_data_type: InputDataType

    :return:


    """
    if input_data_type == InputDataType.FILE:
        all_documents = ingest_files(dataset)

    elif input_data_type == InputDataType.EMAIL:

        all_documents = ingest_emails()

    else:
        raise ValueError('Invalid input data type, must be one of: file, email.')

    if retriever_type == RetrieverType.SQL:
        retriever_prompt_template = retriever_prompt_template or DEFAULT_SQL_RETRIEVAL_PROMPT_TEMPLATE

        rag_pipeline = LangChainRAGPipeline.from_sql_retriever(
            connection_dict=db_connection_dict,
            retriever_prompt_template=retriever_prompt_template,
            rag_prompt_template=rag_prompt_template,
            llm=llm
        )

    else:
        vectorstore = vector_store.from_documents(
            documents=all_documents,
            embedding=embeddings_model,
        )

        if retriever_type == RetrieverType.VECTOR_STORE:
            retriever = vectorstore.as_retriever()

            rag_pipeline = LangChainRAGPipeline.from_retriever(
                retriever=retriever,
                prompt_template=DEFAULT_EVALUATION_PROMPT_TEMPLATE,
                llm=llm
            )

        elif retriever_type == RetrieverType.AUTO:
            retriever_prompt_template = retriever_prompt_template or DEFAULT_AUTO_META_PROMPT_TEMPLATE

            rag_pipeline = LangChainRAGPipeline.from_auto_retriever(
                vectorstore=vectorstore,
                retriever_prompt_template=retriever_prompt_template,
                rag_prompt_template=DEFAULT_EVALUATION_PROMPT_TEMPLATE,
                llm=llm
            )

        else:
            raise ValueError('Invalid retriever type, must be one of: vector_store, auto, sql.')

    rag_chain = rag_pipeline.get_rag_chain()

    # Generate filename based on current datetime.
    dt_string = datetime.now().strftime('%d%m%Y_%H%M%S')
    output_file = f'evaluate_{input_data_type}_rag_{dt_string}.csv'

    qa_file = os.path.join('./data', dataset, 'rag_dataset.json')

    evaluate.evaluate(rag_chain, retriever, qa_file, output_file)


if __name__ == '__main__':
    # Initialize environment.
    if platform.system() == 'Windows':
        # Windows has a problem with asyncio's default EventLoopPolicy (the ragas package uses asyncio).
        # See: https://stackoverflow.com/questions/45600579/asyncio-event-loop-is-closed-when-getting-loop
        import asyncio

        asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy())

    # To use, set the environment variables:
    # OPENAI_API_KEY='<YOUR_API_KEY>'
    # EMAIL_USERNAME='<YOUR_EMAIL_USERNAME>'
    # EMAIL_PASSWORD='<YOUR_PASSWORD>'
    parser = argparse.ArgumentParser(
        prog='Evaluate RAG',
        description='''Evaluates the performance of a RAG pipeline with email or file.
Uses evaluation metrics from the RAGAs library.
        '''
    )
    parser.add_argument(
        '-d', '--dataset', help='Name of QA dataset to use for evaluation (e.g. personal_emails)')
    parser.add_argument(
        '-c', '--connection_dict', help='Connection string for SQL retriever', default=None)
    parser.add_argument('-r', '--retriever_type', help='Type of retriever to use (vector_store, auto, sql)',
                        default='vector_store')
    parser.add_argument('-i', '--input_data_type', help='Type of input data to use (email, file)',
                        default='email')
    parser.add_argument(
        '-l', '--log', help='Logging level to use (default WARNING)', default='WARNING')

    args = parser.parse_args()
    log_level = getattr(logging, args.log.upper())
    logging.basicConfig(level=log_level)

    evaluate_rag(
        dataset=args.dataset,
        db_connection_dict=args.connection_dict,
        retriever_type=args.retriever_type,
        input_data_type=args.input_data_type
    )
