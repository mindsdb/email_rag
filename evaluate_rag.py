from datetime import datetime
from pathlib import Path
import argparse
import logging
import os
import platform
from enum import Enum
from typing import List, Union

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from sqlalchemy import create_engine

from retrievers.multi_vector_retriever import MultiVectorRetrieverMode

from evaluate import evaluate
from loaders.email_loader.email_client import EmailClient
from loaders.email_loader.email_loader import EmailLoader, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from loaders.email_loader.email_search_options import EmailSearchOptions
from loaders.directory_loader.directory_loader import DirectoryLoader
from loaders.vector_store_loader.vector_store_loader import VectorDBLoader
from pipelines.rag_pipelines import LangChainRAGPipeline
from settings import (DEFAULT_LLM,
                      DEFAUlT_VECTOR_STORE,
                      DEFAULT_EVALUATION_PROMPT_TEMPLATE,
                      DEFAULT_EMBEDDINGS,
                      DEFAULT_CONTENT_COLUMN_NAME,
                      DEFAULT_DATASET_DESCRIPTION,
                      DEFAULT_TEST_TABLE_NAME,
                      DEFAULT_POOL_RECYCLE,
                      VectorStoreType
                      )
from utils import documents_to_df, VectorStoreOperator
from visualize.visualize import visualize_evaluation_metrics


# Define the type of retriever to use.
class RetrieverType(Enum):
    VECTOR_STORE = 'vector_store'
    AUTO = 'auto'
    SQL = 'sql'
    MULTI = 'multi'


class InputDataType(Enum):
    EMAIL = 'email'
    FILE = 'file'
    VECTOR_STORE = 'vector_store'


def ingest_files(dataset: str, split_documents: bool = True) -> List[Document]:
    # Define the path to the source files directory
    source_files_path = Path('./data') / dataset / 'source_files'

    source_files = []

    for f in source_files_path.iterdir():

        if f.is_file():
            source_files.append(str(f))

    directory_loader = DirectoryLoader(source_files)
    logging.info('Loading documents from {}'.format(source_files_path))
    if split_documents:
        all_documents = directory_loader.load_and_split()
    else:
        all_documents = directory_loader.load()
    logging.info('Documents loaded')

    return all_documents


def ingest_emails(split_documents: bool = True) -> List[Document]:
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
    email_loader = EmailLoader(email_client, search_options)
    logging.info('Ingesting emails')
    if split_documents:
        all_documents = email_loader.load_and_split()
    else:
        all_documents = email_loader.load()
    logging.info('Ingested')

    return all_documents


def load_vector_store(
        embeddings_model: Embeddings,
        vector_store_config: dict,
        vector_store_type: VectorStoreType
) -> VectorStore:
    vector_store_loader = VectorDBLoader(embeddings_model=embeddings_model,
                                         config=vector_store_config,
                                         vector_store_type=vector_store_type)
    return vector_store_loader.vector_storage


def _ingest_documents(input_data_type: InputDataType, dataset: str, split_documents: bool = True) -> List[Document]:
    if input_data_type == InputDataType.FILE:
        return ingest_files(dataset, split_documents)
    if input_data_type == InputDataType.EMAIL:
        return ingest_emails(split_documents)
    if input_data_type == InputDataType.VECTOR_STORE:
        return
    raise ValueError(
        f'Invalid input data type, must be one of: file, email or vector_store. Got {input_data_type}')


def _get_pipeline_from_retriever(
        all_documents: List[Document],
        content_column_name: str = DEFAULT_CONTENT_COLUMN_NAME,
        dataset_description: str = DEFAULT_DATASET_DESCRIPTION,
        db_connection_string: str = None,
        test_table_name: str = DEFAULT_TEST_TABLE_NAME,
        vector_store: VectorStore = DEFAUlT_VECTOR_STORE,
        llm: BaseChatModel = DEFAULT_LLM,
        embeddings_model: Embeddings = DEFAULT_EMBEDDINGS,
        rag_prompt_template: str = DEFAULT_EVALUATION_PROMPT_TEMPLATE,
        retriever_prompt_template: Union[str, dict] = None,
        retriever_type: RetrieverType = RetrieverType.VECTOR_STORE,
        multi_retriever_mode: MultiVectorRetrieverMode = MultiVectorRetrieverMode.BOTH
) -> LangChainRAGPipeline:
    if retriever_type == RetrieverType.SQL:
        documents_df = documents_to_df(content_column_name,
                                       all_documents,
                                       embeddings_model=embeddings_model,
                                       with_embeddings=True)

        # Save the dataframe to a SQL table.
        alchemyEngine = create_engine(
            db_connection_string, pool_recycle=DEFAULT_POOL_RECYCLE)
        db_connection = alchemyEngine.connect()

        # issues with langchain compatibility with vector type in postgres need to investigate further
        documents_df.to_sql(test_table_name, db_connection,
                            index=False, if_exists='replace')

        return LangChainRAGPipeline.from_sql_retriever(
            connection_string=db_connection_string,
            retriever_prompt_template=retriever_prompt_template,
            rag_prompt_template=rag_prompt_template,
            llm=llm
        )

    if retriever_type == RetrieverType.VECTOR_STORE:
        vector_store_operator = VectorStoreOperator(
            vector_store=vector_store,
            documents=all_documents,
            embeddings_model=embeddings_model
        )
        return LangChainRAGPipeline.from_retriever(
            retriever=vector_store_operator.vector_store.as_retriever(),
            prompt_template=DEFAULT_EVALUATION_PROMPT_TEMPLATE,
            llm=llm
        )

    if retriever_type == RetrieverType.AUTO:
        return LangChainRAGPipeline.from_auto_retriever(
            vectorstore=vector_store,
            data=all_documents,
            data_description=dataset_description,
            content_column_name=content_column_name,
            retriever_prompt_template=retriever_prompt_template,
            rag_prompt_template=DEFAULT_EVALUATION_PROMPT_TEMPLATE,
            llm=llm
        )

    if retriever_type == RetrieverType.MULTI:
        # The splitter to use to embed smaller chunks
        child_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
        return LangChainRAGPipeline.from_multi_vector_retriever(
            documents=all_documents,
            rag_prompt_template=DEFAULT_EVALUATION_PROMPT_TEMPLATE,
            vectorstore=vector_store,
            text_splitter=child_text_splitter,
            llm=llm,
            mode=multi_retriever_mode
        )

    raise ValueError(
        f'Invalid retriever type, must be one of: vector_store, auto, sql, multi. Got {retriever_type}')


def evaluate_rag(dataset: str,
                 content_column_name: str = DEFAULT_CONTENT_COLUMN_NAME,
                 dataset_description: str = DEFAULT_DATASET_DESCRIPTION,
                 db_connection_string: str = None,
                 test_table_name: str = DEFAULT_TEST_TABLE_NAME,
                 vector_store: VectorStore = DEFAUlT_VECTOR_STORE,
                 llm: BaseChatModel = DEFAULT_LLM,
                 embeddings_model: Embeddings = DEFAULT_EMBEDDINGS,
                 rag_prompt_template: str = DEFAULT_EVALUATION_PROMPT_TEMPLATE,
                 retriever_prompt_template: Union[str, dict] = None,
                 retriever_type: RetrieverType = RetrieverType.VECTOR_STORE,
                 input_data_type: InputDataType = InputDataType.EMAIL,
                 show_visualization = False,
                 split_documents = True,
                 multi_retriever_mode: MultiVectorRetrieverMode = MultiVectorRetrieverMode.BOTH,
                 vector_store_type: VectorStoreType = VectorStoreType.CHROMA,
                 vector_store_config: dict = None
                 ):
    """
    Evaluates a RAG pipeline that answers questions from a dataset
    about various emails, depending on the dataset.

    :param vector_store_type:
    :param vector_store_config:
    :param dataset: str
    :param content_column_name: str
    :param dataset_description: str
    :param db_connection_string: str
    :param test_table_name: str
    :param vector_store: VectorStore
    :param llm: BaseChatModel
    :param embeddings_model: Embeddings
    :param rag_prompt_template: str
    :param retriever_prompt_template: Union[str, dict]
    :param retriever_type: RetrieverType
    :param input_data_type: InputDataType
    :param show_visualization: bool
    :param split_documents: bool
    :param multi_retriever_mode: MultiVectorRetrieverMode
    :param vector_store_type: VectorStoreType
    :param vector_store_config: dict

    :return:
    """
    all_documents = _ingest_documents(
        input_data_type, dataset, split_documents=split_documents)

    if all_documents is None and vector_store_config is None:
        raise ValueError(
            'No valid input data or vector store config provided. At least one must be provided.')

    if vector_store_config:
        vector_store = load_vector_store(
            embeddings_model, vector_store_config, vector_store_type)

    rag_pipeline = _get_pipeline_from_retriever(
        all_documents,
        content_column_name=content_column_name,
        dataset_description=dataset_description,
        db_connection_string=db_connection_string,
        test_table_name=test_table_name,
        vector_store=vector_store,
        llm=llm,
        embeddings_model=embeddings_model,
        rag_prompt_template=rag_prompt_template,
        retriever_prompt_template=retriever_prompt_template,
        retriever_type=retriever_type,
        multi_retriever_mode=multi_retriever_mode
    )

    rag_chain = rag_pipeline.rag_with_returned_sources()

    # Generate filename based on current datetime.
    dt_string = datetime.now().strftime('%d%m%Y_%H%M%S')
    output_file = f'evaluate_{input_data_type.value}_{retriever_type.value}_rag_{dt_string}.csv'

    qa_file = os.path.join('./data', dataset, 'rag_dataset.json')

    evaluation_df = evaluate.evaluate(
        rag_chain, qa_file, output_file)
    if show_visualization:
        visualize_evaluation_metrics(output_file, evaluation_df)


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
    parser.add_argument('-dd', '--dataset_description', help='Description of the dataset',
                        default=DEFAULT_DATASET_DESCRIPTION)
    parser.add_argument(
        '-c', '--connection_string', help='Connection string for SQL retriever', default=None)
    parser.add_argument('-cc', '--content_column_name',
                        help='Name of the column containing the content i.e. body of the email',
                        default=DEFAULT_CONTENT_COLUMN_NAME)
    parser.add_argument('-t', '--test_table_name', help='Name of the table to use for testing '
                                                        '(only for SQL retriever)',
                        default=DEFAULT_TEST_TABLE_NAME)
    parser.add_argument('-r', '--retriever_type', help='Type of retriever to use (vector_store, auto, sql, multi)',
                        type=RetrieverType, choices=list(RetrieverType), default=RetrieverType.VECTOR_STORE)
    parser.add_argument("-mr", "--multi_retriever_mode", help="Mode to use for multi retriever",
                        type=MultiVectorRetrieverMode, choices=list(MultiVectorRetrieverMode),
                        default=MultiVectorRetrieverMode.BOTH)
    parser.add_argument('-i', '--input_data_type', help='Type of input data to use (email, file)',
                        type=InputDataType, choices=list(InputDataType), default=InputDataType.EMAIL)
    parser.add_argument('-v', '--show_visualization', type=bool,
                        help='Whether or not to plot and show evaluation metrics', default=False)
    parser.add_argument('-s', '--split_documents', type=bool,
                        help='Whether or not to split documents after they are loaded',
                        default=True)
    parser.add_argument('-vs', '--vector_store_config', help='Configuration for vector store',
                        default=None, type=str)
    parser.add_argument('-vt', '--vector_store_type', help='Type of vector store to use',
                        type=VectorStoreType, choices=list(VectorStoreType), default=VectorStoreType.CHROMA)
    parser.add_argument(
        '-l', '--log', help='Logging level to use (default WARNING)', default='WARNING')

    args = parser.parse_args()
    log_level = getattr(logging, args.log.upper())
    logging.basicConfig(level=log_level)

    logger = logging.getLogger(__name__)

    if args.vector_store_config:
        # convert 'vector_store_config' string to dictionary
        args.vector_store_config = eval(args.vector_store_config)
        # it is not currently possible to use other retriever types when loading in existing vector store

        logger.warning(
            'Vector store config provided, setting retriever type to vector store as other types '
            'are not currently supported.')

        args.retriever_type = RetrieverType.VECTOR_STORE
        # convert 'vector_store_type' string to VectorStoreType enum
        if 'vector_store_type' in args.vector_store_config:
            args.vector_store_type = VectorStoreType(args.vector_store_config.pop('vector_store_type'))

    logger.warning(f'Evaluating RAG pipeline with dataset: {args.dataset}')

    evaluate_rag(dataset=args.dataset, content_column_name=args.content_column_name,
                 dataset_description=args.dataset_description, db_connection_string=args.connection_string,
                 test_table_name=args.test_table_name, retriever_type=args.retriever_type,
                 input_data_type=args.input_data_type, show_visualization=args.show_visualization,
                 split_documents=args.split_documents, multi_retriever_mode=MultiVectorRetrieverMode.BOTH,
                 vector_store_config=args.vector_store_config, vector_store_type=args.vector_store_type)
