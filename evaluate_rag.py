import argparse
import json
import logging
import os
import platform
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
import pathlib
from typing import Dict, List, Optional, Type, Union

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from sqlalchemy import create_engine
from tqdm import tqdm

from evaluate import evaluate
from loaders.directory_loader.directory_loader import DirectoryLoader
from loaders.email_loader.email_client import EmailClient
from loaders.email_loader.email_loader import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, EmailLoader
from loaders.email_loader.email_search_options import EmailSearchOptions
from loaders.vector_store_loader.vector_store_loader import load_vector_store
from pipelines.rag_pipelines import LangChainRAGPipeline
from retrievers.multi_vector_retriever import MultiVectorRetrieverMode
from settings import (DEFAULT_CONTENT_COLUMN_NAME,
                      DEFAULT_DATASET_DESCRIPTION, DEFAULT_EMBEDDINGS, DEFAULT_EVALUATION_PROMPT_TEMPLATE,
                      DEFAULT_LLM, DEFAULT_POOL_RECYCLE, DEFAULT_TEST_TABLE_NAME, DEFAUlT_VECTOR_STORE,
                      InputDataType, RetrieverType, ReRankerType)
from utils import VectorStoreOperator, documents_to_df
from visualize.visualize import visualize_evaluation_metrics

_DEFAULT_ID_KEY = "doc_id"
_SPLIT_DOCS = False
MAX_WORKERS = 5  # Adjust based on your system's capabilities

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_vector_store_and_operator(documents: List[Document], vector_store_class: Type[VectorStore],
                                     embeddings_model: Embeddings = DEFAULT_EMBEDDINGS) -> VectorStoreOperator:
    logger.info(f'Creating vector store with {len(documents)} documents')

    # Remove duplicate documents
    unique_docs = []
    seen_texts = set()
    for doc in documents:
        if doc.page_content not in seen_texts:
            unique_docs.append(doc)
            seen_texts.add(doc.page_content)

    logger.info(f'Found {len(unique_docs)} unique documents')

    # Create the vector store
    logger.info(f'Creating {vector_store_class.__name__} vector store')
    persist_directory = f'{vector_store_class.__name__.lower()}_db'  # You might want to make this configurable

    # Initialize the vector store
    vector_store = vector_store_class(
        embedding_function=embeddings_model,
        persist_directory=persist_directory
    )

    # Create VectorStoreOperator
    vector_store_operator = VectorStoreOperator(
        vector_store=vector_store,
        documents=unique_docs,
        embeddings_model=embeddings_model
    )

    logger.info('Vector store and operator creation complete')
    return vector_store_operator


def load_vector_store_and_operator(vector_store_class: Type[VectorStore],
                                   embeddings_model: Embeddings) -> VectorStoreOperator:
    logger.info(f'Loading existing {vector_store_class.__name__} vector store')
    persist_directory = f'{vector_store_class.__name__.lower()}_db'
    if not os.path.exists(persist_directory):
        raise ValueError(f"No existing vector store found at {persist_directory}")

    vector_store = vector_store_class(
        embedding_function=embeddings_model,
        persist_directory=persist_directory
    )

    vector_store_operator = VectorStoreOperator(
        vector_store=vector_store,
        embeddings_model=embeddings_model
    )

    logger.info('Vector store and operator loaded successfully')
    return vector_store_operator


@dataclass
class GetPipelineArgs:
    all_documents: List[Document]
    content_column_name: str
    dataset_description: str
    db_connection_string: str
    test_table_name: str
    vector_store_operator: VectorStoreOperator
    llm: BaseChatModel
    embeddings_model: Embeddings
    rag_prompt_template: str
    retriever_prompt_template: Union[str, dict]
    retriever_type: RetrieverType
    rerank_documents: ReRankerType
    multi_retriever_mode: MultiVectorRetrieverMode
    retriever_map: dict
    split_documents: bool = True
    _split_docs: List[Document] = field(default_factory=list)
    _doc_ids: List[str] = field(init=False)

    def __post_init__(self):
        self._doc_ids = [str(uuid.uuid4()) for _ in self.all_documents]
        if self.split_documents and self.all_documents and not self._split_docs:
            self._split_documents()
        elif not self.split_documents:
            self._split_docs = self.all_documents
            for doc, doc_id in zip(self._split_docs, self._doc_ids):
                doc.metadata[_DEFAULT_ID_KEY] = doc_id

    @property
    def doc_ids(self) -> List[str]:
        return self._doc_ids

    @property
    def split_docs(self) -> List[Document]:
        return self._split_docs

    def _split_documents(self) -> None:
        """
        Split the documents into sub-documents and generate unique ids for each document.
        """
        child_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)

        for doc, doc_id in zip(self.all_documents, self._doc_ids):
            sub_docs = child_text_splitter.split_documents([doc])
            for sub_doc in sub_docs:
                sub_doc.metadata[_DEFAULT_ID_KEY] = doc_id
            self._split_docs.extend(sub_docs)

        # Update the VectorStoreOperator with the split documents
        self.vector_store_operator.documents = self._split_docs

    def _generate_id_and_split_document(self, doc: Document) -> tuple[str, list[Document]]:
        """
        Generate a unique id for the document and split it into sub-documents.
        """
        child_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
        doc_id = str(uuid.uuid4())
        sub_docs = child_text_splitter.split_documents([doc])
        for sub_doc in sub_docs:
            sub_doc.metadata[_DEFAULT_ID_KEY] = doc_id
        return doc_id, sub_docs


def create_vector_store_and_operator(documents: List[Document], vector_store_class: Type[VectorStore],
                                     embeddings_model: Embeddings = DEFAULT_EMBEDDINGS) -> VectorStoreOperator:
    logger.info(f'Creating vector store with {len(documents)} documents')

    # Remove duplicate documents
    unique_docs = []
    seen_texts = set()
    for doc in documents:
        if doc.page_content not in seen_texts:
            unique_docs.append(doc)
            seen_texts.add(doc.page_content)

    logger.info(f'Found {len(unique_docs)} unique documents')

    # Create the vector store
    logger.info(f'Creating {vector_store_class.__name__} vector store')
    persist_directory = f'{vector_store_class.__name__.lower()}_db'  # You might want to make this configurable

    # Initialize the vector store
    vector_store = vector_store_class(
        embedding_function=embeddings_model,
        persist_directory=persist_directory
    )

    # Create VectorStoreOperator
    vector_store_operator = VectorStoreOperator(
        vector_store=vector_store,
        documents=unique_docs,
        embeddings_model=embeddings_model
    )

    logger.info('Vector store and operator creation complete')
    return vector_store_operator


def ingest_files(dataset: str, split_documents: bool, max_docs: Optional[int] = None) -> List[Document]:
    source_files_path = pathlib.Path('./data') / dataset / 'source_files'
    source_files = [str(f) for f in source_files_path.iterdir() if f.is_file()]

    if max_docs:
        source_files = source_files[:max_docs]

    directory_loader = DirectoryLoader(source_files)
    logger.info(f'Loading documents from {source_files_path}')

    all_documents = []
    for i, doc in enumerate(directory_loader.lazy_load()):
        if max_docs is not None and i >= max_docs:
            break
        all_documents.append(doc)

    logger.info(f'Loaded {len(all_documents)} documents')

    if split_documents:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        all_documents = text_splitter.split_documents(all_documents)
        logger.info(f'Split into {len(all_documents)} chunks')

    return all_documents


def ingest_emails(split_documents: bool = True, max_docs: Optional[int] = None) -> List[Document]:
    username = os.getenv('EMAIL_USERNAME')
    password = os.getenv('EMAIL_PASSWORD')
    if not username or not password:
        raise ValueError("EMAIL_USERNAME and EMAIL_PASSWORD environment variables must be set")

    email_client = EmailClient(username, password)
    search_options = EmailSearchOptions(
        mailbox='INBOX',
        subject=None,
        to_email=None,
        from_email=None,
        since_date=None,
        until_date=None,
        since_email_id=None,
        max_emails=max_docs
    )
    email_loader = EmailLoader(email_client, search_options)
    logger.info('Ingesting emails')
    all_documents = email_loader.load_and_split() if split_documents else email_loader.load()
    logger.info(f'Ingested {len(all_documents)} emails')
    return all_documents

def _ingest_documents(input_data_type: InputDataType, dataset: str, split_documents: bool, max_docs: Optional[int] = None) -> List[Document]:
    if input_data_type == InputDataType.FILE:
        return ingest_files(dataset, split_documents, max_docs)
    elif input_data_type == InputDataType.EMAIL:
        return ingest_emails(split_documents, max_docs)
    elif input_data_type == InputDataType.VECTOR_STORE:
        logger.warning("Vector store ingestion not implemented yet")
        return []
    else:
        raise ValueError(f'Invalid input data type: {input_data_type}')

def _create_retriever(pipeline_args: GetPipelineArgs, retriever_type: RetrieverType) -> LangChainRAGPipeline:
    retriever_creators = {
        RetrieverType.SQL: _create_sql_retriever,
        RetrieverType.VECTOR_STORE: _create_vector_store_retriever,
        RetrieverType.AUTO: _create_auto_retriever,
        RetrieverType.MULTI: _create_multi_retriever,
        RetrieverType.ENSEMBLE: _create_ensemble_retriever,
        RetrieverType.BM25: _create_bm25_retriever,
        RetrieverType.HYBRID: _create_hybrid_search_retriever
    }
    creator = retriever_creators.get(retriever_type)
    if not creator:
        raise ValueError(f'Invalid retriever type: {retriever_type}')
    return creator(pipeline_args)

def _create_vector_store_retriever(pipeline_args: GetPipelineArgs):
    k = 5 if pipeline_args.rerank_documents == ReRankerType.DISABLED else 45
    return LangChainRAGPipeline.from_retriever(
        retriever=pipeline_args.vector_store_operator.vector_store.as_retriever(search_kwargs={"k": k}),
        prompt_template=DEFAULT_EVALUATION_PROMPT_TEMPLATE,
        llm=pipeline_args.llm,
        rerank_documents=pipeline_args.rerank_documents
    )

def _create_bm25_retriever(pipeline_args: GetPipelineArgs):
    if pipeline_args.rerank_documents:
        k = 40
    else:
        k = 5
    bm25_retriever = BM25Retriever.from_documents(pipeline_args.all_documents)
    bm25_retriever.k = k
    return LangChainRAGPipeline.from_retriever(
        retriever=bm25_retriever,
        prompt_template=DEFAULT_EVALUATION_PROMPT_TEMPLATE,
        llm=pipeline_args.llm,
        rerank_documents=pipeline_args.rerank_documents
    )


def _create_hybrid_search_retriever(pipeline_args: GetPipelineArgs):
    pipeline_args.retriever_map = {'bm25':0.5, 'vector_store':0.5}
    pipeline_args.retriever_type = RetrieverType.ENSEMBLE
    return _create_ensemble_retriever(pipeline_args=pipeline_args)


def _create_auto_retriever(pipeline_args: GetPipelineArgs):
    return LangChainRAGPipeline.from_auto_retriever(
        vectorstore=pipeline_args.vector_store,
        data=pipeline_args.all_documents,
        data_description=pipeline_args.dataset_description,
        content_column_name=pipeline_args.content_column_name,
        retriever_prompt_template=pipeline_args.retriever_prompt_template,
        rag_prompt_template=DEFAULT_EVALUATION_PROMPT_TEMPLATE,
        llm=pipeline_args.llm,
        vector_store_operator=pipeline_args.vector_store_operator,
        rerank_documents=pipeline_args.rerank_documents
    )


def _create_sql_retriever(pipeline_args: GetPipelineArgs):
    documents_df = documents_to_df(pipeline_args.content_column_name,
                                   pipeline_args.all_documents,
                                   embeddings_model=pipeline_args.embeddings_model,
                                   with_embeddings=True)

    # Save the dataframe to a SQL table.
    alchemyEngine = create_engine(
        pipeline_args.db_connection_string, pool_recycle=DEFAULT_POOL_RECYCLE)
    db_connection = alchemyEngine.connect()

    # issues with langchain compatibility with vector type in postgres need to investigate further
    documents_df.to_sql(pipeline_args.test_table_name, db_connection,
                        index=False, if_exists='replace')

    return LangChainRAGPipeline.from_sql_retriever(
        connection_string=pipeline_args.db_connection_string,
        retriever_prompt_template=pipeline_args.retriever_prompt_template,
        rag_prompt_template=pipeline_args.rag_prompt_template,
        llm=pipeline_args.llm,
        rerank_documents=pipeline_args.rerank_documents
    )


def _create_multi_retriever(pipeline_args: GetPipelineArgs):
    # The splitter to use to embed smaller chunks
    child_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
    doc_ids = pipeline_args.doc_ids
    split_docs = pipeline_args.split_docs
    return LangChainRAGPipeline.from_multi_vector_retriever(
        documents=split_docs,
        doc_ids=doc_ids,
        rag_prompt_template=DEFAULT_EVALUATION_PROMPT_TEMPLATE,
        vectorstore=pipeline_args.vector_store,
        text_splitter=child_text_splitter,
        llm=pipeline_args.llm,
        mode=pipeline_args.multi_retriever_mode,
        vector_store_operator=pipeline_args.vector_store_operator,
        rerank_documents=pipeline_args.rerank_documents
    )


def _create_ensemble_retriever(pipeline_args: GetPipelineArgs):
    runnable_retrievers: List[Dict] = []
    for type_name, weight in pipeline_args.retriever_map.items():
        pipeline_args.retriever_type = RetrieverType(type_name)
        runnable = _get_pipeline_from_retriever(pipeline_args=pipeline_args)
        runnable_retrievers.append({"weight": weight, "runnable": runnable.retriever_runnable})

    return LangChainRAGPipeline.from_ensemble_retriever(rag_prompt_template=pipeline_args.rag_prompt_template,
                                                        llm=pipeline_args.llm,
                                                        runnable_retrievers=runnable_retrievers,
                                                        rerank_documents = pipeline_args.rerank_documents)


def _get_pipeline_from_retriever(pipeline_args: GetPipelineArgs) -> LangChainRAGPipeline:
    if pipeline_args.retriever_type == RetrieverType.SQL:
        return _create_sql_retriever(pipeline_args=pipeline_args)

    if pipeline_args.retriever_type == RetrieverType.VECTOR_STORE:
        return _create_vector_store_retriever(pipeline_args=pipeline_args)

    if pipeline_args.retriever_type == RetrieverType.AUTO:
        return _create_auto_retriever(pipeline_args=pipeline_args)

    if pipeline_args.retriever_type == RetrieverType.MULTI:
        return _create_multi_retriever(pipeline_args=pipeline_args)

    if pipeline_args.retriever_type == RetrieverType.ENSEMBLE:
        return _create_ensemble_retriever(pipeline_args=pipeline_args)

    if pipeline_args.retriever_type == RetrieverType.BM25:
        return _create_bm25_retriever(pipeline_args=pipeline_args)
    if pipeline_args.retriever_type == RetrieverType.HYBRID:
        return _create_hybrid_search_retriever(pipeline_args=pipeline_args)

    raise ValueError(
        f'Invalid retriever type, must be one of: vector_store, auto, sql, multi, ensemble, hybrid, bm25.  Got {pipeline_args.retriever_type}')


def load_and_limit_qa_samples(qa_file: str, max_qa_samples: Optional[int] = None) -> List[Dict]:

    with open(qa_file, 'r') as f:
        qa_data = json.load(f)

    # If max_qa_samples is specified, limit the data
    if max_qa_samples is not None:
        qa_data['examples'] = qa_data['examples'][:max_qa_samples]

    logger.info(f"Loaded {len(qa_data['examples'])} QA samples for evaluation")

    return qa_data


def evaluate_rag(dataset: str,
                 content_column_name: str = DEFAULT_CONTENT_COLUMN_NAME,
                 dataset_description: str = DEFAULT_DATASET_DESCRIPTION,
                 db_connection_string: Optional[str] = None,
                 test_table_name: str = DEFAULT_TEST_TABLE_NAME,
                 vector_store_class: Type[VectorStore] = DEFAUlT_VECTOR_STORE,
                 llm: BaseChatModel = DEFAULT_LLM,
                 embeddings_model: Embeddings = DEFAULT_EMBEDDINGS,
                 rag_prompt_template: str = DEFAULT_EVALUATION_PROMPT_TEMPLATE,
                 retriever_prompt_template: Union[str, dict] = None,
                 retriever_type: RetrieverType = RetrieverType.VECTOR_STORE,
                 input_data_type: InputDataType = InputDataType.EMAIL,
                 show_visualization: bool = False,
                 split_documents: bool = True,
                 multi_retriever_mode: MultiVectorRetrieverMode = MultiVectorRetrieverMode.BOTH,
                 existing_vector_store: bool = False,
                 rerank_documents: ReRankerType = ReRankerType.DISABLED,
                 retriever_map: Optional[Dict[str, float]] = None,
                 max_input_docs: Optional[int] = None,
                 max_qa_samples: Optional[int] = None
                 ):
    """
    Evaluates a RAG pipeline that answers questions from a dataset.
    """
    try:
        if vector_store_class is None:
            raise ValueError("vector_store_class must be provided")
        if llm is None:
            raise ValueError("llm must be provided")

        all_documents = ingest_files(dataset, split_documents=split_documents, max_docs=max_input_docs)

        if not existing_vector_store:
            vector_store_operator = create_vector_store_and_operator(all_documents, vector_store_class,
                                                                     embeddings_model)
        else:
            vector_store_operator = load_vector_store_and_operator(vector_store_class, embeddings_model)

        pipeline_args = GetPipelineArgs(
            all_documents=all_documents,
            content_column_name=content_column_name,
            dataset_description=dataset_description,
            db_connection_string=db_connection_string,
            test_table_name=test_table_name,
            vector_store_operator=vector_store_operator,
            llm=llm,
            embeddings_model=embeddings_model,
            rag_prompt_template=rag_prompt_template,
            retriever_prompt_template=retriever_prompt_template,
            retriever_type=retriever_type,
            multi_retriever_mode=multi_retriever_mode,
            retriever_map=retriever_map or {},
            rerank_documents=rerank_documents
        )

        rag_pipeline = _create_retriever(pipeline_args, retriever_type)
        rag_chain = rag_pipeline.rag_with_returned_sources()

        dt_string = datetime.now().strftime('%d%m%Y_%H%M%S')
        output_file = f'evaluate_{input_data_type.value}_{retriever_type.value}_rag_{dt_string}.csv'
        qa_file = os.path.join('./data', dataset, 'rag_dataset.json')

        # Load and limit QA samples before evaluation
        qa_samples = load_and_limit_qa_samples(qa_file, max_qa_samples)

        summary_df, individual_scores_df = evaluate.evaluate(rag_chain, qa_samples, output_file)

        if show_visualization:
            visualize_evaluation_metrics(output_file, individual_scores_df)

        logger.info(f"Evaluation complete. Results saved to {output_file}")

    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")
        raise e


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Evaluate RAG',
        description='Evaluates the performance of a RAG pipeline with email or file. Uses evaluation metrics from the RAGAs library.'
    )
    parser.add_argument('-d', '--dataset', required=True,
                        help='Name of QA dataset to use for evaluation (e.g. personal_emails)')
    parser.add_argument('-dd', '--dataset_description', default=DEFAULT_DATASET_DESCRIPTION,
                        help='Description of the dataset')
    parser.add_argument('-c', '--connection_string', default=None, help='Connection string for SQL retriever')
    parser.add_argument('-cc', '--content_column_name', default=DEFAULT_CONTENT_COLUMN_NAME,
                        help='Name of the column containing the content')
    parser.add_argument('-t', '--test_table_name', default=DEFAULT_TEST_TABLE_NAME,
                        help='Name of the table to use for testing (only for SQL retriever)')
    parser.add_argument('-r', '--retriever_type', type=RetrieverType, choices=list(RetrieverType),
                        default=RetrieverType.VECTOR_STORE, help='Type of retriever to use')
    parser.add_argument('-mr', '--multi_retriever_mode', type=MultiVectorRetrieverMode,
                        choices=list(MultiVectorRetrieverMode), default=MultiVectorRetrieverMode.BOTH,
                        help='Mode to use for multi retriever')
    parser.add_argument('-i', '--input_data_type', type=InputDataType, choices=list(InputDataType),
                        default=InputDataType.EMAIL, help='Type of input data to use')
    parser.add_argument('-v', '--show_visualization', action='store_true',
                        help='Whether to plot and show evaluation metrics')
    parser.add_argument('-s', '--split_documents', action='store_true',
                        help='Whether to split documents after they are loaded')
    parser.add_argument('-rd', '--rerank_documents', type=ReRankerType, choices=list(ReRankerType),
                        help='Type of reranker to use, if any', default=ReRankerType.DISABLED)
    parser.add_argument('-evs', '--existing_vector_store', action='store_true',
                        help='If using an existing vector store, update .env file with config')
    parser.add_argument('-er', '--ensemble_retrievers', type=str, default='',
                        help='Comma-separated list of retriever types to use with the Ensemble Retriever')
    parser.add_argument('-ew', '--ensemble_weights', type=str, default='',
                        help='Comma-separated list of weights for the Ensemble Retriever')
    parser.add_argument('-l', '--log', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Logging level to use')
    parser.add_argument('-mid', '--max_input_docs', type=int, default=None,
                        help='Maximum number of input documents to process')
    parser.add_argument('-mqa', '--max_qa_samples', type=int, default=None,
                        help='Maximum number of QA samples to use for evaluation')

    args = parser.parse_args()

    # Set up logging
    try:
        log_level = getattr(logging, args.log.upper())
        if not isinstance(log_level, int):
            raise ValueError(f'Invalid log level: {args.log}')
        logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')
    except (AttributeError, ValueError) as e:
        print(f"Error setting log level: {str(e)}")
        print("Defaulting to INFO level")
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)

    # Process ensemble retriever arguments
    retriever_map = {}
    if args.ensemble_retrievers and args.ensemble_weights:
        try:
            weights = [float(x) for x in args.ensemble_weights.split(',')]
            retrievers = args.ensemble_retrievers.split(',')
            if len(weights) != len(retrievers):
                raise ValueError("Number of weights must match number of retrievers")
            retriever_map = dict(zip(retrievers, weights))
        except ValueError as e:
            logger.error(f"Error processing ensemble retrievers: {str(e)}")
            sys.exit(1)

    # Handle existing vector store
    if args.existing_vector_store:
        logger.warning(
            'Vector store config provided, setting retriever type to vector store as other types are not currently supported.')
        args.retriever_type = RetrieverType.VECTOR_STORE

    logger.info(f'Evaluating RAG pipeline with dataset: {args.dataset}')

    try:
        evaluate_rag(
            dataset=args.dataset,
            content_column_name=args.content_column_name,
            dataset_description=args.dataset_description,
            db_connection_string=args.connection_string,
            test_table_name=args.test_table_name,
            retriever_type=args.retriever_type,
            input_data_type=args.input_data_type,
            show_visualization=args.show_visualization,
            split_documents=args.split_documents,
            multi_retriever_mode=args.multi_retriever_mode,
            existing_vector_store=args.existing_vector_store,
            rerank_documents=args.rerank_documents,
            retriever_map=retriever_map,
            max_input_docs=args.max_input_docs,
            max_qa_samples=args.max_qa_samples
        )
    except Exception as e:
        logger.error(f"An error occurred during evaluation: {str(e)}")
        sys.exit(1)
