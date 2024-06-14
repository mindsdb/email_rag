import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import argparse
import logging
import os
import platform
from typing import List, Union, Dict

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.vectorstores import VectorStore
from langchain_community.retrievers import BM25Retriever

from sqlalchemy import create_engine

from retrievers.multi_vector_retriever import MultiVectorRetrieverMode

from evaluate import evaluate
from loaders.email_loader.email_client import EmailClient
from loaders.email_loader.email_loader import EmailLoader, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from loaders.email_loader.email_search_options import EmailSearchOptions
from loaders.directory_loader.directory_loader import DirectoryLoader
from loaders.vector_store_loader.vector_store_loader import load_vector_store
from pipelines.rag_pipelines import LangChainRAGPipeline
from settings import (DEFAULT_LLM,
                      DEFAUlT_VECTOR_STORE,
                      DEFAULT_EVALUATION_PROMPT_TEMPLATE,
                      DEFAULT_EMBEDDINGS,
                      DEFAULT_CONTENT_COLUMN_NAME,
                      DEFAULT_DATASET_DESCRIPTION,
                      DEFAULT_TEST_TABLE_NAME,
                      DEFAULT_POOL_RECYCLE, RetrieverType, InputDataType,
                      )
from utils import documents_to_df, VectorStoreOperator
from visualize.visualize import visualize_evaluation_metrics

_DEFAULT_ID_KEY = "doc_id"


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


def _ingest_documents(input_data_type: InputDataType, dataset: str, split_documents: bool = True) -> List[Document]:
    if input_data_type == InputDataType.FILE:
        return ingest_files(dataset, split_documents)
    if input_data_type == InputDataType.EMAIL:
        return ingest_emails(split_documents)
    if input_data_type == InputDataType.VECTOR_STORE:
        return
    raise ValueError(
        f'Invalid input data type, must be one of: file, email or vector_store. Got {input_data_type}')



@dataclass
class GetPipelineArgs:
    all_documents: List[Document]
    content_column_name: str
    dataset_description: str
    db_connection_string: str
    test_table_name: str
    vector_store: VectorStore
    llm: BaseChatModel
    embeddings_model: Embeddings
    rag_prompt_template: str
    retriever_prompt_template: Union[str, dict]
    retriever_type: RetrieverType
    rerank_documents: bool
    multi_retriever_mode: MultiVectorRetrieverMode
    retriever_map: dict
    _vector_store_operator: Union[VectorStoreOperator, None] = None
    _split_docs: list[Document] = None,
    _doc_ids: list[str] = None

    @property
    def vector_store_operator(self) -> VectorStoreOperator:
        if not self._doc_ids or not self._split_docs:
            self._split_docs, self._doc_ids = self._split_documents()

        if not self._vector_store_operator:
            self._vector_store_operator = VectorStoreOperator(
                vector_store=self.vector_store,
                documents=self.split_docs,
                embeddings_model=self.embeddings_model
            )
        return self._vector_store_operator

    @property
    def doc_ids(self) -> list[str]:
        if not self._doc_ids:
            self._split_docs, self._doc_ids = self._split_documents()
        return self._doc_ids

    @property
    def split_docs(self) -> list[Document]:
        if not self._split_docs:
            self._split_docs, self._doc_ids = self._split_documents()
        return self._split_docs

    def _split_documents(self) -> tuple[list[Document], list[str]]:
        """
        Split the documents into sub-documents and generate unique ids for each document.
        :return:
        """
        split_info = list(map(self._generate_id_and_split_document, self.all_documents))
        doc_ids, split_docs_lists = zip(*split_info)
        split_docs = [doc for sublist in split_docs_lists for doc in sublist]
        return split_docs, list(doc_ids)

    def _generate_id_and_split_document(self, doc: Document) -> tuple[str, list[Document]]:
        """
        Generate a unique id for the document and split it into sub-documents.
        :param doc:
        :return:
        """
        child_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)
        doc_id = str(uuid.uuid4())
        sub_docs = child_text_splitter.split_documents([doc])
        for sub_doc in sub_docs:
            sub_doc.metadata[_DEFAULT_ID_KEY] = doc_id
        return doc_id, sub_docs


def _create_vector_store_retriever(pipeline_args: GetPipelineArgs):
    if pipeline_args.rerank_documents:
        k = 40
    else:
        k = 5
    return LangChainRAGPipeline.from_retriever(
        retriever=pipeline_args.vector_store_operator.vector_store.as_retriever(search_kwargs={"k": k}),
        prompt_template=DEFAULT_EVALUATION_PROMPT_TEMPLATE,
        llm=pipeline_args.llm,
        rerank_documents = pipeline_args.rerank_documents
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
                 show_visualization=False,
                 split_documents=True,
                 multi_retriever_mode: MultiVectorRetrieverMode = MultiVectorRetrieverMode.BOTH,
                 existing_vector_store: bool = False,
                 rerank_documents=False,
                 retriever_map=None
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
    :param existing_vector_store: bool
    :param retriever_map: dict

    :return:
    """
    all_documents = _ingest_documents(
        input_data_type, dataset, split_documents=split_documents)

    if existing_vector_store:
        vector_store = load_vector_store(embeddings_model)


    pipeline_args = GetPipelineArgs(
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
        multi_retriever_mode=multi_retriever_mode,
        retriever_map=retriever_map,
        rerank_documents=rerank_documents
    )

    rag_pipeline = _get_pipeline_from_retriever(pipeline_args)

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
    parser.add_argument('-rd','--rerank_documents', type=bool,
                        help='Whether or not to use an LLM to rerank documents after pulling.',
                        default=False)

    parser.add_argument('-evs', '--existing_vector_store',
                        help='If using an existing vector store, update .env file with config',
                        type=bool, default=False)

    parser.add_argument('-er', '--ensemble_retrievers', type=str,
                        help='Comma delineated list of retriever types to use with the Ensemble Retriever', default='')
    parser.add_argument('-ew', '--ensemble_weights', type=str,
                        help='Comma delineated list of weights, respective in order to the ensemble_retrievers, '
                             'to use with the Ensemble Retriever', default='')
    parser.add_argument(
        '-l', '--log', help='Logging level to use (default WARNING)', default='WARNING')

    args = parser.parse_args()
    log_level = getattr(logging, args.log.upper())
    logging.basicConfig(level=log_level)


    logger = logging.getLogger(__name__)

    if args.existing_vector_store:
        # Update .env file with vector store config

        logger.warning(
            'Vector store config provided, setting retriever type to vector store as other types '
            'are not currently supported.')

        # Set the retriever type to vector store
        args.retriever_type = RetrieverType.VECTOR_STORE

    logger.warning(f'Evaluating RAG pipeline with dataset: {args.dataset}')

    retriever_map = {}
    if args.ensemble_weights and args.ensemble_retrievers:
        weights = [float(x) for x in args.ensemble_weights.split(',')]
        retrievers = args.ensemble_retrievers.split(',')
        for i, weight in enumerate(weights):
            retriever_map[retrievers[i]] = weight


    evaluate_rag(dataset=args.dataset, content_column_name=args.content_column_name,
                 dataset_description=args.dataset_description, db_connection_string=args.connection_string,
                 test_table_name=args.test_table_name, retriever_type=args.retriever_type,
                 input_data_type=args.input_data_type, show_visualization=args.show_visualization,
                 split_documents=args.split_documents, multi_retriever_mode=MultiVectorRetrieverMode.BOTH,
                 existing_vector_store=args.existing_vector_store,
                 rerank_documents=args.rerank_documents,
                 retriever_map=retriever_map)

