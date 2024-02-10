from datetime import datetime
import argparse
import logging
import os
import platform

from dotenv import load_dotenv, find_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from evaluate import evaluate
from ingestors.file_ingestor import FileIngestor


# Use a basic QA prompt template for file RAG evaluation.
_EVALUATION_PROMPT_TEMPLATE = '''You are an assistant for
question-answering tasks. Use the following pieces of retrieved context
to answer the question. If you don't know the answer, just say that you
don't know. Use two sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:'''


def evaluate_file_rag(dataset: str):
    '''Evaluates a RAG pipeline that answers questions from a dataset
    about various types of files, depending on the dataset.
    '''
    dataset_path = os.path.join('./data', dataset)
    ingestor = FileIngestor(dataset_path)
    logging.info('Loading documents from {}'.format(dataset_path))
    documents = ingestor.ingest()
    logging.info('Documents loaded')

    # For now, use Chroma for evaluation.
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever()

    # Use a simple GPT-3.5 model for evaluation.
    # Use minimum temperature to get more consistent evaluations across multiple runs.
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    prompt = ChatPromptTemplate.from_template(_EVALUATION_PROMPT_TEMPLATE)

    # Setup RAG pipeline
    rag_chain = (
        {'context': retriever,  'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Generate filename based on current datetime.
    dt_string = datetime.now().strftime('%d%m%Y_%H%M%S')
    output_file = 'evaluate_file_rag_{}.csv'.format(dt_string)

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

    # To use, set OPENAI_API_KEY='<YOUR_API_KEY>' in a .env file at the root.
    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser(
        prog='Evaluate File RAG',
        description='''Evaluates the performance of a RAG pipeline with files.
Uses evaluation metrics from the RAGAs library.
        '''
    )
    parser.add_argument(
        '-d', '--dataset', help='Name of QA dataset to use for evaluation (e.g. blockchain_solana)')
    parser.add_argument(
        '-l', '--log', help='Logging level to use (default WARNING)', default='WARNING')

    args = parser.parse_args()
    log_level = getattr(logging, args.log.upper())
    logging.basicConfig(level=log_level)

    evaluate_file_rag(args.dataset)
