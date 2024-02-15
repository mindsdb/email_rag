from datetime import datetime
import argparse
import logging
import os
import platform

from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

from evaluate import evaluate
from ingestors.email_ingestor.email_client import EmailClient
from ingestors.email_ingestor.email_ingestor import EmailIngestor
from ingestors.email_ingestor.email_search_options import EmailSearchOptions


# Use a basic QA prompt template for email RAG evaluation.
_EVALUATION_PROMPT_TEMPLATE = '''You are an assistant for
question-answering tasks. Use the following pieces of retrieved context
to answer the question. If you don't know the answer, just say that you
don't know. Use two sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:'''


def evaluate_email_rag(dataset: str):
    '''Evaluates a RAG pipeline that answers questions from a dataset
    about various emails, depending on the dataset.
    '''
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

    # For now, use Chroma for evaluation.
    vectorstore = Chroma.from_documents(
        documents=all_documents,
        embedding=OpenAIEmbeddings(),
    )
    retriever = vectorstore.as_retriever()

    # Use a simple GPT-3.5 model for evaluation.
    # Use minimum temperature to get more consistent evaluations across multiple runs.
    llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)
    prompt = ChatPromptTemplate.from_template(_EVALUATION_PROMPT_TEMPLATE)

    # Setup simple RAG pipeline
    rag_chain = (
        {'context': retriever,  'question': RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Generate filename based on current datetime.
    dt_string = datetime.now().strftime('%d%m%Y_%H%M%S')
    output_file = 'evaluate_email_rag_{}.csv'.format(dt_string)

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
        prog='Evaluate Email RAG',
        description='''Evaluates the performance of a RAG pipeline with email.
Uses evaluation metrics from the RAGAs library.
        '''
    )
    parser.add_argument(
        '-d', '--dataset', help='Name of QA dataset to use for evaluation (e.g. personal_emails)')
    parser.add_argument(
        '-l', '--log', help='Logging level to use (default WARNING)', default='WARNING')

    args = parser.parse_args()
    log_level = getattr(logging, args.log.upper())
    logging.basicConfig(level=log_level)

    evaluate_email_rag(args.dataset)
