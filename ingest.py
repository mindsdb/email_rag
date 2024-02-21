import argparse
import datetime
import logging
import os

from langchain_community.vectorstores.pgvector import PGVector
import pandas as pd

from loaders.email_loader.email_client import EmailClient
from loaders.email_loader.email_loader import EmailLoader
from loaders.email_loader.email_search_options import EmailSearchOptions


def _ingest_to_csv(email_loader: EmailLoader, output_path: str):
    all_email_docs = email_loader.load()
    all_emails = []
    for doc in all_email_docs:
        meta = doc.metadata
        all_emails.append({
            'body': doc.page_content,
            'from': meta['from_field'],
            'to': meta['to_field'],
            'subject': meta['subject'],
            'date': meta['date']
        })
    df = pd.DataFrame(all_emails)
    df.to_csv(output_path)


def _ingest_to_vector_store(email_loader: EmailLoader, collection_name: str, connection_string: str):
    pgvector_store = PGVector(
        collection_name=collection_name, connection_string=connection_string)
    all_email_docs = email_loader.load_and_split()
    pgvector_store.add_documents(all_email_docs)


def ingest(search_options: EmailSearchOptions,
           connection_string: str = None,
           collection_name: str = None,
           output_path: str = None):
    '''Ingests emails from search options and stores in .csv or Postgres.

    Parameters:
        search_options (EmailSearchOptions): Search options defining emails to ingest.
        connection_string (str): If storing in Postgres, the DB connection string.
        collection_name (str): If storing in Postgres, new or existing collection name to use
        output_path (str): If storing in .csv, output file name to use.
    '''
    username = os.getenv('EMAIL_USERNAME')
    password = os.getenv('EMAIL_PASSWORD')
    email_client = EmailClient(username, password)
    email_loader = EmailLoader(email_client, search_options)

    if output_path:
        _ingest_to_csv(email_loader, output_path)
        return
    _ingest_to_vector_store(email_loader, collection_name, connection_string)


if __name__ == '__main__':
    # To use, set the environment variables:
    # OPENAI_API_KEY='<YOUR_API_KEY>'
    # EMAIL_USERNAME='<YOUR_EMAIL_USERNAME>'
    # EMAIL_PASSWORD='<YOUR_PASSWORD>'
    parser = argparse.ArgumentParser(
        prog='Ingest Emails',
        description='''Ingests emails using an IMAP client with the specified search options.
        Can either output to .csv or to a Postgres database with the pgvector extension.
        '''
    )
    # Email search options
    parser.add_argument(
        '-m', '--mailbox', help='IMAP mailbox to search (default INBOX)', default='INBOX')
    parser.add_argument('-s', '--subject',
                        help='Search by email subject', default=None)
    parser.add_argument(
        '-t', '--to_email', help='Search based on who the email was sent to', default=None)
    parser.add_argument(
        '-f', '--from_email', help='Search based on who the email was sent from', default=None)
    parser.add_argument('-d', '--since_date', help='Searh for all emails received after this date (YYYY-mm-dd format)',
                        type=lambda d: datetime.datetime.strptime(d, '%Y-%m-%d').date(), default=None)
    parser.add_argument('-u', '--until_date', help='Searh for all emails received before this date (YYYY-mm-dd format)',
                        type=lambda d: datetime.datetime.strptime(d, '%Y-%m-%d').date(), default=None)
    parser.add_argument(
        '-i', '--since_id', help='Search for all emails received after this email ID', default=None)

    # Output options
    parser.add_argument(
        '-c', '--connection_string',
        help='Connection string for Postgres vector store (requires pgvector extension)',
        default=None)
    parser.add_argument(
        '-n', '--collection_name',
        help='Collection name to use when storing emails (only used with --connection_string)',
        default=None)
    parser.add_argument(
        '-o', '--output_path',
        help='If set, will write emails to a .csv file instead of Postgres',
        default=None)

    # General options
    parser.add_argument(
        '-l', '--log', help='Logging level to use (default WARNING)', default='WARNING')

    args = parser.parse_args()
    log_level = getattr(logging, args.log.upper())
    logging.basicConfig(level=log_level)

    search_options = EmailSearchOptions(
        mailbox=args.mailbox,
        subject=args.subject,
        to_email=args.to_email,
        from_email=args.from_email,
        since_date=args.since_date,
        until_date=args.until_date,
        since_email_id=args.since_id
    )
    ingest(search_options, connection_string=args.connection_string,
           collection_name=args.collection_name, output_path=args.output_path)
