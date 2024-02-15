
from typing import List

import re

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
import pandas as pd

from ingestors.email_ingestor.email_client import EmailClient
from ingestors.email_ingestor.email_search_options import EmailSearchOptions


class EmailIngestor:
    '''Loads emails into document representation'''

    _DEFAULT_CHUNK_SIZE = 500
    _DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, email_client: EmailClient, search_options: EmailSearchOptions):
        self.email_client = email_client
        self.search_options = search_options

    def _preprocess_raw_html(self, html: str) -> str:
        # Should always have Content-Type as part of body before the actual message.
        content_type_i = html.find('Content-Type')
        html = html[content_type_i:]
        # Get rid of scripts.
        html = re.sub(r'<(script|style).*?</\1>', '', html)
        # Get rid of all other tags.
        html = re.sub(r'<.*?>', '', html)
        # Get rid of extra whitespace that could affect token limit.
        html = re.sub(r'\s\s', ' ', html)
        return html

    def _ingest_email_row(self, row: pd.Series) -> List[Document]:
        if row['body_type'] == 'text/html':
            row['body'] = self._preprocess_raw_html(row['body'])

        email_doc = Document(row['body'])
        email_doc.metadata = {
            'from': row['from'],
            'to': row['to'],
            'subject': row['subject'],
            'date': row['date']
        }

        # Split by ["\n\n", "\n", " ", ""] in order.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=EmailIngestor._DEFAULT_CHUNK_SIZE,
            chunk_overlap=EmailIngestor._DEFAULT_CHUNK_OVERLAP)
        return text_splitter.split_documents([email_doc])

    def ingest(self) -> List[Document]:
        emails_df = self.email_client.search_email(self.search_options)
        all_documents = []
        for _, row in emails_df.iterrows():
            all_documents += self._ingest_email_row(row)
        self.email_client.logout()
        return all_documents
