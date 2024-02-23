
from typing import Iterator, List
import re

from bs4 import BeautifulSoup
import bs4.element
import chardet

from langchain.text_splitter import RecursiveCharacterTextSplitter, TextSplitter
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents.base import Document
import pandas as pd

from loaders.email_loader.email_client import EmailClient
from loaders.email_loader.email_search_options import EmailSearchOptions

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50


class EmailLoader(BaseLoader):
    '''Loads emails into document representation'''


    def __init__(self, email_client: EmailClient, search_options: EmailSearchOptions):
        self.email_client = email_client
        self.search_options = search_options

    def _is_tag_visible(self, element):
        if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
            return False
        if isinstance(element, bs4.element.Comment):
            return False
        return True

    def _preprocess_raw_html(self, html: str) -> str:
        soup = BeautifulSoup(html, 'html.parser')
        texts = soup.find_all(text=True)
        visible_texts = filter(self._is_tag_visible, texts)
        return '\n'.join(t.strip() for t in visible_texts)

    def _ingest_email_row(self, row: pd.Series) -> Document:
        if row['body_content_type'] == 'html':
            # Extract meaningful text from raw HTML.
            row['body'] = self._preprocess_raw_html(row['body'])
        body_str = row['body']
        encoding = None
        if isinstance(body_str, bytes):
            encoding = chardet.detect(body_str)['encoding']
            if 'windows' in encoding.lower():
                # Easier to treat this at utf-8 since str constructor doesn't support all encodings here:
                # https://chardet.readthedocs.io/en/latest/supported-encodings.html.
                encoding = 'utf-8'
            try:
                body_str = str(body_str, encoding=encoding)
            except UnicodeDecodeError:
                # If illegal characters are found, we ignore them.
                # I encountered this issue with some emails that had a mix of encodings.
                body_str = row['body'].decode(encoding, errors='ignore')
        # We split by paragraph so make sure there aren't too many newlines in a row.
        body_str = re.sub(r'[\r\n]\s*[\r\n]', '\n\n', body_str)
        email_doc = Document(body_str)
        email_doc.metadata = {
            'from_field': row['from'],
            'to_field': row['to'],
            'subject': row['subject'],
            'date': row['date']
        }

        # Replacing None values {None: ""}
        for key in email_doc.metadata:
            if email_doc.metadata[key] is None:
                email_doc.metadata[key] = ""
        return email_doc

    def load_and_split(self, text_splitter: TextSplitter = None) -> List[Document]:
        # Split by ["\n\n", "\n", " ", ""] in order.
        if text_splitter is None:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP)

        all_documents = self.load()
        split_documents = []
        for doc in all_documents:
            split_documents += text_splitter.split_documents([doc])
        return split_documents

    def load(self) -> List[Document]:
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        emails_df = self.email_client.search_email(self.search_options)
        for _, row in emails_df.iterrows():
            yield self._ingest_email_row(row)
        self.email_client.logout()
