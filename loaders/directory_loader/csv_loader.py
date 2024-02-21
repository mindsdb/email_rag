import csv

from typing import Iterator, List

from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

_CONTENT_FIELD_NAMES = ['content', 'text', 'body']


class CSVLoader(BaseLoader):

    def __init__(self, path: str):
        self.path = path

    def load(self) -> List[Document]:
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        with open(self.path, 'r', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            headers = next(reader)
            for line in reader:
                doc_body = ''
                meta = {}
                for i, header in enumerate(headers):
                    if header in _CONTENT_FIELD_NAMES and doc_body == '':
                        doc_body = line[i]
                        continue
                    meta[header] = line[i]
                yield Document(page_content=doc_body, metadata=meta)
