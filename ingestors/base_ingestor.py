from typing import List

from langchain_core.documents.base import Document


class BaseIngestor:
    '''Base class for ingesting data into a document represent'''

    def ingest(self) -> List[Document]:
        '''Loads data and returns a list of documents'''
        raise NotImplementedError
