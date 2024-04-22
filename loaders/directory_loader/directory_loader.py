import pathlib
from typing import Iterator, List

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import TextSplitter
from langchain_core.documents.base import Document

from loaders.directory_loader.csv_loader import CSVLoader
from settings import DEFAULT_EMBEDDINGS

from splitter.splitter import AutoSplitter

class DirectoryLoader(BaseLoader):
    '''Loads various file types in a directory into document representation'''

    def __init__(self, paths: List[str]):
        self.paths = paths
        self.splitter = AutoSplitter()
        self.html_loader_options = {
            "mode": "single"
        }
        self.as_text = True
        super().__init__()

    def _get_loader_from_extension(self, extension: str, path: str) -> BaseLoader:
        if self.as_text:
            return TextLoader(path, encoding='utf-8')
        if extension == '.pdf':
            return PyMuPDFLoader(path)
        if extension == '.csv':
            return CSVLoader(path)
        if extension == '.html':
            return UnstructuredHTMLLoader(path, **self.html_loader_options)
        if extension == '.md':
            return UnstructuredMarkdownLoader(path)
        return TextLoader(path, encoding='utf-8')

    def _lazy_load_documents_from_file(self, path: str) -> Iterator[Document]:
        file_extension = pathlib.Path(path).suffix
        loader = self._get_loader_from_extension(file_extension, path)

        for doc in loader.lazy_load():
            doc.metadata['extension'] = file_extension
            yield doc


    def _get_text_splitter_from_extension(self, extension: str) -> TextSplitter:
        if extension == '.pdf':
            return SemanticChunker(DEFAULT_EMBEDDINGS)
        if extension == '.md':
            return MarkdownHeaderTextSplitter(headers_to_split_on=[(
                '#', 'Header 1'), ('##', 'Header 2'), ('###', 'Header 3')])
        if extension == '.html':
            return HTMLHeaderTextSplitter(headers_to_split_on=[
                ('h1', 'Header 1'),
                ('h2', 'Header 2'),
                ('h3', 'Header 3'),
                ('h4', 'Header 4')])
        # Split by ["\n\n", "\n", " ", ""] in order.
        return RecursiveCharacterTextSplitter(
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP)


    def load_and_split(self, text_splitter: TextSplitter = None) -> List[Document]:
        return self.splitter.split_documents(self.load(), text_splitter=text_splitter)

    def load(self) -> List[Document]:
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        for f in self.paths:
            # Load and split each file individually
            for doc in self._lazy_load_documents_from_file(f):
                yield doc
