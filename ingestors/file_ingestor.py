
import pathlib
from typing import List

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import UnstructuredHTMLLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import HTMLHeaderTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_core.documents.base import Document
from langchain_openai.embeddings import OpenAIEmbeddings


class FileIngestor:
    '''Loads various file types into document representation'''

    _DEFAULT_CHUNK_SIZE = 1000
    _DEFAULT_CHUNK_OVERLAP = 50

    def __init__(self, paths: List[str]):
        self.paths = paths
        super().__init__()

    def _load_documents_from_file(self, path: str) -> List[Document]:
        file_extension = pathlib.Path(path).suffix
        loader = TextLoader(path)
        # Split by paragraph by default.
        text_splitter = CharacterTextSplitter(
            chunk_size=FileIngestor._DEFAULT_CHUNK_SIZE,
            chunk_overlap=FileIngestor._DEFAULT_CHUNK_OVERLAP)
        if file_extension == '.pdf':
            loader = PyMuPDFLoader(path)
            # Use semantic chunking to understand the PDf structure.
            text_splitter = SemanticChunker(OpenAIEmbeddings())
        elif file_extension == '.html':
            loader = UnstructuredHTMLLoader(path)
            # Split by h1, h2, h3, h4 for HTML files.
            text_splitter = HTMLHeaderTextSplitter(headers_to_split_on=[
                ('h1', 'Header 1'),
                ('h2', 'Header 2'),
                ('h3', 'Header 3'),
                ('h4', 'Header 4')])
        elif file_extension == '.md':
            loader = UnstructuredMarkdownLoader(path)
            # Split by h1, h2, h3 for .md files.
            text_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[(
                '#', 'Header 1'), ('##', 'Header 2'), ('###', 'Header 3')])

        documents = loader.load()
        return text_splitter.split_documents(documents)

    def ingest(self) -> List[Document]:
        all_documents = []
        for f in self.paths:
            # Load and split each file individually
            all_documents += self._load_documents_from_file(f)

        return all_documents
