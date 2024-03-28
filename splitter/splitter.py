from typing import List

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import MarkdownHeaderTextSplitter, HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter

from settings import DEFAULT_EMBEDDINGS

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 50


class AutoSplitter:
    def __init__(self, headers_to_split_on=None, chunk_size=None, chunk_overlap=None, embedding=None):
        self.headers_to_split_on = headers_to_split_on
        self.chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or DEFAULT_CHUNK_OVERLAP
        self.embedding = embedding or DEFAULT_EMBEDDINGS

        self._extension_map = {
            '.pdf': self.semantic_chunker,
            '.md': self.markdown_splitter,
            '.html': self.html_splitter
        }
        self.default_splitter = self.recursive_splitter

    def split_func_by_extension(self, extension):
        return self._extension_map.get(extension, self.default_splitter)()

    def split_documents(self, documents: List[Document], text_splitter = None, default_failover=True) -> List[Document]:
        split_documents = []
        document: Document
        for document in documents:
            print(document.metadata)
            extension = document.metadata['extension']
            if text_splitter:
                split_func = text_splitter.split_documents
            else:
                split_func = self.split_func_by_extension(extension=extension)
            #This could either be a split_text or a split_documents function
            try: #Try split_documents first
                try:
                    print("trying split docs")
                    split_documents += split_func([document])
                except TypeError: #This throw indicates we are using split_text
                    print("trying split text")
                    split_documents += split_func(document.page_content)
            except Exception as e: #Check if default_failover set, and then try the default splitter.
                if default_failover:
                    print("doing default failover")
                    split_func = self.split_func_by_extension(extension=None)
                    split_documents += split_func([document])
                else:
                    raise e
        return split_documents

    def semantic_chunker(self):
        return SemanticChunker(self.embedding).split_documents

    def markdown_splitter(self):
        if self.headers_to_split_on:
            return MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)
        else:
            return MarkdownHeaderTextSplitter(headers_to_split_on=[(
                '#', 'Header 1'), ('##', 'Header 2'), ('###', 'Header 3')]).split_text

    def html_splitter(self):
        print("DOING HTML SPLITTER")
        if self.headers_to_split_on:
            return HTMLHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)
        else:
            return HTMLHeaderTextSplitter(headers_to_split_on=[
                ('h1', 'Header 1'),
                ('h2', 'Header 2'),
                ('h3', 'Header 3'),
                ('h4', 'Header 4')]).split_text

    def recursive_splitter(self):
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap).split_documents
