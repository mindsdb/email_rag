from enum import Enum

from langchain_community.chat_models.ollama import ChatOllama
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic_settings import BaseSettings

DEFAULT_CARDINALITY_THRESHOLD = 40
DEFAULT_POOL_RECYCLE = 3600

#OpenAI LLM Provider Settings
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_LLM_MODEL = "gpt-3.5-turbo"


#Ollama LLM Provider Settings
#DEFAULT_LLM_PROVIDER = "ollama"
#DEFAULT_LLM_MODEL = "mistral"
#DEFAULT_LLM_MODEL = "gemma:2b"

#Embedding Settings
DEFAULT_EMBEDDINGS = OpenAIEmbeddings()
#DEFAULT_EMBEDDINGS = GPT4AllEmbeddings()


DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_CONTENT_COLUMN_NAME = "body"
DEFAULT_DATASET_DESCRIPTION = "email inbox"
DEFAULT_TEST_TABLE_NAME = "test_email"


DEFAULT_RERANK = False
DEFAUlT_VECTOR_STORE = Chroma


def get_llm(**kwargs):
    if DEFAULT_LLM_PROVIDER == "openai":
        return ChatOpenAI(model_name=DEFAULT_LLM_MODEL, **kwargs)
    if DEFAULT_LLM_PROVIDER == "ollama":
        return ChatOllama(base_url=DEFAULT_OLLAMA_URL, model=DEFAULT_LLM_MODEL, **kwargs)


DEFAULT_LLM = get_llm(temperature=0)  # ChatOpenAI(model_name=DEFAULT_LLM_MODEL, temperature=0)

DEFAULT_AUTO_META_PROMPT_TEMPLATE = """
Below is a json representation of a table with information about {description}. 
Return a JSON list with an entry for each column. Each entry should have 
{{"name": "column name", "description": "column description", "type": "column data type"}}
\n\n{dataframe}\n\nJSON:\n
"""
DEFAULT_EVALUATION_PROMPT_TEMPLATE = '''You are an assistant for
question-answering tasks. Use the following pieces of retrieved context
to answer the question. If you don't know the answer, just say that you
don't know. Use two sentences maximum and keep the answer concise.
Question: {question}
Context: {context}
Answer:'''

DEFAULT_QA_GENERATION_PROMPT_TEMPLATE = '''You are an assistant for
generating sample questions and answers from the given document and metadata. Given
a document and its metadata as context, generate a question and answer from that document and its metadata.

The document will be a string. The metadata will be a JSON string. You need
to parse the JSON to understand it.

Generate a question that requires BOTH the document and metadata to answer, if possible.
Otherwise, generate a question that requires ONLY the document to answer.

Return a JSON dictionary with the question and answer like this:
{{ "question": <the full generated question>, "answer": <the full generated answer> }}

Make sure the JSON string is valid before returning it. You must return the question and answer
in the specified JSON format no matter what.

Document: {document}
Metadata: {metadata}
Answer:'''

DEFAULT_TEXT_2_PGVECTOR_PROMPT_TEMPLATE = """You are a Postgres expert. Given an input question, first create a syntactically correct Postgres query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most 5 results using the LIMIT clause as per Postgres. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use date('now') function to get the current date, if the question involves "today".

You can use an extra extension which allows you to run semantic similarity using <-> operator on tables containing columns named "embeddings".
<-> operator can ONLY be used on embeddings columns.
The embeddings value for a given row typically represents the semantic meaning of that row.
The vector represents an embedding representation of the question, given below. 
Do NOT fill in the vector values directly, but rather specify a `[search_word]` placeholder, which should contain the word that would be embedded for filtering.
For example, if the user asks for songs about 'the feeling of loneliness' the query could be:
'SELECT "[whatever_table_name]"."SongName" FROM "[whatever_table_name]" ORDER BY "embeddings" <-> '[loneliness]' LIMIT 5'

Use the following format:

Question: <Question here>
SQLQuery: <SQL Query to run>
SQLResult: <Result of the SQLQuery>
Answer: <Final answer here>

Only use the following tables:

{schema}
"""

DEFAULT_SQL_RESULT_PROMPT_TEMPLATE = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""

DEFAULT_SQL_RETRIEVAL_PROMPT_TEMPLATE = {
                "sql_query": DEFAULT_TEXT_2_PGVECTOR_PROMPT_TEMPLATE,
                "sql_result": DEFAULT_SQL_RESULT_PROMPT_TEMPLATE
            }

DEFAULT_RERANKING_PROMPT_TEMPLATE = """
A list of documents is shown below. Each document has a number next to it along with a summary of the document. A question is also provided.
  Respond with the numbers of the documents you should consult to answer the question, in order of relevance, as well
  as the relevance score. The relevance score is a number from 1–10 based on how relevant you think the document is to the question.
  Do not include any documents that are not relevant to the question.
  Example format:
  Document 1:
  <summary of document 1>
  Document 2:
  <summary of document 2>
  …
  Document 10:
  <summary of document 10>
  Question: <question>
  Answer:
  Doc: 9, Relevance: 7
  Doc: 3, Relevance: 4
  Doc: 7, Relevance: 3
  Let's try this now:
  {context_str}
  Question: {query_str}
  Answer:
"""

class VectorStoreType(Enum):
    CHROMA = 'chroma'
    PGVECTOR = 'pgvector'


class RetrieverType(Enum):
    VECTOR_STORE = 'vector_store'
    AUTO = 'auto'
    SQL = 'sql'
    MULTI = 'multi'
    BM25 = 'bm25'
    ENSEMBLE = 'ensemble'
    HYBRID = 'hybrid'


class InputDataType(Enum):
    EMAIL = 'email'
    FILE = 'file'
    VECTOR_STORE = 'vector_store'


class VectorStoreConfig(BaseSettings):
    type: VectorStoreType = VectorStoreType.CHROMA
    persist_directory: str = None
    collection_name: str = None
    connection_string: str = None

    class Config:
        arbitrary_types_allowed = True
        extra = "forbid"
        env_file = ".env"
        env_prefix = "vector_store_"
