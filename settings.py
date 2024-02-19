from typing import List

import pandas as pd
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

DEFAULT_CARDINALITY_THRESHOLD = 40
DEFAULT_LLM_MODEL = "gpt-3.5-turbo"
DEFAULT_CONTENT_COLUMN_NAME = "body"
DEFAULT_DATASET_DESCRIPTION = "email inbox"
DEFAULT_TEST_TABLE_NAME = "test_email"
DEFAULT_LLM = ChatOpenAI(model_name=DEFAULT_LLM_MODEL, temperature=0)
DEFAULT_EMBEDDINGS = OpenAIEmbeddings()
DEFAUlT_VECTOR_STORE = Chroma
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
    "sql_query":DEFAULT_TEXT_2_PGVECTOR_PROMPT_TEMPLATE,
    "sql_result":DEFAULT_SQL_RESULT_PROMPT_TEMPLATE
}


def documents_to_df(content_column_name: str,
                    documents: List[Document],
                    embeddings_model: Embeddings = None,
                    with_embeddings: bool = False):
    """
    Given a list of documents, convert it to a dataframe.

    :param content_column_name: str
    :param documents: List[Document]
    :param embeddings_model: Embeddings
    :param with_embeddings: bool

    :return:
    """
    df = pd.DataFrame([doc.metadata for doc in documents])

    df[content_column_name] = [doc.page_content for doc in documents]

    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Reordering the columns to have the content column first.
    df = df[[content_column_name] + [col for col in df.columns if col != content_column_name]]

    if with_embeddings:
        df["embeddings"] = embeddings_model.embed_documents(df[content_column_name].tolist())

    return df

