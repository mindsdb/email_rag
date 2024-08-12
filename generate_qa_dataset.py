from pathlib import Path
from typing import List
import json
import logging
import os

import typer
from langchain_core.documents.base import Document
from loaders.directory_loader.directory_loader import DirectoryLoader
from settings import DEFAULT_LLM, DEFAULT_LLM_MODEL, DEFAULT_QA_GENERATION_PROMPT_TEMPLATE

_MAX_DOCUMENT_TOKENS = 60000

app = typer.Typer()

def _load_files(dataset: str) -> List[Document]:
    # Define the path to the source files directory
    source_files_path = Path('./data') / dataset / 'source_files'
    source_files = []
    for f in source_files_path.iterdir():
        if f.is_file():
            source_files.append(str(f))

    directory_loader = DirectoryLoader(source_files)
    logging.info('Loading documents from {}'.format(source_files_path))
    all_documents = directory_loader.load()
    logging.info('Documents loaded')
    return all_documents

@app.command()
def generate(
    dataset: str = typer.Option(..., help="Name of QA dataset to use for Q&A (e.g. personal_emails)"),
    log: str = typer.Option('INFO', help="Logging level to use (default INFO)")
):
    '''Generates question & answer pairs from a dataset (one per document).'''

    log_level = getattr(logging, log.upper())
    logging.basicConfig(level=log_level)

    all_documents = _load_files(dataset)
    num_documents = len(all_documents)
    all_examples = []
    for i, email_doc in enumerate(all_documents):
        logging.info(
            'Generating Q&A for doc {} / {}'.format(i + 1, num_documents))
        prompt = DEFAULT_QA_GENERATION_PROMPT_TEMPLATE.format(
            document=email_doc.page_content, metadata=json.dumps(email_doc.metadata))
        prompt_len = len(prompt)
        if prompt_len > _MAX_DOCUMENT_TOKENS:
            logging.info('Doc {} above max token limit of {} ({}). Continuing'.format(
                i + 1, _MAX_DOCUMENT_TOKENS, prompt_len))
            continue

        try:
            qa_json_str = DEFAULT_LLM.invoke(input=prompt).content
        except Exception as e:
            logging.info(
                'Failed to generate question. Continuing to next doc. Error: {}'.format(str(e)))
            continue

        try:
            qa_dict = json.loads(qa_json_str)
        except json.decoder.JSONDecodeError:
            logging.info(
                'Did not get expected Q&A JSON format. Continuing to next doc.')
            continue

        example = {
            'query': qa_dict['question'],
            'reference_answer': qa_dict['answer'],
            'query_by': {
                'model_name': DEFAULT_LLM_MODEL,
                'type': 'ai'
            },
            'reference_answer_by': {
                'model_name': DEFAULT_LLM_MODEL,
                'type': 'ai'
            }
        }
        all_examples.append(example)

    dataset_obj = {'examples': all_examples}

    dataset_path = os.path.join('./data', dataset, 'rag_dataset.json')
    with open(dataset_path, 'w') as dataset_file:
        json.dump(dataset_obj, dataset_file, indent=4)
    logging.info('Dataset saved to {}'.format(dataset_path))

if __name__ == '__main__':
    app()