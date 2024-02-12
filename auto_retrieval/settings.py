
DEFAULT_LLM_MODEL = "gpt-3.5-turbo"
DEFAULT_PROMPT_TEMPLATE = """
Below is a json representation of a table with information about {description}. 
Return a JSON list with an entry for each column. Each entry should have 
{{"name": "column name", "description": "column description", "type": "column data type"}}
\n\n{dataframe}\n\nJSON:\n
"""
