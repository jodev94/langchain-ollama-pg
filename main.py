from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from typing_extensions import TypedDict
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

# Initialize DB and LLM
db = SQLDatabase.from_uri("postgresql://postgres:postgres@localhost/postgres")
llm = ChatOllama(model="gemma3")

# TypedDict to track state
class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str

# System message
system_message = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run.
Only return the SQL query, and nothing else.
Limit results to {top_k} unless specified.
You MUST consult the table info provided.
You MUST only return the query string in plaintext. Do not format in any way.
Table Info:
{table_info}
""".strip()

# Combine into chat prompt template
query_prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("user", "Question: {input}")
])

# Main query function
def write_query(state: State):
    prompt = query_prompt_template.invoke({
        "dialect": db.dialect,
        "top_k": 5,
        "table_info": db.get_table_info(),
        "input": state["question"]
    })
    response = llm.invoke(prompt)
    return {"query": response.content.strip()}

# Example run
result = write_query({"question": "How many users are there?", "query": "", "result": "", "answer": ""})
print(result)


def execute_query(state: State):
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}


query_res = execute_query(result)

print(query_res)