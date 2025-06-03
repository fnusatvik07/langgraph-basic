from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
load_dotenv()


@dataclass
class MyState:
    topic: str
    joke: str = ""


llm=init_chat_model("anthropic:claude-3-5-sonnet-latest")

def call_model(state: MyState):
    """Call the LLM to generate a joke about a topic"""
    llm_response = llm.invoke( 
        [
            {"role": "user", "content": f"Generate a joke about {state.topic}"}
        ]
    )
    return {"joke": llm_response.content}

graph = (
    StateGraph(MyState)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .compile()
)

for message_chunk, metadata in graph.stream( 
    {"topic": "ice cream"},
    stream_mode="messages",
):
    if message_chunk.content:
        print(message_chunk.content, end="|", flush=True)