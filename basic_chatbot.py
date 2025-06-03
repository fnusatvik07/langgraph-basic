# Step1: Create State

from typing import TypedDict,Annotated, List
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph,START,END

class State(TypedDict):
    user_question: str
    messages: Annotated[List,add_messages]


# Step 2: Create LLM

from langchain_openai import ChatOpenAI

llm=ChatOpenAI(model="gpt-4")

# Step 3: Create a Blank Graph

graph=StateGraph(State)

# Step 4: Create Node

def llm_response(state:State)-> State:
    question=state["user_question"]
    response=llm.invoke(question)
    return {
        **state,
        "messages":[response.content]
    }

# Step 5: Add Node and Edges to Graph

graph.add_node("AI-ANSWER",llm_response)
graph.add_edge(START,"AI-ANSWER")
graph.add_edge("AI-ANSWER",END)

builder=graph.compile()

#Print the Graph

builder.get_graph().print_ascii()
print(builder.get_graph().draw_mermaid())

builder.get_graph().draw_mermaid_png(output_file_path="graph.png")


for chunk in builder.stream( 
    {"user_question": "How many continents are there in the World"},
    stream_mode="updates", 
):
    print(chunk)
