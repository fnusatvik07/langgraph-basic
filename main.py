from typing import TypedDict, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.messages import ToolMessage, HumanMessage
import json

# Load environment variables
load_dotenv()

# Step 1: Define the State structure
class State(TypedDict):
    question: str
    messages: Annotated[list, add_messages]

# Step 2: Initialize OpenAI LLM and tools
llm = init_chat_model("openai:gpt-4o")
tool = TavilySearch(max_results=5)
tools = [tool]
llm_tool = llm.bind_tools(tools)

# Step 3: Define LLM response node using full message history
def llm_response(state: State) -> State:
    messages = state["messages"]
    response = llm_tool.invoke(messages)
    return {
        **state,
        "messages": [response]
    }

# Step 4: Define Tool Execution Node that returns tool output clearly

class BasicToolNode:
    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        messages = inputs.get("messages", [])
        if not messages:
            raise ValueError("No messages found in input")
        last_msg = messages[-1]
        outputs = []
        for tool_call in last_msg.tool_calls:
            result = self.tools_by_name[tool_call["name"]].invoke(tool_call["args"])
            formatted_result = json.dumps(result, indent=2)  # Optional: pretty format
            outputs.append(
                ToolMessage(
                    content=f"Tool '{tool_call['name']}' returned: {formatted_result}",
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {
            "messages": messages + outputs
        }

# Step 5: Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", llm_response)
graph_builder.add_node("tools", BasicToolNode(tools))
graph_builder.add_edge(START, "chatbot")

def route_tools(state: State):
    messages = state.get("messages", [])
    last_msg = messages[-1] if messages else None
    if last_msg and hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return END

graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END}
)
graph_builder.add_edge("tools", "chatbot")

# Step 6: Compile the graph
graph = graph_builder.compile()

# Step 7: Interactive runner
def stream_graph_updates(user_input: str):
    for event in graph.stream({
        "question": user_input,
        "messages": [HumanMessage(content=user_input)]
    }):
        for value in event.values():
            last_msg = value["messages"][-1]
            print("Assistant:", getattr(last_msg, "content", "[No content]"))

if __name__ == "__main__":
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except Exception as e:
            print("⚠️ Error occurred:", str(e))
            fallback_question = "What do you know about LangGraph?"
            print("User:", fallback_question)
            stream_graph_updates(fallback_question)
            break
