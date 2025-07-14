from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

class State(TypedDict):
    messages: List[HumanMessage]

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

def process_node(state: State) -> State:
    response=llm.invoke(state["messages"])
    print("AI: " + response.content)
    return state

graph_builder = StateGraph(State)
graph_builder.add_node("process",process_node)
graph_builder.add_edge(START, "process")
graph_builder.add_edge("process", END)
graph = graph_builder.compile()

initial_state = State(messages=[HumanMessage(content="Hello, how are you?")])
final_state = graph.invoke(initial_state)
