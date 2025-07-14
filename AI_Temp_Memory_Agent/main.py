import os
from typing import List, TypedDict, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

#This will save memory only for the current session
#If you want to save memory across sessions, use a persistent storage solution like a vector database or a json file.
load_dotenv()

class State(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]
    
llm= ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

def process(state: State) -> State:
    """Process the current state by invoking the LLM with the messages."""
    response = llm.invoke(state["messages"])
    
    state["messages"].append(AIMessage(content=response.content))
    print("AI: " + response.content)
    
    return state

graph_builder= StateGraph(State)
graph_builder.add_node("process", process)
graph_builder.add_edge(START, "process")
graph_builder.add_edge("process", END)
graph = graph_builder.compile()

conversation_history=[]
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    
    conversation_history.append(HumanMessage(content=user_input))
    result = graph.invoke(State(messages=conversation_history))
    
    print(result["messages"][-1].content)  # Print the last AI message
    
    conversation_history.append(AIMessage(content=result["messages"][-1].content))

initial_state = State(messages=[HumanMessage(content="Hello, how are you?")])
final_state = graph.invoke(initial_state)