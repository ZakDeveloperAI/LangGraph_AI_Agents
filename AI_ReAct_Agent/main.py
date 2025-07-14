from typing import Annotated, Sequence, TypedDict 
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage # The foundational class for all message types in LangGraph
from langchain_core.messages import ToolMessage # Passes data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage,HumanMessage # Message for providing instructions to the LLM
from langchain_google_genai import ChatGoogleGenerativeAI # The LLM used for generating responses
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

#REMEMBER THAT LANGCHAIN ALREADY HAVE A PREBUILT REACT AGENT TO USE 

load_dotenv()


#ANNOTATED
    #email = Annotated[str, "Email address of the user"] 
    #print(email.__metadata__)  # Access metadata of the annotated type

#SEQUENCE gestisce automaticamente cambiamenti allo STATE evitando manipolazione manuale 
# esempio: aggiungere messaggi alla chat history

class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages ] #add_messages is a reducer function to not overwrite

@tool
def add(a:int, b:int) -> int:
    """This is an addition function that adds 2 numbers"""
    return a + b    

tools=[add]
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5).bind_tools(tools)


def agent_node(state:State) -> State:
    system_prompt=SystemMessage(content="You are a helpful but synthetics assistant answer only with the necessary info")
    messages = [system_prompt] + list(state["messages"])
    response=llm.invoke(messages)
    state["messages"]=response
    return state


def should_continue(state:State)->str:
    last_message=state["messages"][-1]
    if not last_message.tool_calls:
        return "end"
    else:
        return "continue"
    
    
graph_builder=StateGraph(State)
graph_builder.add_node("agent", agent_node)

tool_node=ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)


graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "tools","end": END}
)
graph_builder.add_edge("tools", "agent")

graph=graph_builder.compile()

initial_state=State(messages=[HumanMessage(content="5-7")])

final_state=graph.invoke(initial_state)

print(final_state["messages"])  # Output the final message from the agent
# The final message should be the response from the LLM after processing the input messages and tool calls.




