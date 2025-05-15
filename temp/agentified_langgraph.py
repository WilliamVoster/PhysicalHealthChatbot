

from langchain.tools import Tool
# from langchain.agents import initialize_agent
# from langgraph.prebuilt import create_react_agent
# from langchain.chat_models import ChatOpenAI
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import Graph, MessagesState

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from IPython.display import Image, display
from typing import TypedDict, List, Any

import os, time
from dotenv import load_dotenv

load_dotenv(dotenv_path="C:\\Repo\\master\\PhysicalHealthChatbotCoT\\aws\\.env")

# class CustomState(TypedDict):
#     messages: List[dict]
#     user_context: List[str]
#     user_aq: float
#     expert_context: List[str]
#     tool_call_ids: List[str]



def extract_symptoms(text: str) -> dict:
    # print(text)
    return {"symptom": "back pain"}

def retrieve_user_context(query: str) -> str:
    # print(query)
    return {"the user had a fever last week."}

def retrieve_expert_context(query: str) -> str:
    # print(query)
    return {"you should listen to your body"}

def retrieve_user_activity_score(query: str) -> str:
    print(query)
    return {"AQ: 12.6"}
    
# def recommend_action(context: str, symptoms: dict) -> str:
#     # print(context, symptoms)
#     return {
#         "response": "LLM: refrain from physical activity for a while", 
#         "status": "FINAL"
#     }


tool_extract_symptoms = Tool(
    name="ExtractSymptoms", 
    func=extract_symptoms, 
    description="Extract health facts from the user prompt to store for later use"
)

tool_retrieve_user_context = Tool(
    name="FetchUserContext", 
    func=retrieve_user_context, 
    description="Fetch context about the user (e.g. symptoms and their situation)"
)

tool_retrieve_expert_context = Tool(
    name="FetchExpertContext", 
    func=retrieve_expert_context, 
    description="Fetch context about physical health from published medical articles."
)

tool_retrieve_user_activity_score = Tool(
    name="FetchUserActivityScore", 
    func=retrieve_user_activity_score, 
    description="Fetch the user's current activity score (aka. activity quotient)"
)

# tool_recommend_action = Tool(
#     name="GenerateAnswer", 
#     func=recommend_action, 
#     description="Generate the final recommendation or answer using all (if any) accumulated context.", 
#     return_direct=True
# )


llm = ChatOllama(
    base_url="http://localhost:11434",
    model="llama3.2:latest",
    temperature=0                      # temp of e.g. 1 or 1.5 gives more random responses, creative, varied
)


def generate_query_or_respond(state: MessagesState):

    messages = [SystemMessage(content=
        "You are an assistant with access to tools. "
        "Only call a tool if it's necessary to answer the user's question. "
        "If it's a simple greeting or a general question not related to physical health, just respond directly."
    )] + state["messages"]

    # print("\n\ngenerate_query_or_respond MESSAGES: \t", messages, "\n\n")

    response = llm\
        .bind_tools([tool_retrieve_expert_context])\
        .invoke(messages)
    
    # print("\n\ngenerate_query_or_respond RESPONSE: \t", response, "\n\n")
    
    return {"messages": [response]}


def generate_answer(state: MessagesState):
    # print("\n\ngenerate_answer ---> ", state, "\n\n")

    question = state["messages"][0].content
    # question = state["messages"][-1].content
    context = ""

    for msg in state["messages"]:
        if type(msg) == ToolMessage:
            context = f"{msg.content} from {msg.name}, {context}"

    prompt = \
        f"You are an assistant. "\
        f"You may use the following pieces of retrieved context to answer the user, or you may answer without the context. "\
        f"If you don't know the answer, just say that you don't know. "\
        f"Use three sentences maximum and keep the answer concise.\n"\
        f"Context: [{context}]"
    
    # print("PROMPT", prompt)
    
    response = llm.invoke([
        {"role": "system", "content": prompt}, 
        {"role": "user", "content": question}
    ])
    # response = llm.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


workflow = StateGraph(MessagesState)
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve_expert_context", ToolNode([tool_retrieve_expert_context]))
workflow.add_node(generate_answer)
workflow.add_edge(START, "generate_query_or_respond")
workflow.add_conditional_edges(
    "generate_query_or_respond",
    tools_condition,
    {"tools": "retrieve_expert_context", "generate_answer": "generate_answer"}
)
workflow.add_edge("retrieve_expert_context", "generate_answer")
workflow.add_edge("generate_answer", END)
graph = workflow.compile()

print(
    "\n##################################################################\n\n",
    graph.get_graph().draw_mermaid(), 
    "\n\n##################################################################\n"
)

messages = graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "Hello",
            }
        ]
    }
)

for chunk in messages:
    for node, update in chunk.items():
        print("Update from node", node)
        update["messages"][-1].pretty_print()
        print("\n\n")


time.sleep(0.1)
exit(0)


######################################################################################################
# from IPython.display import Image, display
# from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

# display(Image(graph.get_graph().draw_mermaid_png()))

# time.sleep(0.1)
# exit(0)
######################################################################################################

input_state = {"messages": [
    # {"role": "system", "content": "Do not use any tools."},
    {"role": "system", "content": "You are a helpful assistant. If the user's message requires using a tool, you may use one. If not, just respond normally."},
    {"role": "user", "content": "hello!"}
]}
response = generate_answer(input_state)
response["messages"][-1].pretty_print()

# generate_query_or_respond(input_state)["messages"][-1].pretty_print()

time.sleep(0.1)
exit(0)

graph = Graph()\
    .add_node(generate_query_or_respond, name="generate_query_or_respond")


executor = GraphExecutor(
    graph,
    stop_on=["generate_answer"],  #name of edge
    max_iterations=5
)




input = {"messages": [{"role": "user", "content": "hello!"}]}

# Basic
generate_query_or_respond(input)["messages"][-1].pretty_print()

# Graph
executor.invoke(input)["messages"][-1].pretty_print()


time.sleep(0.1)
exit(0)




# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent="zero-shot-react-description",
#     # ^^ react means Reason, Act then Repeat
#     verbose=True,
#     max_iterations=5,
#     early_stopping_method="generate"
# )


# How to fix

# make tool outputs clearer
# return {
#     "context": "you should listen to your body",
#     "safe_to_continue": "depends on severity; consult a professional",
#     "status": "FINAL"
# }

# add return_direct true if final step
# Tool(
#     name="FetchExpertContext",
#     func=retrieve_expert_context,
#     description="Fetch expert opinion about exercising with back pain.",
#     return_direct=True
# )













