

from langchain.tools import Tool
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import Graph, MessagesState

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
# from IPython.display import Image, display
from typing import TypedDict, List, Any

import os, time
from dotenv import load_dotenv

MAX_TOOL_CALLS = 2

load_dotenv(dotenv_path="C:\\Repo\\master\\PhysicalHealthChatbotCoT\\aws\\.env")

class CustomState(TypedDict):
    messages: List[dict]
    flag: bool
#     user_context: List[str]
#     user_aq: float
#     expert_context: List[str]
#     tool_call_ids: List[str]


def extract_symptoms(text: str) -> dict:
    # print(text)
    return "symptom: back pain"

def retrieve_user_context(query: str) -> str:
    # print(query)
    return "the user's name is William"

def retrieve_expert_context(query: str) -> str:
    # print(query)
    return "you should listen to your body"

def retrieve_user_activity_score(query: str) -> str:
    return f"user's current AQ / Activity Quotient / physical activity score is 16.4"
    # if query == "william":
    # return f"Could not find AQ for '{query}'"

def temp(query) -> str:
    print("TEMPFUNC passed parameter query: ", query)
    return "TEMPFUNC return value"


tool_extract_symptoms = Tool(
    name="extract_symptoms", 
    func=extract_symptoms, 
    description="Extract health facts from the user prompt to store for later use"
)

tool_retrieve_user_context = Tool(
    name="retrieve_user_context", 
    func=retrieve_user_context, 
    description="Fetch context about the user (e.g. symptoms and their situation in general). \
        This information is intended to help answer questions related to physical activity"
)

tool_retrieve_expert_context = Tool(
    name="retrieve_expert_context", 
    func=retrieve_expert_context, 
    description="Fetch context about physical health concepts from published medical articles."
)

tool_retrieve_user_activity_score = Tool(
    name="retrieve_user_activity_score", 
    func=retrieve_user_activity_score, 
    description="Fetch the user's current AQ / Activity Quotient / physical activity score"
)

tool_generate_asnwer = Tool(
    name="generate_asnwer", 
    func=temp, 
    description="Move to answer the user's query. You have gathered all the context you need for this prompt."
)

tool_no_tool = Tool(
    name="no_tool", 
    func=temp, 
    description="Move to answer the user's query. You have gathered all the context you need for this prompt."
)


llm = ChatOllama(
    base_url="http://localhost:11434",
    # base_url="http://ollama:11434",
    model="llama3.2:latest",
    temperature=0
)


def router(state: MessagesState):

    messages = [SystemMessage(content=
        "You are an assistant with access to tools. "
        "Only call a tool if it's necessary to answer the user's question. "
        # "If multiple tools seem relevant, call them one at a time in separate turns, based on available context. "
        # "e.g. you might need the output of one to query the other. "
        # "You must always use at least one too, even though you might want to answer directly - instead call the 'no_tool' tool"
        "If it's a simple / general question not related to physical health, just respond directly."
    )] + state["messages"]

    print("\n\n ********************Router MESSAGES: \t", messages, "\n\n")

    response = llm\
        .bind_tools([
            tool_no_tool, 
            tool_retrieve_expert_context, 
            tool_retrieve_user_context, 
            tool_retrieve_user_activity_score
        ]).invoke(messages)
    
    print("\n\n *********************Router RESPONSE: \t", response, "\n\n")
    
    return {"messages": state["messages"] + [response]}



def generate_answer(state: MessagesState):
    print("\n\ngenerate_answer ---> ", state, "\n\n")

    question = state["messages"][0].content
    context = ""

    for msg in state["messages"]:
        if type(msg) == ToolMessage:
            if msg.name != "no_tool":
                context = f"[{msg.content}] from {msg.name}, {context}"

    system_prompt = \
        f"You are an assistant. "\
        f"You may use the following pieces of retrieved context to answer the user "\
        f"If you don't know the answer, just say that you don't know. "\
        f"Use three sentences maximum and keep the answer concise.\n"\
        f"Context: [{context}]"
    
    print("PROMPT", system_prompt)
    print("\n\n", question, "\nCCCCCCC\n")
    
    response = llm.invoke([
        {"role": "system", "content": system_prompt}, 
        {"role": "user", "content": question}
    ])

    return {"messages": [response]}


def router_select_tool(state: MessagesState):

    # for msg in state["messages"]:
    print("BBBBBBSSSSSSSSSSSSSSSSS router select tool state", state, "\n\n")

    msg = state["messages"][-1]
    print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX", type(msg), msg)
    tool_calls = msg.tool_calls
    print(type(tool_calls))
    print(len(tool_calls))


    if len(tool_calls) > 0:

        tool_list = []
        for tool in tool_calls:
            tool_list.append(tool["name"])

        print(tool_list)
        return tool_list
        return {"next": tool_list}
        b = tool_calls[0]["name"]
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX_____XXX", type(b), b)
        return b
    
    
    # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX_____", type(a), a)
    return "no_tool"

def router_user_context(state: MessagesState):

    print("KKKKKKKKKKKKKKKKKKKKKKKKKKKK", state)





    return END



def safe_tool_node_factory(tool):
    def node(state):
        print("SAFE TOOOOL", state, "\n\n")
        messages = state["messages"]
        tool_messages = []

        for msg in reversed(messages):

            if isinstance(msg, AIMessage) and len(msg.tool_calls) > 0:

                for call in msg.tool_calls:

                    if call["name"] == tool.name:

                        arg = call["args"].get("__arg1", "")

                        result = tool.func(arg)

                        tool_messages.append(
                            ToolMessage(
                                content=result,
                                name=tool.name,
                                tool_call_id=call["id"],
                            )
                        )
                break

        print("TOOOOL MESSAGESS:::", tool_messages)
        return {"messages": messages + tool_messages}
    return node



workflow = StateGraph(MessagesState)
workflow.add_node(router)
# workflow.add_node("retrieve_user_context", ToolNode([tool_retrieve_user_context]))
# workflow.add_node("retrieve_user_activity_score", ToolNode([tool_retrieve_user_activity_score]))
# workflow.add_node("retrieve_expert_context", ToolNode([tool_retrieve_expert_context]))
# workflow.add_node("retrieve_user_context", SafeToolNode([tool_retrieve_user_context]))
# workflow.add_node("retrieve_user_activity_score", SafeToolNode([tool_retrieve_user_activity_score]))
# workflow.add_node("retrieve_expert_context", SafeToolNode([tool_retrieve_expert_context]))
workflow.add_node("retrieve_user_context", safe_tool_node_factory(tool_retrieve_user_context))
workflow.add_node("retrieve_user_activity_score", safe_tool_node_factory(tool_retrieve_user_activity_score))
workflow.add_node("retrieve_expert_context", safe_tool_node_factory(tool_retrieve_expert_context))
workflow.add_node(generate_answer)
# workflow.add_node("no_tool", ToolNode([tool_no_tool]))
# workflow.add_node("retrieve_expert_context", ToolNode([tool_retrieve_expert_context]))
# workflow.add_node("retrieve_user_context", ToolNode([tool_retrieve_user_context]))

workflow.add_edge(START, "router")
workflow.add_conditional_edges(
    "router",
    router_select_tool,
    {
        "no_tool": "generate_answer", 
        "retrieve_expert_context": "retrieve_expert_context", 
        "retrieve_user_context": "retrieve_user_context",
        "retrieve_user_activity_score": "retrieve_user_activity_score"
    }
)
workflow.add_edge("retrieve_expert_context", "generate_answer")
workflow.add_edge("retrieve_user_context", "generate_answer")
workflow.add_edge("retrieve_user_activity_score", "generate_answer")
# workflow.add_conditional_edges(
#     "retrieve_user_context",
#     router_user_context,
#     {
#         END: "generate_answer", 
#         "retrieve_user_activity_score": "retrieve_user_activity_score"
#     }
# )
# workflow.add_conditional_edges(
#     "evaluate_retrieved_context",
#     router_max_num_tools_reached,
#     {
#         "fetch_more_context": "router",
#         "max_num_tool_calls": "generate_answer"
#     }
# )
# workflow.add_edge("evaluate_retrieved_context", "router")
workflow.add_edge("generate_answer", END)
# workflow.add_edge("retrieve_expert_context", "router")
# workflow.add_edge("retrieve_user_context", "generate_answer")
# workflow.add_edge("no_tool", "generate_answer")


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
                # "content": "hello", 
                # "content": "what are possible symptoms of the drug 'Herbamed solice'?",
                # "content": "am i working out enough?",
                # "content": "do not fetch any expert knowledge. What is my name?",
                # "content": "What is my name, and what is my AQ?",
                "content": "what is my Activity Quotient?",
            }
        ]
    }
)

print("ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ\n\n", messages)

for chunk in messages:
    for node, update in chunk.items():
        print("Node: ", node)
        print("Update: ", update)
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

# input_state = {"messages": [
#     # {"role": "system", "content": "Do not use any tools."},
#     {"role": "system", "content": "You are a helpful assistant. If the user's message requires using a tool, you may use one. If not, just respond normally."},
#     {"role": "user", "content": "hello!"}
# ]}
# response = generate_answer(input_state)
# response["messages"][-1].pretty_print()

# # generate_query_or_respond(input_state)["messages"][-1].pretty_print()

# time.sleep(0.1)
# exit(0)

# graph = Graph()\
#     .add_node(generate_query_or_respond, name="generate_query_or_respond")


# executor = GraphExecutor(
#     graph,
#     stop_on=["generate_answer"],  #name of edge
#     max_iterations=5
# )




# input = {"messages": [{"role": "user", "content": "hello!"}]}

# # Basic
# generate_query_or_respond(input)["messages"][-1].pretty_print()

# # Graph
# executor.invoke(input)["messages"][-1].pretty_print()


# time.sleep(0.1)
# exit(0)




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













