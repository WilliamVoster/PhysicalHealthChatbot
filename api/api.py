

import json
from datetime import datetime, timezone

import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import MetadataQuery, Sort

from fastapi import FastAPI, Query
from fastapi import Request

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_ollama.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableLambda

from langchain.tools import Tool
from langgraph.graph import StateGraph, START, END, Graph, MessagesState

import asyncio
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

app = FastAPI()

client = weaviate.connect_to_custom(
    http_host="weaviate_container",
    http_port=8080,
    http_secure=False,
    grpc_host="weaviate_container",
    grpc_port=50051,
    grpc_secure=False,
    # headers={
    #     "X-OpenAI-Api-Key": os.getenv("OPENAI_APIKEY")  # inference API keys
    # }
)

llm = ChatOllama(
    base_url="http://ollama:11434", 
    model="llama3.2:latest",
    temperature=0
)

llm_with_temperature = ChatOllama(
    base_url="http://ollama:11434", 
    model="llama3.2:latest",
    temperature=4
)

@app.get("/")
async def root():

    return_text = []

    return_text.append("API is running")

    vector_db_schema = "vector db schema:"
    collections = client.collections.list_all()

    return collections


async def process_llm_query(
        data, 
        user_context = "", 
        article_context = "", 
        feedback_box_examples = None,
        custom_system_prompt: str = None, 
        user_aq = None,
        user_aq_goal = None):

    history = []

    if "history" not in data: 
        data["history"] = []

    
    system_prompt = \
        f"You are an expert on physcial health, a personal trainer and motivator! " \
        f"You are not a trained medical professional. "\
        f"Use the provided information and context to answer accurately. "\
        f"Do not answer confidently if you are unsure about a question. "\
        f"If you don't know the answer, just say that you don't know. "\
        f"Use three sentences maximum and keep the answer concise.\n"\
        # f"Context: [{context}]"
    
    if custom_system_prompt is None:
        history.append(SystemMessage(content=system_prompt))
    else:
        history.append(SystemMessage(content=custom_system_prompt))

    for role, content in data["history"]:
        print(role)
        print(content)

        if role == "USER":
            history.append(HumanMessage(content=content))

        elif role == "AI":
            history.append(AIMessage(content=content))

        elif role == "SYSTEM":
            history.append(SystemMessage(content=content))

    if user_aq is not None:
        content = \
            f"The user's activity quotient (AQ score) is currently: {user_aq}. \n" \
            f"The metric is the successor to the PAI score (personal activity intelligence). " \
            f"It scores a persons physcial activity, where 100 points gives the maximum possible health benefits. "

        if user_aq_goal is not None:
            content = f"{content} The user has set a goal for themselves of maintaining an AQ of {user_aq_goal}. "

        history.append(SystemMessage(content=content))
        data["history"].append(["SYSTEM", content])

    if len(user_context) > 0:
        content = f"Context about the user: {user_context}"

        history.append(SystemMessage(content=content))
        data["history"].append(["SYSTEM", content])

    if len(article_context) > 0:
        content = \
            f"Context fetched from raw chunks of published medical articles, "\
            f"please use this information with a pinch of salt, "\
            f"it is also fine to not use these in your response: {article_context}"
        
        history.append(SystemMessage(content=content))
        data["history"].append(["SYSTEM", content])

    if feedback_box_examples is not None:
        content = \
            f"Here are some examples of feedback boxes in JSON format: {feedback_box_examples}"
        
        history.append(SystemMessage(content=content))
        data["history"].append(["SYSTEM", content])


    prompt = data["query"]

    history.append(HumanMessage(content=prompt))
    data["history"].append(["USER", prompt])

    print("HISTORY", history)

    response = llm(history)

    data["history"].append(["AI", response.content])

    print("888888888888888888888888888888888888888888888888 data[history]: ", data["history"])

    return {"response": response.content, "history": data["history"]}


def process_llm_query_from_messages(messages: list):

    # user_context = ""
    # article_context = ""
    # aq = None
    history = []

    for msg in messages:

        if isinstance(msg, HumanMessage):
            history.append(["USER", msg.content])

        if isinstance(msg, ToolMessage):
            history.append(["SYSTEM", msg.content])

        if isinstance(msg, AIMessage):
            if len(msg.content) <= 0: 
                continue
                # history.append(["TOOL", ])

            history.append(["AI", msg.content])



    print("888888888888888888888888888888888888888888888888 agent messages: ", messages)
    print("888888888888888888888888888888888888888888888888 agent history: ", history)
    print("888888888888888888888888888888888888888888888888 agent response: ", messages[-1].content)
    
    return {"response": messages[-1].content, "history": history}


def process_symptom_extraction(user_query: str):

    model = OllamaLLM(
        base_url="http://ollama:11434", 
        model="llama3.2:latest"
    )

    llm_query = f"""You are extracting health and activity-related facts and symptoms from user prompts. 
Only include what is explicitly or clearly implied.
Output only the symptoms as a comma-separated list of objects, JSON format. 
No other words. No explanations.
For queries where no symptoms are present, please respond with the character: 0

An example of a prompt and accompanying desired response is provided:
User prompt: "I use a wheelchair but I want to build upper body strength. I had a rotator cuff injury last year, 
so I’m easing back into it. I feel really good about my current mobility situation now though, it was somewhat harder 
right after the accident and I could only focus on the negatives." 
Output: 
[ 
    {{ 
        "symptom": "wheelchair user",
        "symptom_confidence": "high",
        "recency_specified": "not new",
    }},
    {{
        "symptom": "rotator cuff injury",
        "symptom_confidence": "high",
        "recency_specified": "1 year ago"
    }}
]

A second example is provided:
User prompt: "Can you make me a work-out plan for this week, i want to work on chest and lower back?
I went on a hike last week and stepped wrong, hurts a bit but I had a great time overall."
Output:
[
    {{
        "symptom": "sprained ankle",
        "symptom_confidence": "low",
        "recency_specified": "last week"
    }},
    {{
        "symptom": "lower back issues",
        "symptom_confidence": "low",
        "recency_specified": "no data"
    }}
]

A third example is provided:
User prompt: "I fell down the ladder yesterday and my back hurts today. What football team should I go see next?
Last week I came down with a fever, think I might have had a cold"
Output:
[
    {{
        "symptom": "back pain",
        "symptom_confidence": "high",
        "recency_specified": "yesterday"
    }},
    {{
        "symptom": "the common cold",
        "symptom_confidence": "medium",
        "recency_specified": "last week"
    }}
]

Now your turn, remember that no findings is an ok outcome (respond with 0), here is the new user prompt to be parsed:
User prompt: "{user_query}" """
    

    response = model.invoke(llm_query)

    try:
        extracted_json = json.loads(response)

        if "symptom" not in extracted_json[0]:
            return "", False

        return extracted_json, True

    except Exception as e:
        return e, False


def build_user_context(db_response):

    context_builder = []

    for o in db_response.objects:
        context_builder.append(
            f"Recorded {o.properties['symptom']}, "
            f"with {o.properties['symptom_confidence']} confidence, "
            f"at a time of {o.properties['recency_specified']} - "
            f"recorded at: {o.properties['date_recorded']}"
        )

    context = "\n".join(context_builder)
    return context


def build_article_context(db_response):

    context_builder = []

    for o in db_response.objects:
        context_builder.append(
            f"\nThis raw chunk comes from article {o.properties['full_article_id']}, "
            f"last updated at {o.properties['last_updated'].date()} - "
            f"Recorded chunk: {o.properties['chunk']}"
        )
        
    context = "\n".join(context_builder)
    return context


def build_combined_article_context(objects: tuple[object, float]):
    
    context_builder = []

    for (o, score) in objects:

        if "full_article_id" in o.properties:
            context_builder.append(
                f"This raw chunk comes from article {o.properties['full_article_id']}, "
                f"last updated at {o.properties['last_updated'].date()} - "
            )
        else:
            context_builder.append(
                f"This raw chunk comes from Mia Health's article {o.properties['source']} - "
            )

        context_builder.append(
            f"Recorded chunk: {o.properties['chunk']}"
        )
    
    context = "\n".join(context_builder)
    return context


def rerank_article_chunks(search_term, db_response_pubmed, db_response_miahealth):
    
    def score_article(query: str, chunk: str) -> float:
        prompt = \
            f"Query: \"{query}\"\n\n"\
            f"Document: \"{chunk}\"\n\n"\
            f"How relevant is the document to the provided query?"\
            f"Please provide a rating from 1 to 10."\
            f"Only reply with the number."
        
        response = llm.invoke(prompt)
        
        score = float(response.content)

        return score

    combined_list = []

    for o in db_response_pubmed.objects:
        # score = score_article(search_term, o.properties["chunk"])
        score = o.metadata.distance
        combined_list.append((o, score))

    for o in db_response_miahealth.objects:
        # score = score_article(search_term, o.properties["chunk"])
        score = o.metadata.distance
        combined_list.append((o, score))

    combined_list.sort(key = lambda x : x[1], reverse=False)

    for i in combined_list:
        print(i[0].properties["chunk"][:20], i[1])

    return combined_list


def db_create_symptom_object(payload: list):

    symptoms = client.collections.get("Symptoms")

    uuids = []
    for symptom in payload:

        iso_time_utc = datetime.now(timezone.utc).isoformat()
  
        symptom["date_recorded"] = iso_time_utc
        symptom["location"] = [0, 0]

        print(symptom)

        uuid = symptoms.data.insert(symptom)

        uuids.append(uuid)
    
    return uuids


async def process_db_query_symptoms(term: str):

    collection = client.collections.get("Symptoms")

    response = collection.query.near_text(
        query=term,
        limit=10,
        return_metadata=MetadataQuery(
            distance=True, 
            certainty=True, 
            creation_time=True, 
            last_update_time=True
        ),
        return_properties=[
            "symptom", 
            "symptom_confidence", 
            "date_recorded", 
            "location", 
            "recency_specified"
        ],
        include_vector=False
    )

    # Use autocut 1, 2 or 3
    # only return objects with continual dinstance, e.g. if there is a break in continuity

    
    # Search automatically sorts by distance
    # sort=Sort.by_property(
    #         name="_distance", ascending=True
    #     ).by_property(
    #         name="_creationTimeUnix", ascending=False
    #     ),

    # Use certainty as confidence to display to user

    print("ÆÆÆÆÆÆÆÆ process_db_query_symptoms ", response)

    return response


async def process_db_query_pubmed_articles(term: str):

    collection = client.collections.get("Articles_pubmed")

    response = collection.query.near_text(
        query=term,
        limit=10,
        return_metadata=MetadataQuery(
            distance=True, 
            certainty=True, 
            creation_time=True, 
            last_update_time=True
        ),
        return_properties=[
            "chunk", 
            "full_article_id", 
            "last_updated"
        ],
        include_vector=False
    )

    print("ØØØØØ process_db_query_pubmed_articles ", response)
    return response


async def process_db_query_miahealth_articles(term: str):

    collection = client.collections.get("Articles_miahealth")

    response = collection.query.near_text(
        query=term,
        limit=10,
        return_metadata=MetadataQuery(
            distance=True, 
            certainty=True, 
            creation_time=True, 
            last_update_time=True
        ),
        return_properties=[
            "chunk",
            "source"
        ],
        include_vector=False
    )

    print("ØØØØØ process_db_query_miahealth_articles ", response)
    return response


async def process_db_query_feedback_boxes(term: str):

    collection = client.collections.get("Feedback_boxes_miahealth")

    response = collection.query.near_text(
        query=term,
        limit=10,
        return_metadata=MetadataQuery(
            distance=True, 
            certainty=True, 
            creation_time=True, 
            last_update_time=True
        ),
        return_properties=[
            "rule",
            "tone_of_voice",
            "message"
        ],
        include_vector=False
    )

    print("YYYYY process_db_query_feedback_boxes ", response)
    return response


def filter_db_response_by_distance(response, max_distance: float = 0.5):

    filtered_response = {
        "objects": []
    }
    for o in response.objects:
        if o.metadata.distance < max_distance:
            filtered_response["objects"].append(o)

    return filtered_response
    

def get_activity_quotient() -> float:
    return 12.0

def get_activity_quotient_goal() -> float:
    return 60.0

def get_activity_quotient_string(term: str = "") -> str:
    aq = get_activity_quotient()
    aq_goal = get_activity_quotient_goal()

    return_string = ""

    if aq is not None:
        return_string = f"The user's current activity quotient (AQ) is {aq}. "

    if aq_goal is not None:
        return_string = f"{return_string}The user has set a goal for themselves of maintaining an AQ of {aq_goal}. "

    return return_string

async def retrieve_user_symptoms(term: str) -> str:
    db_response = await process_db_query_symptoms(term)
    context = build_user_context(db_response)
    return context

async def retrieve_articles(term: str) -> str:

    db_response_pubmed_articles = await process_db_query_pubmed_articles(term)
    db_response_miahealth_articles = await process_db_query_miahealth_articles(term)
    combined_articles = rerank_article_chunks(
        term, 
        db_response_pubmed_articles, 
        db_response_miahealth_articles)
    
    article_context = build_combined_article_context(combined_articles)
    return article_context
    

async def retrieve_feedback_box_examples(term: str) -> str:

    db_response = await process_db_query_feedback_boxes(term)

    filtered_response = filter_db_response_by_distance(db_response, max_distance=0.8)

    context = []

    for o in filtered_response["objects"]:
        context.append(o.properties)

    # print("ASVASKDGAJVASDASDASV", str(context))
    # print("ASVASKDGAJVASDASDASV", filtered_response)
    # print("ASVASKDGAJVASDASDASV", db_response)

    return f"Here are examples of feedback messages, i.e. templates and not data about the user.\n\n {str(context)}"


tool_retrieve_user_attributes = Tool(
    name="retrieve_user_attributes",
    func=retrieve_user_symptoms,
    description="Fetch attributes and symptoms about the user to use for additional context."
)

tool_retrieve_articles = Tool(
    name="retrieve_articles",
    func=retrieve_articles,
    description=\
        "Fetch chunks of text from published articles on physical health"
        " to use for additional context. Always supply a search term for this tool. "
)

tool_retrieve_user_activity_quotient = Tool(
    name="retrieve_user_activity_quotient",
    func= get_activity_quotient_string,
    description="Fetch the user's current AQ / Activity Quotient. This is a score from 0-100 of how physically active they are."
)

tool_retrieve_feedback_box_examples = Tool(
    name="retrieve_feedback_box_examples",
    func= retrieve_feedback_box_examples,
    description=\
        "Fetch a list of examples of how to properly write a feedback box message. "
        "The search term can help find examples closer related to a topic/tone/rule. "
        "Always call this tool if user mentions 'feedback'. "
)

tool_no_tool = Tool(
    name="no_tool", 
    func=lambda x : x, 
    description="Move to answer the user's query. You have gathered all the context you need for this prompt."
)


def node_router(state: MessagesState):
    
    messages = [SystemMessage(content=
        "You are an assistant with access to tools. "
        # "Only call a tool if it's necessary to answer the user's question. "
        "Call all the tools you think you would need to answer the user's question. "
        # "If multiple tools seem relevant, call them one at a time in separate turns, based on available context. "
        # "e.g. you might need the output of one to query the other. "
        # "You must always use at least one too, even though you might want to answer directly - instead call the 'no_tool' tool"
        "If it's a simple / general question not related to physical health, just respond directly."
    )] + state["messages"]

    print("\n\n ********************Router MESSAGES: \t", messages, "\n\n")

    response = llm\
        .bind_tools([
            tool_no_tool, 
            tool_retrieve_user_attributes, 
            tool_retrieve_articles, 
            tool_retrieve_user_activity_quotient,
            tool_retrieve_feedback_box_examples
        ]).invoke(messages)
    
    print("\n\n *********************Router RESPONSE: \t", response, "\n\n")
    
    return {"messages": state["messages"] + [response]}

def node_generate_answer(state: MessagesState):
    print("\n\ngenerate_answer ---> ", state, "\n\n")

    question = state["messages"][0].content
    context = None

    for msg in state["messages"]:
        if type(msg) == ToolMessage:
            if msg.name != "no_tool":
                context = f"[{msg.content}] from {msg.name}, the next context provided is:{context}"
    
    system_prompt = \
        f"You are an expert on physcial health, a personal trainer and motivator! " \
        f"You are not a trained medical professional. "\
        f"Use the provided information and context to answer accurately. "\
        f"Do not answer confidently if you are unsure about a question. "\
        f"If you don't know the answer, just say that you don't know. "\
        f"Use three sentences maximum and keep the answer concise.\n"\
        # f"Context: [{context}]"

    system_context = f"Context: {context}"

    print("PROMPT", system_prompt)
    print("\n\n", question, "\nCCCCCCC\n")


    full_prompt = []
    full_prompt.append({"role": "system", "content": system_prompt})
    if context is not None: full_prompt.append({"role": "system", "content": system_context})
    full_prompt.append({"role": "human", "content": question})
    
    response = llm.invoke(full_prompt)

    return {"messages": [response]}

feedback_box_instructions = \
    f"Your task is to generate a short feedback message to a user about their physical activity. "\
    f"The user should have a recorded Activity Quotient (aka. AQ) rating, which is calculated from the last 7 days. "\
    f"Feedback messages can either be a 'Call_to_action', 'Motivational', or 'Informational'. "\
    f"Use any available information about the user's situation to personalize as much as possible. "\
    f"Respond with a maximum of 2 sentences. "\
    f"Respond only with the feedback message and nothing else. "\

def node_router_feedback_box(state: MessagesState):

    messages = [SystemMessage(content=
        "You are an assistant with access to tools. "
        # "Only call a tool if it's necessary to answer the user's question. "
        "Call all the tools you think you would need to answer the user's question. "
        # "If multiple tools seem relevant, call them one at a time in separate turns, based on available context. "
        # "e.g. you might need the output of one to query the other. "
        # "You must always use at least one too, even though you might want to answer directly - instead call the 'no_tool' tool"
        # "If it's a simple / general question not related to physical health, just respond directly."
        f"{feedback_box_instructions}"
    )] + state["messages"]

    print("\n\n ********************Router_feedback_box MESSAGES: \t", messages, "\n\n")

    response = llm\
        .bind_tools([
            tool_no_tool, 
            tool_retrieve_user_attributes, 
            tool_retrieve_articles, 
            tool_retrieve_user_activity_quotient,
            tool_retrieve_feedback_box_examples
        ]).invoke(messages)
    
    print("\n\n *********************Router_feedback_box RESPONSE: \t", response, "\n\n")
    
    return {"messages": state["messages"] + [response]}

def node_generate_feedback_box(state: MessagesState):

    state["messages"].insert(0, SystemMessage(content=feedback_box_instructions))

    print(state["messages"])

    response = llm.invoke(state["messages"])
    # response = llm_with_temperature.invoke(state["messages"])

    return {"messages": [response]}

def router_select_tool(state: MessagesState):

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
    
    # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX_____", type(a), a)

    return "no_tool"

def safe_tool_node_factory(tool):
    async def node(state):
        print("SAFE TOOOOL", state, "\n\n")
        messages = state["messages"]
        tool_messages = []

        for msg in reversed(messages):

            if isinstance(msg, AIMessage) and len(msg.tool_calls) > 0:

                for call in msg.tool_calls:

                    if call["name"] == tool.name:

                        arg = call["args"].get("__arg1", "")

                        if asyncio.iscoroutinefunction(tool.func):
                            result = await tool.func(arg)
                        else:
                            result = tool.func(arg)

                        tool_messages.append(ToolMessage(
                            content=result,
                            name=tool.name,
                            tool_call_id=call["id"],
                        ))
                break

        print("TOOOOL MESSAGESS:::", tool_messages)
        return {"messages": messages + tool_messages}
    return node


workflow = StateGraph(MessagesState)
workflow.add_node("router", node_router)
workflow.add_node("retrieve_articles", safe_tool_node_factory(tool_retrieve_articles))
workflow.add_node("retrieve_user_attributes", safe_tool_node_factory(tool_retrieve_user_attributes))
workflow.add_node("retrieve_user_activity_quotient", safe_tool_node_factory(tool_retrieve_user_activity_quotient))
# workflow.add_node("retrieve_feedback_box_examples", safe_tool_node_factory(tool_retrieve_feedback_box_examples))
workflow.add_node("generate_answer", node_generate_answer)
# workflow.add_node("generate_feedback_box", node_generate_feedback_box)

workflow.add_edge(START, "router")
workflow.add_conditional_edges(
    "router",
    router_select_tool,
    {
        "no_tool": "generate_answer", 
        "retrieve_articles": "retrieve_articles", 
        "retrieve_user_attributes": "retrieve_user_attributes",
        "retrieve_user_activity_quotient": "retrieve_user_activity_quotient",
    }
)
workflow.add_edge("retrieve_articles", "generate_answer")
workflow.add_edge("retrieve_user_attributes", "generate_answer")
workflow.add_edge("retrieve_user_activity_quotient", "generate_answer")
# workflow.add_edge("retrieve_feedback_box_examples", "generate_feedback_box")
workflow.add_edge("generate_answer", END)
# workflow.add_edge("generate_feedback_box", END)

agent_graph = workflow.compile()

workflow = StateGraph(MessagesState)
workflow.add_node("router_feedback_box", node_router_feedback_box)
workflow.add_node("retrieve_articles", safe_tool_node_factory(tool_retrieve_articles))
workflow.add_node("retrieve_user_attributes", safe_tool_node_factory(tool_retrieve_user_attributes))
workflow.add_node("retrieve_user_activity_quotient", safe_tool_node_factory(tool_retrieve_user_activity_quotient))
workflow.add_node("retrieve_feedback_box_examples", safe_tool_node_factory(tool_retrieve_feedback_box_examples))
workflow.add_node("generate_feedback_box", node_generate_feedback_box)

workflow.add_edge(START, "router_feedback_box")
workflow.add_conditional_edges(
    "router_feedback_box",
    router_select_tool,
    {
        "no_tool": "generate_feedback_box", 
        "retrieve_articles": "retrieve_articles", 
        "retrieve_user_attributes": "retrieve_user_attributes",
        "retrieve_user_activity_quotient": "retrieve_user_activity_quotient",
        "retrieve_feedback_box_examples": "retrieve_feedback_box_examples"
    }
)
workflow.add_edge("retrieve_articles", "generate_feedback_box")
workflow.add_edge("retrieve_user_attributes", "generate_feedback_box")
workflow.add_edge("retrieve_user_activity_quotient", "generate_feedback_box")
workflow.add_edge("retrieve_feedback_box_examples", "generate_feedback_box")
workflow.add_edge("generate_feedback_box", END)

agent_graph_feedback_box = workflow.compile()


# Program version: 1 (MVP)
@app.post("/api/query")
async def query(request: Request):

    data = await request.json()

    aq = get_activity_quotient()
    aq_goal = get_activity_quotient_goal()
    
    return await process_llm_query(
        data, 
        user_aq=aq, 
        user_aq_goal=aq_goal
    )


# Variation of Program version: 1 (MVP), used to generate feedback box messages
@app.post("/api/query_feedback_box")
async def query_feedback_box(request: Request):

    data = await request.json()

    aq = get_activity_quotient()
    aq_goal = get_activity_quotient_goal()
    
    return await process_llm_query(
        data, 
        user_aq=aq, 
        user_aq_goal=aq_goal,
        custom_system_prompt=feedback_box_instructions
    )


# Program version: 2 (with user + article context)
@app.post("/api/query_with_context")
async def query_with_context(request: Request):

    data = await request.json()

    user_query = data["query"]
    aq = get_activity_quotient()
    aq_goal = get_activity_quotient_goal()

    # Context fetching
    search_term = user_query + f"\n\nActivity quotient (AQ):{aq}"
    db_response_symptoms = await process_db_query_symptoms(user_query)
    db_response_pubmed_articles = await process_db_query_pubmed_articles(user_query)
    db_response_miahealth_articles = await process_db_query_miahealth_articles(search_term)

    combined_articles = rerank_article_chunks(
        user_query, 
        db_response_pubmed_articles, 
        db_response_miahealth_articles
    )

    # Context building
    user_context = build_user_context(db_response_symptoms)
    article_context = build_combined_article_context(combined_articles)

    user_context = f"{user_context}. The user's current activity quotient (AQ) is {aq}."

    # Context extraction
    # extracted, save_symptoms = process_symptom_extraction(user_query)
    # if save_symptoms: uuids = db_create_symptom_object(extracted)

    
    return await process_llm_query(
        data, 
        user_context, 
        article_context, 
        user_aq=aq,
        user_aq_goal=aq_goal,
    )


# Variation of Program version: 2, used to generate feedback box messages
@app.post("/api/query_with_context_feedback_box")
async def query_with_context_feedback_box(request: Request):

    data = await request.json()

    user_query = data["query"]
    aq = get_activity_quotient()
    aq_goal = get_activity_quotient_goal()

    # Context fetching
    search_term = user_query + f"\n\nActivity quotient (AQ):{aq}"
    db_response_symptoms = await process_db_query_symptoms(term=user_query)
    db_response_pubmed_articles = await process_db_query_pubmed_articles(term=search_term)
    db_response_miahealth_articles = await process_db_query_miahealth_articles(term=search_term)
    db_response_feedback_box_examples = await retrieve_feedback_box_examples(term=search_term)

    combined_articles = rerank_article_chunks(
        user_query, 
        db_response_pubmed_articles, 
        db_response_miahealth_articles
    )

    # Context building
    user_context = build_user_context(db_response_symptoms)
    article_context = build_combined_article_context(combined_articles)

    return await process_llm_query(
        data, 
        user_context, 
        article_context,
        feedback_box_examples=db_response_feedback_box_examples,
        user_aq=aq,
        user_aq_goal=aq_goal,
        custom_system_prompt=feedback_box_instructions
    )


# Program version: 3 (agentified)
@app.post("/api/query_agent")
async def query_agent(request: Request):

    data = await request.json()

    user_query = data["query"]


    print(
        "\n##################################################################\n\n",
        agent_graph.get_graph().draw_mermaid(), 
        "\n\n##################################################################\n"
    )

    messages = await agent_graph.ainvoke({"messages": [{
        "role": "human",
        "content": user_query,
    }]})

    final_response = messages["messages"][-1]
    final_response.pretty_print()

    return process_llm_query_from_messages(messages["messages"])


# Variant of Program version: 3 (agentified), used to generate feedback box messages
@app.post("/api/query_agent_feedback_box")
async def query_agent_feedback_box(request: Request):

    data = await request.json()

    user_query = data["query"]

    print(
        "\n##################################################################\n\n",
        agent_graph_feedback_box.get_graph().draw_mermaid(), 
        "\n\n##################################################################\n"
    )

    messages = await agent_graph_feedback_box.ainvoke({"messages": [{
        "role": "human",
        "content": user_query,
    }]})

    final_response = messages["messages"][-1]
    final_response.pretty_print()

    return process_llm_query_from_messages(messages["messages"])
    


@app.get("/api/create_collection_symptoms")
async def create_collection_symptoms():

    client.collections.delete("Symptoms")

    symptoms = client.collections.create(
        "Symptoms",
        vectorizer_config=Configure.Vectorizer.text2vec_ollama(   
            api_endpoint="http://ollama:11434",    
            model="nomic-embed-text",
        ),
        vector_index_config=Configure.VectorIndex.hnsw(                 # Hierarchical Navigable Small World
            distance_metric=VectorDistances.COSINE                      # Default, and good for NLP
        ),
        # reranker_config=Configure.Reranker.cohere(),                    # Reranker improves ordering of results. SENDS ONLINE API CALLS! 
        properties=[
            Property(name="symptom", data_type=DataType.TEXT),
            Property(name="symptom_confidence", data_type=DataType.TEXT),
            Property(name="location", data_type=DataType.NUMBER_ARRAY),
            Property(name="date_recorded", data_type=DataType.DATE),
            Property(name="recency_specified", data_type=DataType.TEXT),
            Property(name="full_user_text", data_type=DataType.TEXT),
        ]
    )

    print("response", symptoms)
    return {"message": f"created collection: {symptoms}"}


@app.get("/api/create_collection_articles_pubmed")
async def create_collection_articles_pubmed():

    client.collections.delete("Articles_pubmed")

    collection_pubmed = client.collections.create(
        "Articles_pubmed",
        vectorizer_config=Configure.Vectorizer.text2vec_ollama(   
            api_endpoint="http://ollama:11434",    
            model="nomic-embed-text",
        ),
        vector_index_config=Configure.VectorIndex.hnsw(                 # Hierarchical Navigable Small World
            distance_metric=VectorDistances.COSINE                      # Default, and good for NLP
        ),
        properties=[
            Property(name="chunk", data_type=DataType.TEXT),
            Property(name="full_article_id", data_type=DataType.TEXT),
            Property(name="last_updated", data_type=DataType.DATE)
        ]
    )

    print("response", collection_pubmed)
    return {"message": f"created collection: {collection_pubmed}"}


@app.get("/api/create_collection_articles_miahealth")
async def create_collection_articles_miahealth():

    client.collections.delete("Articles_miahealth")

    collection_miahealth = client.collections.create(
        "Articles_miahealth",
        vectorizer_config=Configure.Vectorizer.text2vec_ollama(   
            api_endpoint="http://ollama:11434",    
            model="nomic-embed-text",
        ),
        vector_index_config=Configure.VectorIndex.hnsw(                 # Hierarchical Navigable Small World
            distance_metric=VectorDistances.COSINE                      # Default, and good for NLP
        ),
        properties=[
            Property(name="chunk", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT)
        ]
    )

    print("response", collection_miahealth)
    return {"message": f"created collection: {collection_miahealth}"}


@app.get("/api/create_collection_feedback_boxes")
async def create_collection_articles_miahealth():

    client.collections.delete("Feedback_boxes_miahealth")

    collection_feedback = client.collections.create(
        "Feedback_boxes_miahealth",
        vectorizer_config=Configure.Vectorizer.text2vec_ollama(   
            api_endpoint="http://ollama:11434",    
            model="nomic-embed-text",
        ),
        vector_index_config=Configure.VectorIndex.hnsw(                 # Hierarchical Navigable Small World
            distance_metric=VectorDistances.COSINE                      # Default, and good for NLP
        ),
        properties=[
            Property(name="rule", data_type=DataType.TEXT_ARRAY),
            Property(name="tone_of_voice", data_type=DataType.TEXT),
            Property(name="message", data_type=DataType.TEXT)
        ]
    )

    print("response", collection_feedback)
    return {"message": f"created collection: {collection_feedback}"}


@app.get("/api/create_object")
async def create_object():

    symptoms = client.collections.get("Symptoms")

    uuid = symptoms.data.insert({
        "symptom": "broken right ankle",
        "symptom_confidence": "high",
        "location": [7.0, 15.0],
        "date_recorded": "2025-05-01T12:08:00+00:00",
        "recency_specified": "not sure"
    })

    print("object's id: ", uuid)
    return {"message": f"created object with id: {uuid}"}


@app.post("/api/delete_object")
async def delete_object(request: Request):

    data = await request.json()

    collection = client.collections.get(data["Collection"])

    uuid = data["uuid"]

    try:
        collection.data.delete_by_id(uuid)

        return {"message": f"Object deleted with id: {uuid}"}
    
    except weaviate.exceptions.UnexpectedStatusCodeError as e:

        return {
            "message": f"Could not delete object with id: {uuid} in collection: {collection['name']}", 
            "error": e.message,
            "status_code": e.status_code}
        

@app.get("/api/get_all")
async def get_all():

    symptoms = client.collections.get("Symptoms")
    return_data = []

    for item in symptoms.iterator():
        return_data.append({"uuid": item.uuid, "properties": item.properties})

    return {"message": return_data}


@app.get("/api/get_all_chunks")
async def get_all_chunks():
    
    collection = client.collections.get("Articles_pubmed")
    return_data = []

    for item in collection.iterator():
        return_data.append({"uuid": item.uuid, "properties": item.properties})

    return {"message": return_data}


@app.get("/api/get_symptomp_near")
async def get_symptom_near(term: str = Query(..., description="Search-term for Weaviate")):

    response = await process_db_query_symptoms(term)

    return {"response": response}


@app.get("/api/get_pubmed_article_near")
async def get_article_near(term: str = Query(..., description="Search-term for Weaviate")):

    response = await process_db_query_pubmed_articles(term)

    return {"response": response}


@app.get("/api/get_miahealth_article_near")
async def get_article_near(term: str = Query(..., description="Search-term for Weaviate")):

    response = await process_db_query_miahealth_articles(term)

    return {"response": response}


@app.get("/api/get_feedback_boxes_near")
async def get_article_near(term: str = Query(..., description="Search-term for Weaviate")):

    response = await process_db_query_feedback_boxes(term)

    filtered_response = filter_db_response_by_distance(response, max_distance=0.5)

    return {"response": filtered_response}

