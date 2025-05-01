

import json
from datetime import datetime, timezone

import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import MetadataQuery, Sort

from fastapi import FastAPI, Query
from fastapi import Request

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_ollama.chat_models import ChatOllama
from langchain.schema import HumanMessage, SystemMessage, AIMessage

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


@app.get("/")
async def root():

    return_text = []

    return_text.append("API is running")

    vector_db_schema = "vector db schema:"
    collections = client.collections.list_all()

    return collections


async def process_llm_query(data, context = ""):

    history = []

    if "history" not in data: 
        data["history"] = []

    ai_role_message = \
        """You are a medical assistant, personal trainer and motivator!
        Use the provided information and context to answer accurately.
        Do not answer confidently if you are unsure about a question."""
    
    history.append(SystemMessage(content=ai_role_message))

    chat = ChatOllama(
        base_url="http://ollama:11434", 
        model="llama3.2:latest"
    )

    for role, content in data["history"]:
        print(role)
        print(content)

        if role == "USER":
            history.append(HumanMessage(content=content))

        elif role == "AI":
            history.append(AIMessage(content=content))

        elif role == "SYSTEM":
            history.append(SystemMessage(content=content))

    if len(context) > 0:
        history.append(SystemMessage(content=f"Context about the user: {context}"))
        data["history"].append(["SYSTEM", f"Context about the user: {context}"])

    prompt = data["query"]

    history.append(HumanMessage(content=prompt))
    data["history"].append(["USER", prompt])

    print("HISTORY", history)

    response = chat(history)

    data["history"].append(["AI", response.content])

    return {"response": response.content, "history": data["history"]}


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


def db_create_object(payload: list, collection: str):

    symptoms = client.collections.get(collection)

    uuids = []
    for symptom in payload:

        iso_time_utc = datetime.now(timezone.utc).isoformat()
  
        symptom["date_recorded"] = iso_time_utc
        symptom["location"] = [0, 0]

        print(symptom)

        uuid = symptoms.data.insert(symptom)

        uuids.append(uuid)
    
    return uuids


# Program version: 1 (MVP)
@app.post("/api/query")
async def query(request: Request):

    data = await request.json()
    
    return await process_llm_query(data)


# Program version: 2 (with context)
@app.post("/api/query_with_context")
async def query_with_context(request: Request):

    data = await request.json()

    user_query = data["query"]

    # Context fetching
    db_response = await process_db_query(user_query, "Symptoms")
    
    # Context extraction
    extracted, save_symptoms = process_symptom_extraction(user_query)

    print(extracted)
    if save_symptoms:
        uuids = db_create_object(extracted, collection="Symptoms")

    # Context builder
    context_builder = []
    for o in db_response.objects:
        context_builder.append(
            f"Recorded {o.properties['symptom']}, \
            with {o.properties['symptom_confidence']} confidence, \
            at a time of {o.properties['recency_specified']} - \
            recorded at: {o.properties['date_recorded']}"
        )
    context = "\n".join(context_builder)


    return await process_llm_query(data, context)


@app.get("/api/create_collection")
async def create_collection():

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


# @app.post("/api/create_object")
# async def create_object_with_payload(request: Request):

#     data = await request.json()

#     payload = data["payload"]

#     uuids = db_create_object(payload=payload, collection=data["collection"])

#     return f"created object(s) with id(s): {uuids}"



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


async def process_db_query(term: str, collection_name: str):

    collection = client.collections.get(collection_name)

    response = collection.query.near_text(
        query=term,
        limit=10,
        return_metadata=MetadataQuery(distance=True, certainty=True, creation_time=True, last_update_time=True),
        return_properties=["symptom", "symptom_confidence", "date_recorded", "location", "recency_specified"],
        include_vector=True
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

    return response


@app.get("/api/get_near")
async def get_near(term: str = Query(..., description="Search term for Weaviate")):

    response = await process_db_query(term, "Symptoms")

    return {"response": response}

 




