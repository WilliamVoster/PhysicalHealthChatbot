


import weaviate
from weaviate.classes.config import Configure, Property, DataType, VectorDistances
from weaviate.classes.query import MetadataQuery, Sort
from fastapi import FastAPI, Query
from fastapi import Request
# from neo4j import GraphDatabase
from langchain_ollama import OllamaEmbeddings
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


# neo4j_driver = GraphDatabase.driver("bolt://neo4j:7687", auth=("neo4j", "password"))


@app.get("/")
async def root():

    return_text = []

    return_text.append("API is running")

    vector_db_schema = "vector db schema:"
    collections = client.collections.list_all()

    return collections

    for col_name in collections:

        # vector_db_schema += f"\n{collections[col_name]}"
        vector_db_schema += f"\n{collections[col_name].properties}"
        vector_db_schema += f"\n{collections[col_name].vectorizer}"
        vector_db_schema += f"\n{collections[col_name].vectorizer_config}"

    return_text.append(vector_db_schema)

    return "\n".join(return_text)

    # ollama_embedder = OllamaEmbeddings(
    #     base_url="http://ollama:11434", 
    #     model="llama3.2:latest"
    # )
    # prompt = "Please tell me a joke about the sun."
    # response = ollama_embedder.embed_query(prompt)
    # return {"message": f"Embedding:{response}"}

# @app.get("/graph")
# async def get_graph():
#     with neo4j_driver.session() as session:
#         result = session.run("MATCH (n) RETURN n LIMIT 1")
#         return [record["n"] for record in result]


async def process_llm_query(data):

    history = []

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

    prompt = data["query"]

    history.append(HumanMessage(content=prompt))
    data["history"].append(["USER", prompt])

    response = chat(history)

    data["history"].append(["AI", response.content])

    return {"response": response.content, "history": data["history"]}


@app.post("/api/query")
async def query(request: Request):

    data = await request.json()
    
    return await process_llm_query(data)


@app.post("/api/query_with_context")
async def query_with_context(request: Request):

    data = await request.json()

    search_term = data["query"]

    db_response = await process_db_query(search_term, "Symptoms")

    context_builder = []
    for o in db_response.objects:

        context_builder.append(f"{o.properties['entity']} is {o.properties['problem']}")
    
    context = "\n".join(context_builder)

    data["query"] = f"Context: \n{context} \n\n Query: {search_term}"

    return await process_llm_query(data)



@app.get("/create_object")
async def create_object():

    symptoms = client.collections.get("Symptoms")

    uuid = symptoms.data.insert({
        "entity": "right ankle",
        "problem": "broken",
        "location": [7.0, 15.0]
    })

    print("object's id: ", uuid)
    return {"message": f"object's id: {uuid}"}


@app.get("/create_collection")
async def create_collection():

    #     generative_config = Configure.Generative.ollama(            # generative integration
    #         api_endpoint="http://host.docker.internal:11434",       # Allow Weaviate from within a Docker container to contact your Ollama instance
    #         model="llama3.2",
    #     )
    # )

    client.collections.delete("Symptoms")
    symptoms = client.collections.create(
        "Symptoms",
        vectorizer_config=Configure.Vectorizer.text2vec_ollama(   
            api_endpoint="http://host.docker.internal:11434",    
            model="nomic-embed-text",
        ),
        vector_index_config=Configure.VectorIndex.hnsw(                 # Hierarchical Navigable Small World
            distance_metric=VectorDistances.COSINE                      # Default, and good for NLP
        ),
        # reranker_config=Configure.Reranker.cohere(),                    # Reranker improves ordering of results. SENDS ONLINE API CALLS! 
        properties=[
            Property(name="entity", data_type=DataType.TEXT),
            Property(name="problem", data_type=DataType.TEXT),
            Property(name="location", data_type=DataType.NUMBER_ARRAY),
            Property(name="date", data_type=DataType.DATE),
        ]
    )

    print("response", symptoms)
    return {"message": f"created collection: {symptoms}"}


@app.post("/api/create_object")
async def create_object(request: Request):

    data = await request.json()

    collection = client.collections.get(data["Collection"])

    uuid = collection.data.insert(data["Object"])

    print("object's id: ", uuid)
    return {"message": f"object's id: {uuid}"}


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
        

    

@app.get("/get_all")
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
        return_metadata=MetadataQuery(distance=True, creation_time=True, last_update_time=True),
        return_properties=["entity", "problem", "date", "location"],
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

    return response


@app.get("/get_near")
async def get_near(term: str = Query(..., description="Search term for Weaviate")):

    response = await process_db_query(term, "Symptoms")

    return {"response": response}

 




