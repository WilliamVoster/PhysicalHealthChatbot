


import weaviate
from weaviate.classes.config import Configure, Property, DataType
from fastapi import FastAPI
# from neo4j import GraphDatabase

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
    return {"message": "Hello, person!"}


# @app.get("/graph")
# async def get_graph():
#     with neo4j_driver.session() as session:
#         result = session.run("MATCH (n) RETURN n LIMIT 1")
#         return [record["n"] for record in result]

@app.get("/set_test")
async def set_test():

    client.collections.create(
        "Article",
        properties=[]
    )


@app.get("/create_collection")
async def create_collection():

    client.collections.delete("Question")
    
    questions = client.collections.create(

        name="Question",

        vectorizer_config = Configure.Vectorizer.text2vec_ollama(   # embedding integration
            api_endpoint="http://host.docker.internal:11434",       # Allow Weaviate from within a Docker container to contact your Ollama instance
            model="nomic-embed-text",
        ),

        generative_config = Configure.Generative.ollama(            # generative integration
            api_endpoint="http://host.docker.internal:11434",       # Allow Weaviate from within a Docker container to contact your Ollama instance
            model="llama3.2",
        )
    )
    print("response", questions)


@app.get("/get_vector")
async def get_vector():

    # result = client.query.get("Article", ["title"]).with_limit(1).do()

    result = client.collections.get("Article").query.near_text(
        query="example",
        limit=1,
        return_properties=["title"]
    )

    # result = client.collections.get("Article").query.bm25(        # keyword-based search
    #     query="example",
    #     limit=1,
    #     return_properties=["title"]
    # )

    # result = client.collections.get("Article").query.hybrid(      # vector and keyword search
    #     query="example",
    #     limit=1,
    #     return_properties=["title"]
    # )

    return result["data"]["Get"]["Article"]



# print("Closing...")
# client.close()













