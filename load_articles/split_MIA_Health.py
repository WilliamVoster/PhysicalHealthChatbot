import weaviate
import os, sys
from langchain.text_splitter import RecursiveCharacterTextSplitter


if len(sys.argv) < 2:
    print("Usage: python split_MIA-Health.py <directory_with_.txt_files>")
    sys.exit(1)

# data_dir_path = "./mia_health_articles"
data_dir_path = sys.argv[1]

client = weaviate.connect_to_custom(
    http_host="localhost",
    http_port=8080,
    http_secure=False,
    grpc_host="localhost",
    grpc_port=50051,
    grpc_secure=False,
)

db_collection = client.collections.get("Articles_miahealth")

splitter_default = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 75)


if not os.path.isdir(data_dir_path):
    print(f"Error: {data_dir_path} is not valid")


for filename in os.listdir(data_dir_path):
    filepath = os.path.join(data_dir_path, filename)

    if os.path.isfile(filepath):
        try: 
            with open(filepath, "r", encoding="utf-8") as file:
                raw_text = file.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

        chunks = splitter_default.split_text(raw_text)

        object = {}
        for chunk in chunks:

            object["chunk"] = chunk
            object["source"] = filename

            uuid = db_collection.data.insert(object)

            print(f"inserted chunk with uuid: {uuid}")

    else:
        print(f"Skipping {filepath}")

client.close()

