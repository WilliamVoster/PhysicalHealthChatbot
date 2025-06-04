import weaviate
import os, sys, json


if len(sys.argv) < 2:
    print("Usage: python load_feedback_exmaples.py <feedback_box_examples.json>")
    sys.exit(1)

file_path = sys.argv[1]

client = weaviate.connect_to_custom(
    http_host="localhost",
    http_port=8080,
    http_secure=False,
    grpc_host="localhost",
    grpc_port=50051,
    grpc_secure=False,
)

db_collection = client.collections.get("Feedback_boxes_miahealth")


if not os.path.isfile(file_path):
    print(f"Error: {file_path} is not valid")

with open(file_path, "r") as file:
    data = json.load(file)

for feedback_box in data["feedback_messages"]:

    # print(feedback_box)

    uuid = db_collection.data.insert(feedback_box)

    print(f"inserted chunk with uuid: {uuid}")


client.close()

