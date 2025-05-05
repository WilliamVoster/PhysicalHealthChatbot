
import os
import pandas as pd
import sys
import re
import weaviate
import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_body_section(text):
    pattern = r"==== Body\s*\n[^\n]*\n(.*)"

    match = re.search(pattern, text, flags=re.DOTALL)

    front_removed = match.group(1).strip()
    
    pattern = r"==== Refs.*"

    body = re.sub(pattern, "", front_removed, flags=re.DOTALL)

    return body


if len(sys.argv) < 3:
    # print("Usage: python your_script.py <csv_filename_of_PMC-OA_filelisting> <num_articles_to_parse>")
    print("Usage: python your_script.py <csv_filename_of_PMC-OA_filelisting>")
    sys.exit(1)

data_dir_path = ["C:\\", "Users", "willi", "Documents", "NTNU", "Master", "data"]

# csv_path = "oa_comm_txt.PMC011xxxxxx.baseline.2024-12-17.filelist.csv"
csv_path = sys.argv[1]
# PMID_to_start_from = sys.argv[2]
num_articles_to_parse = sys.argv[2]


client = weaviate.connect_to_custom(
    http_host="localhost",
    http_port=8080,
    http_secure=False,
    grpc_host="localhost",
    grpc_port=50051,
    grpc_secure=False,
)

db_collection = client.collections.get("Articles_pubmed")

db_result = db_collection.query.fetch_objects(
    return_properties=["chunk", "full_article_id", "last_updated"],
    sort=weaviate.classes.query.Sort.by_property(name="_creationTimeUnix", ascending=False),
    limit=1
)

PMC_to_start_from = db_result.objects[0].properties["full_article_id"]

print("Starting from article:", PMC_to_start_from)

df = pd.read_csv(csv_path)
csv_index = df["AccessionID"].astype(str) == str(PMC_to_start_from)
match_index = df[csv_index].index[0]

for index, row in df.iloc[match_index:].iterrows():
    # print(index)
    if index - match_index > num_articles_to_parse:
        break

    article_path = os.path.join(*data_dir_path, row.iloc[0])
    article_PMCID = row.iloc[2]
    dt = datetime.datetime.strptime(row.iloc[3], "%Y-%m-%d %H:%M:%S")
    article_last_updated = dt.replace(tzinfo=datetime.timezone.utc).isoformat()

    print("Loading: ", article_path, article_PMCID, article_last_updated)

    with open(article_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    body_text = extract_body_section(raw_text)


    splitter_default = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 75)

    chunks = splitter_default.split_text(body_text)
    object = {}

    for chunk in chunks:
        # print("\n\n====================================================================")
        # print(chunk)

        object["chunk"] = chunk
        object["full_article_id"] = row.iloc[2]
        object["last_updated"] = article_last_updated

        uuid = db_collection.data.insert(object)


client.close()

