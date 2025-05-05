

import tarfile

filepath = "./data/oa_comm_txt.PMC011xxxxxx.baseline.2024-12-17.tar.gz"
destination_path = ""
with tarfile.open(filepath, "r:gz") as tar:
    tar.extractall(path="./data")

    for member in tar.getmembers():
        print(member.name)






