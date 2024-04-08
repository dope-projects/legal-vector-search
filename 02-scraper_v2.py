import json
import numpy

with open("opinions_5k.json", "r") as file:
    # Load the JSON data
    data = json.load(file)

# Extract specific fields using GPU vectorization
flattened_data = [{
    "resource_uri": result["resource_uri"],
    "id": result["id"],
    "absolute_url": result["absolute_url"],
    "date_created": result["date_created"],
    "date_modified": result["date_modified"],
    "page_count": result["page_count"],
    "download_url": result["download_url"],
    "local_path": result["local_path"],
    "author": result["author"],
    "plain_text": result["plain_text"],
    "html_with_citations": result["html_with_citations"]
} for result in data["results"]]

# Check the length of the flattened data
data_length = len(flattened_data)
print("Length of flattened data:", data_length)

with open("opinions_flattened_5k.json", "w") as file:
    json.dump(flattened_data, file)

# print(flattened_data)