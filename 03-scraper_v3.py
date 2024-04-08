import json

with open("opinions_2k.json", "r") as file:
    # Load the JSON data
    data = json.load(file)

# Extract specific fields using GPU vectorization
flattened_data = []

# Iterate through each page's results

for result in data:
    flattened_result = {
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
    }
    flattened_data.append(flattened_result)

# Check the length of the flattened data
data_length = len(flattened_data)
print("Length of flattened data:", data_length)

with open("opinions_flattened_2k.json", "w") as file:
    json.dump(flattened_data, file)
