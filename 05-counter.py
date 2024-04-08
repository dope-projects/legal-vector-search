import json

# Open the JSON file
with open("opinions.json", "r") as file:
    # Load the JSON data
    data = json.load(file)

# Check the length of the JSON data
data_length = len(data["results"])
print("Length of JSON data:", data_length)