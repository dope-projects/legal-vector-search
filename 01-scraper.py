import requests
import json

# Define the API endpoint
api_url = "https://www.courtlistener.com/api/rest/v3/opinions/"

# Set parameters to limit the dataset
params = {
    "limit": 20,  # Limit the number of opinions per page (max allowed by the API)
    "ordering": "-date_filed",  # Order by date filed, descending
    # You can add more parameters like jurisdiction, date range, etc.
}

# List to store all the results
all_results = []

# Total limit required (5k)
total_limit = 5000

# Number of pages required
num_pages = total_limit // params["limit"] + (1 if total_limit % params["limit"] else 0)

# Paginate through the pages
for page_number in range(1, num_pages + 1):
    params["page"] = page_number
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        # Parse the JSON response
        data = response.json()
        print(page_number)
        # Append the results to the list
        all_results.extend(data["results"])
    else:
        print("Error:", response.status_code)
        break

# Save all the results to a file
with open("opinions_5k.json", "w") as file:
    json.dump(all_results, file)
print("JSON data saved to opinions.json")
