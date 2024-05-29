import requests
from bs4 import BeautifulSoup
import json
import os


def extractollamamodels():
    url = "https://ollama.com/library"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    config_file = os.path.join('pages', 'src', 'Database', 'config.json')
    with open(config_file, 'r') as file:
        config_data = json.load(file)
    ul = soup.find("ul", {"role": "list"})

    data = []
    names=[]
    # Iterate over each li tag within the ul
    for li in ul.find_all("li"):
        # Extract the name and metadata
        name = li.find("h2").text.strip()
        names.append(name)

        metadata = li.find("p").text.strip()
        
        # Append the data to the list
        data.append({"name": name, "metadata": metadata})


    with open(config_file, 'w') as file:
        config_data['opensource_llms']=names
        json.dump(config_data, file, indent=4)

