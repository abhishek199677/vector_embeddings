import os
import requests
from chromadb import HttpClient
from dotenv import load_dotenv



load_dotenv()

EURI_API_KEY = os.getenv("EURI_API_KEY") 

#for using chromadb
#connection betweeen chromadb and python code
client = HttpClient(host="localhost", port=8000)     #connection
collection = client.get_or_create_collection("abhishek_sample_data")    #collection



# Using requests library for embeddings
import requests
import numpy as np

def generate_embeddings(text):
    url = "https://api.euron.one/api/v1/euri/alpha/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}"
    }
    payload = {
        "input": "text",
        "model": "text-embedding-3-small"
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    
    # Convert to numpy array for vector operations
    embedding = np.array(data['data'][0]['embedding'])
    
    return embedding



def searchchroma(query_text):
    query_embed = generate_embeddings(query_text)
    #getting the data from chromadb
    result = collection.query(query_embeddings = [query_embed], n_results=2, include=["documents"])
    print(result)

searchchroma("Born and raised in a Muslim family in Rameswaram, Tamil Nadu, Kalam studied physics and aerospace engineering")

