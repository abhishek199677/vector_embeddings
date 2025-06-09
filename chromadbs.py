import os
import requests
from chromadb import HttpClient
from dotenv import load_dotenv
import numpy as np

load_dotenv()

EURI_API_KEY = os.getenv("EURI_API_KEY")  

#connection betweeen chromadb and python code
client = HttpClient(host="localhost", port=8000)   #connection
collection = client.get_or_create_collection("abhishek_sample_data")    #collection

print(client)


#converting textual data in to embeddings
def generate_embeddings(text_list):
    url = "https://api.euron.one/api/v1/euri/alpha/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}"
    }
    payload = {
        "input": text_list,
        "model": "text-embedding-3-small"
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    
    # Convert to numpy array for vector operations
    embeddings = [item['embedding']for item in response.json()['data']]
    
    
    return embeddings

document = [
    "Born and raised in a Muslim family in Rameswaram, Tamil Nadu, Kalam studied physics and aerospace engineering.",
    "He spent the next four decades as a scientist and science administrator, mainly at the Defence Research and Development Organisation (DRDO) and the Indian Space Research Organisation (ISRO).",
    "He was the country's first civilian space scientist and was intimately involved in India's civilian space programme and military missile development efforts.",
    'He was known as the "Missile Man of India" for his work on the development of ballistic missile and launch vehicle technology.',
    "He also played a pivotal organisational, technical, and political role in the Pokhran-II nuclear tests in 1998, India's second such test after the first test in 1974.",
    "Kalam was elected as the President of India in 2002 with the support of both the ruling Bharatiya Janata Party and the then-opposition Indian National Congress.",
    "He was widely referred to as the People's President. He engaged in teaching, writing, and public service after his presidency. He was a recipient of several awards, including the Bharat Ratna, India's highest civilian honour."
]




all_embeddings = generate_embeddings(document)
print(all_embeddings)
print(len(all_embeddings[0]))


#storing all the embeddings into chromadb database using enumerate
enumerate(zip(document,all_embeddings))
for idx, (doc, emb) in enumerate(zip(document,all_embeddings)):
    collection.add(
        documents=[doc],
        embeddings=[emb],
        metadatas=[{"source": "abhishek_sample_data"}],
        ids= [f"doc_{idx}"]
    )

print("Embeddings added to chromadb")
#fetching data from chromadb
print(collection.get(include=["documents", "embeddings"]))



