""" script creates a collection in the chromadb vector database """
import os
import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from dotenv import load_dotenv
import openai


# api key
env_path = Path(".") / ".env"
load_dotenv(dotenv_path=env_path)

OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

client = chromadb.PersistentClient(path="../emb/")

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-ada-002"
            )

collection = client.create_collection(name="bodniak_v3", embedding_function=openai_ef)

documents = []
ids = []
metadatas = []

txt_path = Path("..") / "text" / "bodniak_output_clear.txt"
with open(txt_path, 'r', encoding='utf-8') as f:
    text_lines = f.readlines()

source = 'Bodniak S., Polska a Bałtyk za ostatniego Jagiellona, Pamiętnik Biblioteki Kórnickiej 3, 42-276, 1939-1946'
current_chapter = ''
current_page = ''

for i, line in enumerate(text_lines):
    if line.strip() != '':
        if line.startswith('[CHAPTER:'):
            current_chapter = line.replace('[CHAPTER:','').replace(']','').strip()
            continue
        if line.startswith('[PAGE:'):
            current_page = line.replace('[PAGE:','').replace(']','').strip()

        documents.append(line)
        ids.append('id_bodniak_1946_' + str(i).zfill(10))
        metadatas.append({"source": source, "chapter": current_chapter, "page": current_page})

collection.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)
