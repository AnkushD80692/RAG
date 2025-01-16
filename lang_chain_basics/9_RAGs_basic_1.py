import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir,"The-Monk-Who-Sold-His-Ferrari_Robin-Sharma_16.04.2020.txt")
persistent_directory = os.path.join(current_dir,"db","chroma_db")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist, Initializing vector store...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"the file{file_path} does not eixist, please check path"
        )

    loader = TextLoader(file_path)
    documents = loader.load()

    text_splitter =  CharacterTextSplitter(chunk_size = 1000,chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    print("Number of document chunk : ",len(docs))
    
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    db= Chroma.from_documents(
        docs[:50], embeddings, persist_directory = persistent_directory
    )
    
else:
    print("vector store already exists. no need to initialize")




