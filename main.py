#necessary Libraries
import os
import groq
from dotenv import load_dotenv
load_dotenv()

# pdfreader for PDFs
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader

# load docreader for Docs
from langchain_community.document_loaders import Docx2txtLoader

# load textreader
from langchain_community.document_loaders import TextLoader

# load document splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

#llm
from langchain_groq import ChatGroq

# load embedding model
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
#database to store vectors from embeddings
from pymongo import MongoClient

# load chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

#load memory buffer
from langchain.memory import ConversationBufferMemory
from src.exceptions.operationshandler import system_logger




# loading api key
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

#storing up the models to be used in our RAG
models = [
    # "llama-3.1-405b-reasoning",
    "llama-3.1-70b-versatile",
    "claude-3-5-sonnet@20240620",
    "claude-3-opus@20240229",
    "gemini-1.5-pro-001",
    "mistral-large@2407"
]

#def document_upload(dir):