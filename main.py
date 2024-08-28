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
from langchain_community.embeddings import HuggingFaceBgeEmbeddings, OpenAIEmbeddings

#database to store vectors from embeddings
from pymongo import MongoClient

# load chain
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain