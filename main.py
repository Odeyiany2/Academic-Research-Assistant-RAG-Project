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
from langchain.vectorstores import Chroma

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
    "llama-3.1-70b-versatile",
    "claude-3-5-sonnet@20240620",
    "claude-3-opus@20240229",
    "gemini-1.5-pro-001",
    "mistral-large@2407"
]

def document_processing(dir = "data"):

    # Initialize an empty list to store document contents
    docs_before_split = []

    try:

        # Iterate through all files in the 'docs' directory
        for file in os.listdir(dir):
            # Check if the file is a PDF
            if file.endswith('.pdf'):
                # Construct the full path to the PDF file
                pdf_path = os.path.join(dir, file)
                # Create a PDF loader
                loader = PyPDFLoader(pdf_path)
                # Load the PDF and extend the documents list with its contents
                docs_before_split.extend(loader.load())
            # Check if the file is a Word document
            elif file.endswith('.docx') or file.endswith('.doc'):
                # Construct the full path to the Word document
                doc_path = os.path.join(dir, file)
                # Create a Word document loader
                loader = Docx2txtLoader(doc_path)
                # Load the Word document and extend the documents list with its contents
                docs_before_split.extend(loader.load())
            # Check if the file is a text file
            elif file.endswith('.txt'):
                # Construct the full path to the text file
                text_path = os.path.join(dir, file)
                # Create a text file loader
                loader = TextLoader(text_path)
                # Load the text file and extend the documents list with its contents
                docs_before_split.extend(loader.load())
        
    except Exception as e:
        message = f"Error during document processing: {e}"
        system_logger.log(str(e))  # Logging the error to the custom system logger
    
    return docs_before_split



# set the splitter parameters and instantiate it
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=20)
# split the document into chunks
document_chunks = text_splitter.split_documents(document_processing())

#initializing the embeddingg model
huggingface_embeddings = HuggingFaceBgeEmbeddings(
    model_name="sentence-transformers/all-MiniLM-l6-v2",  
    model_kwargs={'device':'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

CHROMA_PATH = r"C:\Academic_Research_Assistant_RAG\Academic-Research-Assistant-RAG-Project\chromadb"  # ChromaDB Path
# Embed the chunks and load them into the ChromaDB
try:
    message = "Starting embedding and storing in ChromaDB..."
    db_chroma = Chroma.from_documents(document_chunks, huggingface_embeddings, persist_directory=CHROMA_PATH)
    db_chroma.persist()
    system_logger.log("Embedding and storage completed successfully.")
except Exception as e:
    message = f"Error during embedding or storage: {e}"
    system_logger.error(message, str(e))  # Logging the error to your custom system logger



#instantiating our llm (only groq for now)
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0.1, #keep on playing with it. 
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

prompt_template = """ You are an academic reserach assistant for students, professors and lecturers. You assisit them with answering questions
baed on the document they provide to you.
Use the following pieces of context to answer the question at the end. Please follow the following rules:
1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer but you may want to check the following links".
2. If you find the answer, write the answer in a concise way with ten sentences maximum.

{context}

Question: {question}

Helpful Answer:
"""

PROMPT = PromptTemplate(
 template=prompt_template, input_variables=["context", "question"]
)

retrievalQA = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db_chroma.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)