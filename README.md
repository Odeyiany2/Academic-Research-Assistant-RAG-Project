# RAG Project: Academic Research Assistant ( Business and Finance )

## 1. Objective
As a penultimate student who is studying Accounting, the use case for my RAG chatbot will be based on Business and Finance. The goal is to build a simple RAG chatbot that takes in documents talking about business and finance via file upload, generates embeddings of the documents, stores the embeddings in a vector store (MongoDB), and retrieves relevant embeddings to answer the user's query. 

## 2. Implementation Components
* FastAPI: Serves as the backend for the application, handling file uploads, query processing, and interacting with LangChain and MongoDB.

* LangChain: generate embeddings for the doument and process user queries
  
* ChromaDB: store the generated embeddings in vector store

* Streamlit: For the deployment of the chatbot

### Models Used to be Used
- Via Groq:
    - `llama-3.1-70b-versatile`
- Via Vertex AI on GCP (Not used yet):
    - `gemini-1.5-pro-001`
    - `mistral-large@2407`
- Via AnthropicVertex GCP (Not used yet):
    - `claude-3-opus@20240229`
    - `claude-3-5-sonnet@20240620`

## Future Considerations to Build a Robust RAG
- Training with more files to help the LLM learn more. 
- Possibility of expanding the courses to more.
- Getting access to more LLMS as the last two on the Model section to get better comparisons on how different models work

## To Run Locally
Follow the steps below to run the codes locally and replicate the results. 
