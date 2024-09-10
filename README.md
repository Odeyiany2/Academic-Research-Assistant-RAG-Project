# RAG Project: 📚 Academic Research Assistant ( Business and Finance )

## 1. Objective
As a penultimate student who is studying Accounting, the use case for my RAG chatbot will be based on Business and Finance. The goal is to build a simple RAG chatbot that takes in documents talking about business and finance via file upload, generates embeddings of the documents, stores the embeddings in a vector store, and retrieves relevant embeddings to answer the user's query. As a student I have had to read very large files for my courses and most times I find it difficult and time consuming to get answers from the files. With the chatbot access to answers will be easier, faster and more efficient. 

## 2. Implementation Components
* FastAPI: Serves as the backend for the application, handling file uploads, query processing, and interacting with LangChain and ChromaDB.

* LangChain and Hugging Face Emeddings: generate embeddings for the doument and process user queries
  
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

## To Run Locally 👩🏽‍💻
Follow the steps below to run the codes locally and replicate the results. 
- Clone the repo to your local machine
   - VS code: use the link below to clone on vs code
     ```
     https://github.com/Odeyiany2/Academic-Research-Assistant-RAG-Project.git
     ```
   - Git Bash
     
     ```
     gh repo clone Odeyiany2/Academic-Research-Assistant-RAG-Project
     ```
- Create a virtual environment, activate it and install the requirements.txt
  ```
  pythom -m env venv
  ```
  
  ```
  venv\Scripts\activate
  ```
  
  ```
  pip install -r requirements.txt
  ```
- Run the app.py file
  ```
  uvicorn app:app --host 127.0.0.1 --port 5000 --reload
  ```
- Run the streamlit_app.py file
  ```
  streamlit run streamlit_app.py
  ```

**Note: Ensure to create a file to store your API keys and access them.**

## 🚀 Future Considerations to Build a more Robust RAG
- Training with more large files to help the LLM
- Ensuring the RAG can take larger files from users and efficiently summarize important details. 
- Possibility of expanding the number of courses
- Getting access to more LLMS as the last two on the Model section to get better comparisons on how different models perform.
  
  
### Streamlit Demo Video
https://github.com/user-attachments/assets/4f8720e8-b2ba-40e3-95ad-2ad3de958da5


