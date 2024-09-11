import os, time, traceback, tempfile
from typing import List, Literal, Any
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends
from fastapi.responses import  PlainTextResponse, JSONResponse
import groq
from src.exceptions.operationshandler import llmresponse_logger, userops_logger, evaluation_logger
from main import qa_engine, Chroma, huggingface_embeddings,document_processing,text_splitter, ChatGroq, get_session_history, chat_chain
from utils.helpers import allowed_file, QueryEngineError, system_logger, upload_files
from main import *
from utils.evaluation import *
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

CHROMA_PATH = r"C:\Academic-Research-Assistant-RAG-2\Academic-Research-Assistant-RAG-Project\chromadb"


@app.get('/healthz')
async def health():
    return {
        "application": "Academic Research Assistant LLM API",
        "message":"Welcome to the RAG-based Research Assistant API!"
    }

@app.post('/upload')
async def upload_documents(
    projectUuid: str = Form(...),
    files: List[UploadFile] = None, 
):

    try:
        # os.makedirs("temp_docs", exist_ok=True)
        # # Save the uploaded files
        # for file in files:
        #     file_path = os.path.join("temp_docs", file.filename)
        #     with open(file_path, "wb") as f:
        #         f.write(await file.read())

        #with tempfile.TemporaryDirectory() as temp_dir:

        _uploaded = await upload_files(files, "temp_docs")

        if _uploaded["status_code"]==200:
                # Process the documents
            document_chunks = text_splitter.split_documents(document_processing(dir="temp_docs"))
                
            document_chunks = [doc for doc in document_chunks if doc.page_content and doc.page_content.strip()]

            if not document_chunks:
                raise ValueError("No valid documents to process. All documents have empty or invalid content.")
                # Embed the chunks and load them into the ChromaDB
            vector_store = Chroma.from_documents(document_chunks, huggingface_embeddings, persist_directory=CHROMA_PATH)
            vector_store.persist()

                
            return "File uploaded successfully"
        # {
                    # "detail": "Embeddings generated and stored succesfully",
                    # "status_code": 200
                    # }
        else:
            return _uploaded # returns status dict

    except Exception as e:
        error_message = traceback.format_exc()
        system_logger.error(f"Error during document upload: {error_message}")
        return {
            "detail": f"Could not generate embeddings: {e}",
            "status_code": 500
        }



@app.post('/query')
async def query_model(
    request: Request
):
    try:
        json_content = await request.json()
        query = json_content.get("question")
        model = json_content.get("model", "llama-3.1-70b-versatile")  # Default model if not provided
        temperature = json_content.get("temperature", 0.1)
        vector_store = Chroma(embedding_function=huggingface_embeddings, persist_directory=CHROMA_PATH)

        llm_client = ChatGroq(
            model=model,
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        ) 
        retrievalQA = RetrievalQA.from_chain_type(
        llm=llm_client,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
        result = retrievalQA.invoke({"query": query})
        # Logging the response
        llmresponse_logger.log(logging.INFO, result["result"])  
    
        # Evaluate the model's response
        evaluate_model_response(query, model, result["result"])
        return PlainTextResponse(content=result["result"], status_code=200)
    
    except Exception as e:
        message = f"An error occured where {llm_client} was trying to generate a response: {e}",
        system_logger.error(
            message,
            exc_info=1
        )
        raise QueryEngineError(message)


@app.post('/chat')
async def chat_with_assistant(user_input: str):
    try:
        response = chat_chain({"input": user_input})
        llmresponse_logger.log(response['output_text'])  # Log the LLM response
        return {
            "response": response['output_text'],
            "source_documents": response.get('source_documents', [])
        }
    except Exception as e:
        system_logger.error(f"Error during chat: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": "Chat error occurred"})


@app.post('/reset')
async def reset_chat(request: Request):
    try:
        session_id = request.query_params.get("session_id")

        if session_id:
            chat_history = get_session_history(session_id)
            chat_history.clear()  # Clear the chat history

            return {"detail": f"Chat history for session {session_id} has been reset"}
        else:
            return {"detail": "Session ID not provided"}, 400

    except Exception as e:
        system_logger.error(f"Error during reset: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": "Reset error occurred"})
    

if __name__ == "__main__":
    import uvicorn
    print("Starting Academic Research Assistant LLM API")
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
