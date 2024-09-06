import os, time, traceback, tempfile
from typing import List, Literal, Any
from fastapi import FastAPI, Request, UploadFile, File, Form, Depends
from fastapi.responses import  PlainTextResponse, JSONResponse
import groq
from src.exceptions.operationshandler import llmresponse_logger, userops_logger, evaluation_logger
from main import qa_engine, Chroma, huggingface_embeddings,document_processing,text_splitter, ChatGroq, db_chroma, conversation_memory,chat_chain
from utils.helpers import allowed_file, QueryEngineError, system_logger, upload_files
from main import *
from utils.evaluation import *
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

CHROMA_PATH = r"C:\Academic_Research_Assistant_RAG\Academic-Research-Assistant-RAG-Project\chromadb"


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

        with tempfile.TemporaryDirectory() as temp_dir:

            _uploaded = await upload_files(files, temp_dir)

            if _uploaded["status_code"]==200:
                # Process the documents
                document_chunks = text_splitter.split_documents(document_processing(dir="temp_docs"))
                # Embed the chunks and load them into the ChromaDB
                db_chroma = Chroma.from_documents(document_chunks, huggingface_embeddings, persist_directory=CHROMA_PATH)
                #db_chroma.persist()

                            
                return {
                        "detail": "Embeddings generated and stored succesfully",
                        "status_code": 200
                    }
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

    query = await request.json()
    model = query["model"]
    temperature = query["temperature"]

    llm_client = ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0.1,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )
    
    # Initialize ChromaDB
    chroma_retriever = db_chroma.as_retriever()

    # Calculate collection size
    collection_size = chroma_retriever.index.embeddings.shape[0]
    print(f"Retrieved collection size: {collection_size}...")

    # experiment with choice_k to find something optimal
    choice_k = 40 if collection_size>150 \
                    else 15 if collection_size>50 \
                        else 10 if collection_size>20 \
                            else 5

    try:
        response = qa_engine(
            query["question"], 
            db_chroma,
            llm_client, 
            choice_k=choice_k
            # model=model
        )
        # Logging the response
        llmresponse_logger.log(response["result"])  # Log the generated response 
        print(response.response)

        # Evaluate the model's response
        evaluate_model(query["question"], response["result"])
        
        return PlainTextResponse(content=response.response, status_code=200)
    
    except Exception as e:
        message = f"An error occured where {model} was trying to generate a response: {e}",
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
async def reset_chat():
    try:
        conversation_memory.clear()
        return {"detail": "Chat history has been reset"}
    except Exception as e:
        system_logger.error(f"Error during reset: {str(e)}")
        return JSONResponse(status_code=500, content={"detail": "Reset error occurred"})


if __name__ == "__main__":
    import uvicorn
    print("Starting Academic Research Assistant LLM API")
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)
