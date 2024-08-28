import os, time, traceback
from typing import List, Literal, Any
from fastapi import FastAPI, Request, HTTPException, UploadFile
from fastapi.responses import  PlainTextResponse, JSONResponse
import groq
from src.exceptions.operationshandler import llmresponse_logger, userops_logger, evaluation_logger
from main import *
from dotenv import load_dotenv
load_dotenv()



# initialize Applications
app = FastAPI()
#chat_bot
# @app.get("/upload")
# async def process_file():

# @app.get("/query")
# async def get_user_query():




# import tempfile, traceback
# from typing import List, Literal, Any
# from fastapi import FastAPI, Request, UploadFile
# from week_2.day_1_robust_rag.main import *
# from week_2.day_1_robust_rag.helpers import upload_file

# app = FastAPI()

# @app.get('/healthz')
# async def health():
#     return {
#         "application": "Simple LLM API",
#         "message": "running succesfully"
#     }

# @app.post('/upload')
# async def process(
#     files: List[UploadFile] = None,
#     urls: List[str] = None
# ):

#     # query = await request.json()
#         try:
#             with tempfile.TemporaryDirectory() as temp_dir:
                
#                 _uploaded = upload_file(files, temp_dir)
#                 if _uploaded["status_code"]==200:
#                     documents = SimpleDirectoryReader(temp_dir).load_data()
#                     app.embeddings = VectorStoreIndex.from_documents(documents)
                            
#                     return {
#                         "detail": "Embeddings generated succesfully",
#                         "status_code": 200
#                     }
#                 else:
#                     return _uploaded # returns status dict

#         except Exception as e:
#             return {
#                 "detail": "Could not generated embeddings",
#                 "status_code": 500
#             }


# @app.post('/generate')
# async def generate_chat(request: Request):

#     query = await request.json()
    
#     pass



# if _name_ == "_main_":
#     import uvicorn
#     print("Starting LLM API")
#     uvicorn.run(app, host="0.0.0.0", reload=True)