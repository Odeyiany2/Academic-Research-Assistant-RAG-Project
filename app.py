import os, time, traceback
from fastapi import FastAPI, Request, HTTPException, UploadFile
from fastapi.responses import  PlainTextResponse, JSONResponse
import groq
from src.exceptions.operationshandler import llmresponse_logger,system_logger, userops_logger
from main import *
from dotenv import load_dotenv
load_dotenv()



# initialize Applications
app = FastAPI()
#chat_bot
@app.get("/upload")
async def upload_file():
    try:




    except:








@app.get("/query")
async def get_user_query():
