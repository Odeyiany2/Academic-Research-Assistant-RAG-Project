import os, time
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, PlainTextResponse, JSONResponse
import groq
from main import *
from dotenv import load_dotenv
load_dotenv()



# initialize Applications
app = FastAPI()
#chat_bot