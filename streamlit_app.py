# import streamlit as st
# import requests

# # FastAPI backend URL


# st.title("Academic Research Assistant (Business and Finance)")

# # File upload section
# st.subheader("Upload a Business or Finance Document")
# uploaded_file = st.file_uploader("Choose a file")

# if uploaded_file is not None:
#     files = {'file': uploaded_file}
#     response = requests.post(f"{api_url}/upload", files=files)
#     st.write(response.json()["message"])

# # Query section
# st.subheader("Ask a Question")
# query = st.text_input("Enter your question:")

# if st.button("Search"):
#     if query:
#         response = requests.post(f"{api_url}/query", json={"q": query})
#         result = response.json()
#         if "document" in result:
#             st.write(f"Relevant document: {result['document']} (Chunk index: {result['chunk_index']})")
#         else:
#             st.write(result["message"])



import streamlit as st
import requests
import json
import uuid
#from utils.models import LLMClient
from app import *

# Set your API endpoint URLs
UPLOAD_URL = "http://127.0.0.1:5000/upload"
QUERY_URL = "http://127.0.0.1:5000/query"
CHAT_URL = "http://127.0.0.1:5000/chat"
RESET_URL = "http://127.0.0.1:5000/reset"

st.title("📚 Academic Research Assistant - Business and Finance")

# # Initialize session state for tracking user input and responses
# if 'responses' not in st.session_state:
#     st.session_state.responses = []

# Sidebar for file upload and model selection
st.sidebar.header(" ✨ Necessary Inputs")
project_uuid = st.sidebar.text_input("Project UUID", value=str(uuid.uuid4()))
st.sidebar.subheader("Select Model to Use")
# Extract model names (keys) from the map_client_to_model method
model_names = ["llama-3.1-70b-versatile", "llama-3.1-8b-instant"]
selected_model = st.sidebar.selectbox(
    "Select a Model", model_names)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
st.sidebar.subheader("Upload your Documents on Business or Finance")
uploaded_files = st.sidebar.file_uploader("Choose files", accept_multiple_files=True)

if st.sidebar.button("Upload"):
    if uploaded_files and project_uuid :
        files = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
        response = requests.post(UPLOAD_URL, data = {"projectUuid": project_uuid}, files= files) 
        st.sidebar.write(response.json())
    else:
        st.sidebar.error("Please upload files")

# Main area for querying the model
#st.subheader("Query the model based on your uploaded document")
user_input = st.text_input("Enter your question: ")

if st.button("Submit Query"):
    if uploaded_files and user_input:
        query_payload = {
            "model": selected_model,
            "temperature": temperature,
            "question": user_input,
            "projectUuid": project_uuid
        }
        try:
            response = requests.post(QUERY_URL, json=query_payload)

            # Check if the request was successful
            if response.status_code == 200:
                try:
                    # Try to parse the JSON response
                    result = response.json()
                    st.write("Model's Response:")
                    st.write(result)
                except ValueError:
                    # If JSON decoding fails, print the raw content for debugging
                    st.error("Failed to decode JSON response")
                    st.write(f"Raw response content: {response.text}")
            else:
                st.error(f"Error: Received status code {response.status_code}")
                st.write(f"Response content: {response.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")


# Reset the conversation
if st.button("Reset Chat"):
    response = requests.post(RESET_URL)
    if response.status_code == 200:
        st.success("Chat history has been reset.")
    else:
        st.error("Error resetting chat history.")

