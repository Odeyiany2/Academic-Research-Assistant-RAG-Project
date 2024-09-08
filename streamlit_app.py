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

# Set your API endpoint URLs
UPLOAD_URL = "http://127.0.0.1:5000/upload"
QUERY_URL = "http://127.0.0.1:5000/query"
CHAT_URL = "http://127.0.0.1:5000/chat"
RESET_URL = "http://127.0.0.1:5000/reset"

st.title("Academic Research Assistant - Business and Finance")

# Sidebar for file upload
st.sidebar.header("Upload Documents")
project_uuid = st.sidebar.text_input("Project UUID", value=str(uuid.uuid4()))
st.sidebar.write("If you don't have a Project UUID, one has been automatically generated for you.")
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5)
uploaded_files = st.sidebar.file_uploader("Choose files", accept_multiple_files=True)

if st.sidebar.button("Upload"):
    if uploaded_files and project_uuid:
        files = [("files", (file.name, file.getvalue(), file.type)) for file in uploaded_files]
        response = requests.post(UPLOAD_URL, data={"projectUuid": project_uuid}, files=files)
        st.sidebar.write(response.json())
    else:
        st.sidebar.error("Please upload files and provide a Project UUID.")

# Main area for querying the model
st.header("Query the Model")

model = st.selectbox("Choose the model", ["llama-3.1-70b-versatile"])
# temperature = st.slider("Temperature", 0.0, 1.0, 0.5)
question = st.text_area("Enter your question")

if st.button("Submit Query"):
    if question:
        query_payload = {
            "model": model,
            "temperature": temperature,
            "question": question
        }
        response = requests.post(QUERY_URL, json=query_payload)
        # st.write(f"Response Status Code: {response.status_code}")  # Add this line to check status code
        # st.write(f"Response Content: {response.text}") 
        if response.status_code == 200:
            st.write("Model's Response:")
            st.write(response.text)
        else:
            st.error("Error in generating response. Please try again.")
    else:
        st.error("Please enter a question.")

# Chat with the Assistant
st.header("Chat with the Assistant")

user_input = st.text_input("You:")

if st.button("Send"):
    if user_input:
        response = requests.post(CHAT_URL, json={"user_input": user_input})
        if response.status_code == 200:
            chat_response = response.json()
            st.write("Assistant:")
            st.write(chat_response["response"])
            if chat_response["source_documents"]:
                st.write("Source Documents:")
                st.json(chat_response["source_documents"])
        else:
            st.error("Error during chat. Please try again.")
    else:
        st.error("Please enter a message.")

# Reset the conversation
if st.button("Reset Chat"):
    response = requests.post(RESET_URL)
    if response.status_code == 200:
        st.success("Chat history has been reset.")
    else:
        st.error("Error resetting chat history.")

