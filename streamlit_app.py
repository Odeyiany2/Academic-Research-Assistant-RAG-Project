import streamlit as st
import requests

# FastAPI backend URL
api_url = "http://localhost:8000"

st.title("Academic Research Assistant (Business and Finance)")

# File upload section
st.subheader("Upload Business or Finance Document")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    files = {'file': uploaded_file}
    response = requests.post(f"{api_url}/upload", files=files)
    st.write(response.json()["message"])

# Query section
st.subheader("Ask a Question")
query = st.text_input("Enter your question:")

if st.button("Search"):
    if query:
        response = requests.post(f"{api_url}/query", json={"q": query})
        result = response.json()
        if "document" in result:
            st.write(f"Relevant document: {result['document']} (Chunk index: {result['chunk_index']})")
        else:
            st.write(result["message"])

