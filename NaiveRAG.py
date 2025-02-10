import streamlit as st
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    CSVLoader,
    
)

from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
import tempfile

# Load environment variables
load_dotenv()
groq_api_key = st.secrets['GROQ_API_KEY']
openai_api_key = st.secrets['OPENAI_API_KEY']


# Initialize LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)

# Streamlit app title
st.title("Document Chatbot")

# Initialize session state for conversation history
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# File upload, URL input, directory input, and YouTube URL input in the sidebar
pdf_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")
web_url = st.sidebar.text_input("Enter a URL to load data from:")
csv_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")



# Load documents and create vector store
if pdf_file or web_url or csv_file:
    pages = []
    
    # Load PDF file
    if pdf_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(pdf_file.getbuffer())
            tmp_file_path = tmp_file.name

        try:
            loader = PyPDFLoader(tmp_file_path)
            pages.extend(loader.load())
        finally:
            os.unlink(tmp_file_path)

    # Load web URL
    if web_url:
        try:
            loader = WebBaseLoader(web_url)
            pages.extend(loader.load())
        except Exception as e:
            st.error(f"Error loading URL: {e}")

    # Load CSV file
    if csv_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_file:
            tmp_file.write(csv_file.getbuffer())
            tmp_file_path = tmp_file.name

        try:
            loader = CSVLoader(tmp_file_path)
            pages.extend(loader.load())
        finally:
            os.unlink(tmp_file_path)
  

    # Create a vector store
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    vector_store = FAISS.from_documents(pages, embeddings)

    # Store the retriever in session state
    st.session_state.retriever = vector_store.as_retriever()

# Chat input
query = st.chat_input("Ask a question about the document:")

# Display conversation history
for message in st.session_state.conversation:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Process user input
if query:
    # Add user query to conversation history
    st.session_state.conversation.append({"role": "user", "content": query})

    # Display user message
    with st.chat_message("user"):
        st.write(query)

    # Retrieve relevant documents
    if "retriever" in st.session_state:
        relevant_docs = st.session_state.retriever.get_relevant_documents(query)
        context = "\n".join([doc.page_content for doc in relevant_docs])
    else:
        context = "No documents loaded."

    # Define prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant. Use the following context to answer questions."),
        ("user", "{query}\n\nContext:\n{context}")
    ])

    # Generate response
    response = llm.invoke(prompt.format(query=query, context=context))

    # Add bot response to conversation history
    st.session_state.conversation.append({"role": "assistant", "content": response.content})

    # Display bot response
    with st.chat_message("assistant"):
        st.write(response.content)