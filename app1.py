import os
import streamlit as st
import requests
import pickle
import time
import yaml
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re

# Function to load secrets from secrets.yaml
def load_secrets(file_path="secrets.yaml"):
    with open(file_path, "r") as file:
        secrets = yaml.safe_load(file)
    return secrets

# Load API key from secrets.yaml
secrets = load_secrets()
os.environ["OPENAI_API_KEY"] = secrets["OPENAI_API_KEY"]

# Function to get all subpage links
def get_all_links(url, base_url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    links = []

    for a_tag in soup.find_all("a", href=True):
        href = a_tag['href']
        if href.startswith(base_url) or href.startswith("/"):
            full_url = href if href.startswith(base_url) else base_url + href
            links.append(full_url)

    return list(set(links))

# Function to load all pages and create a FAISS vector store
def create_faiss_index(main_url):
    base_url = re.match(r"https?://[^/]+", main_url).group(0)
    all_links = get_all_links(main_url, base_url)
    all_docs = []

    for link in all_links:
        try:
            loader = WebBaseLoader(web_paths=[link])
            docs = loader.load()
            all_docs.extend(docs)
        except Exception as e:
            print(f"Error loading {link}: {e}")

    # Split documents and create embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(all_docs)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embedding=embeddings)

    # Save the FAISS index locally
    vectorstore.save_local("faiss_index")
    print("FAISS index created and saved.")
    return vectorstore

# Check if the FAISS index already exists
FAISS_INDEX_DIR = "faiss_index"
# MAIN_URL = "https://engineering.buffalo.edu/computer-science-engineering/information-for-faculty-and-staff/health-and-wellness.html"
MAIN_URL = "https://engineering.buffalo.edu/computer-science-engineering.html"
if os.path.exists(FAISS_INDEX_DIR):
    try:
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(FAISS_INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

        print("FAISS index loaded from file.")
    except Exception as e:
        print(f"Error loading FAISS index: {e}. Recreating index...")
        vectorstore = create_faiss_index(MAIN_URL)
else:
    print("FAISS index not found. Creating new index...")
    vectorstore = create_faiss_index(MAIN_URL)

# Initialize LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2
)

# System Prompt Template
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create the retrieval chain
retriever = vectorstore.as_retriever()
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Streamlit UI
st.title("Ask your RAG-Enhanced LLM")

if 'messages' not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# User prompt input
user_prompt = st.chat_input("Go ahead and ask me something...")

if user_prompt:
    st.chat_message('user').markdown(user_prompt)
    st.session_state.messages.append({'role': 'user', 'content': user_prompt})
    response = rag_chain.invoke({'input': user_prompt})
    # print("retrived context", response["context"])
    st.chat_message('assistant').markdown(response['answer'])
    st.session_state.messages.append({'role': 'assistant', 'content': response['answer']})
