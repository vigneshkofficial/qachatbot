%%writefile app.py
import streamlit as st
import requests
from PyPDF2 import PdfReader
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
import google.generativeai as genai
import os
import json
from collections import Counter

# Set the API key for Google Generative AI
api_key = "AIzaSyABBPLGcxeNjPzFrFwCzvqozoV2ZEDuYpE"
if api_key:
    genai.configure(api_key=api_key)

# Function to load and extract text from a PDF
def load_pdf(file_path):
    try:
        pdf_reader = PdfReader(file_path)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error loading PDF: {e}")
        return ""

# Function to split text into manageable chunks
def split_text_recursively(text, max_length=1000, chunk_overlap=0):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + max_length
        if end < text_length:
            end = text.rfind(' ', start, end) + 1
            if end <= start:
                end = start + max_length
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - chunk_overlap
        if start >= text_length:
            break
    return chunks

# Initialize ChromaDB client
try:
    google_ef = embedding_functions.GoogleGenerativeAiEmbeddingFunction(api_key=api_key)
    client = chromadb.PersistentClient(path="embeddings/gemini")
except Exception as e:
    st.error(f"Error initializing ChromaDB client: {e}")
    google_ef = None
    client = None

# Retrieve or create the collection for RAG
if client:
    collection_name = "pdf_rag"
    try:
        collection = client.get_or_create_collection(name=collection_name, embedding_function=google_ef)
    except Exception as e:
        st.error(f"Error retrieving or creating collection: {e}")
        collection = None
else:
    collection = None

# Cache embeddings
embedding_cache_path = "embedding_cache.json"
if os.path.exists(embedding_cache_path):
    with open(embedding_cache_path, "r") as f:
        cached_embeddings = json.load(f)
else:
    cached_embeddings = {}

# Function to process multiple uploaded PDFs
def process_pdf(uploaded_files):
    if not collection:
        st.error("ChromaDB collection is not available.")
        return

    all_text = ""
    for uploaded_file in uploaded_files:
        file_path = uploaded_file.name
        try:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            pdf_text = load_pdf(file_path)
            all_text += pdf_text
        except Exception as e:
            st.error(f"Error processing file {file_path}: {e}")

    chunks = split_text_recursively(all_text, max_length=1000, chunk_overlap=200)

    # Index the chunks into the ChromaDB collection (only if not already cached)
    for i, chunk in enumerate(chunks):
        if str(i) not in cached_embeddings:
            try:
                collection.add(documents=[chunk], ids=[str(i)])
                cached_embeddings[str(i)] = chunk
            except Exception as e:
                st.error(f"Error adding chunk to collection: {e}")

    # Save cached embeddings
    try:
        with open(embedding_cache_path, "w") as f:
            json.dump(cached_embeddings, f, indent=4)
    except Exception as e:
        st.error(f"Error saving cached embeddings: {e}")

# Function to find relevant context for the user query
def find_relevant_context(query, db, n_results=3):
    if not db:
        return ""

    try:
        results = db.query(query_texts=[query], n_results=n_results)
        return "\n\n".join(results['documents'][0])
    except Exception as e:
        st.error(f"Error querying database: {e}")
        return ""

# Function to create a prompt for the generative model
def create_prompt_for_gemini(query, context):
    return f"""
    Answer the question below based on the provided information from the PDFs. If the answer is not fully available, provide what’s relevant from the text and suggest consulting a doctor. Make sure the answer is clear and simple for any user to understand.
    Question: {query}
    Context: {context}
    Answer:
    """

# Function to generate an answer using the Gemini model
def generate_answer_from_gemini(prompt):
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        result = model.generate_content(prompt)
        return result.text
    except Exception as e:
        st.error(f"Error generating answer from Gemini model: {e}")
        return "I'm sorry, I couldn't generate an answer at this time."

# Initialize session state variables
if "qa_cache" not in st.session_state:
    st.session_state.qa_cache = {}

if "conversation" not in st.session_state:
    st.session_state.conversation = []

if "question_counter" not in st.session_state:
    st.session_state.question_counter = Counter()

if "faqs" not in st.session_state:
    st.session_state.faqs = {}

# Sidebar setup
st.sidebar.title("Menu")
st.sidebar.info("Upload your PDF files and click on the Submit & Process button")
uploaded_files = st.sidebar.file_uploader("Drag and drop files here", type="pdf", accept_multiple_files=True)
process_button = st.sidebar.button("Submit & Process")
clear_chat_button = st.sidebar.button("Clear Chat History")

# Process the PDFs based on user input
if process_button and uploaded_files:
    process_pdf(uploaded_files)

if clear_chat_button:
    st.session_state.conversation.clear()

# Update UI with a more user-friendly theme for drugs and medicine
st.markdown(
    """
    <style>
        /* Set background color for the entire page */
        .stApp {
            background-color: #1E1E1E; /* Dark background */
        }

        /* Customize the sidebar background */
        .sidebar .sidebar-content {
            background-color: #2D2D2D; /* Slightly lighter dark background */
        }

        /* Customize the main title and other texts */
        .css-10trblm {
            color: #E5E5E5; /* Light gray text */
        }

        /* Customize the text input box */
        .css-1dp5vir {
            background-color: #3D3D3D; /* Dark input background */
            color: #FFFFFF; /* White text */
        }

        /* Customize the submit button */
        .css-1offfwp > button {
            background-color: #6200EA; /* Purple background for the button */
            color: #FFFFFF; /* White text */
            border-radius: 5px;
        }

        /* Change button color on hover */
        .css-1offfwp > button:hover {
            background-color: #3700B3; /* Darker purple on hover */
        }

        /* Customize the chat bubbles */
        .user-message {
            text-align: right;
            background-color: #1E88E5; /* Blue user message */
            color: #FFFFFF; /* White text */
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
        }

        .bot-message {
            text-align: left;
            background-color: #43A047; /* Green bot message */
            color: #FFFFFF; /* White text */
            border-radius: 8px;
            padding: 10px;
            margin: 10px 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Drug Information Chatbot")
st.write("Welcome to the chat!")
st.write("Upload your PDFs, and ask me a question")

# User input
user_input = st.text_input("Ask a question about the content:")

# Disable the "Ask" button until the user inputs something
ask_button_disabled = not user_input.strip()
ask_button_pressed = st.button("Ask", disabled=ask_button_disabled)

if user_input and ask_button_pressed:
    # Track question occurrences
    st.session_state.question_counter[user_input] += 1

    # Check if the query is already cached
    if user_input in st.session_state.qa_cache:
        # Notify the user about repeated questions
        if st.session_state.question_counter[user_input] > 1:
            st.warning(f"You already asked: {user_input}")

        # Skip generating a new answer
        answer = ""
    else:
        context = find_relevant_context(user_input, collection)
        prompt = create_prompt_for_gemini(user_input, context)
        answer = generate_answer_from_gemini(prompt)
        # Cache the answer
        st.session_state.qa_cache[user_input] = answer

    # Update conversation history
    st.session_state.conversation.append({"question": user_input, "answer": answer})

    # Display the conversation history (including the new entry)
    st.write("")
    for chat in st.session_state.conversation:
        if chat["answer"]:
            st.markdown(f'<div class="user-message">You: {chat["question"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bot-message">Bot: {chat["answer"]}</div>', unsafe_allow_html=True)
