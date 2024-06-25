# chatbot_app.py

import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
from datetime import datetime
import streamlit as st

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure the Google Generative AI
genai.configure(api_key=api_key)

# Model directory where the single PDF file is located
PDF_FILE_PATH = os.path.join("Workhub24 Support Framework 8c92ec8e015b431dadd90fd771efc070 1.pdf")

def get_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf_file:
        for page in pdf_file.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def answer_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    # Initialize or load chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    # Load text from the single PDF file
    if 'loaded' not in st.session_state:
        raw_text = get_pdf_text(PDF_FILE_PATH)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        st.session_state['loaded'] = True

    # Display chat messages in a conversational manner
    bot_logo = 'https://pbs.twimg.com/profile_images/1739538983112048640/4NzIg1h6_400x400.jpg'

    for message in st.session_state['chat_history']:
        if message["role"] == 'bot':
            with st.chat_message(message["role"], avatar=bot_logo):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # User input section
    user_question = st.chat_input("Please ask your question here:")

    if user_question:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        response = answer_question(user_question)
        st.session_state.chat_history.append({"role": "bot", "content": response})
        with st.chat_message("bot", avatar=bot_logo):
            st.markdown(response)

    # Acknowledge when user stops entering questions
    if st.button("End Chat"):
        st.session_state.chat_history.append({"role": "bot", "content": "Thank you for chatting with me. Have a great day!"})
        with st.chat_message("bot", avatar=bot_logo):
            st.markdown("Thank you for chatting with me. Have a great day!")

if __name__ == "__main__":
    main()
