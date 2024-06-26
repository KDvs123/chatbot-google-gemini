import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Configure the Google Generative AI
genai.configure(api_key=api_key)

def get_pdf_text(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf_file:
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
    chain = model  # Placeholder for the chain, as direct usage is simulated here

    return chain

def answer_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def get_question_generation_chain():
    prompt_template = """
    Generate a question related to the provided context.\n\n
    Context:\n {context}.\n

    Question:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    chain = model  # Placeholder for the chain, as direct usage is simulated here

    return chain

def generate_questions(text, num_questions=4):
    sentences = text.split('.')  # Split by sentences for example
    questions = []
    question_chain = get_question_generation_chain()

    for sentence in sentences:
        if len(questions) >= num_questions:
            break
        
        context = sentence.strip()
        response = question_chain({"context": context}, return_only_outputs=True)
        questions.append(response["output_text"])
    
    return questions[:num_questions]

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    # Initialize or load chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = [{"role": "bot", "content": "Hi, welcome to Workhub 24 Support Care System. How may I help you?"}]

    # Upload PDF file
    uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])

    if uploaded_file:
        raw_text = get_pdf_text(uploaded_file)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

        # Generate questions button
        if st.button("Generate 4 Questions"):
            st.session_state['generated_questions'] = generate_questions(raw_text)

    # Display generated questions
    if 'generated_questions' in st.session_state:
        st.subheader("Generated Questions:")
        for idx, question in enumerate(st.session_state['generated_questions'], 1):
            st.markdown(f"{idx}. {question}")

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
    user_question = st.text_input("Please ask your question here:")

    if user_question:
        st.session_state['chat_history'].append({"role": "user", "content": user_question})
        response = answer_question(user_question)
        st.session_state['chat_history'].append({"role": "bot", "content": response})

    # Acknowledge when user stops entering questions
    if st.button("End Chat"):
        st.session_state['chat_history'].append({"role": "bot", "content": "Thank you for chatting with me. Have a great day!"})

if __name__ == "__main__":
    main()
