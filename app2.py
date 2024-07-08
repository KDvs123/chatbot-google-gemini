import os
import json
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

# Function to extract text from PDF
def get_pdf_text(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf_file:
        for page in pdf_file.pages:
            text += page.extract_text()
    return text

# Function to split text into manageable chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to generate questions based on text chunks
def generate_questions(pdf_text, num_questions=4):
    text_chunks = get_text_chunks(pdf_text)
    questions = []
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)

    prompt_template = """
    Generate a thoughtful question based on the information provided below:\n\n
    Text:\n {text}\n

    Question:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

    try:
        for _ in range(num_questions):
            response = model.invoke(prompt.format(text=pdf_text))
            question_content = response['content'] if isinstance(response, dict) and 'content' in response else str(response)
            questions.append(question_content.strip())
    except Exception as e:
        st.error(f"Error generating question: {e}")
        questions.append("Error generating question")

    return questions

# Function to save questions to JSON
def save_questions_to_json(questions, filename="custom_questions.json"):
    with open(filename, "w") as f:
        json.dump(questions, f, indent=4)
    st.success(f"Custom questions saved to {filename}")

# Main function to run Streamlit app
def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    # Initialize or load chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = [{"role": "bot", "content": "Hi, welcome to Workhub 24 Support Care System. How may I help you?"}]

    # Display chat messages in a conversational manner
    bot_logo = 'https://pbs.twimg.com/profile_images/1739538983112048640/4NzIg1h6_400x400.jpg'
    for message in st.session_state['chat_history']:
        if message["role"] == 'bot':
            with st.chat_message(message["role"], avatar=bot_logo):
                st.markdown(message["content"])
        else:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Prompt user to upload a PDF file
    if 'uploaded_file' not in st.session_state:
        st.session_state['chat_history'].append({"role": "bot", "content": "Please upload a PDF file to start."})
        uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])
        if uploaded_file:
            st.session_state['uploaded_file'] = uploaded_file
            st.session_state['chat_history'].append({"role": "user", "content": "PDF file uploaded successfully!"})
            raw_text = get_pdf_text(uploaded_file)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.experimental_rerun()

    if 'uploaded_file' in st.session_state:
        # Generate questions button
        if st.button("Generate Questions"):
            questions = generate_questions(get_pdf_text(st.session_state['uploaded_file']))
            st.session_state['generated_questions'] = questions
            for question in questions:
                st.session_state['chat_history'].append({"role": "bot", "content": question})
            st.experimental_rerun()

    # Display generated questions
    if 'generated_questions' in st.session_state:
        st.subheader("Generated Questions:")
        for idx, question in enumerate(st.session_state['generated_questions'], 1):
            st.text_area(f"Question {idx}", value=question, height=100)

        # Offer options for customization
        st.session_state['chat_history'].append({"role": "bot", "content": "Do you want to keep these questions or customize them?"})
        if st.button("Keep Questions"):
            st.session_state['chat_history'].append({"role": "user", "content": "I want to keep these questions."})
        if st.button("Customize Questions"):
            st.session_state['chat_history'].append({"role": "user", "content": "I want to customize these questions."})
            custom_questions = []
            for i in range(4):
                custom_questions.append(st.text_input(f"Custom Question {i+1}"))

            if st.button("Save Custom Questions"):
                if any(not question for question in custom_questions):
                    st.error("All custom questions must be filled out.")
                else:
                    save_questions_to_json(custom_questions)
                    st.session_state['chat_history'].append({"role": "bot", "content": "Custom questions saved successfully."})

    # Acknowledge when user stops entering questions
    if st.button("End Chat"):
        st.session_state['chat_history'].append({"role": "bot", "content": "Thank you for chatting with me. Have a great day!"})

if __name__ == "__main__":
    main()
