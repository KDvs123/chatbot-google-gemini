import os
import pdfplumber
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

def get_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf_file:
        for page in pdf_file.pages:
            text += page.extract_text()
    return text.strip()

def clean_text(text):
    cleaned_text = text.replace("\n", " ").replace("\r", "")
    return cleaned_text

def generate_suggested_questions(pdf_text):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", api_key=api_key, temperature=0.3)

    prompt = "Generate four suggested questions based on the following document content:\n\n"+pdf_text+"\n\nSuggested Questions:"
    input_data = [{"role": "system", "content": prompt}]
    

    try:
        response = model.invoke(input_data)
        st.write(f"Model Response: {response}")  # Debug: Show the raw response from the model

        questions = response["messages"]
        suggested_questions = [message["content"] for message in questions if message["role"] == "assistant"]
        return suggested_questions
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []

def main():
    st.set_page_config(page_title="Chat PDF")
    st.header("Chat with PDF using Gemini")

    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = [{"role": "bot", "content": "Hi, welcome to Workhub 24 Support Care System. How may I help you?"}]

    uploaded_file = st.file_uploader("Upload your PDF file", type="pdf")

    if uploaded_file:
        raw_text = get_pdf_text(uploaded_file)
        st.write(f"Extracted Text:\n{raw_text}")  # Debug: Show the extracted text
        cleaned_text = clean_text(raw_text)
        st.write(f"Cleaned Text:\n{cleaned_text}")  # Debug: Show the cleaned text

        if st.button("Generate Questions"):
            suggested_questions = generate_suggested_questions(cleaned_text)
            st.write(f"Suggested Questions:\n{suggested_questions}")  # Debug: Show the suggested questions

            if suggested_questions:
                st.subheader("Suggested Questions")
                st.write("Select a question to explore further:")
                selected_question = st.selectbox("Choose a question:", options=suggested_questions)

                if selected_question:
                    st.success(f"You selected: {selected_question}")
            else:
                st.info("No suggested questions found.")
    else:
        st.info("Please upload a PDF file to generate suggested questions.")

if __name__ == "__main__":
    main()
