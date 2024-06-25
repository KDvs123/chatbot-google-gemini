# Import setup_ml_model from ml_model.py
from ml_model import setup_ml_model
import streamlit as st

# Setup ML model and get all functions
ml_functions = setup_ml_model()

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    # Load text from the single PDF file
    raw_text = ml_functions['get_pdf_text'](ml_functions['PDF_FILE_PATH'])  # Correctly access PDF_FILE_PATH
    text_chunks = ml_functions['get_text_chunks'](raw_text)
    ml_functions['get_vector_store'](text_chunks)

    # Greet user
    st.text("Hello! How can I assist you today?")

    user_question = st.text_input("Ask a Question from the PDF File")

    if user_question:
        response = ml_functions['user_input'](user_question)
        st.write("Reply: ", response)

    # Acknowledge when user leaves
    st.text("Thank you for chatting with me. Have a great day!")

    # Display chat history
    st.subheader("Chat History")
    for chat in ml_functions['chat_history']:
        st.write(f"{chat['timestamp']} - You: {chat['question']}")
        st.write(f"{chat['timestamp']} - Bot: {chat['answer']}")

if __name__ == "__main__":
    main()
