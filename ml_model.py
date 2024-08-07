import os
import fitz  # PyMuPDF
import logging
from datetime import datetime
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Model directory where the single PDF file is located
PDF_FILE_PATH = "Workhub24 Support Framework 8c92ec8e015b431dadd90fd771efc070 1.pdf"

# Initialize chat history
chat_history = []

def setup_ml_model():
    # Configure the Google Generative AI
    genai.configure(api_key=api_key)

    def get_pdf_text(pdf_path):
        document = fitz.open(pdf_path)
        text = ""
        for page_num in range(document.page_count):
            page = document.load_page(page_num)
            text += page.get_text()
        return text

    def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        logging.debug(f"Generated {len(chunks)} text chunks")
        return chunks

    def get_vector_store(text_chunks):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
            vector_store.save_local("faiss_index")
        except Exception as e:
            logging.error(f"Error creating FAISS vector store: {e}")
            raise
        return vector_store

    def get_conversational_chain():
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        try:
            model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
            chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        except Exception as e:
            logging.error(f"Error creating conversational chain: {e}")
            raise
        return chain

    def save_chat_history(question, answer):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_history.append({"timestamp": timestamp, "question": question, "answer": answer})

    def user_input(user_question):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            if not docs:
                raise ValueError(f"No documents found for query: {user_question}")

            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            save_chat_history(user_question, response["output_text"])  # Save chat history
        except Exception as e:
            logging.error(f"Error during user input processing: {e}")
            raise
        return response["output_text"]

    # Load PDF and create vector store during initialization
    try:
        pdf_text = get_pdf_text(PDF_FILE_PATH)
        text_chunks = get_text_chunks(pdf_text)
        get_vector_store(text_chunks)
    except Exception as e:
        logging.error(f"Error during ML model setup: {e}")
        raise

    return {
        'get_pdf_text': get_pdf_text,
        'get_text_chunks': get_text_chunks,
        'get_vector_store': get_vector_store,
        'get_conversational_chain': get_conversational_chain,
        'save_chat_history': save_chat_history,
        'user_input': user_input,
        'PDF_FILE_PATH': PDF_FILE_PATH,
        'chat_history': chat_history 
    }

ml_functions = setup_ml_model()
