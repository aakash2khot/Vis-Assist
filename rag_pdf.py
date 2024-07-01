import streamlit as st
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain_community.llms import Ollama
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os
import time
from audiorecorder import audiorecorder
from asr import query  # Import the query function from asr.py
from TTS.api import TTS
import spacy
import PyPDF2
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# Function to create structured PDF
def create_structured_pdf(text, filename):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    pdf = SimpleDocTemplate(f"{filename}.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    content = []
    for sent in doc.sents:
        content.append(Paragraph(sent.text, styles['Normal']))
        content.append(Spacer(1, 12))  # Add space after paragraph
    pdf.build(content)
    return f"{filename}.pdf"

# Function to process audio and convert it to text
def process_audio_and_convert_to_text(audio):
    if len(audio) > 0:
        audio.export("audio.flac", format="flac")
        text = query("audio.flac")
        return text

# Function to autoplay audio
def autoplay_audio(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

# Function to analyze uploaded PDF
def analyze_pdf(uploaded_file):
    if uploaded_file is not None:
        if not os.path.isfile("files/"+uploaded_file.name+".pdf"):
            with st.status("Analyzing your document..."):
                bytes_data = uploaded_file.read()
                f = open("files/"+uploaded_file.name, "wb")
                f.write(bytes_data)
                f.close()
                loader = PyPDFLoader("files/"+uploaded_file.name)
                data = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1500,
                    chunk_overlap=200,
                    length_function=len
                )
                all_splits = text_splitter.split_documents(data)
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=all_splits,
                    embedding=OllamaEmbeddings(model="llama3")
                )
                st.session_state.vectorstore.persist()
            st.session_state.retriever = st.session_state.vectorstore.as_retriever()
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                chain_type='stuff',
                retriever=st.session_state.retriever,
                verbose=True,
                chain_type_kwargs={
                    "verbose": True,
                    "prompt": st.session_state.prompt,
                    "memory": st.session_state.memory,
                }
            )
            st.success("Document analysis completed.")

# Session state initialization
if 'template' not in st.session_state:
    st.session_state.template = """You are a knowledgeable chatbot, here to help with questions of the user. Your tone should be professional and informative.
    Context: {context}
    History: {history}
    User: {question}
    Chatbot:"""
if 'prompt' not in st.session_state:
    st.session_state.prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=st.session_state.template,
    )
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True,
        input_key="question")
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = Chroma(persist_directory='jj',
                                          embedding_function=OllamaEmbeddings(base_url='http://localhost:11434',
                                                                              model="llama3")
                                          )
if 'llm' not in st.session_state:
    st.session_state.llm = Ollama(base_url="http://localhost:11434",
                                  model="llama3",
                                  verbose=True,
                                  callback_manager=CallbackManager(
                                      [StreamingStdOutCallbackHandler()]),
                                  )

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("PDF Chatbot")

# Upload a PDF file and analyze it
uploaded_file = st.file_uploader("Upload your PDF", type='pdf')
analyze_pdf(uploaded_file)

# Trigger recording and downloading tailored content when the button is clicked
if uploaded_file is not None:
    if st.button("Record"):
        audio = audiorecorder("Record", "Stop")
    if st.button("Generate Tailored Content"):
        st.session_state.is_downloading = True
        response = st.session_state.qa_chain("Based on this document create a detailed study material for blind children. The study material should be adapted in such a way so that blind children can understand more easily.")
        if response:
            content_text = response['result']
            pdf_filename = create_structured_pdf(content_text, "tailored_content")
            st.success(f"Tailored content has been created.")
            st.download_button(label="Download Tailored Content", data=open(pdf_filename, 'rb'), file_name=pdf_filename)
