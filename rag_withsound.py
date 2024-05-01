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
from IPython.display import Audio, display
import pygame
# Create directories if they don't exist
if not os.path.exists('files'):
    os.mkdir('files')

if not os.path.exists('jj'):
    os.mkdir('jj')
def autoplay_audio(filename):
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()

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

# Upload a PDF file
uploaded_file = st.file_uploader("Upload your PDF", type='pdf')

# Define function to process audio and convert it to text
def process_audio_and_convert_to_text(audio):
    if len(audio) > 0:
        audio.export("audio.flac", format="flac")
        # Call the query function to convert audio to text
        text = query("audio.flac")
        return text

# # Initialize the QA chain
# if 'qa_chain' not in st.session_state:
#     st.session_state.qa_chain = None

if uploaded_file is not None:
    if not os.path.isfile("files/"+uploaded_file.name+".pdf"):
        with st.status("Analyzing your document..."):
            bytes_data = uploaded_file.read()
            f = open("files/"+uploaded_file.name, "wb")
            f.write(bytes_data)
            f.close()
            loader = PyPDFLoader("files/"+uploaded_file.name)
            data = loader.load()
            # Initialize text splitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,
                chunk_overlap=200,
                length_function=len
            )
            all_splits = text_splitter.split_documents(data)
            # Create and persist the vector store
            st.session_state.vectorstore = Chroma.from_documents(
                documents=all_splits,
                embedding=OllamaEmbeddings(model="llama3")
            )
            st.session_state.vectorstore.persist()

    st.session_state.retriever = st.session_state.vectorstore.as_retriever()

    if 'qa_chain' not in st.session_state:
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

# Define placeholder for storing recorded audio
audio = audiorecorder("Record", "Stop")

# If audio is not empty (i.e., recording is stopped)
if len(audio) > 0:
    text_input = process_audio_and_convert_to_text(audio)
    print("Converted text:", text_input)  # Print the converted text
    if text_input:
        user_message = {"role": "user", "message": text_input}
        st.session_state.chat_history.append(user_message)
        with st.chat_message("user"):
            st.markdown(text_input)
        print("Type of qa_chain:", type(st.session_state.qa_chain))
        with st.chat_message("assistant"):
            with st.spinner("Assistant is typing..."):
                response = st.session_state.qa_chain(text_input)

            # Generate speech from the bot's response
            speech_text = response['result']
            tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

# generate speech by cloning a voice using default settings
            tts.tts_to_file(text=speech_text,
                file_path="output.wav",
                speaker_wav="sample.wav",
                language="en")
            
            # st.audio("output.wav",start_time=0)
            # display(Audio(filename="output.wav", aut
            # Load audio file
            # audio_file ="output.wav"

            # # Display audio player with autoplay
            # # st.audio(audio_file, start_time=0, autoplay=True)
            # Audio(audio_file, autoplay=True)
            audio_file = "output.wav"
            autoplay_audio(audio_file)
 
            message_placeholder = st.empty()
            full_response = ""
            for chunk in response['result'].split():
                full_response += chunk + " "
                time.sleep(0.05)
                # Add a blinking cursor to simulate typing
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

        chatbot_message = {"role": "assistant", "message": response['result']}
        st.session_state.chat_history.append(chatbot_message)

else:
    st.write("Please upload a PDF file.")
