import os
import re
import glob
import warnings
import streamlit as st
import smtplib
from email.message import EmailMessage
from typing import List
import tempfile
from datetime import datetime
from io import BytesIO

from openai import OpenAI
import tiktoken

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

warnings.filterwarnings("ignore")

GENERATION_MODEL = "gpt-4o"
EMBEDDING_MODEL_NAME = "text-embedding-3-small"
GENERATION_ARGS = dict(temperature=0.0, max_tokens=500)

# ------------------ API Setup ------------------
def get_api_key() -> str:
    return os.getenv("OPENAI_API_KEY") 
    
API_KEY = get_api_key()
client = OpenAI(api_key=API_KEY)

# ------------------ Session State Initialization ------------------
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'saved_filepath' not in st.session_state:
    st.session_state.saved_filepath = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'last_audio_bytes' not in st.session_state:
    st.session_state.last_audio_bytes = None

# ------------------ Audio Recording Settings ------------------
audio_output_dir = "audio_submissions"

# Create audio output directory if it doesn't exist
if not os.path.exists(audio_output_dir):
    try:
        os.makedirs(audio_output_dir)
    except Exception as e:
        st.error(f"Error creating audio directory: {e}")


# ------------------ Tokenizer ------------------
def _get_encoding():
    try:
        return tiktoken.get_encoding("o200k_base")
    except:
        return tiktoken.get_encoding("cl100k_base")

def split_text_by_tokens(text: str, max_tokens: int = 300) -> List[str]:
    enc = _get_encoding()
    lines = [re.sub(r"^\d+(\.\d+)*\s*", "", l.strip())
             for l in text.splitlines() if l.strip() and len(l.strip()) >= 20]
    chunks, cur, cur_tokens = [], [], 0
    for line in lines:
        t = len(enc.encode(line))
        if cur_tokens + t <= max_tokens:
            cur.append(line)
            cur_tokens += t
        else:
            if cur:
                chunks.append("\n".join(cur))
            if t > max_tokens:
                ids = enc.encode(line)
                chunks.append(enc.decode(ids[:max_tokens]))
                rest = enc.decode(ids[max_tokens:])
                cur, cur_tokens = ([rest], len(enc.encode(rest))) if rest.strip() else ([], 0)
            else:
                cur, cur_tokens = [line], t
    if cur:
        chunks.append("\n".join(cur))
    return chunks


# ------------------ Load Documents ------------------
def load_single_document(file_path: str) -> List[Document]:
    """Load a single document based on its file extension"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path)
            return loader.load()
        elif file_extension == ".docx":
            loader = Docx2txtLoader(file_path)
            return loader.load()
        elif file_extension in [".txt", ".text"]:
            loader = TextLoader(file_path, encoding='utf-8')
            return loader.load()
        else:
            st.warning(f"Unsupported file type: {file_extension} for file {file_path}")
            return []
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return []

def load_and_chunk_documents(folder: str, max_tokens: int = 300):
    """Load and chunk documents from multiple file formats"""
    all_docs = []
    
    # Supported file patterns
    file_patterns = [
        os.path.join(folder, "*.pdf"),
        os.path.join(folder, "*.docx"),
        os.path.join(folder, "*.txt"),
        os.path.join(folder, "*.text")
    ]
    
    # Load all supported files
    for pattern in file_patterns:
        for file_path in glob.glob(pattern):
            docs = load_single_document(file_path)
            if docs:
                # Add metadata about the source file
                for doc in docs:
                    doc.metadata.update({
                        "source_file": os.path.basename(file_path),
                        "file_type": os.path.splitext(file_path)[1].lower()
                    })
                all_docs.extend(docs)
    
    if not all_docs:
        st.warning(f"No supported documents found in {folder}")
        return []
    
    st.success(f"Loaded {len(all_docs)} document pages from Maasai Cultural Database")
    
    # Chunk all documents
    all_chunks = []
    for doc in all_docs:
        chunks = split_text_by_tokens(doc.page_content, max_tokens)
        for chunk_text in chunks:
            # Create new document with chunk and preserve metadata
            chunk_doc = Document(
                page_content=chunk_text,
                metadata=doc.metadata.copy()
            )
            all_chunks.append(chunk_doc)
    
    return all_chunks


def build_vectorstore(chunks: List[Document]):
    embedding = OpenAIEmbeddings(model=EMBEDDING_MODEL_NAME, openai_api_key=API_KEY)
    return FAISS.from_documents(documents=chunks, embedding=embedding)

@st.cache_resource(show_spinner=False)
def get_retriever(folder_path: str, max_tokens_per_chunk: int = 300):
    chunks = load_and_chunk_documents(folder_path, max_tokens_per_chunk)
    vs = build_vectorstore(chunks)
    return vs.as_retriever(search_kwargs={"k": 5})

# ------------------ RAG Query ------------------
def run_rag_query(query: str, retriever):
    retrieved = retriever.get_relevant_documents(query)
    context = "\n\n".join(doc.page_content for doc in retrieved)
    messages = [
        {"role": "system", "content":
         ("You are an experienced anthropologist who specializes in studying indigenous cultures. "
          "Explain in a respectful, clear, and simple way.\n"
          "Only use the info below.\n"
          "If no answer can be found, say:\n“No relevant information can be found.”")},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
    ]
    response = client.chat.completions.create(model=GENERATION_MODEL, messages=messages, **GENERATION_ARGS)
    return response.choices[0].message.content

# ------------------ Email Upload ------------------
EMAIL_SENDER = "larkhoon.leem@gmail.com"
EMAIL_RECEIVER = "larkhoon.leem@gmail.com"
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

def send_email_with_attachment(file):
    msg = EmailMessage()
    msg["Subject"] = "File from MAASAI ChatGPT"
    msg["From"] = EMAIL_SENDER
    msg["To"] = EMAIL_RECEIVER
    msg.set_content("A new file has been uploaded.")
    msg.add_attachment(file.read(), maintype="application", subtype="octet-stream", filename=file.name)

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
        st.success(f"📨 Successfully sent '{file.name}' for MAASAI Knowledge Collection!")
    except Exception as e:
        st.error(f"❌ Email failed: {e}")


# ------------------ Audio Recording Functions ------------------
def transcribe_audio_openai_api(audio_bytes):
    """Transcribe audio using OpenAI Whisper API"""
    try:
        with st.spinner("🤖 Transcribing audio for knowledge collection..."):
            # Create a temporary file-like object from the bytes
            audio_file = BytesIO(audio_bytes)
            audio_file.name = "audio.wav"  # OpenAI API needs a filename
            
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            return transcript.text
    except Exception as e:
        st.error(f"Transcription error: {e}")
        if "api_key" in str(e).lower():
            st.error("Please check your API key is correct and has sufficient credits")
        return None

def save_audio_bytes(audio_bytes, filename):
    """Save audio bytes to file"""
    try:
        with open(filename, 'wb') as f:
            f.write(audio_bytes)
        return True
    except Exception as e:
        st.error(f"Error saving audio: {e}")
        return False

def process_audio_submission(audio_bytes):
    """Process audio submission: save audio, transcribe, and send via email"""
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    audio_filename = f"maasai_audio_{timestamp}.wav"
    audio_filepath = os.path.join(audio_output_dir, audio_filename)
    
    # Create progress placeholder
    progress_placeholder = st.empty()
    
    # Step 1: Save audio file
    progress_placeholder.info("💾 Saving audio file...")
    if save_audio_bytes(audio_bytes, audio_filepath):
        st.session_state.saved_filepath = audio_filepath
        
        # Step 2: Transcribe
        progress_placeholder.info("🤖 Transcribing audio...")
        transcription = transcribe_audio_openai_api(audio_bytes)
        
        if transcription:
            st.session_state.transcription = transcription
            
            # Step 3: Save transcription
            progress_placeholder.info("📝 Saving transcription...")
            text_filename = f"maasai_transcription_{timestamp}.txt"
            text_filepath = os.path.join(audio_output_dir, text_filename)
            
            try:
                with open(text_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"Audio Transcription - {timestamp}\n")
                    f.write("="*50 + "\n\n")
                    f.write(transcription)
                
                # Step 4: Send email
                progress_placeholder.info("📨 Sending to database...")
                send_audio_submission_email(audio_filepath, text_filepath, transcription, audio_bytes)
                
                # Clear progress and show success
                progress_placeholder.empty()
                st.success("✅ Successfully submitted audio knowledge to MAASAI Cultural Database!")
                
                # Display transcription
                with st.expander("📝 View Transcription"):
                    st.write(transcription)
                
            except Exception as e:
                progress_placeholder.empty()
                st.error(f"Error saving transcription: {e}")
            
            return transcription
        else:
            progress_placeholder.empty()
            st.error("❌ Transcription failed")
            return None
    else:
        progress_placeholder.empty()
        st.error("❌ Failed to save audio file")
        return None

def send_audio_submission_email(audio_filepath, text_filepath, transcription_text, audio_bytes):
    """Send audio submission and transcription via email"""
    try:
        msg = EmailMessage()
        msg["Subject"] = "MAASAI Audio Knowledge Submission"
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        
        # Email body with transcription preview
        email_body = f"""
New Maasai cultural knowledge submitted via audio recording.

Transcription Preview:
{transcription_text[:500]}{'...' if len(transcription_text) > 500 else ''}

Files attached:
- Audio recording (WAV format)
- Full transcription (TXT format)

Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        """
        
        msg.set_content(email_body)
        
        # Attach audio file directly from bytes
        msg.add_attachment(audio_bytes, maintype="audio", subtype="wav", 
                         filename=os.path.basename(audio_filepath))
        
        # Attach transcription file
        with open(text_filepath, "rb") as text_file:
            text_data = text_file.read()
            msg.add_attachment(text_data, maintype="text", subtype="plain", 
                             filename=os.path.basename(text_filepath))
        
        # Send email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
            smtp.send_message(msg)
        
        st.success("📨 Successfully submitted audio to MAASAI Cultural Database!")
        
    except Exception as e:
        st.error(f"❌ Email submission failed: {e}")


# ------------------ UI ------------------
st.markdown("<h1 style='font-size:36px;'>MAASAI ChatGPT v0.1</h1>", unsafe_allow_html=True)

# File upload section
st.markdown(
    "<div style='font-weight: 600; font-size: 16px; margin-bottom: 0px;'>"
    "📁 Submit a file to update MAASAI Knowledge"
    "</div>",
    unsafe_allow_html=True,
)
uploaded_file = st.file_uploader("", type=["pdf", "txt", "docx"])
if uploaded_file:
    send_email_with_attachment(uploaded_file)

# Audio recording section
st.markdown(
    "<div style='font-weight: 600; font-size: 16px; margin-top: 20px; margin-bottom: 10px;'>"
    "🎤 Submit a voice recording to update MAASAI Knowledge"
    "</div>",
    unsafe_allow_html=True,
)

# Use st.audio_input for recording
audio_value = st.audio_input("Click to start recording, speak your knowledge, then click stop when done")

if audio_value:
    # Check if this is new audio (different from last processed)
    audio_bytes = audio_value.getvalue()
    
    if audio_bytes != st.session_state.last_audio_bytes:
        st.session_state.last_audio_bytes = audio_bytes
        st.session_state.processing = True
        
        # Process the audio
        with st.container():
            st.info("🔄 Processing your recording...")
            transcription = process_audio_submission(audio_bytes)
            st.session_state.processing = False
            
            if transcription:
                # Option to clear the recording
                if st.button("🔄 Record Another", type="secondary"):
                    st.session_state.last_audio_bytes = None
                    st.session_state.transcription = ""
                    st.rerun()

# Chat interface
st.markdown(
    "<div style='font-weight: 600; font-size: 16px; margin-top: 30px; margin-bottom: 10px;'>"
    "💬 Ask Questions About Maasai Culture"
    "</div>",
    unsafe_allow_html=True,
)

documents_folder = "./documents"
retriever = get_retriever(documents_folder)

if query := st.chat_input("Ask me anything about the Maasai:"):
    st.chat_message("user").write(query)
    answer = run_rag_query(query, retriever)
    st.chat_message("assistant").write(answer)