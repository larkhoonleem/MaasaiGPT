import os
import re
import glob
import warnings
import streamlit as st
import smtplib
from email.message import EmailMessage
from typing import List

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time
import tempfile
from datetime import datetime

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
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'sample_rate' not in st.session_state:
    st.session_state.sample_rate = 44100
if 'transcription' not in st.session_state:
    st.session_state.transcription = ""
if 'saved_filepath' not in st.session_state:
    st.session_state.saved_filepath = None
if 'recording_object' not in st.session_state:
    st.session_state.recording_object = None
# ADD THIS LINE:
if 'processing' not in st.session_state:
    st.session_state.processing = False

# ------------------ Audio Recording Settings ------------------
duration = 30  # Increased duration for data collection
sample_rate = 44100
channels = 1
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
            #st.info(f"Loading: {os.path.basename(file_path)}")
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
    
    #st.info(f"Created {len(all_chunks)} text chunks")
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
def transcribe_audio_openai_api(audio_file_path):
    """Transcribe audio using OpenAI Whisper API"""
    try:
        with st.spinner("🤖 Transcribing audio for knowledge collection..."):
            with open(audio_file_path, "rb") as audio_file:
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

def record_audio(duration, sample_rate, channels):
    """Record audio for specified duration"""
    try:
        # Create a placeholder for recording status
        recording_placeholder = st.empty()
        progress_placeholder = st.empty()
        
        with recording_placeholder.container():
            st.info(f"🔴 Recording for {duration} seconds... Share your Maasai knowledge!")
        
        # Show countdown
        progress_bar = progress_placeholder.progress(0)
        
        # Record audio
        audio_data = sd.rec(int(duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=channels, 
                           dtype='float64')
        
        # Update progress bar during recording
        for i in range(duration):
            time.sleep(1)
            progress = (i + 1) / duration
            progress_bar.progress(progress)
        
        # Wait for recording to complete
        sd.wait()
        
        # Clear placeholders
        recording_placeholder.empty()
        progress_placeholder.empty()
        
        return audio_data, sample_rate
    except Exception as e:
        st.error(f"Error during recording: {e}")
        return None, None


def save_audio(audio_data, sample_rate, filename):
    """Save audio data to file"""
    try:
        # Convert to 16-bit integers
        audio_int = np.int16(audio_data * 32767)
        
        # Save as WAV file
        write(filename, sample_rate, audio_int)
        return True
    except Exception as e:
        st.error(f"Error saving audio: {e}")
        return False

def process_audio_submission():
    """Process audio submission: save audio, transcribe, and send via email"""
    if st.session_state.audio_data is not None:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"maasai_audio_{timestamp}.wav"
        audio_filepath = os.path.join(audio_output_dir, audio_filename)
        
        # Save audio file
        if save_audio(st.session_state.audio_data, st.session_state.sample_rate, audio_filepath):
            st.session_state.saved_filepath = audio_filepath
            #st.success(f"💾 Audio saved as: {audio_filename}")
            
            # Automatically transcribe
            transcription = transcribe_audio_openai_api(audio_filepath)
            
            if transcription:
                st.session_state.transcription = transcription
                #st.success("✅ Transcription completed!")
                
                # Save transcription as text file
                text_filename = f"maasai_transcription_{timestamp}.txt"
                text_filepath = os.path.join(audio_output_dir, text_filename)
                
                try:
                    with open(text_filepath, 'w', encoding='utf-8') as f:
                        f.write(f"Audio Transcription - {timestamp}\n")
                        f.write("="*50 + "\n\n")
                        f.write(transcription)
                    
                    # Send both audio and transcription via email
                    send_audio_submission_email(audio_filepath, text_filepath, transcription)
                    
                except Exception as e:
                    st.error(f"Error saving transcription: {e}")
                
                return transcription
            else:
                st.error("❌ Transcription failed")
                return None
        else:
            st.error("❌ Failed to save audio file")
            return None

def send_audio_submission_email(audio_filepath, text_filepath, transcription_text):
    """Send audio submission and transcription via email"""
    EMAIL_SENDER = "larkhoon.leem@gmail.com"
    EMAIL_RECEIVER = "larkhoon.leem@gmail.com"
    EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
    
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
        
        # Attach audio file
        with open(audio_filepath, "rb") as audio_file:
            audio_data = audio_file.read()
            msg.add_attachment(audio_data, maintype="audio", subtype="wav", 
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
        
        st.success("📨 Successfully submitted audio knowledge to MAASAI Cultural Database!")
        
    except Exception as e:
        st.error(f"❌ Email submission failed: {e}")


def process_audio_submission_with_progress():
    """Process audio submission with progress updates"""
    if st.session_state.audio_data is not None:
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_filename = f"maasai_audio_{timestamp}.wav"
        audio_filepath = os.path.join(audio_output_dir, audio_filename)
        
        # Create progress placeholder
        progress_placeholder = st.empty()
        
        # Step 1: Save audio file
        progress_placeholder.info("💾 Saving audio file...")
        if save_audio(st.session_state.audio_data, st.session_state.sample_rate, audio_filepath):
            st.session_state.saved_filepath = audio_filepath
            
            # Step 2: Transcribe
            progress_placeholder.info("🤖 Transcribing audio...")
            transcription = transcribe_audio_openai_api(audio_filepath)
            
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
                    send_audio_submission_email(audio_filepath, text_filepath, transcription)
                    
                    # Clear progress and show success
                    progress_placeholder.empty()
                    
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





# ------------------ UI ------------------
st.markdown("<h1 style='font-size:36px;'>MAASAI ChatGPT v0.1</h1>", unsafe_allow_html=True)

#st.markdown("###### 📁 Submit a file to update MAASAI Knowledge")
st.markdown(
    "<div style='font-weight: 600; font-size: 16px; margin-bottom: 0px;'>"
    "📁 Submit a file to update MAASAI Knowledge"
    "</div>",
    unsafe_allow_html=True,
)
#st.markdown("<div style='margin-top: -10px'></div>", unsafe_allow_html=True)  # adjusts top spacing
uploaded_file = st.file_uploader("", type=["pdf", "txt", "docx"])
if uploaded_file:
    send_email_with_attachment(uploaded_file)


# # Recording controls
# st.markdown("<div style='margin-bottom: 4px'></div>", unsafe_allow_html=True)  # adds space to match uploader
# st.markdown("###### 🎤 Submit a voice recording to update MAASAI Knowledge\n")
# st.markdown("<div style='margin-bottom: 4px'></div>", unsafe_allow_html=True)  # adds space to match uploader


# Show recording status
# Show detailed status
if st.session_state.recording:
    st.info("🔴 **Recording in progress...** Click 'Stop Recording' when finished.")
elif st.session_state.processing:
    st.info("🔄 **Processing your recording...** Please wait while we transcribe and submit your knowledge.")
else:
    st.info("🎤 **Ready to record.** Click 'Start Recording' to share your Maasai knowledge.")


# Create two columns for side-by-side buttons
col1, col2 = st.columns(2)

with col1:
    start_disabled = st.session_state.recording or st.session_state.processing
    start_label = ("🎤 Start Recording" if not start_disabled 
                  else "🎤 Recording..." if st.session_state.recording 
                  else "🎤 Processing...")
    
    if st.button(start_label, 
                 use_container_width=True, 
                 type="primary",
                 disabled=start_disabled):
        try:
            # Clear previous data
            st.session_state.transcription = ""
            st.session_state.saved_filepath = None
            st.session_state.audio_data = None
            st.session_state.processing = False
            
            # Start recording
            recording_obj = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                dtype='float64'
            )
            
            # Store the recording object and set state
            st.session_state.recording_object = recording_obj
            st.session_state.recording = True
            
            st.rerun()
            
        except Exception as e:
            st.session_state.recording = False
            st.session_state.recording_object = None
            st.session_state.processing = False
            st.error(f"Recording error: {e}")

with col2:
    stop_disabled = not st.session_state.recording or st.session_state.processing
    
    if st.button("⏹️ Stop Recording", 
                 use_container_width=True, 
                 type="secondary",
                 disabled=stop_disabled):
        try:
            # Stop the recording and set processing state
            sd.stop()
            st.session_state.recording = False
            st.session_state.processing = True
            
            # Get the audio data
            st.session_state.audio_data = st.session_state.recording_object
            st.session_state.sample_rate = sample_rate
            st.session_state.recording_object = None
            
            # Force UI update to show processing state
            st.rerun()
                
        except Exception as e:
            st.session_state.recording = False
            st.session_state.recording_object = None
            st.session_state.processing = False
            st.error(f"Error stopping recording: {e}")

# Process audio if we have data and are in processing state
if st.session_state.processing and st.session_state.audio_data is not None:
    try:
        process_audio_submission_with_progress()
        # Reset processing state after completion
        st.session_state.processing = False
        st.success("✅ All done! Ready for next recording.")
        st.rerun()  # Update UI to re-enable start button
    except Exception as e:
        st.session_state.processing = False
        st.error(f"Error processing audio: {e}")
        st.rerun()

documents_folder = "./documents"
retriever = get_retriever(documents_folder)

if query := st.chat_input("Ask me anything about the Maasai:"):
    st.chat_message("user").write(query)
    answer = run_rag_query(query, retriever)
    st.chat_message("assistant").write(answer)
