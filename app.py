import os
import re
import json
import zipfile
import tempfile
from pathlib import Path
from datetime import datetime

import streamlit as st
import gdown
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from pypdf import PdfReader
from docx import Document


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Daraz RAG Assistant",
    page_icon="🛍️",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

DRIVE_FILE_ID = "1jkr0sPZ3WEzYAMOgWeX7syCpppJYoP_3"

# Current preferred Groq model
GROQ_MODEL = "openai/gpt-oss-120b"

# Alternative:
# GROQ_MODEL = "qwen/qwen3.6-27b"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


# ============================================================
# DIRECTORIES
# ============================================================

BASE_DIR = Path(tempfile.gettempdir()) / "daraz_rag"

KB_DIR = BASE_DIR / "knowledge_base"
INDEX_DIR = BASE_DIR / "faiss_index"

INDEX_FILE = INDEX_DIR / "daraz.index"
METADATA_FILE = INDEX_DIR / "metadata.json"

KB_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(text):
    """Clean unnecessary spaces and blank lines."""

    if not text:
        return ""

    text = text.replace("\x00", " ")

    # Normalize spaces
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def extract_txt(file_path):
    """Extract text from TXT file."""

    with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
        return file.read()


def extract_pdf(file_path):
    """Extract text from PDF."""

    reader = PdfReader(str(file_path))

    pages = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages.append(text)

    return "\n\n".join(pages)


def extract_docx(file_path):
    """Extract text from DOCX."""

    document = Document(str(file_path))

    paragraphs = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            paragraphs.append(paragraph.text)

    return "\n".join(paragraphs)


def extract_file_text(file_path):
    """Extract text according to file type."""

    extension = file_path.suffix.lower()

    try:
        if extension == ".txt":
            return extract_txt(file_path)

        elif extension == ".pdf":
            return extract_pdf(file_path)

        elif extension == ".docx":
            return extract_docx(file_path)

        else:
            return ""

    except Exception as error:
        st.warning(f"Could not read {file_path.name}: {error}")
        return ""


def detect_topic(filename):
    """Detect Daraz topic from filename."""

    name = filename.lower()

    topic_map = {
        "return": "Returns and Refunds",
        "refund": "Returns and Refunds",
        "cancel": "Order Cancellation",
        "buyer": "Buyer Policy",
        "payment": "Payment Methods",
        "wallet": "Daraz Wallet",
        "seller": "Seller Policy",
        "privacy": "Privacy",
        "contact": "Contact and Help",
        "help": "Contact and Help",
        "faq": "Frequently Asked Questions",
        "overview": "Platform Overview",
        "platform": "Platform Overview",
        "source": "Sources",
        "metadata": "Metadata",
    }

    for keyword, topic in topic_map.items():
        if keyword in name:
            return topic

    return "General"


# ============================================================
# MEANINGFUL CHUNKING
# ============================================================

def create_chunks(text, chunk_size=900, overlap=150):
    """
    Create meaningful chunks using paragraphs and sentences.
    """

    text = clean_text(text)

    if not text:
        return []

    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current_chunk = ""

    for paragraph in paragraphs:

        sentences = re.split(
            r"(?<=[.!?])\s+",
            paragraph
        )

        for sentence in sentences:

            sentence = sentence.strip()

            if not sentence:
                continue

            # Add sentence if it fits
            if len(current_chunk) + len(sentence) + 1 <= chunk_size:
                current_chunk += (
                    sentence + " "
                ).strip() + " "

            else:

                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                # Keep a small overlap
                words = current_chunk.strip().split()

                overlap_words = words[-30:]

                current_chunk = (
                    " ".join(overlap_words)
                    + " "
                    + sentence
                    + " "
                )

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


# ============================================================
# DOWNLOAD KNOWLEDGE BASE
# ============================================================

def download_knowledge_base():
    """
    Download the Daraz knowledge-base ZIP from Google Drive.
    """

    download_path = BASE_DIR / "daraz_knowledge_base.zip"

    # Download only if it doesn't already exist
    if not download_path.exists():

        url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"

        try:
            result = gdown.download(
                url,
                str(download_path),
                quiet=False
            )

            if result is None:
                raise RuntimeError(
                    "Google Drive download failed."
                )

        except Exception as error:
            raise RuntimeError(
                f"Could not download the knowledge base: {error}"
            )

    return download_path


# ============================================================
# EXTRACT KNOWLEDGE BASE
# ============================================================

def prepare_knowledge_base():
    """
    Download and extract the Daraz knowledge base.
    """

    zip_path = download_knowledge_base()

    marker_file = KB_DIR / ".extracted"

    if not marker_file.exists():

        # Clear old files
        for item in KB_DIR.iterdir():

            if item.is_file():
                item.unlink()

            elif item.is_dir():
                import shutil
                shutil.rmtree(item)

        try:

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(KB_DIR)

        except zipfile.BadZipFile:
            raise RuntimeError(
                "The Google Drive file is not a valid ZIP file."
            )

        marker_file.write_text(
            datetime.now().isoformat(),
            encoding="utf-8"
        )


# ============================================================
# LOAD DOCUMENTS
# ============================================================

def load_documents():

    documents = []

    for file_path in KB_DIR.rglob("*"):

        if not file_path.is_file():
            continue

        if file_path.name.startswith("."):
            continue

        extension = file_path.suffix.lower()

        if extension not in [".txt", ".pdf", ".docx"]:
            continue

        text = extract_file_text(file_path)

        text = clean_text(text)

        if not text:
            continue

        documents.append(
            {
                "source": file_path.name,
                "topic": detect_topic(file_path.name),
                "text": text,
            }
        )

    return documents


# ============================================================
# CREATE METADATA
# ============================================================

def create_metadata(documents):

    metadata = []

    for document in documents:

        chunks = create_chunks(
            document["text"],
            CHUNK_SIZE,
            CHUNK_OVERLAP
        )

        for index, chunk in enumerate(chunks):

            metadata.append(
                {
                    "chunk_id": len(metadata),
                    "source": document["source"],
                    "topic": document["topic"],
                    "chunk_number": index + 1,
                    "text": chunk,
                    "created_at": datetime.now().isoformat(),
                }
            )

    return metadata


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# ============================================================
# BUILD FAISS INDEX
# ============================================================

def build_vector_database(metadata):

    model = load_embedding_model()

    texts = [
        item["text"]
        for item in metadata
    ]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    embeddings = np.asarray(
        embeddings,
        dtype="float32"
    )

    dimension = embeddings.shape[1]

    # Inner Product + normalized vectors
    # = cosine similarity
    index = faiss.IndexFlatIP(dimension)

    index.add(embeddings)

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    with open(
        METADATA_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            ensure_ascii=False,
            indent=2
        )

    return index


# ============================================================
# LOAD OR BUILD RAG DATABASE
# ============================================================

@st.cache_resource
def initialize_rag():

    prepare_knowledge_base()

    # If existing index and metadata are available,
    # load them instead of rebuilding.
    if INDEX_FILE.exists() and METADATA_FILE.exists():

        index = faiss.read_index(
            str(INDEX_FILE)
        )

        with open(
            METADATA_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(file)

        return index, metadata

    # Otherwise build database
    documents = load_documents()

    if not documents:
        raise RuntimeError(
            "No TXT, PDF, or DOCX knowledge-base files were found."
        )

    metadata = create_metadata(documents)

    if not metadata:
        raise RuntimeError(
            "No chunks were created from the knowledge base."
        )

    index = build_vector_database(
        metadata
    )

    return index, metadata


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    question,
    index,
    metadata,
    top_k=TOP_K
):

    model = load_embedding_model()

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for score, index_number in zip(
        scores[0],
        indices[0]
    ):

        if index_number < 0:
            continue

        item = metadata[index_number].copy()

        item["score"] = float(score)

        results.append(item)

    return results


# ============================================================
# GROQ CLIENT
# ============================================================

@st.cache_resource
def get_groq_client():

    try:
        api_key = st.secrets["GROQ_API_KEY"]

    except Exception:
        api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. "
            "Add it in Streamlit Secrets."
        )

    return Groq(
        api_key=api_key
    )


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, retrieved_chunks):

    client = get_groq_client()

    context_parts = []

    for item in retrieved_chunks:

        context_parts.append(
            f"""
Source: {item['source']}
Topic: {item['topic']}
Similarity: {item['score']:.4f}

Content:
{item['text']}
"""
        )

    context = "\n\n".join(
        context_parts
    )

    system_prompt = """
You are a helpful Daraz knowledge-base assistant.

Answer the user's question using ONLY the
provided knowledge-base context.

Rules:
1. Do not invent policies or information.
2. If the answer is not present in the context,
   clearly say that the information is not available
   in the current knowledge base.
3. Give a concise and beginner-friendly answer.
4. When useful, mention the relevant topic.
5. Do not claim that this knowledge base is an
   official Daraz policy document.
"""

    user_prompt = f"""
Knowledge Base Context:
{context}

User Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_tokens=700
    )

    return response.choices[0].message.content


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🛍️ Daraz RAG")

    st.markdown(
        """
### About

This application uses:

- 📚 Daraz knowledge base
- ✂️ Meaningful text chunking
- 🏷️ Metadata
- 🧠 Sentence Transformers
- 🔎 FAISS semantic search
- 🤖 Groq LLM
- 🌐 Streamlit
        """
    )

    st.divider()

    st.write(
        f"**Embedding:** `{EMBEDDING_MODEL}`"
    )

    st.write(
        f"**LLM:** `{GROQ_MODEL}`"
    )

    st.write(
        f"**Top-K:** `{TOP_K}`"
    )

    st.divider()

    st.caption(
        "Educational RAG project. "
        "Policies may change and should be "
        "verified against current official information."
    )


# ============================================================
# MAIN APPLICATION
# ============================================================

st.title("🛍️ Daraz RAG Assistant")

st.write(
    "Ask questions about the Daraz knowledge base "
    "and get answers using semantic retrieval."
)


# ============================================================
# INITIALIZE DATABASE
# ============================================================

try:

    with st.spinner(
        "Loading knowledge base and FAISS database..."
    ):

        index, metadata = initialize_rag()

    st.success(
        f"Knowledge base ready — {len(metadata)} chunks loaded."
    )

except Exception as error:

    st.error(
        f"Application initialization failed: {error}"
    )

    st.info(
        "Check your Google Drive file and Streamlit Secrets."
    )

    st.stop()


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about Daraz..."
)


if question:

    # User message
    with st.chat_message("user"):
        st.write(question)

    # Retrieve relevant chunks
    with st.spinner(
        "Searching the knowledge base..."
    ):

        retrieved_chunks = semantic_search(
            question,
            index,
            metadata,
            TOP_K
        )

    # Generate answer
    with st.chat_message("assistant"):

        try:

            with st.spinner(
                "Generating answer..."
            ):

                answer = generate_answer(
                    question,
                    retrieved_chunks
                )

            st.write(answer)

        except Exception as error:

            st.error(
                f"Could not generate the answer: {error}"
            )


    # ========================================================
    # RETRIEVED SOURCES
    # ========================================================

    with st.expander(
        "🔎 View retrieved knowledge-base chunks"
    ):

        for number, item in enumerate(
            retrieved_chunks,
            start=1
        ):

            st.markdown(
                f"### Chunk {number}"
            )

            st.write(
                f"**Source:** {item['source']}"
            )

            st.write(
                f"**Topic:** {item['topic']}"
            )

            st.write(
                f"**Similarity:** {item['score']:.4f}"
            )

            st.write(
                item["text"]
            )

            st.divider()
