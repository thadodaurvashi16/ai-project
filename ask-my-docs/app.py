from pathlib import Path
import os
import numpy as np
import streamlit as st

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# 1. LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please check your .env file.")
    st.stop()


# ============================================================
# 2. CREATE GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# 3. STREAMLIT PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Ask My Docs",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Ask My Docs")
st.write("Ask questions from your documents using Gemini AI.")


# ============================================================
# 4. LOAD DOCUMENTS
# ============================================================

docs_folder = Path("docs")

documents = []

for file_path in sorted(docs_folder.glob("*.txt")):

    text = file_path.read_text(
        encoding="utf-8"
    )

    documents.append({
        "filename": file_path.name,
        "text": text
    })


# Check documents

if len(documents) == 0:

    st.warning(
        "No .txt files found inside the docs folder."
    )

    st.stop()


# ============================================================
# 5. SPLIT TEXT INTO CHUNKS
# ============================================================

def split_text(text, chunk_size=20):

    words = text.split()

    chunks = []

    for i in range(0, len(words), chunk_size):

        chunk = " ".join(
            words[i:i + chunk_size]
        )

        chunks.append(chunk)

    return chunks


chunks = []

for document in documents:

    document_chunks = split_text(
        document["text"],
        chunk_size=20
    )

    for chunk in document_chunks:

        chunks.append({
            "filename": document["filename"],
            "text": chunk
        })


# ============================================================
# 6. CREATE DOCUMENT EMBEDDINGS
# ============================================================

@st.cache_data
def create_document_embeddings(chunk_list):

    embeddings = []

    for chunk in chunk_list:

        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=chunk["text"],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )

        embedding = np.array(
            result.embeddings[0].values
        )

        embeddings.append(embedding)

    return np.array(embeddings)


doc_embeddings = create_document_embeddings(
    chunks
)


# ============================================================
# 7. CREATE QUERY EMBEDDING
# ============================================================

def get_query_embedding(question):

    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=question,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY"
        )
    )

    return np.array(
        result.embeddings[0].values
    )


# ============================================================
# 8. SEARCH RELEVANT DOCUMENT
# ============================================================

def search_documents(question):

    question_embedding = get_query_embedding(
        question
    )

    similarities = (
        doc_embeddings @ question_embedding
        /
        (
            np.linalg.norm(
                doc_embeddings,
                axis=1
            )
            *
            np.linalg.norm(
                question_embedding
            )
        )
    )

    best_index = np.argmax(
        similarities
    )

    return (
        chunks[best_index],
        similarities[best_index]
    )


# ============================================================
# 9. GENERATE FINAL ANSWER USING GEMINI
# ============================================================

def generate_answer(
    question,
    relevant_chunk
):

    context = relevant_chunk["text"]

    prompt = f"""
You are a helpful document assistant.

Answer the user's question using ONLY the information
available in the document context below.

If the answer is not available in the document context,
say:

"I could not find the answer in the provided documents."

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Give a simple and clear answer.
"""

    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )

    return response.text


# ============================================================
# 10. STREAMLIT FRONTEND
# ============================================================

st.subheader("Ask a Question")

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is machine learning?"
)


# ============================================================
# 11. ASK BUTTON
# ============================================================

if st.button("🔍 Ask"):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        # Search document

        with st.spinner(
            "Searching documents..."
        ):

            relevant_chunk, score = search_documents(
                question
            )


        # Generate answer

        with st.spinner(
            "Generating answer..."
        ):

            answer = generate_answer(
                question,
                relevant_chunk
            )


        # ====================================================
        # DISPLAY ANSWER
        # ====================================================

        st.subheader("💡 Answer")

        st.write(answer)


        # ====================================================
        # DISPLAY SOURCE
        # ====================================================

        st.subheader("📄 Source")

        st.info(
            relevant_chunk["filename"]
        )


        # ====================================================
        # DISPLAY SIMILARITY SCORE
        # ====================================================

        st.subheader("📊 Similarity Score")

        st.write(
            round(
                float(score),
                4
            )
        )


        # ====================================================
        # DISPLAY RELEVANT CHUNK
        # ====================================================

        with st.expander(
            "📖 View relevant document text"
        ):

            st.write(
                relevant_chunk["text"]
            )