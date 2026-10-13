from pathlib import Path
import os
import pickle
import time

import numpy as np
import streamlit as st

from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Ask My Docs",
    page_icon="📚",
    layout="centered"
)


# ============================================================
# 2. LOAD API KEY
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:

    st.error(
        "GEMINI_API_KEY not found. "
        "Please check your .env file."
    )

    st.stop()


# ============================================================
# 3. CREATE GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# 4. LOAD SAVED EMBEDDINGS
# ============================================================

EMBEDDING_FILE = Path(
    "data/embeddings.pkl"
)

if not EMBEDDING_FILE.exists():

    st.error(
        "embeddings.pkl not found inside the data folder."
    )

    st.info(
        "Please run your embedding creation code first."
    )

    st.stop()


with open(
    EMBEDDING_FILE,
    "rb"
) as file:

    saved_chunks = pickle.load(
        file
    )


# ============================================================
# 5. APP TITLE
# ============================================================

st.title("📚 Ask My Docs")

st.write(
    "Ask questions from your documents "
    "using Gemini AI."
)


# ============================================================
# 6. SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📁 Document Information")

    st.write(
        f"Total chunks: **{len(saved_chunks)}**"
    )

    sources = sorted(
        set(
            chunk["source"]
            for chunk in saved_chunks
        )
    )

    st.write(
        f"Total documents: **{len(sources)}**"
    )

    st.write("")

    st.subheader("Documents")

    for source in sources:

        st.write(
            f"📄 {source}"
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

    embedding = np.array(
        result.embeddings[0].values,
        dtype=np.float32
    )

    return embedding


# ============================================================
# 8. RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_relevant_chunks(
    question,
    top_k=3
):

    question_embedding = get_query_embedding(
        question
    )

    scored_chunks = []

    for chunk in saved_chunks:

        chunk_embedding = np.array(
            chunk["embedding"],
            dtype=np.float32
        )

        chunk_norm = np.linalg.norm(
            chunk_embedding
        )

        question_norm = np.linalg.norm(
            question_embedding
        )

        if (
            chunk_norm == 0
            or question_norm == 0
        ):

            score = 0.0

        else:

            score = np.dot(
                question_embedding,
                chunk_embedding
            ) / (
                question_norm
                * question_embedding.dot(
                    question_embedding
                ) ** 0.0
                * chunk_norm
            )

        scored_chunks.append(
            {
                "score": float(score),
                "text": chunk["text"],
                "source": chunk["source"]
            }
        )


    scored_chunks.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    return scored_chunks[:top_k]


# ============================================================
# 9. ASK GEMINI
# ============================================================

def ask_my_docs(question):

    retrieved_chunks = retrieve_relevant_chunks(
        question,
        top_k=3
    )


    # Keep useful chunks

    useful_chunks = [

        chunk

        for chunk in retrieved_chunks

        if chunk["score"] >= 0.30

    ]


    if not useful_chunks:

        return (
            "I don't know based on the provided documents.",
            []
        )


    # Create context

    context_parts = []

    for chunk in useful_chunks:

        context_parts.append(

            f"Source: {chunk['source']}\n"
            f"Content: {chunk['text']}"

        )


    context = "\n\n---\n\n".join(
        context_parts
    )


    # Create prompt

    prompt = f"""
You are a document-based question answering assistant.

Answer the user's question using ONLY the information
provided in the CONTEXT below.

Do not use outside knowledge.

If the answer cannot be found in the context,
say exactly:

"I don't know based on the provided documents."

Do not make up information.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


    # Gemini API

    for attempt in range(3):

        try:

            response = client.models.generate_content(

                model="gemini-3.8-flash",

                contents=prompt

            )

            answer = response.text

            return (
                answer,
                useful_chunks
            )


        except Exception as error:

            error_message = str(error)


            # 429

            if (
                "429" in error_message
                or "RESOURCE_EXHAUSTED"
                in error_message
            ):

                return (
                    "Gemini API quota has been exceeded. "
                    "Please try again later.",
                    useful_chunks
                )


            # 503

            elif (
                "503" in error_message
                or "UNAVAILABLE"
                in error_message
            ):

                if attempt < 2:

                    wait_time = 5 * (
                        attempt + 1
                    )

                    time.sleep(
                        wait_time
                    )

                else:

                    return (
                        "Gemini is temporarily busy. "
                        "Please try again later.",
                        useful_chunks
                    )


            # 404

            elif (
                "404" in error_message
                or "NOT_FOUND"
                in error_message
            ):

                return (
                    "The selected Gemini model is "
                    "not available for this API key.",
                    useful_chunks
                )


            # 401 / 403

            elif (
                "401" in error_message
                or "403" in error_message
                or "PERMISSION_DENIED"
                in error_message
            ):

                return (
                    "Gemini API authentication error. "
                    "Please check your API key.",
                    useful_chunks
                )


            else:

                return (
                    f"Gemini API error: {error_message}",
                    useful_chunks
                )


# ============================================================
# 10. QUESTION INPUT
# ============================================================

st.subheader("Ask a Question")

question = st.text_input(
    "Enter your question:",
    placeholder="Example: What is machine learning?"
)


# ============================================================
# 11. ASK BUTTON
# ============================================================

if st.button(
    "🔍 Ask",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "Please enter a question."
        )

    else:

        # Search + answer

        with st.spinner(
            "Searching documents and generating answer..."
        ):

            answer, sources = ask_my_docs(
                question
            )


        # ====================================================
        # ANSWER
        # ====================================================

        st.subheader("💡 Answer")

        st.success(
            answer
        )


        # ====================================================
        # SOURCES
        # ====================================================

        if sources:

            st.subheader("📄 Sources")

            for source in sources:

                st.write(
                    f"📄 **{source['source']}**"
                )

                st.write(
                    f"Similarity: "
                    f"**{source['score']:.3f}**"
                )

                with st.expander(
                    "View relevant text"
                ):

                    st.write(
                        source["text"]
                    )

                st.divider()