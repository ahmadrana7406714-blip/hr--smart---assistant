import os
import re
import io

import streamlit as st
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from groq import Groq


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="HR Smart Assistant",
    page_icon="👩‍💼",
    layout="wide"
)


# =========================================================
# TITLE
# =========================================================

st.title("👩‍💼 HR Smart Assistant")
st.write(
    "Upload an HR PDF and ask questions. "
    "The assistant will search the document using RAG "
    "and answer using GPT-OSS-20B."
)


# =========================================================
# GROQ API KEY
# =========================================================

# For Streamlit Cloud:
# st.secrets["GROQ_API_KEY"]

# For local/Colab:
# environment variable GROQ_API_KEY

try:
    api_key = st.secrets["GROQ_API_KEY"]
except Exception:
    api_key = os.getenv("GROQ_API_KEY")


if not api_key:
    st.warning(
        "GROQ_API_KEY is not configured. "
        "Add your API key in Streamlit Secrets."
    )
    st.stop()


client = Groq(api_key=api_key)

MODEL_NAME = "openai/gpt-oss-20b"


# =========================================================
# PDF TEXT EXTRACTION
# =========================================================

def extract_text_from_pdf(pdf_file):
    """
    Read all text from an uploaded PDF.
    """

    pdf_bytes = pdf_file.read()

    reader = PdfReader(io.BytesIO(pdf_bytes))

    pages_text = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages_text.append(text)

    return "\n".join(pages_text)


# =========================================================
# TEXT CLEANING
# =========================================================

def clean_text(text):
    """
    Remove unnecessary spaces and blank lines.
    """

    text = re.sub(r"\s+", " ", text)

    return text.strip()


# =========================================================
# CREATE CHUNKS
# =========================================================

def create_chunks(text, chunk_size=800, overlap=150):
    """
    Split document into smaller chunks.

    RAG works better when we search smaller pieces
    instead of the complete document.
    """

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(words[start:end])

        if chunk.strip():
            chunks.append(chunk)

        start = end - overlap

    return chunks


# =========================================================
# RAG RETRIEVAL
# =========================================================

def retrieve_relevant_chunks(question, chunks, top_k=4):

    if not chunks:
        return []

    documents = chunks + [question]

    vectorizer = TfidfVectorizer(
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(documents)

    question_vector = matrix[-1]

    document_vectors = matrix[:-1]

    similarities = cosine_similarity(
        question_vector,
        document_vectors
    )[0]

    top_indexes = similarities.argsort()[-top_k:][::-1]

    results = []

    for index in top_indexes:

        if similarities[index] > 0:

            results.append({
                "text": chunks[index],
                "score": float(similarities[index])
            })

    return results


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(question, retrieved_chunks):

    if not retrieved_chunks:

        return (
            "I could not find relevant information in the "
            "uploaded HR document."
        )

    context = "\n\n".join(
        [
            f"DOCUMENT SECTION {i + 1}:\n{item['text']}"
            for i, item in enumerate(retrieved_chunks)
        ]
    )

    system_prompt = """
You are an HR Smart Assistant.

Answer the user's question ONLY using the information
provided in the document context.

Rules:

1. Do not invent HR policies.
2. Do not make up information.
3. If the answer is not present in the document,
   clearly say that the information was not found.
4. Give a simple and clear answer.
5. Mention important details such as numbers,
   dates, rules, or conditions when available.
"""

    user_prompt = f"""
DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

Answer the question using only the document context.
"""

    response = client.chat.completions.create(
        model=MODEL_NAME,
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
        max_tokens=800
    )

    return response.choices[0].message.content


# =========================================================
# PDF UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📄 Upload your HR PDF",
    type=["pdf"]
)


# =========================================================
# PROCESS PDF
# =========================================================

if uploaded_file:

    with st.spinner("Reading HR PDF..."):

        try:

            raw_text = extract_text_from_pdf(
                uploaded_file
            )

            text = clean_text(raw_text)

            if not text:

                st.error(
                    "Could not extract text from this PDF. "
                    "Please upload a text-based PDF."
                )

                st.stop()

            chunks = create_chunks(text)

            st.success(
                f"PDF processed successfully! "
                f"Created {len(chunks)} text chunks."
            )

            # Store chunks in session
            st.session_state["chunks"] = chunks
            st.session_state["pdf_name"] = uploaded_file.name

        except Exception as e:

            st.error(
                f"Could not process the PDF: {str(e)}"
            )


# =========================================================
# QUESTION AREA
# =========================================================

if "chunks" in st.session_state:

    st.divider()

    st.subheader("💬 Ask your HR question")

    question = st.text_input(
        "Example: How many casual leaves are allowed?"
    )

    if st.button("Ask Assistant"):

        if not question.strip():

            st.warning("Please enter a question.")

        else:

            with st.spinner(
                "Searching HR document and generating answer..."
            ):

                retrieved = retrieve_relevant_chunks(
                    question,
                    st.session_state["chunks"],
                    top_k=4
                )

                answer = generate_answer(
                    question,
                    retrieved
                )

            st.subheader("🤖 Assistant Answer")

            st.write(answer)

            # Show retrieved information
            with st.expander("🔎 View RAG Retrieved Sections"):

                for i, item in enumerate(retrieved):

                    st.markdown(
                        f"**Section {i + 1} "
                        f"(score: {item['score']:.3f})**"
                    )

                    st.write(item["text"])

else:

    st.info(
        "👆 Upload an HR PDF to start using the assistant."
    )


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("About")

    st.write(
        "This application uses RAG "
        "(Retrieval-Augmented Generation)."
    )

    st.write("### How it works")

    st.write(
        """
        1. Upload HR PDF
        2. Extract PDF text
        3. Split text into chunks
        4. Retrieve relevant chunks
        5. Send context to GPT-OSS-20B
        6. Generate answer
        """
    )

    st.write("### Model")

    st.code(MODEL_NAME)

    st.write("### Project")

    st.write("HR Smart Assistant using RAG + GitHub + Streamlit")
