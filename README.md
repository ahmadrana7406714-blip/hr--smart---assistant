# HR Smart Assistant using RAG

An AI-powered HR Smart Assistant that allows users to upload an HR PDF and ask questions about the document.

## Features

- Upload HR PDF
- Extract text from PDF
- Split document into chunks
- Retrieve relevant information using RAG
- Generate answers using GPT-OSS-20B
- Streamlit user interface
- GitHub deployment
- Streamlit Community Cloud deployment

## Technologies

- Python
- Streamlit
- RAG
- TF-IDF
- Scikit-learn
- PyPDF
- Groq API
- GPT-OSS-20B
- GitHub

## How it works

User uploads an HR PDF.

The application:

1. Extracts the text.
2. Splits the text into chunks.
3. Searches for relevant chunks.
4. Sends the relevant context to GPT-OSS-20B.
5. Generates an answer based on the HR document.

## Model

openai/gpt-oss-20b

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
