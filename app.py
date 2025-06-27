import gradio as gr
import os
import pytesseract
import requests
import PyPDF2
import pandas as pd
from docx import Document
from PIL import Image
import whisper
from langchain_community.document_loaders import WebBaseLoader
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
pytesseract.pytesseract.tesseract_cmd = os.getenv("TESSERACT_CMD")

model = whisper.load_model("base")

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".txt":
        return open(file_path, "r", encoding="utf-8").read()
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs)
    elif ext == ".csv":
        df = pd.read_csv(file_path)
        return df.to_string(index=False)
    elif ext == ".pdf":
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())
    else:
        return "Unsupported file type."
        
def chat_with_file(file, question):
    content = extract_text_from_file(file.name)
    prompt = f"{content}\n\nQuestion: {question}"
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "Answer based on the file content."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def chat_with_url(url, question):
    loader = WebBaseLoader(url)
    docs = loader.load()
    content = "\n".join([doc.page_content for doc in docs])
    prompt = f"{content}\n\nQuestion: {question}"
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": "Answer based on the website content."},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result["text"]

with gr.Blocks() as demo:
    gr.Markdown("# Nexus AI")

    with gr.Tab("Chat with File"):
        file = gr.File()
        file_q = gr.Textbox(placeholder="Ask a question based on the file")
        file_btn = gr.Button("Ask")
        file_a = gr.Textbox(label="Answer")
        file_btn.click(chat_with_file, [file, file_q], file_a)

    with gr.Tab("Chat with Website"):
        url = gr.Textbox(placeholder="Enter a URL")
        url_q = gr.Textbox(placeholder="Ask a question based on the page")
        url_btn = gr.Button("Ask")
        url_a = gr.Textbox(label="Answer")
        url_btn.click(chat_with_url, [url, url_q], url_a)

    with gr.Tab("Extract Text from Image"):
        image = gr.Image(type="filepath")
        image_btn = gr.Button("Extract Text")
        image_out = gr.Textbox(label="Detected Text")
        image_btn.click(extract_text_from_image, image, image_out)

    with gr.Tab("Transcribe Audio"):
        audio = gr.Audio(type="filepath")
        audio_btn = gr.Button("Transcribe")
        audio_out = gr.Textbox(label="Transcript")
        audio_btn.click(transcribe_audio, audio, audio_out)

demo.launch()
