from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from groq import Groq
from langchain_community.document_loaders import WebBaseLoader

import os
import io
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import whisper

# Load environment variables
load_dotenv()

# Tesseract path
tesseract_cmd = os.getenv("TESSERACT_CMD")
if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

# FFmpeg path (adjust this path to match where you extracted FFmpeg)
ffmpeg_path = r"C:\Users\aliab\Downloads\ffmpeg-7.1.1-essentials_build\ffmpeg-7.1.1-essentials_build\bin"
if os.path.exists(ffmpeg_path):
    os.environ["PATH"] += os.pathsep + ffmpeg_path

# File reading libraries
from docx import Document
import pandas as pd
import PyPDF2

app = FastAPI()

# Use Groq API key from .env
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_FILE_SIZE_MB = 10

# ---------- File Text Extraction ----------
def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    elif ext == ".docx":
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    elif ext == ".csv":
        df = pd.read_csv(file_path)
        return df.to_string(index=False)
    elif ext == ".pdf":
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    else:
        return "❌ Unsupported file type."

# ---------- Chat with File ----------
@app.post("/chat-with-file")
async def chat_with_file(file: UploadFile = File(...), question: str = Form(...)):
    try:
        contents = await file.read()

        if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
            return JSONResponse(status_code=400, content={"error": "❌ File too large. Max size is 10MB."})

        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            f.write(contents)

        file_content = extract_text_from_file(file_path)

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the uploaded file content to answer questions."},
                {"role": "user", "content": f"{file_content}\n\nQuestion: {question}"}
            ]
        )

        return {"answer": response.choices[0].message.content}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------- Chat with URL ----------
class URLQuery(BaseModel):
    url: str
    question: str

@app.post("/chat-with-url")
async def chat_with_url(data: URLQuery):
    try:
        os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        loader = WebBaseLoader(data.url)
        documents = loader.load()
        web_content = "\n".join([doc.page_content for doc in documents])

        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the website content to answer the user's question."},
                {"role": "user", "content": f"Website Content:\n{web_content}\n\nNow answer this question:\n{data.question}"}
            ]
        )
        return {"answer": response.choices[0].message.content}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------- Extract Text from Image ----------
@app.post("/extract-text-from-image")
async def extract_text_from_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        text = pytesseract.image_to_string(image)
        return {"answer": text.strip() or "⚠️ No text extracted."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---------- Transcribe Audio ----------
@app.post("/transcribe-audio")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        audio_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(audio_path, "wb") as f:
            f.write(contents)

        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return {"answer": result["text"] if result.get("text") else "⚠️ No transcript returned."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
