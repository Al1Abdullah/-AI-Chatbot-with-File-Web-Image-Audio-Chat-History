# 🧠 AI Chatbot with File, Web, OCR & Audio RAG (FastAPI + Gradio)

An AI assistant built with FastAPI and Gradio that can:
- Answer questions from uploaded `.txt`, `.pdf`, `.docx`, `.csv` files.
- Extract and use text from websites.
- Perform OCR on images using Tesseract.
- Transcribe audio using OpenAI's Whisper.
- Respond using Groq's LLaMA models.

## 🚀 Features
- Chat with documents
- RAG from URLs
- OCR from images
- Audio transcription
- Chat history memory

## 🛠️ Tech Stack
- FastAPI
- Gradio
- Whisper
- Tesseract OCR
- LangChain
- Groq LLaMA

## 📦 Setup Instructions

1. Clone the repo  
   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
2. Install dependencies
   ```bash
   pip install -r requirements.txt
3. Create .env file
   ```bash
   GROQ_API_KEY=your_groq_api_key
   TESSERACT_CMD=C:\\Program Files\\Tesseract-OCR\\tesseract.exe
4. Start the FastAPI server
   ```bash
   uvicorn main:app --reload
5. Run the Gradio frontend
   ```bash
   python app.py
