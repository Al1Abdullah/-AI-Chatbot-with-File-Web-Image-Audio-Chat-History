import gradio as gr
import requests
import mimetypes
import os

FILE_API_URL = "http://127.0.0.1:8000/chat-with-file"
URL_API_URL = "http://127.0.0.1:8000/chat-with-url"
IMAGE_API_URL = "http://127.0.0.1:8000/extract-text-from-image"
AUDIO_API_URL = "http://127.0.0.1:8000/transcribe-audio"

chat_sessions = {"file": [], "url": []}

def ask_from_file(file_obj, question, session_id):
    if not file_obj or not question.strip():
        return "⚠️ Please upload a file and enter a question.", chat_sessions["file"]

    if os.path.getsize(file_obj.name) > 10 * 1024 * 1024:
        return "❌ File too large. Please upload files under 10MB.", chat_sessions["file"]

    try:
        mime_type, _ = mimetypes.guess_type(file_obj.name)
        with open(file_obj.name, "rb") as f:
            files = {"file": (file_obj.name, f, mime_type or "application/octet-stream")}
            history = chat_sessions["file"]
            context = "\n".join([f"Q: {q}\nA: {a}" for q, a in history]) + f"\nQ: {question}"
            data = {"question": context}
            response = requests.post(FILE_API_URL, files=files, data=data)
            answer = response.json().get("answer", "⚠️ No answer returned.")
            chat_sessions["file"].append((question, answer))
            return format_chat(chat_sessions["file"]), chat_sessions["file"]
    except Exception as e:
        return f"❌ Error: {str(e)}", chat_sessions["file"]

def ask_from_url(url, question, session_id):
    if not url.strip() or not question.strip():
        return "⚠️ Please enter both a valid URL and a question.", chat_sessions["url"]
    try:
        history = chat_sessions["url"]
        context = "\n".join([f"Q: {q}\nA: {a}" for q, a in history]) + f"\nQ: {question}"
        response = requests.post(URL_API_URL, json={"url": url, "question": context})
        answer = response.json().get("answer", "⚠️ No answer returned.")
        chat_sessions["url"].append((question, answer))
        return format_chat(chat_sessions["url"]), chat_sessions["url"]
    except Exception as e:
        return f"❌ Error: {str(e)}", chat_sessions["url"]

def extract_text_from_image(image_path):
    try:
        with open(image_path, "rb") as f:
            files = {"file": ("image.png", f, "image/png")}
            response = requests.post(IMAGE_API_URL, files=files)
            return response.json().get("answer", "⚠️ No text extracted.")
    except Exception as e:
        return f"❌ Error: {str(e)}"

def transcribe_audio(audio_path):
    try:
        with open(audio_path, "rb") as f:
            files = {"file": ("audio.wav", f, "audio/wav")}
            response = requests.post(AUDIO_API_URL, files=files)
            return response.json().get("answer", "⚠️ No transcript returned.")
    except Exception as e:
        return f"❌ Error: {str(e)}"

def format_chat(history):
    return "\n\n".join([f"👤 {q}\n🤖 {a}" for q, a in history])

def clear_file_chat():
    chat_sessions["file"] = []
    return "", chat_sessions["file"]

def clear_url_chat():
    chat_sessions["url"] = []
    return "", chat_sessions["url"]

custom_css = """
body, .gradio-container {
    background-color: #111111 !important;
    color: white !important;
}
footer { display: none !important; }
h1 {
    font-size: 2.2em;
    font-weight: bold;
    text-align: center;
    color: white;
    margin-bottom: 20px;
}
.gr-button {
    background-color: #00FF88;
    color: black;
    border: none;
    border-radius: 6px;
    font-weight: bold;
}
.gr-button:hover {
    background-color: #00cc70;
}
.gr-input, .gr-textbox, .gr-file, textarea, input {
    background-color: #1e1e1e !important;
    color: white !important;
    border: 1px solid #00FF88 !important;
    border-radius: 6px !important;
}
.gr-tabitem[data-selected="true"] > button {
    background-color: #00FF88 !important;
    color: black !important;
}
.gr-tabitem > button {
    background-color: transparent !important;
    color: white !important;
}
"""

with gr.Blocks(css=custom_css) as demo:
    gr.Markdown("# 🤖 AI Chatbot with File, Web, Image, Audio & Chat History")

    with gr.Tab("📂 Chat with File"):
        file_input = gr.File(label="Upload File (.txt, .csv, .docx, .pdf)", file_types=[".txt", ".csv", ".docx", ".pdf"])
        file_question = gr.Textbox(label="Your Question", placeholder="Ask something based on file")
        file_button = gr.Button("Ask from File")
        file_output = gr.Textbox(label="Chat History", lines=10)
        clear_file = gr.Button("New File Chat")

        file_button.click(fn=ask_from_file, inputs=[file_input, file_question, gr.State("file")], outputs=[file_output, gr.State("file")])
        clear_file.click(fn=clear_file_chat, inputs=[], outputs=[file_output, gr.State("file")])

    with gr.Tab("🌐 Chat with Website"):
        url_input = gr.Textbox(label="Website URL", placeholder="https://example.com")
        url_question = gr.Textbox(label="Your Question", placeholder="Ask something based on webpage")
        url_button = gr.Button("Ask from URL")
        url_output = gr.Textbox(label="Chat History", lines=10)
        clear_url = gr.Button("New URL Chat")

        url_button.click(fn=ask_from_url, inputs=[url_input, url_question, gr.State("url")], outputs=[url_output, gr.State("url")])
        clear_url.click(fn=clear_url_chat, inputs=[], outputs=[url_output, gr.State("url")])

    with gr.Tab("🖼 Extract Text from Image"):
        image_input = gr.Image(label="Upload Image", type="filepath")
        image_button = gr.Button("Extract Text")
        image_output = gr.Textbox(label="Extracted Text", lines=8)
        image_button.click(fn=extract_text_from_image, inputs=image_input, outputs=image_output)

    with gr.Tab("🎤 Transcribe Audio"):
        audio_input = gr.Audio(label="Upload Audio", type="filepath")
        audio_button = gr.Button("Transcribe Audio")
        audio_output = gr.Textbox(label="Transcript", lines=8)
        audio_button.click(fn=transcribe_audio, inputs=audio_input, outputs=audio_output)

demo.launch(share=True)
