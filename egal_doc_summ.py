import os
from dotenv import load_dotenv
import fitz
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import gradio as gr
from PyPDF2 import PdfReader
from fpdf import FPDF
import speech_recognition as sr
import google.generativeai as genai
import asyncio

# Load environment variables
load_dotenv()

# CONFIG
pdf_folder = "D:/MSC_PROJECTS/Legal_CP/dataset"
faiss_db_path = "./faiss_index"
chunk_size = 500
chunk_overlap = 50
supported_languages = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Marathi": "mr",
    "Punjabi": "pa",
    "Urdu": "ur"
}

# Initialize recognizer
recognizer = sr.Recognizer()

# Gemini config
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel('gemini-2.0-flash')

# TEXT PROCESSING
def extract_text_from_uploaded_pdfs(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip() or None

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [Document(page_content=chunk) for chunk in splitter.split_text(text)]

# VECTOR STORES
def get_main_vector_store():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists(faiss_db_path):
        return FAISS.load_local(faiss_db_path, embedding_model, allow_dangerous_deserialization=True)
    return None

def create_temp_vector_store(text_chunks):
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(text_chunks, embedding_model)

# TRANSLATION USING GEMINI
async def translate_text_async(text, lang):
    if not text or lang == "English":
        return text
    try:
        prompt = f"Translate this to {lang}:\n{text}"
        response = await gemini_model.generate_content_async(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Translation error: {str(e)}"

# PDF CREATION
def create_pdf(text, filename="output.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, text)
    pdf.output(filename)
    return filename

# QUERY HANDLING
async def generate_with_gemini(prompt):
    try:
        response = await gemini_model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

async def process_query_async(query, lang="English", context=None):
    if not query.strip():
        return "❌ Please enter a legal query.", ""

    prompt = f"""
You are a legal expert assistant. Provide accurate, concise legal information.
Query: {query}
Context: {context if context else "Use general legal knowledge."}
Answer professionally with citations if possible. If unsure, say so.
"""
    response = await generate_with_gemini(prompt)
    translated = await translate_text_async(response, lang)
    return response, translated

# VOICE
def transcribe_voice(audio_state=None):
    if not audio_state:
        return ""
    try:
        with sr.AudioFile(audio_state) as source:
            audio = recognizer.record(source)
            return recognizer.recognize_google(audio)
    except Exception as e:
        return f"Voice recognition error: {str(e)}"

# VECTOR INIT
main_vectorstore = get_main_vector_store()

# GRADIO INTERFACE
custom_css = """
h1 { color: #7e22ce !important; }
.primary { background: #7e22ce !important; border-color: #7e22ce !important; }
.primary:hover { background: #6b21a8 !important; border-color: #6b21a8 !important; }
"""

async def handle_general_query_async(query, voice_input, lang):
    if voice_input:
        query = transcribe_voice(voice_input)
    if not query.strip():
        return "❌ Please enter a legal query.", "", None

    context = None
    if main_vectorstore:
        context = "\n\n".join([doc.page_content for doc in main_vectorstore.similarity_search(query, k=3)])

    tasks = await asyncio.gather(
        process_query_async(query, lang, context)
    )
    response, translated = tasks[0]
    pdf_file = create_pdf(response, "general_response.pdf")
    return response, translated, pdf_file

async def handle_case_query_async(case_files, query_text, voice_input, lang):
    query = transcribe_voice(voice_input) if voice_input else query_text
    if not query.strip():
        return "❌ Please enter a query about the case.", "", None

    context = None
    if case_files:
        text = extract_text_from_uploaded_pdfs(case_files)
        if text:
            chunks = split_text(text)
            temp_store = create_temp_vector_store(chunks)
            context = "\n\n".join([doc.page_content for doc in temp_store.similarity_search(query, k=3)])

    tasks = await asyncio.gather(
        process_query_async(query, lang, context)
    )
    response, translated = tasks[0]
    pdf_file = create_pdf(response, "case_response.pdf")
    return response, translated, pdf_file

with gr.Blocks(title="Legal Document Assistant", theme=gr.themes.Soft(), css=custom_css) as app:
    gr.Markdown("""
    # 📚 Legal Document Assistant
    <div style="color: #7e22ce">Ask general legal questions or analyze specific case files</div>
    """)

    with gr.Tab("📄 Case File Analysis"):
        with gr.Row():
            with gr.Column(scale=1):
                case_files = gr.File(label="Upload Case Files", file_count="multiple", file_types=[".pdf"])
                case_query = gr.Textbox(label="Text Query", placeholder="Ask about this specific case...")
                case_voice = gr.Audio(label="Or Voice Query", sources=["microphone"], type="filepath")
                target_lang = gr.Dropdown(label="Output Language", choices=list(supported_languages.keys()), value="English")
                case_submit = gr.Button("Analyze Case", variant="primary")
            with gr.Column(scale=1):
                case_output = gr.Textbox(label="English Response", interactive=False)
                case_translated = gr.Textbox(label="Translated Response", interactive=False)
                case_pdf = gr.File(label="Download Analysis", interactive=False)

    with gr.Tab("❓ General Legal Questions"):
        with gr.Row():
            with gr.Column(scale=1):
                general_query = gr.Textbox(label="Text Query", placeholder="Type your general legal question...")
                general_voice = gr.Audio(label="Or Voice Query", sources=["microphone"], type="filepath")
                general_lang = gr.Dropdown(label="Output Language", choices=list(supported_languages.keys()), value="English")
                general_submit = gr.Button("Get Answer", variant="primary")
            with gr.Column(scale=1):
                general_output = gr.Textbox(label="English Response", interactive=False)
                general_translated = gr.Textbox(label="Translated Response", interactive=False)
                general_pdf = gr.File(label="Download PDF", interactive=False)

    general_submit.click(fn=handle_general_query_async, inputs=[general_query, general_voice, general_lang], outputs=[general_output, general_translated, general_pdf])
    case_submit.click(fn=handle_case_query_async, inputs=[case_files, case_query, case_voice, target_lang], outputs=[case_output, case_translated, case_pdf])

if __name__ == "__main__":
    app.launch()
