import gradio as gr
from sentence_transformers import SentenceTransformer
import faiss
import fitz  
from openai import OpenAI
import numpy as np
import os
from dotenv import load_dotenv


load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

embed_model = SentenceTransformer("all-MiniLM-L6-v2")


def load_pdf(file_path):
    doc = fitz.open(file_path)
    texts = []
    for page in doc:
        texts.append(page.get_text())
    return texts

def chunk_text(texts, chunk_size=500, overlap=50):
    chunks = []
    for page_text in texts:
        words = page_text.split()
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i+chunk_size])
            if chunk.strip():
                chunks.append(chunk)
    return chunks

def build_faiss_index(chunks):
    embeddings = embed_model.encode(chunks, convert_to_numpy=True, show_progress_bar=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embeddings

def retrieve_chunks(query, index, embeddings, chunks, top_k=3):
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    D, I = index.search(query_vec, top_k)
    return [chunks[i] for i in I[0]]




def answer_with_openai(query, context_chunks):
    context_text = "\n\n".join(context_chunks)
    prompt = f"""Answer the question based on the following context:

{context_text}

Question: {query}
Answer:"""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content




state = {"index": None, "embeddings": None, "chunks": None}

def upload_pdf(file):
    texts = load_pdf(file.name)
    chunks = chunk_text(texts)
    index, embeddings = build_faiss_index(chunks)
    state["index"] = index
    state["embeddings"] = embeddings
    state["chunks"] = chunks
    return f"✅ PDF uploaded & processed! Ready to chat."

def ask_question(audio, text, history):
    if audio is not None:
        with open(audio, "rb") as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
            query = transcription.text
    elif text:
        query = text
    else:
        return history + [[None, "⚠️ Please provide a question either via text or speech."]]

    if state["index"] is None:
        return history + [[query, "⚠️ Please upload a PDF first."]]

    retrieved_chunks = retrieve_chunks(query, state["index"], state["embeddings"], state["chunks"])
    answer = answer_with_openai(query, retrieved_chunks)

    history = history + [[query, answer]]
    return history




customCSS = """
body {
    background: linear-gradient(135deg, #6366f1, #3b82f6);
    font-family: 'Segoe UI', sans-serif;
}

#chat-card {
    max-width: 800px; 
    margin: auto;
    background: white; 
    border-radius: 20px;
    padding: 20px; 
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}

#title {
    text-align:center; 
    padding:10px 0;
}
#title h1 {
    font-size:2em; 
    color:#4338ca; 
    margin:0;
}
#title p {
    font-size:1em; 
    color:#555;
}

/* Chat styling */
.chatbot {
    height: 450px !important; 
    border-radius: 16px;
}
.message.user {
    background: #3b82f6 !important; 
    color: white !important;
    border-radius: 16px 16px 0 16px !important; 
    align-self: flex-end !important;
}
.message.bot {
    background: #f3f4f6 !important; 
    color: black !important;
    border-radius: 16px 16px 16px 0 !important; 
    align-self: flex-start !important;
}

/* Upload box (only "Click to Upload") */
#upload-box { 
    height:70px !important;   /* increased a little */
    border: 2px dashed #6366f1; 
    text-align:center; 
    position: relative;
}

#upload-box .label,
#upload-box .file-drop-label {
    display: none !important;  /* hide default texts */
}
#upload-box::before {
    content: "Drop File Here"; 
    display: block; 
    font-size: 14px; 
    color: #4338ca; 
    padding: 18px 0;
}


/* Input section */
#input-section {margin-top: 15px;}
#send-btn {
    background: #3b82f6; 
    color: white; 
    border-radius: 10px; 
    padding: 6px 16px;
}
#send-btn:hover {
    background: #4338ca;
}
"""


with gr.Blocks(css=customCSS) as demo:
    with gr.Column(elem_id="chat-card"):
        gr.HTML(
            """<div id="title">
                <h1>📘 PDF Q&A Assistant</h1>
                <p>Upload a PDF, then ask questions by typing or speaking 🎤</p>
            </div>"""
        )

        
        pdf_input = gr.File(file_types=['.pdf'], elem_id="upload-box", label="")
        upload_btn = gr.Button("🚀 Process PDF")
        upload_output = gr.Textbox(label="Status", interactive=False)

        chatbot = gr.Chatbot(label="Conversation", bubble_full_width=False)

        
        with gr.Column(elem_id="input-section"):
            text_input = gr.Textbox(
                placeholder="Type your question here...",
                lines=1,
                label="✍️ Text Query"
            )
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="🎤 Voice Query"
            )
            send_btn = gr.Button("💡 Send", elem_id="send-btn")

        state_history = gr.State([])

        
        upload_btn.click(upload_pdf, inputs=pdf_input, outputs=upload_output)

        
        send_btn.click(
            ask_question,
            inputs=[audio_input, text_input, state_history],
            outputs=[chatbot],
        ).then(
            lambda h: h,
            inputs=[chatbot],
            outputs=[state_history],
        )

demo.launch(share=True)
