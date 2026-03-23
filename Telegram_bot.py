from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image


TOKEN = "8634865972:AAFyoJZpVnainsYa5vjs9iFzEXHAVgz4q5g"

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("""
Commands:
/ask <query> - Ask questions
/image - Upload image for caption
/help - Show this message
""")

async def ask(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = " ".join(context.args)
    response = rag_pipeline(query)
    await update.message.reply_text(response)

async def image_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    photo = update.message.photo[-1]
    file = await photo.get_file()
    path = "image.jpg"
    await file.download_to_drive(path)

    caption, tags = vision_pipeline(path)
    await update.message.reply_text(f"{caption}\nTags: {', '.join(tags)}")

loader = PyPDFLoader(r"D:\RAG Pride and Prejudce\Corrected Book.pdf")
docs = loader.load()

texts = [doc.page_content for doc in docs]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(texts)

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))


def retrieve(query, k=2):
    q_emb = embedding_model.encode([query])
    distances, indices = index.search(np.array(q_emb), k)
    return [docs[i] for i in indices[0]]

def generate_answer(context, query):
    prompt = f"""
Answer the question using context:
{context}

Question: {query}
"""
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": "mistral", "prompt": prompt, "stream": False }
    )
    data = response.json()
    print("DEBUG:", data)  # 👈 VERY IMPORTANT for debugging
    return data.get("response", "⚠️ No response from model")    

def rag_pipeline(query):
    chunks = retrieve(query)
    context = "\n".join([chunk.page_content for chunk in chunks])
    return generate_answer(context, query)

app = ApplicationBuilder().token(TOKEN).build()

app.add_handler(CommandHandler("help", help_cmd))
app.add_handler(CommandHandler("ask", ask))
app.add_handler(MessageHandler(filters.PHOTO, image_handler))
print("✅ Bot is running...")

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
vision_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def vision_pipeline(image_path):
    image = Image.open(image_path)

    inputs = processor(image, return_tensors="pt")
    output = vision_model.generate(**inputs)

    caption = processor.decode(output[0], skip_special_tokens=True)

    tags = caption.split()[:3]  # simple tagging
    return caption, tags

app.run_polling()
