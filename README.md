# Pride-and-Prejudice-RAG-Image-Captioning-Telegram-Bot

A lightweight GenAI Telegram bot that combines:

📖 Mini-RAG (Retrieval-Augmented Generation) for answering questions from a PDF
🖼️ Image Captioning using a vision model

**🚀 Features:**

/ask <query> → Ask questions from Pride and Prejudice (PDF-based RAG)
📷 Upload image → Get caption + tags
/help → Usage instructions

**🧠 System Overview:**

🔹 1. RAG Pipeline
Load PDF using PyPDFLoader
Split into chunks (page-level)
Generate embeddings using SentenceTransformer
Store embeddings in FAISS index
Retrieve top-k relevant chunks
Pass context + query to LLM (Mistral via Ollama)
Return generated answer
🔹 2. Vision Pipeline
User uploads image
Image processed using BLIP model
Generate:
Caption
Tags (first 3 keywords)
Send response back to user

**🏗️ Architecture (Simple Flow):**

User → Telegram Bot → Router
                     ├── /ask → RAG Pipeline → Ollama (Mistral) → Response
                     └── Image → BLIP Model → Caption + Tags

**⚙️ Setup Instructions:**

🔹 1. Clone Repository
git clone <your-repo-url>
cd <repo-folder>

🔹 2. Install Dependencies
pip install python-telegram-bot langchain sentence-transformers faiss-cpu transformers pillow requests

🔹 3. Install & Run Ollama

Download Ollama from: https://ollama.com

Run:
ollama run mistral

Make sure it runs at:

http://localhost:11434

🔹 4. Add Your Telegram Bot Token
Replace in code:
TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"

🔹 5. Add Your PDF
Update path:
loader = PyPDFLoader("path_to_your_pdf")

🔹 6. Run the Bot
python app.py

You should see:
✅ Bot is running...

**📸 Usage:**

🔹 Ask Questions (RAG)
/ask Who is Elizabeth Bennet?

🔹 Image Captioning
Send any image to the bot
Receive:
Caption
Tags

🔹 Help Command
/help
