# RagChatAPP 🧠🤖

RagChatAPP is a modular, high-performance Retrieval-Augmented Generation (RAG) chatbot application designed to answer questions based on your custom local documents. It features dense semantic vector retrieval paired with a responsive and interactive Streamlit UI.

---

## 🚀 Key Features

* **Multi-Format Document Ingest**: Automatically processes, cleans, and chunks **TXT**, **PDF** (powered by fast PyMuPDF extraction), and Word **DOCX** files placed in your data directory.
* **Vector Semantic Search**: Utilizes Hugging Face's `all-MiniLM-L6-v2` transformer model to produce 384-dimensional dense embeddings and indexes them using **FAISS (Facebook AI Similarity Search)**.
* **Modern Gemini Client**: Harnesses the brand new **`google-genai`** SDK to stream and generate precise answers using the `gemini-2.5-flash` model.
* **Interactive UI**:
  * Persistent chat bubbles styled with clean borders and modern colors.
  * Sidebar configuration to input and update your Gemini API key dynamically.
  * Clear chat logs button.
  * **Source Citation**: Displays exactly which documents were referenced to formulate the LLM's answer.

---

## 📂 Restructured Folder Layout

```text
RAGChatApp/
├── app/
│   └── ui/
│       └── streamlit_app.py   # Refactored interactive web frontend
├── data/
│   ├── CA_conservation.txt    # Sample TXT document
│   ├── startup.txt            # Sample TXT document
│   └── [Your PDFs/DOCXs]      # Place your custom files here (e.g. PDFs)
├── db/
│   └── faiss/
│       ├── index.bin          # Saved FAISS index database
│       └── metadata.pkl       # Saved document chunks and source mappings
├── src/
│   ├── ingestion/
│   │   ├── parser.py          # Document parsers (PyMuPDF, python-docx, txt)
│   │   └── chunker.py         # Text chunking and cleaning helpers
│   ├── search/
│   │   ├── embedder.py        # SentenceTransformer embedding encoder wrapper
│   │   └── vector_store.py    # FAISS read, write, and retrieval engine
│   ├── llm/
│   │   └── gemini_client.py   # Modern google-genai client wrapper
│   └── build_index.py         # Main script to trigger data indexing pipeline
├── requirement.txt            # Python dependencies
└── README.md                  # This file
```

---

## 🛠️ Installation & Setup

### 1. Install Dependencies
Run the command below in the project root directory:
```bash
pip install -r requirement.txt
pip install google-genai pymupdf
```

### 2. Configure Your Gemini API Key
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_gemini_api_key
```
*Note: You can also enter the API key directly in the sidebar of the Streamlit application.*

---

## 💻 How to Run

### Step 1: Ingest Documents & Build Vector Index
Place all text, PDF, and DOCX documents you wish to query inside the `data/` directory. Then compile the vector index by running:
```bash
python src/build_index.py
```
This parses all documents, builds the FAISS vector database, and saves it inside `db/faiss/`.

### Step 2: Start the Chatbot
To start the interactive chat UI, run:
```bash
streamlit run app/ui/streamlit_app.py
```
## Result

![Bhagwad Geeta Response](C:\Users\dell\Desktop\RL_practice\RAGChatApp\result\result1.png)

Streamlit will automatically open a tab in your web browser at `http://localhost:8501`.

---

## ⚙️ Technical Stack
* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 Dimensions).
* **Vector Store**: `FAISS` (IndexFlatIP using Cosine Similarity).
* **LLM**: Google Gemini `gemini-2.5-flash` via the official `google-genai` package.
* **PDF Parser**: `PyMuPDF (fitz)`.
* **Word Parser**: `python-docx`.
