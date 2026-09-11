# 🧠 RAG-Based Knowledge Retrieval System

A Retrieval-Augmented Generation (RAG) tool that lets you feed in one or more web URLs, builds a searchable knowledge base from their content, and answers natural-language questions using that content — complete with cited sources.

Built with **LangChain**, **ChromaDB**, **HuggingFace Embeddings**, and **Groq (LLaMA/OSS models)**.

---

## 📌 Overview

Traditional LLMs only know what they were trained on. This project solves that limitation by combining:

- **Retrieval** — pulling relevant, up-to-date content from user-provided URLs
- **Augmentation** — injecting that content as context into the LLM
- **Generation** — producing accurate, source-backed answers

The result is a lightweight "chat with any webpage(s)" assistant that avoids hallucination by grounding every answer in real, retrievable text.

---

## ⚙️ Project Workflow

1. **User submits URLs** through the UI
2. Pages are scraped and loaded (`WebBaseLoader`)
3. Content is split into overlapping chunks (`RecursiveCharacterTextSplitter`)
4. Chunks are embedded into vectors (`HuggingFaceEmbeddings`)
5. Vectors are stored/persisted in a **Chroma** vector database
6. User asks a question
7. The most relevant chunks are retrieved and passed to the LLM
8. The LLM (via **Groq**) generates an answer **with source citations**

```
URLs → Loader → Text Splitter → Embeddings → Vector Store (Chroma)
                                                     │
User Question ──────────────────────────────────────┘
                                                     │
                                        Retriever → LLM → Answer + Sources
```

---

## ✨ Key Features

- 🔗 Process multiple URLs at once
- ✂️ Automatic chunking with configurable size/overlap
- 🧬 Semantic search using vector embeddings
- 💾 Persistent vector storage (no need to reprocess URLs every run)
- 🤖 Fast, low-cost inference using Groq-hosted open models
- 📚 Every answer includes its original source links for verification
- 🖥️ Simple web UI for entering URLs and asking questions

---

## 🛠️ Technologies Used

| Component            | Tool / Library                                      |
|-----------------------|------------------------------------------------------|
| Orchestration         | LangChain (`langchain-classic`, `langchain-community`) |
| LLM Inference          | Groq API (`langchain-groq`, `openai/gpt-oss-20b`)    |
| Embeddings             | HuggingFace (`sentence-transformers/all-MiniLM-L6-v2`) |
| Vector Database        | Chroma (`langchain-chroma`)                          |
| Text Splitting         | `RecursiveCharacterTextSplitter`                     |
| Web Scraping           | `WebBaseLoader`                                      |
| Environment Config     | `python-dotenv`                                      |

---

## 🔍 How It Works

### 1. Processing URLs
```python
def process_urls(urls):
    initialize_components()
    loader = WebBaseLoader(urls)
    data = loader.load()
    textsplitter = RecursiveCharacterTextSplitter(
        chunk_size=ch_size, chunk_overlap=ch_overlap
    )
    docs = textsplitter.split_documents(data)
    vector_store.add_documents(docs)
```

### 2. Generating Answers
```python
def generate_answer(query):
    chain = RetrievalQAWithSourcesChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever()
    )
    result = chain.invoke({"question": query}, return_only_output=True)
    return result['answer'], result.get("sources", "")
```

The `RetrievalQAWithSourcesChain` automatically retrieves the most relevant chunks from the vector store and prompts the LLM to answer **using only that retrieved context**, then reports which source documents were used.

---

## 📁 Project Structure

```
rag-url-assistant/
│
├── rag_backend.py          # Core RAG logic (loading, embedding, retrieval, QA chain)
├── app.py                  # Frontend/UI entry point (Streamlit or similar)
├── resources/
│   └── vectorstore/        # Persisted Chroma vector database
├── screenshots/
│   ├── url-input.png
│   ├── question.png
│   └── answer-sources.png
├── .env                     # API keys (not committed)
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/rag-url-assistant.git
cd rag-url-assistant

# Create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Key dependencies** (add to `requirements.txt` if not already present):
```
langchain-community
langchain-text-splitters
langchain-chroma
langchain-groq
langchain-huggingface
langchain-classic
python-dotenv
```

---

## 🔑 Environment Setup

Create a `.env` file in the project root with your Groq API key:

```
GROQ_API_KEY=your_groq_api_key_here
```

> Get a free API key at [console.groq.com](https://console.groq.com)

---

## ▶️ Usage

1. **Start the app** (adjust based on your UI framework, e.g. Streamlit):
   ```bash
   streamlit run app.py
   ```

2. **Enter one or more URLs** you want the assistant to learn from, then click **Process URLs**.

3. **Ask a question** related to the content of those URLs and click **Ask Question**.

4. **Read the answer** along with the exact sources it was generated from.

---

## 💬 Example

**URLs processed:**
- `https://en.wikipedia.org/wiki/GPT-6_Astra`

**Question:**
> "What is AI Astra?"

**Answer:**
> AI Astra is OpenAI's next-generation large language model designed to act as an autonomous "agent" that can interact directly with software and websites — reading screens, executing tasks, and completing multi-step workflows more efficiently than prior models.

**Sources:**
- Medium article on GPT-6 Astra
- Blockchain Council article

---

## 🔮 Future Improvements

- [ ] Support PDF and text file uploads in addition to URLs
- [ ] Add conversation memory for multi-turn follow-up questions
- [ ] Allow model selection (swap between Groq models)
- [ ] Add streaming responses in the UI
- [ ] Deploy as a hosted web app (Streamlit Cloud / HuggingFace Spaces)
- [ ] Add unit tests for the ingestion and retrieval pipeline

---

## 👤 Author

Built as a demonstration of practical Retrieval-Augmented Generation architecture — combining web scraping, vector search, and LLM inference into a working end-to-end application.

If you found this project interesting, feel free to ⭐ the repo or connect!
