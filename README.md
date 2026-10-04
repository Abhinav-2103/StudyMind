# StudyMind RAG

**StudyMind RAG** is a Retrieval-Augmented Generation (RAG) project that allows users to ask questions about their study notes and PDF documents.

The project is being built step-by-step to understand how a RAG system works internally, from PDF processing and embeddings to vector search and LLM-based answer generation.

## 🎯 Current Goal

StudyMind is designed around the idea of **"Ask My Notes"**:

> Upload or provide study material, ask a question, retrieve the most relevant information from the notes, and generate an answer based on that retrieved context.

---

## 🧠 How RAG Works in StudyMind

The current pipeline is:

```text
PDF
 ↓
Extract Text
 ↓
Chunk Text
 ↓
Generate Embeddings
 ↓
Store in ChromaDB
 ↓
User Question
 ↓
Generate Question Embedding
 ↓
Similarity Search
 ↓
Retrieve Relevant Chunks
 ↓
Build Context + Prompt
 ↓
Gemini
 ↓
Answer
```

The important idea is that the LLM is given relevant information retrieved from the user's notes instead of relying only on its general knowledge.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| pdfplumber | Extract text from PDF files |
| Sentence Transformers | Generate text embeddings |
| `all-MiniLM-L6-v2` | Embedding model |
| ChromaDB | Store embeddings and perform similarity search |
| Google Gemini API | Generate answers from retrieved context |
| `google-genai` | Python SDK for Gemini |
| python-dotenv | Load API keys from `.env` |
| Git & GitHub | Version control |

---

## 📁 Project Structure

```text
StudyMind(Rag)/
│
├── .gitignore
│
└── ask-my-notes/
    │
    ├── data/
    │   └── sample.pdf
    │
    ├── ingest.py
    ├── test.py
    ├── .env
    ├── chroma_db/
    ├── venv/
    └── __pycache__/
```

### Important

The following files/folders are intentionally excluded from GitHub:

```text
.env
venv/
chroma_db/
__pycache__/
```

The `.env` file contains the Gemini API key and should never be uploaded to GitHub.

---

# 📚 Development Progress

## Day 1 — PDF Processing & Embeddings ✅

### 1. PDF Text Extraction

Used `pdfplumber` to extract text from PDF files.

Each extracted page is stored with metadata:

```python
{
    "page": 1,
    "text": "...",
    "source": "sample.pdf"
}
```

This metadata will later help identify where retrieved information came from.

### 2. Text Chunking

Large PDF text is divided into smaller chunks.

Current configuration:

```python
chunk_size = 300
```

The text is split into words and grouped into chunks of approximately 300 words.

Example:

```text
PDF text
   ↓
Page
   ↓
Words
   ↓
300-word chunks
```

Each chunk keeps its:

- text
- source
- page number

### 3. Embeddings

Used:

```python
SentenceTransformer("all-MiniLM-L6-v2")
```

Each text chunk is converted into a numerical vector.

Current embedding size:

```text
384 dimensions
```

This allows the system to compare the semantic meaning of text mathematically.

---

# 🗄️ Day 2 — ChromaDB & RAG Pipeline ✅

## 1. Persistent ChromaDB

ChromaDB was added as the vector database.

```python
chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)
```

A collection called:

```text
study_notes
```

is used to store the PDF chunks and their embeddings.

## 2. Store Embedded Chunks

Each chunk is stored with:

- ID
- original text
- embedding
- source
- page number

Example structure:

```text
chunk_0
 ├── document
 ├── embedding
 └── metadata
      ├── source
      └── page
```

## 3. Question Embedding

When the user asks a question, the question is also converted into a 384-dimensional embedding.

Example:

```python
question_embedding = model.encode(question)
```

The system does **not** compare the question with chunks using exact text matching.

Instead, it compares their **semantic meaning represented as vectors**.

## 4. Similarity Search

ChromaDB searches for the most relevant chunks.

Example:

```python
results = collection.query(
    query_embeddings=[question_embedding.tolist()],
    n_results=2
)
```

The retrieved chunks become the context for the LLM.

## 5. Context Creation

Multiple retrieved chunks are combined:

```python
retrieved_text = "\n".join(
    results["documents"][0]
)
```

This creates the context that will be passed to Gemini.

## 6. RAG Prompt

The retrieved context and question are placed into a prompt:

```text
Context:
<retrieved notes>

Question:
<user question>

If the answer is not present in the context,
say:
"I couldn't find the answer in the provided notes."

Answer:
```

## 7. Gemini Integration

Gemini is used to generate the final answer.

Current model:

```text
gemini-3.8-flash
```

The Gemini API key is loaded securely from `.env`.

The system successfully generated an answer from the retrieved PDF context.

### Example

Question:

```text
What is the assignment submission date?
```

Retrieved PDF information:

```text
Assignment Submission: 6/09/2026
```

Generated answer:

```text
The assignment submission date is 6/09/2026.
```

---

# 🔐 Environment Variables

Create a `.env` file inside:

```text
ask-my-notes/
```

Example:

```env
GEMINI_API_KEY=your_api_key_here
```

The API key is loaded using:

```python
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
```

**Never commit the `.env` file to GitHub.**

---

# ▶️ Running the Project

## 1. Activate Virtual Environment

From the `ask-my-notes` directory:

```powershell
venv\Scripts\activate
```

You should see:

```text
(venv) PS E:\StudyMind(Rag)\ask-my-notes>
```

## 2. Run the RAG Pipeline

```powershell
python ingest.py
```

The current pipeline:

1. Extracts the PDF
2. Creates chunks
3. Generates embeddings
4. Stores them in ChromaDB
5. Embeds the question
6. Performs similarity search
7. Builds the context
8. Sends the context to Gemini
9. Prints the final answer

---

# 🧪 Current Test Result

The current sample PDF contains a Computer Networks assignment.

The system successfully retrieved the relevant information and answered:

```text
The assignment submission date is 6/09/2026.
```

Current successful pipeline statistics:

```text
Total pages extracted: 1
Total chunks created: 2
Embedding size: 384
ChromaDB records: 2
```

---

# 🗺️ Learning Roadmap

The project is being developed in phases.

### Week 1 — Foundations

- Python file handling
- PDF text extraction
- Text chunking
- Understanding embeddings
- Sentence Transformers

### Week 2 — Vector Database & Retrieval

- ChromaDB
- Vector storage
- Similarity search
- Retrieval
- Metadata
- Improving ingestion

### Week 3 — LLM & RAG

- Prompt engineering
- Gemini API
- RAG pipeline
- User questions
- Dynamic retrieval
- Context handling

### Week 4 — Final System

- Evaluation
- Citations
- Streamlit UI
- Better project structure
- GitHub README
- Final documentation

---

# 🚧 Current Limitations

The current version is a learning prototype.

Some improvements are still planned:

- Separate PDF ingestion and question-answering code
- Allow users to enter questions dynamically
- Handle multiple PDF files
- Improve chunking strategy
- Make database ingestion safe for repeated runs
- Retrieve and display citations for all relevant chunks
- Add RAG evaluation
- Build a Streamlit interface
- Improve error handling
- Add a proper README with final architecture and usage instructions

---

# 📌 Project Status

**Current status: Working RAG prototype**

```text
PDF Processing          ✅
Text Chunking           ✅
Embeddings              ✅
ChromaDB                ✅
Similarity Search       ✅
Context Retrieval       ✅
Prompt Construction     ✅
Gemini Integration      ✅
End-to-End RAG          ✅
Dynamic Questions       ⏳
Multiple PDFs           ⏳
Citations               ⏳
Evaluation              ⏳
Streamlit UI            ⏳
```

---

## 👨‍💻 Author

**Abhinav Thakur**

B.Tech — Information Technology

This project is being developed as a hands-on learning project to understand Retrieval-Augmented Generation and modern AI application development.
