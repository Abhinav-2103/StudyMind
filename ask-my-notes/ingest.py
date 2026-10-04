import pdfplumber  # used to extract text from PDF files
import os  # gives Python tools for interacting with the operating system.
import chromadb  # used to store and manage embeddings
from dotenv import load_dotenv  # used to load environment variables from .env
from sentence_transformers import SentenceTransformer  # used to create embeddings
from google import genai  # used to interact with Google's Generative AI API


# Load environment variables from .env
load_dotenv()

# Get Gemini API key from .env
api_key = os.getenv("GEMINI_API_KEY")

# Create Gemini client
gemini_client = genai.Client(api_key=api_key)


def extract_text_from_pdf(pdf_path):
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        # with statement opens the PDF and ensures it is properly closed
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()

            if text:
                pages.append({
                    # Store text + information about where it came from
                    "page": i + 1,
                    "text": text,
                    "source": os.path.basename(pdf_path)
                })

    return pages


def chunk_text(pages, chunk_size=300):
    chunks = []

    for page in pages:
        words = page["text"].split()

        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])

            chunks.append({
                "text": chunk,
                "source": page["source"],
                "page": page["page"]
            })

    return chunks


# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Create persistent ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Get or create the collection
collection = chroma_client.get_or_create_collection(name="study_notes")

# Store embedded chunks
embedded_chunks = []


if __name__ == "__main__":

    # PDF file
    pdf_path = "data/sample.pdf"

    # Extract text from PDF
    pages = extract_text_from_pdf(pdf_path)

    # Create chunks
    chunks = chunk_text(pages)

    print(f"Total pages extracted: {len(pages)}")
    print(f"Total chunks created: {len(chunks)}")

    # Display a sample chunk
    print(f"\nSample chunk:\n{chunks[0]['text'][:200]}")
    print(f"Source: {chunks[0]['source']}, Page: {chunks[0]['page']}")

    # Create embeddings and store them in ChromaDB
    for i, chunk in enumerate(chunks):

        chunk_id = f"chunk_{i}"

        # Convert chunk text into embedding
        embedding = model.encode(chunk["text"])

        # Store embedding information in Python list
        embedded_chunks.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "page": chunk["page"],
            "embedding": embedding
        })

        # Store chunk in ChromaDB
        collection.add(
            ids=[chunk_id],
            documents=[chunk["text"]],
            embeddings=[embedding.tolist()],
            metadatas=[{
                "source": chunk["source"],
                "page": chunk["page"]
            }]
        )

    # Display embedding information
    print(f"Embedded chunks: {len(embedded_chunks)}")
    print(f"First embedding size: {len(embedded_chunks[0]['embedding'])}")
    print(f"First 5 values: {embedded_chunks[0]['embedding'][:5]}")
    print(f"ChromaDB records: {collection.count()}")

    # -----------------------------
    # USER QUESTION
    # -----------------------------

    question = "What is the assignment submission date?"

    # Convert question into embedding
    question_embedding = model.encode(question)

    # Search ChromaDB for relevant chunks
    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=2
    )

    # Combine retrieved chunks into one context
    retrieved_text = "\n".join(results["documents"][0])

    # -----------------------------
    # CREATE PROMPT
    # -----------------------------

    prompt = f"""
Use the following context to answer the question.

Context:
{retrieved_text}

Question:
{question}

If the answer is not present in the context, say:
"I couldn't find the answer in the provided notes."

Answer:
"""

    # Display the prompt
    print("\nPrompt:")
    print(prompt)

    # Display retrieved context
    print("\nCombined Context:")
    print(retrieved_text)

    # Display source information
    print("\nSource:", results["metadatas"][0][0]["source"])
    print("Page:", results["metadatas"][0][0]["page"])

    # -----------------------------
    # GEMINI GENERATION
    # -----------------------------

    response = gemini_client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )

    # Display final answer
    print("\nAnswer:")
    print(response.text)