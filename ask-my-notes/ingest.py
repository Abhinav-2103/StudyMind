import pdfplumber #used to extract text from PDF files
import os #gives Python tools for interacting with the operating system.

from sentence_transformers import SentenceTransformer # used to convert text into numerical vectors (embeddings) that capture the semantic meaning of the text.

def extract_text_from_pdf(pdf_path):
    pages = []
    with pdfplumber.open(pdf_path) as pdf: #with statement is used to open the PDF file and ensure it is properly closed after processing.
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages.append({       #We're storing text + information about where it came from. 
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
            chunk = " ".join(words[i:i+chunk_size])
            chunks.append({
                "text": chunk,
                "source": page["source"],
                "page": page["page"]
            })
    return chunks

# Test it
model = SentenceTransformer("all-MiniLM-L6-v2") # This line initializes a pre-trained model from the Sentence Transformers library, which is used to convert text into embeddings. The model "all-MiniLM-L6-v2" is a lightweight model that provides good performance for generating embeddings.

if __name__ == "__main__":
    pdf_path = "data/sample.pdf"  # drop any PDF in your data/ folder
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(pages)
    
    print(f"Total pages extracted: {len(pages)}")
    print(f"Total chunks created: {len(chunks)}")

    print(f"\nSample chunk:\n{chunks[0]['text'][:200]}")
    print(f"Source: {chunks[0]['source']}, Page: {chunks[0]['page']}")

    embedding = model.encode(chunks[0]["text"])
    
    print(f"Embedding size: {len(embedding)}")
    print(f"First 5 values: {embedding[:5]}")