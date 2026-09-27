import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer

print("pdfplumber ✅")
print("chromadb ✅")

model = SentenceTransformer('all-MiniLM-L6-v2')
test = model.encode("hello world")
print(f"sentence-transformers ✅ — vector size: {len(test)}")