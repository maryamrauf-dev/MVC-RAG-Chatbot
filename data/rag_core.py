import os
import re
import json
import time
import numpy as np
import google.generativeai as genai
import fitz  # PyMuPDF
from typing import List, Dict, Optional, Any
from PIL import Image
import io
from dotenv import load_dotenv

# Load Env
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'RAG system', '.env.local'))
load_dotenv()

# Constants
VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vector_store")
DB_FILE = os.path.join(VECTOR_STORE_DIR, "embeddings_cache.json")
PDF_PATH = os.path.join(os.path.dirname(__file__), "..", "public", "calculus.pdf")

def get_api_key():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API Key not found in .env.local")
    return api_key

def configure_api():
    # Use client_options to explicitly set the stable v1 API version
    from google.api_core import client_options
    options = client_options.ClientOptions(api_version='v1')
    genai.configure(api_key=get_api_key(), transport='rest', client_options=options)

# Configure once at module level
try:
    configure_api()
except Exception as e:
    print(f"Initial API configuration failed: {e}")

def get_embedding(text: str) -> List[float]:
    """Gemini API method for embeddings"""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        return result['embedding']
    except Exception as e:
        print(f"Error embedding: {e}")
        return []

def get_query_embedding(query: str) -> List[float]:
    """Gemini API method for query embeddings"""
    try:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )
        return result['embedding']
    except Exception:
        return []

class SimpleLocalVectorStore:
    """Persistent vector store implementation for Vision-based RAG"""
    def __init__(self):
        self.data = []
        if not os.path.exists(VECTOR_STORE_DIR):
            os.makedirs(VECTOR_STORE_DIR)
        self.load()

    def add_item(self, text: str, page_number: int, embedding: List[float]):
        self.data.append({
            "text": text,
            "page": page_number,
            "embedding": embedding
        })

    def save(self):
        with open(DB_FILE, "w", encoding="utf-8") as f:
            json.dump(self.data, f)

    def load(self):
        if os.path.exists(DB_FILE):
            try:
                with open(DB_FILE, "r", encoding="utf-8") as f:
                    self.data = json.load(f)
            except:
                self.data = []

    def search(self, query: str, k: int = 3) -> List[Dict]:
        q_vec = np.array(get_query_embedding(query))
        if q_vec.size == 0 or not self.data:
            return []
            
        scores = []
        for item in self.data:
            d_vec = np.array(item["embedding"])
            # Dot product similarity
            score = np.dot(q_vec, d_vec)
            scores.append((score, item))
        
        scores.sort(key=lambda x: x[0], reverse=True)
        return [score_item[1] for score_item in scores[:k]]

_PDF_DOC = None

def get_pdf_doc():
    global _PDF_DOC
    if _PDF_DOC is None:
        if not os.path.exists(PDF_PATH):
            return None
        _PDF_DOC = fitz.open(PDF_PATH)
    return _PDF_DOC

def extract_page_as_image(page_index: int, dpi: int = 150) -> Image.Image:
    """Extract a specific page from the PDF as a PIL Image"""
    doc = get_pdf_doc()
    if doc is None:
        raise FileNotFoundError(f"PDF missing at {PDF_PATH}")
    page = doc[page_index]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data))

def get_page_description_vision(image: Image.Image) -> str:
    """Use Gemini Vision to extract text and keywords from a page image"""
    model = genai.GenerativeModel('gemini-2.5-flash')
    prompt = "Extract all text, mathematical formulas, theorems, and exercise numbers from this calculus textbook page. Be very detailed so this content can be used for a search index."
    
    try:
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        print(f"Vision OCR Error: {e}")
        return ""

def build_knowledge_base(progress_callback=None):
    """
    New Vision-based indexing pipeline:
    1. Extract page image
    2. Vision OCR with Gemini
    3. Embed OCR results
    4. Save to vector store
    """
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF missing at {PDF_PATH}")

    doc = fitz.open(PDF_PATH)
    total_pages = len(doc)
    doc.close()
    
    store = get_vector_store()
    
    # We index page by page
    for i in range(total_pages):
        if progress_callback:
            progress_callback(int((i / total_pages) * 100), f"Processing Page {i+1}/{total_pages} (Multimodal OCR)...")
        
        # 1. Extract
        img = extract_page_as_image(i, dpi=100) # Low DPI for OCR indexing
        
        # 2. Describe (Gemini Vision)
        description = get_page_description_vision(img)
        
        if description:
            # 3. Embed
            emb = get_embedding(description)
            if emb:
                store.add_item(description, i + 1, emb)
        
        # Rate limit protection
        time.sleep(1.0) 
        
        # Periodic save
        if i % 5 == 0:
            store.save()
            
    store.save()
    if progress_callback:
        progress_callback(100, "Vision Indexing Complete!")
    return [{"status": "ready"}]

def load_knowledge_base():
    if os.path.exists(DB_FILE):
        return [{"status": "ready"}]
    return []

# Global vector store instance for caching
_VECTOR_STORE = None

def get_vector_store():
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = SimpleLocalVectorStore()
    return _VECTOR_STORE

def retrieve_context_multimodal(query: str, top_k: int = 2) -> List[Dict]:
    """Retrieve relevant page descriptions and numbers"""
    store = get_vector_store()
    return store.search(query, k=top_k)

def generate_answer(query: str, retrieved_items: List[Dict], user_image=None):
    """
    Final Answer Generation (Streaming):
    1. Take the top retrieved pages.
    2. Extract HIGH-RES images of those pages.
    3. Send images + query to Gemini for the actual book-based answer.
    """
    model = genai.GenerativeModel('gemini-2.5-flash')
    
    content_to_send = [f"User Query: {query}\n\nI have retrieved the following relevant pages from the 'Thomas Finney Calculus' textbook. Use these images to answer the query accurately following the book's methods."]
    
    # Extract high-res images for the best pages
    for item in retrieved_items:
        page_num = item['page']
        print(f"Fetching High-Res Page {page_num}...")
        high_res_img = extract_page_as_image(page_num - 1, dpi=200)
        content_to_send.append(f"--- TEXTBOOK PAGE {page_num} ---")
        content_to_send.append(high_res_img)
        
    if user_image:
        content_to_send.append("The user also provided this image related to the problem:")
        content_to_send.append(user_image)
        
    try:
        response = model.generate_content(content_to_send, stream=True)
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"Error generating answer: {e}"
