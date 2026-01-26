#  Calculus AI Tutor (RAG-powered)

An intelligent RAG (Retrieval-Augmented Generation) chatbot designed to help students master Calculus. This application leverages **Google Gemini** and a custom knowledge base built from **Thomas Finney Calculus (9th Edition)** to provide textbook-accurate explanations and solutions.

---

## Features

- ** RAG-Driven Responses**: Answers questions based on the Thomas Finney Calculus textbook for high accuracy.
- ** Vision Support**: Upload images of math problems to get step-by-step solutions.
- ** Conversation History**: Maintain multiple chat sessions in a sidebar for easy reference.
- ** Fast Retrieval**: Uses cosine similarity with Gemini embeddings for efficient context matching.
- ** Modern UI**: Clean, intuitive interface built with Streamlit.

---

##  Tech Stack

- **Core**: Python
- **LLM**: Google Gemini 2.0 Flash
- **Embeddings**: Text Embedding 004
- **PDF Parsing**: PyPDF
- **Frontend**: Streamlit
- **Environment**: Dotenv for secure key management

---

##  Project Structure

```text
mvc-master-rag-chatbot/
├── data/                   # Core Application Logic
│   ├── app.py             # Streamlit UI
│   ├── cli.py             # CLI Interface
│   └── rag_core.py        # Backend RAG Logic
├── public/                 # Static Assets
│   └── calculus.pdf       # Textbook Source
├── RAG system/             # System Configuration & Setup
│   ├── .env.local         # API Keys
│   ├── .gitignore         # Git ignore rules
│   ├── requirements.txt   # Dependencies
│   └── start_app.bat      # Quick launch script
├── vector_store/           # Search Database
│   └── embeddings_cache.json
└── README.md               # Documentation
```

---

##  Local Setup

### Prerequisites
- Python 3.9+
- [Google AI Studio API Key](https://aistudio.google.com/app/apikey)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd mvc-master-rag-chatbot
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Key**:
   Create a `.env.local` file in the root directory and add your key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

---

##  Usage

- **Querying**: Ask any calculus question in the chat input.
- **Images**: Upload a photo of a problem, then ask "Solve this" or "Explain this concept".
- **History**: Use the sidebar to switch between different problem-solving sessions.

---

##  License
This project is for educational purposes. All rights to the contents of the textbook used in the demonstration belong to the respective copyright holders.

