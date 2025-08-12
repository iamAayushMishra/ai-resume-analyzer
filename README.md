# 📄 AI-Powered Resume Analysis Tool

A Flask + Socket.IO application that analyzes resumes against job descriptions using Google's Gemini API and ChromaDB for semantic search.  
It highlights matched keywords in the uploaded resume PDF and provides a detailed match score.

---

## 🚀 Features
- **Live Status Updates** — See step-by-step progress via WebSocket.
- **RAG-based Matching** — Uses resume chunking + vector search for relevant context.
- **Keyword Highlighting** — Generates a highlighted PDF showing matched terms.
- **Detailed Report** — Match score, requirement-wise breakdown, and extracted keywords.
- **Multi-step Processing** — Parsing → Vector DB → AI Analysis → Report Generation.

---

## 🛠️ Tech Stack
- **Backend:** Python, Flask, Flask-SocketIO
- **AI:** Google Gemini API (Generative AI)
- **Vector DB:** ChromaDB + SentenceTransformers
- **PDF Processing:** PyMuPDF (fitz)
- **Frontend:** HTML, CSS, JS (Socket.IO)

---

## 📂 Project Structure
```
.
├── main.py                 # Flask backend & AI analysis logic
├── templates/              # HTML templates
│   ├── index.html           # Upload page
│   ├── loading.html         # Loading/Progress UI
│   └── report.html          # Final analysis report
├── uploads/                # Stores uploaded & highlighted resumes
│   └── README.md            # Info about uploads folder
├── .env                     # API keys & secrets (not committed)
└── requirements.txt         # Python dependencies
```

---

## ⚙️ Setup & Run
1. **Clone the repo**  
   ```bash
   git clone https://github.com/iamAayushMishra/ai-resume-analyzer.git

   cd ai-resume-analyzer
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Create `.env` file**  
   ```
   GOOGLE_API_KEY=your_gemini_api_key
   FLASK_SECRET_KEY=your_flask_secret
   ```

4. **Run the app**  
   ```bash
   python main.py
   ```
   or  
   ```bash
   flask run
   ```

5. **Open in browser**  
   ```
   http://localhost:5000
   ```

---

## 📌 Notes
- The `uploads/` folder is **auto-created** and stores original & highlighted resumes.  
- Add `uploads/*` to `.gitignore` (except `uploads/README.md`) to prevent committing sensitive files.
- Requires Python 3.9+ for best compatibility.

---
