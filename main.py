import os
import re
import uuid
import chromadb
import google.generativeai as genai
import fitz  # PyMuPDF
import markdown2
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from flask_socketio import SocketIO, emit
from google.api_core import exceptions as google_exceptions
from chromadb.utils import embedding_functions

# --- FLASK APP & SOCKETIO CONFIGURATION ---
app = Flask(__name__)
# Load the secret key from an environment variable for consistency
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "a-default-fallback-key-if-not-set")
app.config['UPLOAD_FOLDER'] = 'uploads'
socketio = SocketIO(app, async_mode='eventlet')

# In-memory storage instead of sessions
analysis_tasks = {}
analysis_results = {}

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- CORE RAG & AI CONFIGURATION ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Model names
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "gemini-2.5-pro" # Corrected model name

# Initialize the embedding function once to be reused
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME
)

def parse_pdf(file_path: str) -> str:
    """Extracts text content from a PDF file."""
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[str]:
    """Splits text into smaller, overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def highlight_keywords_in_pdf(original_path: str, keywords: list[str]) -> str:
    """Finds keywords in a PDF and saves a new version with highlights."""
    doc = fitz.open(original_path)
    unique_keywords = list(set(kw for kw in keywords if kw))

    if not unique_keywords:
        return os.path.basename(original_path)

    for page in doc:
        for keyword in unique_keywords:
            areas = page.search_for(keyword, quads=False)
            for area in areas:
                page.add_highlight_annot(area)
    
    dir_name = os.path.dirname(original_path)
    base_name = os.path.basename(original_path)
    highlighted_filename = f"highlighted_{base_name}"
    highlighted_path = os.path.join(dir_name, highlighted_filename)
    
    doc.save(highlighted_path)
    doc.close()
    
    return highlighted_filename

def perform_analysis(socketio_instance, task_id: str):
    """
    Performs RAG-based analysis using data from the in-memory task dictionary.
    """
    # This 'with' block provides the necessary application context for the background task.
    with app.app_context():
        task_data = analysis_tasks.get(task_id)
        if not task_data:
            print(f"Error: Could not find task data for task_id {task_id}")
            return

        sid = task_data['sid']
        resume_path = task_data['resume_path']
        jd_text = task_data['jd_text']

        try:
            # Step 1: Parsing and Chunking
            socketio_instance.emit('status_update', {'step': 1, 'message': 'Parsing and chunking resume...'}, room=sid)
            resume_text = parse_pdf(resume_path)
            resume_chunks = chunk_text(resume_text)

            # Step 2: Initializing Vector DB
            socketio_instance.emit('status_update', {'step': 2, 'message': 'Initializing vector database...'}, room=sid)
            client = chromadb.Client()
            if "resume_collection" in [c.name for c in client.list_collections()]:
                client.delete_collection(name="resume_collection")
            collection = client.create_collection(name="resume_collection", embedding_function=sentence_transformer_ef)
            collection.add(documents=resume_chunks, ids=[str(i) for i in range(len(resume_chunks))])
            
            # Step 3: Generating Analysis
            socketio_instance.emit('status_update', {'step': 3, 'message': 'Generating analysis with AI...'}, room=sid)
            job_requirements = [line.strip() for line in jd_text.split('\n') if line.strip()]
            llm = genai.GenerativeModel(GENERATION_MODEL_NAME)
            
            total_score, requirements_count, detailed_analysis, all_keywords = 0, 0, "", []

            for req in job_requirements:
                results = collection.query(query_texts=[req], n_results=3)
                context = "\n---\n".join(results['documents'][0])
                prompt = f"""
                You are an expert HR analyst. Your task is to analyze a candidate's resume against a specific job requirement.
                **Job Requirement:** {req}
                **Relevant excerpts from the candidate's resume:** {context}
                **Your Analysis:**
                1.  Provide a concise, short crisp one-paragraph analysis explaining your reasoning.
                2.  On a new line, provide a numerical confidence score like this: "Confidence Score: [score]/100".
                3.  On a final new line, list the specific keywords from the resume excerpts that match the job requirement, in this format: "Keywords: [keyword1, keyword2, ...]". If no direct keywords are found, write "Keywords: []".
                """
                response = llm.generate_content(prompt)
                
                score_match = re.search(r'Confidence Score: (\d+)/100', response.text)
                if score_match:
                    total_score += int(score_match.group(1))
                    requirements_count += 1
                
                keywords_match = re.search(r'Keywords: \[(.*?)\]', response.text, re.DOTALL)
                if keywords_match and keywords_match.group(1):
                    all_keywords.extend([k.strip().strip("'\"") for k in keywords_match.group(1).split(',')])

                detailed_analysis += f"### Requirement: {req}\n{response.text}\n\n"
            
            # Step 4: Finalizing Report
            socketio_instance.emit('status_update', {'step': 4, 'message': 'Finalizing report and score...'}, room=sid)
            overall_score = total_score / requirements_count if requirements_count > 0 else 0
            analysis_report = f"## 📝 Detailed Analysis\n\n**Candidate:** `{os.path.basename(resume_path)}`\n\n---\n\n{detailed_analysis}"
            html_report = markdown2.markdown(analysis_report, extras=["tables", "fenced-code-blocks", "break-on-newline"])

            # Step 5: Highlighting PDF
            socketio_instance.emit('status_update', {'step': 5, 'message': 'Highlighting keywords in PDF...'}, room=sid)
            highlighted_filename = highlight_keywords_in_pdf(resume_path, all_keywords)

            # Store results in the global dictionary
            analysis_results[task_id] = {
                'score': int(overall_score),
                'report': html_report,
                'pdf_filename': highlighted_filename
            }
            socketio_instance.emit('analysis_complete', {'url': url_for('report', task_id=task_id)}, room=sid)

        except google_exceptions.InternalServerError as e:
            print(f"An API error occurred: {e}")
            error_message = "The AI service is currently experiencing issues. Please try again in a few moments."
            socketio_instance.emit('analysis_error', {'error': error_message}, room=sid)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            error_message = "An unexpected error occurred during the analysis. Please check the logs."
            socketio_instance.emit('analysis_error', {'error': error_message}, room=sid)
        finally:
            # Clean up the task data once analysis is complete or fails
            if task_id in analysis_tasks:
                del analysis_tasks[task_id]

# --- FLASK & SOCKETIO ROUTES ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    """Handles file uploads and creates a task in the in-memory dictionary."""
    if 'resume' not in request.files or not request.form['job_description']:
        return jsonify({'success': False, 'error': 'Missing files or job description.'}), 400

    resume_file = request.files['resume']
    job_description = request.form['job_description']

    resume_filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
    resume_file.save(resume_filename)

    task_id = str(uuid.uuid4())
    analysis_tasks[task_id] = {
        'resume_path': resume_filename,
        'jd_text': job_description,
        'sid': None # Will be populated on connect
    }
    
    return jsonify({'success': True, 'redirect_url': url_for('loading', task_id=task_id)})

@app.route('/loading/<task_id>')
def loading(task_id):
    if task_id not in analysis_tasks:
        return redirect(url_for('index'))
    return render_template('loading.html', task_id=task_id)

@app.route('/report/<task_id>')
def report(task_id):
    results = analysis_results.get(task_id)
    if not results:
        return redirect(url_for('index'))
    
    # Clean up results after they have been fetched
    if task_id in analysis_results:
        del analysis_results[task_id]
    
    return render_template('report.html', **results)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@socketio.on('start_analysis')
def handle_start_analysis(data):
    """Triggered by the loading page to start the analysis in the background."""
    task_id = data.get('task_id')
    task_data = analysis_tasks.get(task_id)
    if task_data:
        # Associate the client's session ID with the task
        task_data['sid'] = request.sid
        socketio.start_background_task(perform_analysis, socketio, task_id)
    else:
        print(f"Warning: 'start_analysis' received for unknown task_id: {task_id}")

if __name__ == '__main__':
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("FLASK_SECRET_KEY"):
        print("Error: Make sure GOOGLE_API_KEY and FLASK_SECRET_KEY are set in your .env file.")
    else:
        # Disable the reloader to prevent connection issues with eventlet
        socketio.run(app, debug=True, use_reloader=False)
