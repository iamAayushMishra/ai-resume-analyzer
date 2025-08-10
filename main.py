import os
import re
import uuid
import time
import chromadb
import google.generativeai as genai
import fitz  # PyMuPDF
import markdown2
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
from flask_socketio import SocketIO
from google.api_core import exceptions as google_exceptions
from chromadb.utils import embedding_functions

# --- FLASK APP & SOCKETIO CONFIGURATION ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("FLASK_SECRET_KEY", "a-default-fallback-key-if-not-set")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SERVER_NAME'] = 'localhost:5000'   # So url_for can work outside request contexts if needed
app.config['PREFERRED_URL_SCHEME'] = 'http'

socketio = SocketIO(app, async_mode='eventlet')

# In-memory task/result storage
analysis_tasks = {}
analysis_results = {}

# Ensure uploads folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- AI CONFIGURATION ---
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
GENERATION_MODEL_NAME = "gemini-2.5-pro"

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME
)

# --- Helper Functions ---
def parse_pdf(file_path: str) -> str:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

def chunk_text(text: str, chunk_size: int = 512, chunk_overlap: int = 50) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
    return chunks

def highlight_keywords_in_pdf(original_path: str, keywords: list[str]) -> str:
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

# --- Main Analysis Task ---
def perform_analysis(socketio_instance, task_id: str):
    """
    Performs RAG-based analysis using data from the in-memory task dictionary.
    This runs as a background task started by Socket.IO.
    """
    # app context for url_for or other Flask needs
    with app.app_context():
        task_data = analysis_tasks.get(task_id)
        if not task_data:
            print(f"Error: Could not find task data for task_id {task_id}")
            return

        sid = task_data.get('sid')
        resume_path = task_data.get('resume_path')
        jd_text = task_data.get('jd_text')

        try:
            # Step 1
            print("\nStep 1: Parsing and Chunking\n")
            socketio_instance.emit('status_update', {'step': 1, 'message': 'Parsing and chunking resume...'}, room=sid)
            resume_text = parse_pdf(resume_path)
            resume_chunks = chunk_text(resume_text)

            # Step 2
            print("\nStep 2: Initializing Vector DB\n")
            socketio_instance.emit('status_update', {'step': 2, 'message': 'Initializing vector database...'}, room=sid)
            client = chromadb.Client()
            if "resume_collection" in [c.name for c in client.list_collections()]:
                client.delete_collection(name="resume_collection")
            collection = client.create_collection(name="resume_collection", embedding_function=sentence_transformer_ef)
            collection.add(documents=resume_chunks, ids=[str(i) for i in range(len(resume_chunks))])
            
            # Step 3
            print("\nStep 3: Generating Analysis\n")
            socketio_instance.emit('status_update', {'step': 3, 'message': 'Generating analysis with AI...'}, room=sid)

            # Dynamic splitting of JD into requirements:
            raw_requirements = re.split(r'\n|•|-', jd_text or "")
            job_requirements = [req.strip() for req in raw_requirements if req.strip()]

            llm = genai.GenerativeModel(GENERATION_MODEL_NAME)
            total_score, requirements_count, detailed_analysis, all_keywords = 0, 0, "", []

            # loop over each requirement and emit a per-requirement progress update
            for idx, req in enumerate(job_requirements, start=1):
                socketio_instance.emit('status_update', {
                    'step': 3,
                    'message': f'Analyzing requirement {idx}/{len(job_requirements)}...'
                }, room=sid)

                print(f"Processing requirement [{idx}/{len(job_requirements)}]: {req}")
                start_time = time.time()

                try:
                    results = collection.query(query_texts=[req], n_results=3)
                    # results['documents'] is list of lists (one per query). take first query results.
                    context = "\n---\n".join(results['documents'][0]) if results and 'documents' in results and results['documents'] else ""
                    prompt = f"""
You are an expert HR analyst. Your task is to analyze a candidate's resume against a specific job requirement.
Job Requirement: {req}
Relevant excerpts from the candidate's resume: {context}
Your Analysis:
1. Provide a concise, short crisp one-paragraph analysis explaining your reasoning.
2. On a new line, provide a numerical confidence score like this: "Confidence Score: [score]/100".
3. On a final new line, list the specific keywords from the resume excerpts that match the job requirement, in this format: "Keywords: [keyword1, keyword2, ...]". If no direct keywords are found, write "Keywords: []".
"""
                    response = llm.generate_content(prompt)

                    # guard: if API returned nothing useful, skip this req
                    if not getattr(response, "candidates", None) or not response.candidates[0].content.parts:
                        print(f"No content returned for requirement: {req}")
                        continue

                    # Safely extract text
                    answer_text = response.text if hasattr(response, 'text') else ""
                    # extract score and keywords
                    score_match = re.search(r'Confidence Score:\s*(\d{1,3})/100', answer_text)
                    if score_match:
                        total_score += int(score_match.group(1))
                        requirements_count += 1

                    keywords_match = re.search(r'Keywords:\s*\[(.*?)\]', answer_text, re.DOTALL)
                    if keywords_match and keywords_match.group(1):
                        all_keywords.extend([k.strip().strip("'\"") for k in keywords_match.group(1).split(',')])

                    detailed_analysis += f"### Requirement: {req}\n{answer_text}\n\n"
                    print(f"Processed requirement in {time.time() - start_time:.2f}s")

                except Exception as e:
                    print(f"Error processing requirement '{req}': {e}")
                    # continue to next requirement (don't crash whole analysis)
                    continue
            
            # Step 4 finalizing
            print("\nStep 4: Finalizing Report\n")
            socketio_instance.emit('status_update', {'step': 4, 'message': 'Finalizing report and score...'}, room=sid)
            overall_score = total_score / requirements_count if requirements_count > 0 else 0
            analysis_report = f"## 📝 Detailed Analysis\n\n**Candidate:** `{os.path.basename(resume_path)}`\n\n---\n\n{detailed_analysis}"
            html_report = markdown2.markdown(analysis_report, extras=["tables", "fenced-code-blocks", "break-on-newline"])

            # Step 5 highlighting
            print("\nStep 5: Highlighting PDF\n")
            socketio_instance.emit('status_update', {'step': 5, 'message': 'Highlighting keywords in PDF...'}, room=sid)
            highlighted_filename = highlight_keywords_in_pdf(resume_path, all_keywords)

            # Store results
            analysis_results[task_id] = {
                'score': int(overall_score),
                'report': html_report,
                'pdf_filename': highlighted_filename
            }

            # Build manual report URL (avoid url_for outside request if any)
            report_url = f"/report/{task_id}"
            # Emit final complete event
            socketio_instance.emit('analysis_complete', {'url': report_url}, room=sid)

        except google_exceptions.InternalServerError as e:
            print(f"An API error occurred: {e}")
            socketio_instance.emit('analysis_error', {'error': "The AI service is currently experiencing issues. Please try again."}, room=sid)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            socketio_instance.emit('analysis_error', {'error': "An unexpected error occurred during the analysis."}, room=sid)
        # NOTE: do NOT delete analysis_tasks here — keep until report is viewed to avoid reconnect issues.

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    if 'resume' not in request.files or not request.form.get('job_description'):
        return jsonify({'success': False, 'error': 'Missing files or job description.'}), 400

    resume_file = request.files['resume']
    job_description = request.form['job_description']

    resume_filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
    resume_file.save(resume_filename)

    task_id = str(uuid.uuid4())
    # include started flag so we don't spawn duplicate background tasks
    analysis_tasks[task_id] = {
        'resume_path': resume_filename,
        'jd_text': job_description,
        'sid': None,
        'started': False
    }
    
    return jsonify({'success': True, 'redirect_url': url_for('loading', task_id=task_id)})

@app.route('/loading/<task_id>')
def loading(task_id):
    # If task doesn't exist and results exist, redirect to report
    if task_id not in analysis_tasks and task_id in analysis_results:
        return redirect(url_for('report', task_id=task_id))
    if task_id not in analysis_tasks:
        return redirect(url_for('index'))
    return render_template('loading.html', task_id=task_id)

@app.route('/report/<task_id>')
def report(task_id):
    results = analysis_results.get(task_id)
    if not results:
        return redirect(url_for('index'))
    
    # Cleanup after viewing
    analysis_results.pop(task_id, None)
    analysis_tasks.pop(task_id, None)
    
    return render_template('report.html', **results)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@socketio.on('start_analysis')
def handle_start_analysis(data):
    task_id = data.get('task_id')
    task_data = analysis_tasks.get(task_id)
    
    if task_data:
        task_data['sid'] = request.sid

        # If results are already ready, send them immediately
        if task_id in analysis_results:
            print(f"Resending completed analysis for task {task_id} to sid {request.sid}")
            report_url = f"/report/{task_id}"
            socketio.emit('analysis_complete', {'url': report_url}, room=request.sid)
            return
        
        # If no results yet, start background analysis
        if not task_data.get('analysis_started', False):
            task_data['analysis_started'] = True
            print(f"Starting background analysis for task {task_id} on sid {request.sid}")
            socketio.start_background_task(perform_analysis, socketio, task_id)
        else:
            print(f"Analysis already started for task {task_id}; updated sid to {request.sid}")
    else:
        print(f"Warning: 'start_analysis' received for unknown task_id: {task_id}")


if __name__ == '__main__':
    if not os.getenv("GOOGLE_API_KEY") or not os.getenv("FLASK_SECRET_KEY"):
        print("Error: Make sure GOOGLE_API_KEY and FLASK_SECRET_KEY are set in your .env file.")
    else:
        socketio.run(app, debug=True, use_reloader=False)
