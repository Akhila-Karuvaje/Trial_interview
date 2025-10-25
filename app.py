from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import google.generativeai as genai
import os
import secrets
import re
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import speech_recognition as sr

# ============================================================
# ===== Flask Setup =========================================
# ============================================================
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# ============================================================
# ===== Gemini API ==========================================
# ============================================================
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY environment variable required!")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# ============================================================
# ===== Whisper Setup ========================================
# ============================================================
WHISPER_ENABLED = False
whisper_model = None

try:
    import whisper
    whisper_model = whisper.load_model("tiny", device="cpu", download_root="/tmp/whisper_cache")
    WHISPER_ENABLED = True
except Exception as e:
    print(f"⚠️ Whisper not available: {e}")

# ============================================================
# ===== NLTK Setup ===========================================
# ============================================================
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

# ============================================================
# ===== Utility Functions ===================================
# ============================================================
def detect_fillers(text):
    fillers = {"um", "uh", "like", "you know", "so", "actually", "basically", "literally", "well", "hmm"}
    words = word_tokenize(text.lower())
    used = [w for w in words if w in fillers]
    return ", ".join(set(used)) if used else "None"

def SpeechToText():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
        return r.recognize_google(audio, language='en-IN')
    except Exception as e:
        return f"Error: {str(e)}"

def clean_answer(answer):
    try:
        words = word_tokenize(answer)
        stop_words = set(stopwords.words('english'))
        return ' '.join([word for word in words if word.lower() not in stop_words])
    except:
        return answer

# ============================================================
# ===== Routes ===============================================
# ============================================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    job = request.form.get('job', '').strip()
    level = request.form.get('level', 'medium')
    if not job:
        return jsonify({"error": "Job title required"}), 400
    session['job_title'] = job
    session['difficulty'] = level
    return redirect(url_for('regenerate_questions'))

@app.route('/regenerate_questions')
def regenerate_questions():
    job = session.get('job_title', 'Software Developer')
    level = session.get('difficulty', 'medium')

    prompt = f"""
Generate exactly 10 interview questions for the job role: {job} 
with difficulty level: {level}. 
Only return the 10 questions in plain text, numbered 1 to 10.
Do not include any introduction or extra comments.
"""
    try:
        response = model.generate_content(prompt)
        raw_questions = response.text.strip().split("\n")
        questions = [re.sub(r'^\d+[\).\s-]+', '', q).strip() for q in raw_questions if q.strip()]
        session['questions'] = questions[:10]
        session['results'] = []
        return redirect(url_for('questions'))
    except Exception as e:
        return jsonify({"error": f"Failed to generate questions: {e}"}), 500

@app.route('/questions')
def questions():
    questions = session.get('questions', [])
    if not questions:
        return redirect(url_for('index'))
    return render_template('questions.html', questions=list(enumerate(questions, start=1)),
                           job_title=session.get('job_title', 'N/A'),
                           difficulty=session.get('difficulty', 'N/A'))

@app.route('/interview/<int:qid>')
def interview(qid):
    questions = session.get('questions', [])
    question = questions[qid-1] if 1 <= qid <= len(questions) else 'No question found'
    return render_template('interview.html', question=question, qid=qid)

@app.route('/submit_answer/<qid>', methods=['POST'])
def submit_answer(qid):
    user_answer = request.form.get('answer', '').strip()
    if not user_answer:
        return jsonify({"error": "Answer required"}), 400
    questions = session.get('questions', [])
    question_text = questions[int(qid)-1] if questions else "Interview question"

    prompt = f"""
You are a strict technical interviewer evaluating an interview answer.

Question: {question_text}
User's Answer: "{user_answer}"

Return ONLY valid JSON:
{{
    "correct_answer": "Brief ideal answer",
    "validation": "Valid/Partial-High/Partial-Low/Invalid",
    "score": 75,
    "fillers_used": ["um", "like"],
    "feedback": "Brief explanation"
}}
"""
    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip().replace('```json', '').replace('```', '').strip()
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        result = json.loads(json_match.group()) if json_match else {}
    except:
        result = {"correct_answer":"N/A","validation":"Invalid","score":0,"feedback":"Error"}

    results = session.get('results', [])
    results.append({
        "Q.ID": qid,
        "Question": question_text,
        "User Answer": user_answer,
        "Score": result.get('score',0),
        "Validation": result.get('validation','Invalid'),
        "Feedback": result.get('feedback',''),
        "Fillers Used": result.get('fillers_used',[])
    })
    session['results'] = results

    return jsonify(result)

# ============================================================
# ===== Video Interview ======================================
# ============================================================
@app.route('/video_interview')
def video_interview():
    return render_template('video_interview.html')

@app.route('/submit_video_answer/<qid>', methods=['POST'])
def submit_video_answer(qid):
    if 'video' not in request.files:
        return jsonify({"error":"No video uploaded"}),400
    file = request.files['video']
    os.makedirs("/tmp/uploads", exist_ok=True)
    filepath = os.path.join("/tmp/uploads", f"answer_{qid}_{os.getpid()}.webm")
    file.save(filepath)

    transcript = "Transcription unavailable"
    if WHISPER_ENABLED and whisper_model:
        try:
            result = whisper_model.transcribe(filepath, fp16=False)
            transcript = result.get('text','').strip()
        except:
            transcript = "Transcription failed"

    # Evaluate with Gemini if transcript valid
    validation_result = {
        "correct_answer": "",
        "validation": "Invalid",
        "score": 0,
        "feedback": "",
        "confidence_score": 0.0,
        "content_relevance": 0.0,
        "fluency_score": 0.0
    }

    if transcript and len(transcript)>5:
        try:
            prompt = f"""
You are a strict technical interviewer evaluating a VIDEO interview answer.

Question: {session.get('questions',[qid])[int(qid)-1]}
User's Answer (from video): "{transcript}"

Provide COMPLETE evaluation in JSON format:
{{
    "correct_answer": "Brief ideal answer to the question",
    "validation": "Valid/Partial-High/Partial-Low/Invalid",
    "score": 75,
    "feedback": "2-3 sentences explaining the evaluation",
    "confidence_score": 0.8,
    "content_relevance": 0.7,
    "fluency_score": 0.9
}}
"""
            response = model.generate_content(prompt)
            raw_text = response.text.strip().replace('```json','').replace('```','').strip()
            json_match = re.search(r'\{.*?\}', raw_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                validation_result.update(result)
        except Exception as e:
            validation_result['feedback'] = f"Evaluation error: {e}"

    results = session.get('results', [])
    results.append({
        "Q.ID": qid,
        "Question": session.get('questions',[qid])[int(qid)-1],
        "User Answer": transcript,
        "Score": validation_result.get('score',0),
        "Validation": validation_result.get('validation','Invalid'),
        "Feedback": validation_result.get('feedback',''),
        "Confidence Score": validation_result.get('confidence_score',0),
        "Content Relevance": validation_result.get('content_relevance',0),
        "Fluency Score": validation_result.get('fluency_score',0)
    })
    session['results'] = results

    os.remove(filepath)
    return jsonify({
        "user_answer": transcript,
        "validation_result": validation_result,
        "fillers_used": detect_fillers(transcript).split(', ')
    })

# ============================================================
# ===== Results & Health =====================================
# ============================================================
@app.route('/result')
def result():
    return render_template('result.html', results=session.get('results',[]),
                           job_title=session.get('job_title','N/A'),
                           difficulty=session.get('difficulty','N/A'))

@app.route('/get_results')
def get_results():
    return jsonify(session.get('results',[]))

@app.route('/health')
def health():
    return jsonify({
        "status":"healthy",
        "whisper_enabled":WHISPER_ENABLED,
        "whisper_loaded":whisper_model is not None,
        "api_configured": bool(GOOGLE_API_KEY)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
