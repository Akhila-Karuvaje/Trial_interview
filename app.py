from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import google.generativeai as genai
import os
import speech_recognition as sr
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import json
import whisper

# Download NLTK data
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    print("✅ NLTK data downloaded")
except:
    pass

app = Flask(__name__)
app.secret_key = 'my_super_secret_key_456789'

# Gemini API config
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', 'AIzaSyACpD3waeAbKickkjJb7gBHqegPhGGB-VE')
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Load Whisper model at startup
print("🔄 Loading Whisper tiny model...")
model_whisper = whisper.load_model("tiny")
print("✅ Whisper loaded!")

# ============================================================
# ===== Utility Functions ====================================
# ============================================================

def SpeechToText():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            print("Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source)
            print("Listening...")
            audio = r.listen(source)
        print("Recognizing...")
        query = r.recognize_google(audio, language='en-IN')
        return query
    except sr.UnknownValueError:
        return "Could not understand the audio."
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def clean_answer(answer):
    try:
        words = word_tokenize(answer)
        stop_words = set(stopwords.words('english'))
        return ' '.join([word for word in words if word.lower() not in stop_words])
    except:
        return answer

def detect_fillers(text):
    try:
        common_fillers = {"um", "uh", "like", "you know", "so", "actually", "basically", "literally", "well", "hmm"}
        words = word_tokenize(text.lower())
        used_fillers = [w for w in words if w in common_fillers]
        return ", ".join(set(used_fillers)) if used_fillers else "None"
    except:
        return "None"

# ============================================================
# ===== Routes ===============================================
# ============================================================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_questions', methods=['POST'])
def generate_questions():
    job = request.form['job']
    level = request.form['level']
    session['job_title'] = job
    session['difficulty'] = level
    return redirect(url_for('regenerate_questions'))

@app.route('/regenerate_questions')
def regenerate_questions():
    job = session.get('job_title')
    level = session.get('difficulty')

    prompt = f"""
    Generate exactly 10 interview questions for the job role: {job} 
    with difficulty level: {level}. 
    Only return the 10 questions in plain text, numbered 1 to 10. 
    Do not include any introduction or extra comments.
    """
    response = model.generate_content(prompt)

    raw_questions = response.text.strip().split("\n")
    questions = []
    for q in raw_questions:
        match = re.match(r'^\d+[\).\s-]+(.*)', q.strip())
        if match:
            questions.append(match.group(1).strip())

    questions = questions[:10]
    session['questions'] = questions

    return redirect(url_for('questions'))

@app.route('/questions')
def questions():
    questions = session.get('questions', [])
    job = session.get('job_title')
    difficulty = session.get('difficulty')
    question_list = list(enumerate(questions, start=1))
    return render_template('questions.html', questions=question_list, job_title=job, difficulty=difficulty)

@app.route('/interview/<int:qid>')
def interview(qid):
    questions = session.get('questions', [])
    if 1 <= qid <= len(questions):
        question = questions[qid - 1]
    else:
        question = 'No question found'
    return render_template('interview.html', question=question, qid=qid)

@app.route('/get_analysis', methods=['POST'])
def get_analysis():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file"}), 400

    audio_file = request.files['audio']
    audio_path = "user_audio.wav"
    audio_file.save(audio_path)

    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        transcribed_text = recognizer.recognize_google(audio)
        duration = 10
    except sr.UnknownValueError:
        return jsonify({"error": "Could not understand audio."}), 400
    except sr.RequestError as e:
        return jsonify({"error": f"Speech recognition failed: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500

    return jsonify({
        "transcription": transcribed_text,
        "duration": duration
    })

@app.route('/submit_answer/<qid>', methods=['POST'])
def submit_answer(qid):
    user_answer = request.form.get('answer', '').strip()
    questions = session.get('questions', [])
    
    # Get the actual question text
    try:
        question_text = questions[int(qid) - 1]
    except:
        question_text = "Interview question"

    prompt = f"""
You are a strict technical interviewer evaluating an interview answer.

Question: {question_text}
User's Answer: "{user_answer}"

EVALUATION RULES:
- "Valid": Answer correctly addresses the question with accurate information
- "Partial": Answer is somewhat related but incomplete or has errors
- "Invalid": Answer is wrong, off-topic, nonsense, or gibberish

Examples of INVALID:
- Random words like "asdfgh xyz"
- Completely wrong information
- Off-topic answers

Return ONLY valid JSON (no markdown):
{{
    "correct_answer": "Brief ideal answer",
    "validation": "Valid/Invalid/Partial",
    "fillers_used": ["um", "like"],
    "feedback": "Brief explanation"
}}
"""
    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        # Remove markdown if present
        raw_text = raw_text.replace('```json', '').replace('```', '').strip()
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            result = json.loads(json_str)
        else:
            raise ValueError("No JSON object found")
    except Exception as e:
        print(f"Error: {e}")
        result = {
            "correct_answer": "Unable to parse response.",
            "validation": "Unknown",
            "fillers_used": [],
            "feedback": "N/A"
        }

    return jsonify({
        'user_answer': user_answer,
        'validation_result': {
            'correct_answer': result.get('correct_answer', ''),
            'validation': result.get('validation', ''),
            'feedback': result.get('feedback', '')
        },
        'fillers_used': result.get('fillers_used', [])
    })

# ============================================================
# ===== Video Interview Routes ===============================
# ============================================================

@app.route('/video_interview')
def video_interview():
    return render_template('video_interview.html')

@app.route('/submit_video_answer/<qid>', methods=['POST'])
def submit_video_answer(qid):
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files['video']
    os.makedirs("uploads", exist_ok=True)
    filepath = os.path.join("uploads", f"answer_{qid}.webm")
    file.save(filepath)
    
    print(f"📹 Processing video for Q{qid}")

    # Get question
    questions = session.get('questions', [])
    try:
        question_text = questions[int(qid) - 1]
    except:
        question_text = "Interview question"

    # Step 1: Transcribe using Whisper
    try:
        print("🎤 Transcribing...")
        result = model_whisper.transcribe(filepath, fp16=False)
        transcript = result['text'].strip()
        print(f"✅ Transcript: {transcript[:100]}")
    except Exception as e:
        print(f"❌ Transcription error: {e}")
        transcript = f"Transcription failed: {str(e)}"

    # Step 2: Evaluate with Gemini
    prompt = f"""
You are an expert interviewer analyzing a video interview answer.

Question: {question_text}
User's Answer (from video): "{transcript}"

Evaluate with these scores (0.0 to 1.0):
- Confidence Score: Speaker's clarity and confidence
- Content Relevance: How well it addresses the question
- Fluency Score: Language fluency and coherence

Return ONLY valid JSON (no markdown):
{{
    "Confidence Score": 0.8,
    "Content Relevance": 0.7,
    "Fluency Score": 0.9
}}
"""

    try:
        response = model.generate_content(prompt)
        raw_text = response.text.strip()
        raw_text = raw_text.replace('```json', '').replace('```', '').strip()
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            scores = json.loads(json_str)
        else:
            scores = {
                "Confidence Score": 0.7,
                "Content Relevance": 0.7,
                "Fluency Score": 0.7
            }
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        scores = {
            "Confidence Score": 0.7,
            "Content Relevance": 0.7,
            "Fluency Score": 0.7
        }

    # Calculate final score
    try:
        final_eval = round(
            (scores["Confidence Score"] +
             scores["Content Relevance"] +
             scores["Fluency Score"]) / 3 * 100, 2
        )
    except:
        final_eval = 70.0

    print(f"📊 Final: {final_eval}%")

    return jsonify({
        "Confidence Score": scores["Confidence Score"],
        "Content Relevance": scores["Content Relevance"],
        "Fluency Score": scores["Fluency Score"],
        "Final Evaluation": final_eval,
        "Transcript": transcript
    })

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
