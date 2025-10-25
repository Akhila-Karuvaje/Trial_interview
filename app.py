from flask import Flask, render_template, request, redirect, url_for, jsonify, session
import google.generativeai as genai
import os
import speech_recognition as sr
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import json
import secrets
# Download NLTK data
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except:
    pass

app = Flask(__name__)

# Security: Use environment variable for secret key
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# Gemini API config - MUST be in environment variable for Render
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY environment variable required! Set it in Render dashboard.")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# Whisper configuration for Render
WHISPER_ENABLED = os.environ.get('WHISPER_ENABLED', 'true').lower() == 'true'
whisper_model = None

if WHISPER_ENABLED:
    print("🔄 Loading Whisper model (tiny for Render compatibility)...")
    try:
        import whisper
        # Use TINY model instead of SMALL - much less memory
        whisper_model = whisper.load_model("tiny", download_root="/tmp/whisper_cache", device="cpu")
        print("✅ Whisper tiny model loaded!")
    except Exception as e:
        print(f"⚠️ Whisper failed to load: {e}")
        WHISPER_ENABLED = False
        whisper_model = None

# ============================================================
# ===== Utility Functions ====================================
# ============================================================

def SpeechToText():
    r = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
        query = r.recognize_google(audio, language='en-IN')
        return query
    except Exception as e:
        return f"Error: {str(e)}"

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
        questions = []
        
        for q in raw_questions:
            match = re.match(r'^\d+[\).\s-]+(.*)', q.strip())
            if match:
                questions.append(match.group(1).strip())
        
        questions = questions[:10]
        
        if len(questions) < 5:
            raise ValueError("Too few questions generated")
        
        session['questions'] = questions
        session['results'] = []  # Initialize results
        
        return redirect(url_for('questions'))
    
    except Exception as e:
        print(f"Error generating questions: {e}")
        return jsonify({"error": "Failed to generate questions"}), 500

@app.route('/questions')
def questions():
    questions = session.get('questions', [])
    if not questions:
        return redirect(url_for('index'))
    
    job = session.get('job_title', 'N/A')
    difficulty = session.get('difficulty', 'N/A')
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

@app.route('/submit_answer/<qid>', methods=['POST'])
def submit_answer(qid):
    user_answer = request.form.get('answer', '').strip()
    questions = session.get('questions', [])
    
    if not user_answer:
        return jsonify({"error": "Answer is required"}), 400
    
    try:
        question_text = questions[int(qid) - 1]
    except:
        question_text = "Interview question"

    prompt = f"""
You are a strict technical interviewer evaluating an interview answer with a detailed scoring system.

Question: {question_text}
User's Answer: "{user_answer}"

EVALUATION RULES WITH SCORES:
1. "Valid" (76-100%): Answer correctly addresses the question with accurate, relevant information
2. "Partial-High" (50-75%): Answer is related with some correct information but incomplete
3. "Partial-Low" (30-49%): Has some keywords but is vague or mostly incorrect
4. "Invalid" (0-29%): Completely wrong, off-topic, nonsense, or gibberish

Return ONLY valid JSON (no markdown):
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
        raw_text = response.text.strip()
        raw_text = raw_text.replace('```json', '').replace('```', '').strip()
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group()
            result = json.loads(json_str)
            
            # Normalize validation and score
            validation = result.get('validation', 'Invalid')
            score = result.get('score', 0)
            
            # Ensure score matches category
            if validation == 'Valid' and score < 76:
                score = 76
            elif validation == 'Partial-High' and (score < 50 or score > 75):
                score = 65
            elif validation == 'Partial-Low' and (score < 30 or score >= 50):
                score = 40
            elif validation == 'Invalid' and score >= 30:
                score = 15
            
            # Simplify validation for frontend
            if validation in ['Partial-High', 'Partial-Low']:
                simple_validation = 'Partial'
            else:
                simple_validation = validation
            
            result['score'] = score
            result['simple_validation'] = simple_validation
        else:
            raise ValueError("No JSON found")
    
    except Exception as e:
        print(f"Error parsing Gemini: {e}")
        result = {
            "correct_answer": "Unable to evaluate",
            "validation": "Invalid",
            "simple_validation": "Invalid",
            "score": 0,
            "fillers_used": [],
            "feedback": "Evaluation error"
        }

    # Store result in session
    results = session.get('results', [])
    results.append({
        "Q.ID": qid,
        "Question": question_text,
        "User Answer": user_answer,
        "Score": result.get('score', 0),
        "Validation": result.get('simple_validation', 'Invalid'),
        "Feedback": result.get('feedback', '')
    })
    session['results'] = results

    return jsonify({
        'user_answer': user_answer,
        'validation_result': {
            'correct_answer': result.get('correct_answer', ''),
            'validation': result.get('simple_validation', 'Invalid'),
            'score': result.get('score', 0),
            'feedback': result.get('feedback', '')
        },
        'fillers_used': result.get('fillers_used', [])
    })

# ============================================================
# ===== Video Interview (OPTIMIZED FOR RENDER) ==============
# ============================================================

@app.route('/video_interview')
def video_interview():
    return render_template('video_interview.html')

@app.route('/submit_video_answer/<qid>', methods=['POST'])
def submit_video_answer(qid):
    print(f"📹 Video submission Q{qid}")
    
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files['video']
    
    # Check file size BEFORE saving (max 15MB for free tier)
    file.seek(0, os.SEEK_END)
    file_size_bytes = file.tell()
    file.seek(0)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print(f"📦 Video size: {file_size_mb:.2f} MB")
    
    if file_size_mb > 15:
        return jsonify({"error": "Video too large. Max 1 minute (15MB)"}), 400
    
    # Save to /tmp (Render requirement)
    os.makedirs("/tmp/uploads", exist_ok=True)
    filepath = os.path.join("/tmp/uploads", f"answer_{qid}_{os.getpid()}.webm")
    
    try:
        file.save(filepath)
        print(f"✅ Video saved: {file_size_mb:.2f} MB")
    except Exception as e:
        print(f"❌ Save error: {e}")
        return jsonify({"error": f"Save failed: {str(e)}"}), 500

    # Get question
    questions = session.get('questions', [])
    try:
        question_text = questions[int(qid) - 1]
    except:
        question_text = "Interview question"

    # === TRANSCRIPTION with Whisper ===
    transcript = "Transcription unavailable"
    
    if WHISPER_ENABLED and whisper_model:
        try:
            print("🎤 Transcribing with Whisper tiny...")
            
            result = whisper_model.transcribe(
                filepath,
                fp16=False,
                language='en',
                verbose=False,
                condition_on_previous_text=False,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.6
            )
            
            transcript = result['text'].strip()
            
            if not transcript or len(transcript) < 5:
                transcript = "No clear speech detected"
            
            print(f"✅ Transcript: {len(transcript)} chars")
            
        except Exception as e:
            print(f"❌ Whisper error: {e}")
            transcript = "Transcription failed. Please upgrade hosting for better support."
    else:
        transcript = "Video recorded. Transcription disabled on free tier - upgrade to Starter plan for AI transcription."

    # Cleanup video immediately
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print("🗑️ Video deleted")
    except:
        pass

    # === COMPREHENSIVE EVALUATION with Gemini ===
    # Get both validation AND detailed scores
    validation_result = {
        "correct_answer": "",
        "validation": "Invalid",
        "score": 0,
        "feedback": "",
        "confidence_score": 0.70,
        "content_relevance": 0.70,
        "fluency_score": 0.70
    }

    # Only evaluate if we have real transcript
    if WHISPER_ENABLED and len(transcript) > 10 and "failed" not in transcript.lower() and "unavailable" not in transcript.lower():
        try:
            print("🤖 Comprehensive evaluation with Gemini...")
            
            prompt = f"""You are a strict technical interviewer evaluating a VIDEO interview answer.

Question: {question_text}
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

SCORING RULES:
- Valid (76-100%): Correct answer with accurate information
- Partial-High (50-75%): Related but incomplete
- Partial-Low (30-49%): Vague or mostly incorrect
- Invalid (0-29%): Wrong, off-topic, or nonsense

DETAILED SCORES (0.0-1.0):
- confidence_score: How confident/clear the speaker sounds
- content_relevance: How well answer addresses the question
- fluency_score: Speech clarity and coherence

Return ONLY the JSON object."""
            
            response = model.generate_content(prompt)
            raw_text = response.text.strip().replace('```json', '').replace('```', '').strip()
            json_match = re.search(r'\{.*?\}', raw_text, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group())
                
                # Normalize validation and score
                validation = result.get('validation', 'Invalid')
                score = result.get('score', 0)
                
                # Ensure score matches category
                if validation == 'Valid' and score < 76:
                    score = 76
                elif validation == 'Partial-High' and (score < 50 or score > 75):
                    score = 65
                elif validation == 'Partial-Low' and (score < 30 or score >= 50):
                    score = 40
                elif validation == 'Invalid' and score >= 30:
                    score = 15
                
                # Simplify validation for frontend
                if validation in ['Partial-High', 'Partial-Low']:
                    simple_validation = 'Partial'
                else:
                    simple_validation = validation
                
                validation_result = {
                    "correct_answer": result.get('correct_answer', ''),
                    "validation": simple_validation,
                    "score": score,
                    "feedback": result.get('feedback', ''),
                    "confidence_score": float(result.get('confidence_score', 0.70)),
                    "content_relevance": float(result.get('content_relevance', 0.70)),
                    "fluency_score": float(result.get('fluency_score', 0.70))
                }
                
                print(f"✅ Evaluation complete: Score {score}%, Validation: {simple_validation}")
                
        except Exception as e:
            print(f"⚠️ Evaluation error: {e}")
    else:
        # No valid transcript - return default scores
        validation_result = {
            "correct_answer": "Transcription unavailable",
            "validation": "Invalid",
            "score": 0,
            "feedback": "Could not transcribe video audio",
            "confidence_score": 0.0,
            "content_relevance": 0.0,
            "fluency_score": 0.0
        }

    print(f"📊 Final Score: {validation_result['score']}%")

    # Store in session
    results = session.get('results', [])
    results.append({
        "Q.ID": qid,
        "Question": question_text,
        "User Answer": transcript,
        "Score": validation_result['score'],
        "Validation": validation_result['validation'],
        "Feedback": validation_result['feedback'],
        "Confidence Score": validation_result['confidence_score'],
        "Content Relevance": validation_result['content_relevance'],
        "Fluency Score": validation_result['fluency_score']
    })
    session['results'] = results

    # Return unified format (same as text/speech + extra scores)
    return jsonify({
        'user_answer': transcript,
        'validation_result': {
            'correct_answer': validation_result['correct_answer'],
            'validation': validation_result['validation'],
            'score': validation_result['score'],
            'feedback': validation_result['feedback']
        },
        'fillers_used': detect_fillers(transcript).split(', ') if transcript else [],
        'extra_scores': {
            'confidence_score': validation_result['confidence_score'],
            'content_relevance': validation_result['content_relevance'],
            'fluency_score': validation_result['fluency_score']
        }
    })

@app.route('/result')
def result():
    results = session.get('results', [])
    job = session.get('job_title', 'N/A')
    difficulty = session.get('difficulty', 'N/A')
    return render_template('result.html', results=results, job_title=job, difficulty=difficulty)

@app.route('/get_results')
def get_results():
    results = session.get('results', [])
    return jsonify(results)

# Health check for Render
@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "whisper_enabled": WHISPER_ENABLED,
        "api_configured": bool(GOOGLE_API_KEY)
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
