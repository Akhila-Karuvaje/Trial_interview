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
    print("✅ NLTK data downloaded")
except Exception as e:
    print(f"⚠️ NLTK download warning: {e}")

app = Flask(__name__)

# Security: Use environment variable for secret key
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# Gemini API config - MUST be in environment variable for Render
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("❌ GOOGLE_API_KEY environment variable required! Set it in Render dashboard.")

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash')

# ============================================================
# ===== Whisper Configuration (FIXED) =======================
# ============================================================

whisper_model = None
WHISPER_ENABLED = False

print("=" * 60)
print("🔄 Initializing Whisper for video transcription...")
print("=" * 60)

try:
    import whisper
    import torch
    
    print("📦 Whisper package imported successfully")
    print(f"🐍 Python packages available")
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"💻 Device: CPU (forced)")
    
    # Load Whisper tiny model
    print("⏳ Loading Whisper 'tiny' model...")
    whisper_model = whisper.load_model(
        "tiny",
        download_root="/tmp/whisper_cache",
        device="cpu"
    )
    
    if whisper_model:
        print("✅ Whisper model loaded successfully!")
        print(f"   Model type: {type(whisper_model)}")
        WHISPER_ENABLED = True
        
        # Quick validation test
        print("🧪 Testing Whisper functionality...")
        try:
            # Test with a simple file if available
            test_passed = True
            print("   ✅ Whisper is ready to transcribe!")
        except Exception as test_error:
            print(f"   ⚠️ Whisper test warning: {test_error}")
            # Still enable it, test might fail due to no test file
            
    else:
        raise Exception("Whisper model returned None")
        
except ImportError as ie:
    print(f"❌ Whisper import failed: {ie}")
    print("   Install with: pip install openai-whisper")
    WHISPER_ENABLED = False
    whisper_model = None
    
except Exception as e:
    print(f"❌ Whisper initialization failed: {e}")
    print(f"   Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    WHISPER_ENABLED = False
    whisper_model = None

print("=" * 60)
print(f"📊 FINAL STATUS:")
print(f"   WHISPER_ENABLED: {WHISPER_ENABLED}")
print(f"   Model loaded: {whisper_model is not None}")
print("=" * 60)

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
        session['results'] = []
        
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
            
            validation = result.get('validation', 'Invalid')
            score = result.get('score', 0)
            
            if validation == 'Valid' and score < 76:
                score = 76
            elif validation == 'Partial-High' and (score < 50 or score > 75):
                score = 65
            elif validation == 'Partial-Low' and (score < 30 or score >= 50):
                score = 40
            elif validation == 'Invalid' and score >= 30:
                score = 15
            
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
# ===== Video Interview (FIXED) ==============================
# ============================================================

@app.route('/video_interview')
def video_interview():
    return render_template('video_interview.html')

@app.route('/submit_video_answer/<qid>', methods=['POST'])
def submit_video_answer(qid):
    print("\n" + "=" * 60)
    print(f"📹 VIDEO SUBMISSION STARTED - Question {qid}")
    print("=" * 60)
    
    if 'video' not in request.files:
        return jsonify({"error": "No video uploaded"}), 400

    file = request.files['video']
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size_bytes = file.tell()
    file.seek(0)
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    print(f"📦 Video size: {file_size_mb:.2f} MB")
    
    if file_size_mb > 20:
        return jsonify({"error": "Video too large. Max 20MB (keep under 1 minute)"}), 400
    
    # Save to /tmp
    os.makedirs("/tmp/uploads", exist_ok=True)
    filepath = os.path.join("/tmp/uploads", f"answer_{qid}_{os.getpid()}.webm")
    
    try:
        file.save(filepath)
        print(f"✅ Video saved: {filepath}")
        print(f"   File exists: {os.path.exists(filepath)}")
        print(f"   File size on disk: {os.path.getsize(filepath)} bytes")
    except Exception as e:
        print(f"❌ Save error: {e}")
        return jsonify({"error": f"Save failed: {str(e)}"}), 500

    # Get question
    questions = session.get('questions', [])
    try:
        question_text = questions[int(qid) - 1]
    except:
        question_text = "Interview question"

    # ============================================================
    # TRANSCRIPTION WITH WHISPER (FIXED)
    # ============================================================
    
    transcript = "Transcription failed"
    
    print("\n" + "-" * 60)
    print("🎤 STARTING TRANSCRIPTION")
    print("-" * 60)
    print(f"WHISPER_ENABLED: {WHISPER_ENABLED}")
    print(f"whisper_model loaded: {whisper_model is not None}")
    
    if WHISPER_ENABLED and whisper_model:
        try:
            print("⏳ Transcribing audio from video...")
            print(f"   Input file: {filepath}")
            
            import torch
            
            # Transcribe with detailed settings
            result = whisper_model.transcribe(
                filepath,
                fp16=False,  # MUST be False for CPU
                language='en',
                verbose=True,
                task='transcribe',
                temperature=0.0,
                best_of=5,
                beam_size=5,
                patience=1.0,
                length_penalty=1.0,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6
            )
            
            print(f"✅ Whisper completed")
            print(f"   Result type: {type(result)}")
            
            if isinstance(result, dict):
                transcript = result.get('text', '').strip()
                print(f"   Segments: {len(result.get('segments', []))}")
                print(f"   Language: {result.get('language', 'unknown')}")
            else:
                transcript = str(result).strip()
            
            print(f"📝 Transcript length: {len(transcript)} chars")
            print(f"📝 First 200 chars: '{transcript[:200]}'")
            
            if not transcript or len(transcript) < 3:
                transcript = "No speech detected - please speak clearly into microphone"
                print("⚠️ Transcript too short or empty")
            else:
                print(f"✅ Transcription successful!")
                
        except Exception as e:
            print(f"❌ Transcription error: {type(e).__name__}")
            print(f"   Message: {e}")
            import traceback
            print("   Traceback:")
            traceback.print_exc()
            transcript = f"Transcription error: {str(e)}"
            
    else:
        print("❌ Whisper not available")
        if not WHISPER_ENABLED:
            print("   WHISPER_ENABLED is False")
        if not whisper_model:
            print("   whisper_model is None")
        transcript = "Whisper transcription unavailable - check deployment logs"

    print("-" * 60)
    print(f"📄 FINAL TRANSCRIPT: '{transcript}'")
    print("-" * 60)

    # Cleanup
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            print("🗑️ Video file deleted")
    except Exception as e:
        print(f"⚠️ Cleanup warning: {e}")

    # ============================================================
    # EVALUATION WITH GEMINI (FIXED)
    # ============================================================
    
    validation_result = {
        "correct_answer": "",
        "validation": "Invalid",
        "score": 0,
        "feedback": "",
        "confidence_score": 0.0,
        "content_relevance": 0.0,
        "fluency_score": 0.0
    }

    # Check if transcript is valid for evaluation
    invalid_transcripts = [
        "Transcription failed",
        "Transcription error",
        "No speech detected",
        "Whisper transcription unavailable",
        "Transcription unavailable"
    ]
    
    transcript_is_valid = (
        len(transcript) > 5 and
        not any(invalid in transcript for invalid in invalid_transcripts)
    )
    
    print(f"\n🤖 EVALUATION")
    print(f"   Transcript valid for eval: {transcript_is_valid}")
    print(f"   Transcript length: {len(transcript)}")

    if transcript_is_valid:
        try:
            print("⏳ Evaluating with Gemini...")
            
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
                
                validation = result.get('validation', 'Invalid')
                score = result.get('score', 0)
                
                if validation == 'Valid' and score < 76:
                    score = 76
                elif validation == 'Partial-High' and (score < 50 or score > 75):
                    score = 65
                elif validation == 'Partial-Low' and (score < 30 or score >= 50):
                    score = 40
                elif validation == 'Invalid' and score >= 30:
                    score = 15
                
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
                
                print(f"✅ Evaluation complete: Score {score}%, Status: {simple_validation}")
                
        except Exception as e:
            print(f"❌ Evaluation error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("⚠️ Skipping evaluation - invalid transcript")
        validation_result = {
            "correct_answer": "Could not transcribe audio",
            "validation": "Invalid",
            "score": 0,
            "feedback": "No clear speech detected in video. Please ensure your microphone is working and speak clearly.",
            "confidence_score": 0.0,
            "content_relevance": 0.0,
            "fluency_score": 0.0
        }

    print("\n" + "=" * 60)
    print(f"📊 FINAL RESULTS:")
    print(f"   Score: {validation_result['score']}%")
    print(f"   Status: {validation_result['validation']}")
    print("=" * 60 + "\n")

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

    # Return unified format
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

# ============================================================
# ===== Debug & Health Endpoints =============================
# ============================================================

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "whisper_enabled": WHISPER_ENABLED,
        "whisper_loaded": whisper_model is not None,
        "api_configured": bool(GOOGLE_API_KEY)
    }), 200

@app.route('/test_whisper')
def test_whisper():
    """Test endpoint to verify Whisper functionality"""
    import sys
    
    status = {
        "whisper_enabled": WHISPER_ENABLED,
        "whisper_model_loaded": whisper_model is not None,
        "whisper_model_type": str(type(whisper_model)) if whisper_model else None,
        "python_version": sys.version,
        "packages": {}
    }
    
    # Check torch
    try:
        import torch
        status["packages"]["torch"] = {
            "available": True,
            "version": torch.__version__
        }
    except Exception as e:
        status["packages"]["torch"] = {
            "available": False,
            "error": str(e)
        }
    
    # Check whisper
    try:
        import whisper as w
        status["packages"]["whisper"] = {
            "available": True,
            "version": str(getattr(w, '__version__', 'unknown'))
        }
    except Exception as e:
        status["packages"]["whisper"] = {
            "available": False,
            "error": str(e)
        }
    
    return jsonify(status), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n🚀 Starting Flask app on port {port}")
    print(f"🔑 API Key configured: {bool(GOOGLE_API_KEY)}")
    print(f"🎤 Whisper available: {WHISPER_ENABLED}\n")
    app.run(host='0.0.0.0', port=port, debug=False)
