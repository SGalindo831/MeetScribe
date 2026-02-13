from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
from werkzeug.utils import secure_filename
import threading
import json
import base64
import whisper
import requests

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024
app.config['ALLOWED_EXTENSIONS'] = {'mp3', 'wav', 'mp4', 'm4a', 'webm', 'ogg'}
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///meetings.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
socketio = SocketIO(app, cors_allowed_origins="*", max_http_buffer_size=10e8)

os.makedirs('uploads', exist_ok=True)
os.makedirs('transcriptions', exist_ok=True)
os.makedirs('summaries', exist_ok=True)
os.makedirs('temp_recordings', exist_ok=True)

processing_tasks = {}

# Load Whisper model once at startup
print("Loading Whisper model... This may take a moment on first run.")
whisper_model = whisper.load_model("base")
print("Whisper model loaded successfully!")

# Database Model
class Meeting(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    task_id = db.Column(db.String(100), unique=True, nullable=False)
    filename = db.Column(db.String(200), nullable=False)
    file_path = db.Column(db.String(300), nullable=False)
    status = db.Column(db.String(50), nullable=False)
    transcript = db.Column(db.Text)
    summary_overview = db.Column(db.Text)
    summary_data = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    recording_type = db.Column(db.String(20))
    
    def to_dict(self):
        summary = json.loads(self.summary_data) if self.summary_data else None
        return {
            'id': self.id,
            'task_id': self.task_id,
            'filename': self.filename,
            'status': self.status,
            'transcript': self.transcript,
            'summary': summary,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'recording_type': self.recording_type
        }

with app.app_context():
    db.create_all()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        unique_filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        
        file.save(filepath)
        
        task_id = timestamp
        
        meeting = Meeting(
            task_id=task_id,
            filename=unique_filename,
            file_path=filepath,
            status='uploaded',
            recording_type='upload'
        )
        db.session.add(meeting)
        db.session.commit()
        
        processing_tasks[task_id] = {'status': 'uploaded', 'filename': unique_filename}
        
        thread = threading.Thread(target=process_audio_file, args=(filepath, task_id))
        thread.start()
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': 'File uploaded successfully. Processing started.'
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/status/<task_id>')
def check_status(task_id):
    meeting = Meeting.query.filter_by(task_id=task_id).first()
    if meeting:
        return jsonify(meeting.to_dict())
    return jsonify({'error': 'Task not found'}), 404

@app.route('/meetings')
def get_meetings():
    meetings = Meeting.query.order_by(Meeting.created_at.desc()).all()
    return jsonify([meeting.to_dict() for meeting in meetings])

@app.route('/meeting/<int:meeting_id>')
def get_meeting(meeting_id):
    meeting = Meeting.query.get_or_404(meeting_id)
    return jsonify(meeting.to_dict())

@app.route('/meeting/<int:meeting_id>', methods=['DELETE'])
def delete_meeting(meeting_id):
    meeting = Meeting.query.get_or_404(meeting_id)
    
    try:
        if os.path.exists(meeting.file_path):
            os.remove(meeting.file_path)
    except Exception as e:
        print(f"Error deleting file: {e}")
    
    db.session.delete(meeting)
    db.session.commit()
    
    return jsonify({'success': True, 'message': 'Meeting deleted'})

def process_audio_file(filepath, task_id):
    with app.app_context():
        try:
            meeting = Meeting.query.filter_by(task_id=task_id).first()
            meeting.status = 'transcribing'
            db.session.commit()
            processing_tasks[task_id]['status'] = 'transcribing'
            
            print(f"Starting transcription for {filepath}")
            transcript = transcribe_audio(filepath)
            print(f"Transcription complete. Length: {len(transcript)} characters")
            
            transcript_path = f"transcriptions/{task_id}_transcript.txt"
            with open(transcript_path, 'w') as f:
                f.write(transcript)
            
            meeting.transcript = transcript
            meeting.status = 'summarizing'
            db.session.commit()
            processing_tasks[task_id]['status'] = 'summarizing'
            processing_tasks[task_id]['transcript'] = transcript
            
            print(f"Starting summarization for task {task_id}")
            summary = summarize_transcript(transcript)
            print(f"Summarization complete")
            
            summary_path = f"summaries/{task_id}_summary.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            meeting.summary_overview = summary['overview']
            meeting.summary_data = json.dumps(summary)
            meeting.status = 'completed'
            meeting.completed_at = datetime.utcnow()
            db.session.commit()
            
            processing_tasks[task_id]['status'] = 'completed'
            processing_tasks[task_id]['summary'] = summary
            
        except Exception as e:
            print(f"Error processing audio: {str(e)}")
            import traceback
            traceback.print_exc()
            meeting = Meeting.query.filter_by(task_id=task_id).first()
            if meeting:
                meeting.status = 'error'
                db.session.commit()
            
            processing_tasks[task_id]['status'] = 'error'
            processing_tasks[task_id]['error'] = str(e)

@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connection_response', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('start_recording')
def handle_start_recording(data):
    session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    meeting = Meeting(
        task_id=session_id,
        filename=f"{session_id}_recording.webm",
        file_path=f"uploads/{session_id}_recording.webm",
        status='recording',
        recording_type='live'
    )
    db.session.add(meeting)
    db.session.commit()
    
    processing_tasks[session_id] = {
        'status': 'recording',
        'type': 'live',
        'chunks': [],
        'audio_blobs': []
    }
    emit('recording_started', {'session_id': session_id})

@socketio.on('audio_data')
def handle_audio_data(data):
    """Receive complete audio blob from client"""
    session_id = data.get('session_id')
    audio_blob = data.get('audio_blob')
    
    if session_id in processing_tasks:
        try:
            # Decode base64 audio data
            audio_bytes = base64.b64decode(audio_blob.split(',')[1] if ',' in audio_blob else audio_blob)
            
            # Save to file
            filepath = f"uploads/{session_id}_recording.webm"
            with open(filepath, 'wb') as f:
                f.write(audio_bytes)
            
            print(f"Saved audio recording: {filepath} ({len(audio_bytes)} bytes)")
            emit('audio_saved', {'session_id': session_id, 'size': len(audio_bytes)})
            
        except Exception as e:
            print(f"Error saving audio: {str(e)}")
            emit('audio_error', {'error': str(e)})

@socketio.on('stop_recording')
def handle_stop_recording(data):
    session_id = data.get('session_id')
    
    if session_id in processing_tasks:
        with app.app_context():
            meeting = Meeting.query.filter_by(task_id=session_id).first()
            if meeting:
                meeting.status = 'processing'
                db.session.commit()
        
        processing_tasks[session_id]['status'] = 'processing'
        emit('recording_stopped', {'session_id': session_id})
        
        # Start processing in background thread
        thread = threading.Thread(target=process_live_recording, args=(session_id,))
        thread.start()

def process_live_recording(session_id):
    with app.app_context():
        try:
            filepath = f"uploads/{session_id}_recording.webm"
            
            # Check if file exists
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Recording file not found: {filepath}")
            
            meeting = Meeting.query.filter_by(task_id=session_id).first()
            meeting.status = 'transcribing'
            db.session.commit()
            processing_tasks[session_id]['status'] = 'transcribing'
            
            print(f"Transcribing live recording: {filepath}")
            transcript = transcribe_audio(filepath)
            print(f"Transcription complete: {len(transcript)} characters")
            
            meeting.transcript = transcript
            meeting.status = 'summarizing'
            db.session.commit()
            processing_tasks[session_id]['transcript'] = transcript
            processing_tasks[session_id]['status'] = 'summarizing'
            
            print(f"Summarizing transcript...")
            summary = summarize_transcript(transcript)
            print(f"Summarization complete")
            
            meeting.summary_overview = summary['overview']
            meeting.summary_data = json.dumps(summary)
            meeting.status = 'completed'
            meeting.completed_at = datetime.utcnow()
            db.session.commit()
            
            processing_tasks[session_id]['summary'] = summary
            processing_tasks[session_id]['status'] = 'completed'
            
            socketio.emit('processing_complete', {
                'session_id': session_id,
                'summary': summary,
                'transcript': transcript
            })
            
        except Exception as e:
            print(f"Error in live recording: {str(e)}")
            import traceback
            traceback.print_exc()
            meeting = Meeting.query.filter_by(task_id=session_id).first()
            if meeting:
                meeting.status = 'error'
                db.session.commit()
            
            processing_tasks[session_id]['status'] = 'error'
            processing_tasks[session_id]['error'] = str(e)

def transcribe_audio(filepath):
    """Use local Whisper to transcribe audio"""
    try:
        print(f"Transcribing {filepath} with Whisper...")
        result = whisper_model.transcribe(filepath)
        return result["text"]
    except Exception as e:
        print(f"Transcription error: {str(e)}")
        raise

def summarize_transcript(transcript):
    """Use Ollama with Llama3 to summarize transcript"""
    try:
        prompt = f"""Analyze this meeting transcript and provide a structured summary in JSON format.

Transcript:
{transcript}

Please provide a JSON response with exactly these fields:
1. "overview": A brief 2-3 sentence summary of the meeting
2. "key_points": An array of the main discussion points (3-6 items)
3. "action_items": An array of tasks or actions mentioned (if any)
4. "decisions": An array of decisions made during the meeting (if any)

IMPORTANT: Return ONLY the raw JSON object. Do not include any markdown formatting, code blocks, or explanatory text before or after the JSON."""


        response = requests.post('http://localhost:11434/api/generate',
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            },
            timeout=120)
        
        result = response.json()['response']
        
        try:
            if '```json' in result:
                result = result.split('```json').split('```').strip()[1]
            elif '```' in result:
                result = result.split('```')[1].split('```')[0].strip()
            
            summary = json.loads(result)
            
            if not all(k in summary for k in ['overview', 'key_points', 'action_items', 'decisions']):
                raise ValueError("Missing required fields")
            
            return summary
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON parse error: {e}")
            print(f"Raw response: {result}")
            # Extract the the content just from the JSON
            try:
                import re
                json_match = re.search(r'\{[\s\S]*\}', result)
                if json_match:
                    summary = json.loads(json_match.group())
                    return summary
            except:
                pass

            # fallback to basic structure
            return {
                'overview': 'See full transcript for details',
                'key_points': ['See full transcript'],
                'action_items': [],
                'decisions': []
            }
                
            
    except Exception as e:
        print(f"Summarization error: {str(e)}")
        return {
            'overview': f'Error generating summary: {str(e)}',
            'key_points': ['See full transcript'],
            'action_items': [],
            'decisions': []
        }

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Meeting Audio Summarizer - LOCAL AI VERSION")
    print("="*60)
    print("\n Using local Whisper for transcription")
    print(" Using Ollama + Llama3 for summarization")
    print("\n Database: SQLite (meetings.db)")
    print(" Opening server at: http://localhost:5001")
    print("\n Upload or record audio to get real AI summaries!")
    print("="*60 + "\n")
    socketio.run(app, debug=True, host='0.0.0.0', port=5001)