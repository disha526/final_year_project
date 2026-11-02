import os
import io
import csv
import time
import datetime
import cv2
import numpy as np
import mysql.connector
from mysql.connector import Error
from flask import Flask, render_template, Response, request, jsonify, send_file, send_from_directory, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash

from crime_detector import CrimeDetector, CRIME_CLASSES
from video_manager import VideoManager
from email_notifier import email_notifier

app = Flask(__name__)
app.config['SECRET_KEY'] = 'sentinel-ai-yolo11-sky-blue-2026'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
app.config['ALERTS_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'alerts')
app.config['SAMPLE_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sample_videos')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['ALERTS_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'm4v'}

# =============================================================================
# MySQL Database Configuration & Initialization
# =============================================================================
MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
MYSQL_USER = os.environ.get('MYSQL_USER', 'root')
MYSQL_PASS = os.environ.get('MYSQL_PASSWORD', 'root')
MYSQL_DB = os.environ.get('MYSQL_DB', 'crime_surveillance_db')
MYSQL_PORT = int(os.environ.get('MYSQL_PORT', 3306))

def get_db_connection():
    """Returns a connected MySQL database connection or None if unavailable."""
    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASS,
            database=MYSQL_DB,
            port=MYSQL_PORT
        )
        return conn
    except Error as e:
        # Graceful fallback: log warning and continue without crashing server
        print(f"[MySQL Warning] Connection error: {e}")
        return None

def init_db():
    """Initializes MySQL database tables on server startup."""
    try:
        # First ensure database exists
        admin_conn = mysql.connector.connect(
            host=MYSQL_HOST,
            user=MYSQL_USER,
            password=MYSQL_PASS,
            port=MYSQL_PORT
        )
        admin_cur = admin_conn.cursor()
        admin_cur.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DB}`")
        admin_conn.commit()
        admin_cur.close()
        admin_conn.close()

        # Connect to database and create required schema
        conn = get_db_connection()
        if conn:
            cur = conn.cursor()
            cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                fullname VARCHAR(120) NOT NULL,
                email VARCHAR(120) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                role VARCHAR(50) DEFAULT 'Investigator',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS upload_history (
                id INT AUTO_INCREMENT PRIMARY KEY,
                filename VARCHAR(255) NOT NULL,
                dominant_crime VARCHAR(100),
                peak_confidence FLOAT,
                threat_level VARCHAR(50),
                incident_count INT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)

            cur.execute("""
            CREATE TABLE IF NOT EXISTS crime_incidents (
                id INT AUTO_INCREMENT PRIMARY KEY,
                video_source VARCHAR(255),
                crime_type VARCHAR(100),
                confidence FLOAT,
                severity VARCHAR(50),
                snapshot_path VARCHAR(255),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            conn.commit()
            cur.close()
            conn.close()
            print("[MySQL] Database & tables initialized successfully.")
    except Exception as e:
        print(f"[MySQL Initialization Error] {e}")

# Initialize Database Schema
init_db()

# Initialize AI Engine & Video Manager
detector = CrimeDetector(models_dir="models", alerts_dir="alerts")
video_manager = VideoManager(detector)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_standby_frame():
    """Generates a modern clean standby frame when stream is paused."""
    img = np.full((540, 960, 3), 248, dtype=np.uint8)  # Pristine light background
    # Subtle sky blue grid lines
    for y in range(0, 540, 40):
        cv2.line(img, (0, y), (960, y), (230, 242, 254), 1)
    for x in range(0, 960, 40):
        cv2.line(img, (x, 0), (x, 540), (230, 242, 254), 1)
        
    cv2.putText(img, "[ SURVEILLANCE FEED STANDBY ]", (270, 240), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (2, 132, 199), 2, cv2.LINE_AA)
    cv2.putText(img, "Choose a video file and click 'Predict' to start analysis", (210, 285), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (71, 85, 105), 1, cv2.LINE_AA)
    cv2.putText(img, "Or switch to Live Webcam for real-time camera surveillance", (215, 320), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (148, 163, 184), 1, cv2.LINE_AA)
    
    ret, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()

# =============================================================================
# Web Page Navigation Routes
# =============================================================================

@app.route('/')
def index():
    """Portfolio-style landing page with 3D effects & interactive system overview."""
    user = session.get('user')
    return render_template('index.html', crime_classes=CRIME_CLASSES, user=user)

@app.route('/upload')
def upload_page():
    """Dedicated video upload and prediction page (starts in clean refreshed standby)."""
    video_manager.stop_source()
    detector.reset_state(clear_incidents=True, clear_saved_report=True)
    email_notifier.reset_alert_guard()
    sample_files = []
    if os.path.exists(app.config['SAMPLE_FOLDER']):
        sample_files = [f for f in os.listdir(app.config['SAMPLE_FOLDER']) if f.endswith(('.mp4', '.avi', '.mov'))]
    return render_template('upload.html', samples=sample_files, crime_classes=CRIME_CLASSES, user=session.get('user'))

@app.route('/live')
def live_page():
    """Dedicated live webcam real-time prediction page (starts in clean refreshed standby)."""
    video_manager.stop_source()
    detector.reset_state(clear_incidents=True, clear_saved_report=True)
    email_notifier.reset_alert_guard()
    return render_template('live.html', crime_classes=CRIME_CLASSES, user=session.get('user'))

@app.route('/history')
def history_page():
    """Redirects to the unified video studio with real-time graphs."""
    return redirect(url_for('upload_page'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User Login route connected to MySQL."""
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()

        if not email or not password:
            return render_template('login.html', error="Please provide both email and password.")

        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor(dictionary=True)
                cur.execute("SELECT * FROM users WHERE email = %s", (email,))
                user = cur.fetchone()
                cur.close()
                conn.close()

                if user and check_password_hash(user['password_hash'], password):
                    session['user'] = {
                        'id': user['id'],
                        'fullname': user['fullname'],
                        'email': user['email'],
                        'role': user.get('role', 'Investigator')
                    }
                    return redirect(url_for('upload_page'))
                else:
                    return render_template('login.html', error="Invalid email address or password.")
            except Exception as e:
                return render_template('login.html', error=f"Database error: {e}")
        else:
            # Fallback demo login if MySQL is disconnected
            session['user'] = {'id': 1, 'fullname': 'Security Officer', 'email': email, 'role': 'Investigator'}
            return redirect(url_for('upload_page'))

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User Registration route connected to MySQL."""
    if request.method == 'POST':
        fullname = request.form.get('fullname', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        role = request.form.get('role', 'Investigator').strip()

        if not fullname or not email or not password:
            return render_template('signup.html', error="All fields are required.")

        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor(dictionary=True)
                cur.execute("SELECT id FROM users WHERE email = %s", (email,))
                if cur.fetchone():
                    cur.close()
                    conn.close()
                    return render_template('signup.html', error="An account with this email already exists.")

                pwd_hash = generate_password_hash(password)
                cur.execute(
                    "INSERT INTO users (fullname, email, password_hash, role) VALUES (%s, %s, %s, %s)",
                    (fullname, email, pwd_hash, role)
                )
                conn.commit()
                user_id = cur.lastrowid
                cur.close()
                conn.close()

                session['user'] = {'id': user_id, 'fullname': fullname, 'email': email, 'role': role}
                return redirect(url_for('upload_page'))
            except Exception as e:
                return render_template('signup.html', error=f"Registration failed: {e}")
        else:
            session['user'] = {'id': 1, 'fullname': fullname, 'email': email, 'role': role}
            return redirect(url_for('upload_page'))

    return render_template('signup.html')

@app.route('/logout')
def logout():
    """Logs out current user session."""
    session.clear()
    return redirect(url_for('login'))

# =============================================================================
# Video Stream & API Routes
# =============================================================================

@app.route('/video_feed')
def video_feed():
    """Multipart MJPEG video feed endpoint."""
    def frame_generator():
        standby_bytes = generate_standby_frame()
        while True:
            if video_manager.is_running:
                frame_bytes = video_manager.get_latest_frame_bytes()
                if frame_bytes:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    time.sleep(0.02)
            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + standby_bytes + b'\r\n')
                time.sleep(0.4)

    return Response(frame_generator(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/reset', methods=['POST'])
def reset_system():
    """Stops video playback and resets detector to clean standby."""
    video_manager.stop_source()
    detector.reset_state(clear_incidents=True, clear_saved_report=True)
    email_notifier.reset_alert_guard()
    return jsonify({"success": True, "message": "System reset to clean standby"})

@app.route('/api/source', methods=['POST'])
def set_source():
    """Switches active video source (sample, webcam, upload, url, or stop)."""
    data = request.json or {}
    source_type = data.get('source_type', 'sample')

    if source_type == 'webcam':
        cam_idx = int(data.get('camera_index', 0))
        detector.active_video_name = f"Live Camera Feed #{cam_idx}"
        email_notifier.reset_alert_guard()
        success = video_manager.start_source(cam_idx, source_type='webcam', camera_index=cam_idx)
        return jsonify({"success": success, "message": f"Webcam index {cam_idx} started", "source_type": "webcam"})

    elif source_type == 'sample':
        sample_name = data.get('sample_name', 'cctv_fight_1.mp4')
        sample_path = os.path.join(app.config['SAMPLE_FOLDER'], sample_name)
        if not os.path.exists(sample_path):
            return jsonify({"success": False, "error": f"Sample file {sample_name} not found"}), 404
        detector.active_video_name = sample_name
        detector.reset_state(clear_incidents=True)
        email_notifier.reset_alert_guard()
        success = video_manager.start_source(sample_path, source_type='sample')
        return jsonify({"success": success, "message": f"Sample {sample_name} loaded", "source_type": "sample"})

    elif source_type == 'url':
        url = data.get('url', '').strip()
        if not url:
            return jsonify({"success": False, "error": "URL parameter missing"}), 400
        detector.active_video_name = url
        email_notifier.reset_alert_guard()
        success = video_manager.start_source(url, source_type='url')
        return jsonify({"success": success, "message": "Stream URL connected", "source_type": "url"})

    elif source_type == 'stop':
        video_manager.stop_source()
        detector.reset_state()
        email_notifier.reset_alert_guard()
        return jsonify({"success": True, "message": "Video stream stopped and reset to standby", "source_type": "none"})

    return jsonify({"success": False, "error": "Unknown source type"}), 400

@app.route('/api/test_email', methods=['POST', 'GET'])
def test_email_alert():
    """Manually test or verify authorities email alert notification with location."""
    email_notifier.reset_alert_guard()
    success = email_notifier.trigger_crime_alert(
        crime_type="Fighting",
        confidence=95.0,
        severity="CRITICAL",
        snapshot_filename=None,
        alerts_dir=app.config['ALERTS_FOLDER'],
        source_name="Manual System Verification"
    )
    loc = email_notifier.get_surveillance_location()
    return jsonify({
        "success": success,
        "message": "Forensic test alert triggered to concerned authorities",
        "recipients": ["vishwa.977@gmail.com", "vishwastudies7@gmail.com"],
        "location": loc
    })

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handles video file upload, records to MySQL, and begins prediction."""
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"success": False, "error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        orig_filename = secure_filename(file.filename)
        timestamp = int(time.time())
        save_name = f"upload_{timestamp}_{orig_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], save_name)
        file.save(filepath)

        # Set active video name and reset analytics state for clean reporting
        detector.active_video_name = orig_filename
        detector.reset_state(clear_incidents=True, clear_saved_report=True)
        detector.active_video_name = orig_filename
        email_notifier.reset_alert_guard()

        success = video_manager.start_source(filepath, source_type='file')

        # Record upload event into MySQL
        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO upload_history (filename, threat_level) VALUES (%s, %s)",
                    (orig_filename, "ANALYZING")
                )
                conn.commit()
                cur.close()
                conn.close()
            except Exception as e:
                print(f"[MySQL Log Error] {e}")

        return jsonify({
            "success": success,
            "message": f"Uploaded and analyzing {orig_filename}",
            "filename": orig_filename,
            "source_type": "file"
        })
    else:
        return jsonify({"success": False, "error": "Invalid file format. Upload MP4, AVI, MOV, or MKV."}), 400

@app.route('/api/upload_analytics', methods=['GET'])
def get_upload_analytics():
    """Returns real-time distribution and area chart data ONLY for uploaded video analysis."""
    data = detector.get_video_analytics()
    # Update MySQL upload_history if connected
    kpis = data.get("kpis", {})
    if data.get("active_video") and kpis.get("total_points", 0) > 0:
        conn = get_db_connection()
        if conn:
            try:
                cur = conn.cursor()
                cur.execute(
                    """
                    UPDATE upload_history 
                    SET dominant_crime = %s, peak_confidence = %s, threat_level = %s, incident_count = %s 
                    WHERE filename = %s 
                    ORDER BY id DESC LIMIT 1
                    """,
                    (
                        kpis.get("dominant_crime", "Normal activity"),
                        kpis.get("peak_conf", 0.0),
                        kpis.get("threat_level", "SAFE"),
                        kpis.get("incident_count", 0),
                        data.get("active_video")
                    )
                )
                conn.commit()
                cur.close()
                conn.close()
            except Exception:
                pass

    data["is_running"] = video_manager.is_running
    data["source_type"] = video_manager.source_type
    return jsonify(data)

@app.route('/api/status', methods=['GET'])
def get_status():
    """Returns telemetry and real-time 11-category probabilities."""
    status = detector.get_status()
    status["source_type"] = video_manager.source_type
    status["is_running"] = video_manager.is_running
    status["camera_index"] = video_manager.camera_index
    status["active_video_name"] = detector.active_video_name
    return jsonify(status)

@app.route('/api/incidents', methods=['GET'])
def get_incidents():
    """Returns the recorded crime incident alerts."""
    return jsonify({
        "total": len(detector.incidents),
        "incidents": detector.incidents[:50]
    })

@app.route('/api/clear_incidents', methods=['POST'])
def clear_incidents():
    """Clears accumulated incident logs."""
    detector.incidents.clear()
    return jsonify({"success": True, "message": "Incident log cleared."})

@app.route('/api/export_incidents', methods=['GET'])
def export_incidents():
    """Exports incidents log as CSV."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Incident ID", "Timestamp", "Activity / Crime Type", "Severity", "Confidence (%)", "Snapshot Filename"])

    for inc in detector.incidents:
        writer.writerow([
            inc.get("id"),
            inc.get("timestamp"),
            inc.get("label"),
            inc.get("severity"),
            inc.get("confidence"),
            inc.get("snapshot")
        ])

    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment;filename=crime_incident_report.csv"}
    )

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Updates detector thresholds, switches, and alarm zones."""
    data = request.json or {}
    detector.set_config(
        conf=data.get('confidence'),
        violence=data.get('violence'),
        weapons=data.get('weapons'),
        intrusion=data.get('intrusion'),
        fallen=data.get('fallen'),
        bags=data.get('abandoned_bag'),
        zone=data.get('intrusion_zone')
    )
    return jsonify({"success": True, "config": detector.get_status()["config"]})

@app.route('/alerts/<path:filename>')
def serve_alert(filename):
    """Serves forensic snapshot evidence."""
    return send_from_directory(app.config['ALERTS_FOLDER'], filename)

@app.route('/sample_videos/<path:filename>')
def serve_sample(filename):
    """Serves sample test videos if folder exists."""
    if not os.path.exists(app.config['SAMPLE_FOLDER']):
        return "Sample folder not found", 404
    return send_from_directory(app.config['SAMPLE_FOLDER'], filename)

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  SENTINEL-AI: AUTONOMOUS CRIME DETECTION (YOLO11 & MySQL)")
    print("  Theme: Pure White & Sky Blue Portfolio Edition")
    print("="*70)
    print("  Web Portal: http://localhost:5000")
    print("  Navigation:")
    print("    • Portfolio / Home: http://localhost:5000/")
    print("    • Video Upload:     http://localhost:5000/upload")
    print("    • Live Webcam:      http://localhost:5000/live")
    print("    • Video Analytics:  http://localhost:5000/history")
    print("    • Login / Sign Up:  http://localhost:5000/login")
    print("="*70 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
