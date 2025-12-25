import os
import io
import math
import time
import json
import pickle
import base64
import re
import csv
import secrets
import hashlib
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from collections import defaultdict
from datetime import datetime, timedelta
from functools import wraps

# NEW: Load environment variables
from dotenv import load_dotenv
load_dotenv()

# NEW: Email support
from flask_mail import Mail, Message

# NEW: Rate limiting
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import jwt
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

# SECURITY: NO DEFAULT SECRETS - Fail fast if not provided
JWT_SECRET = os.getenv("JWT_SECRET")
SECRET_KEY = os.getenv("FLASK_SECRET_KEY")

if not JWT_SECRET or not SECRET_KEY:
    raise RuntimeError(
        "CRITICAL: JWT_SECRET and FLASK_SECRET_KEY must be set in environment variables. "
        "Run generate_secrets.py to create secure values."
    )

JWT_ALGO = "HS256"
JWT_EXP_HOURS = 1  # Reduced from 8 to 1 hour

# SECURITY: Database config from environment
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'database': os.getenv('DB_NAME', 'ehr_system'),
    'port': int(os.getenv('DB_PORT', 3306))
}

if not DB_CONFIG['user'] or not DB_CONFIG['password']:
    raise RuntimeError("DB_USER and DB_PASSWORD must be set in environment variables")

# Email configuration
MAIL_CONFIG = {
    'MAIL_SERVER': os.getenv('SMTP_HOST', 'smtp.gmail.com'),
    'MAIL_PORT': int(os.getenv('SMTP_PORT', 587)),
    'MAIL_USE_TLS': True,
    'MAIL_USERNAME': os.getenv('SMTP_USER'),
    'MAIL_PASSWORD': os.getenv('SMTP_PASSWORD'),
    'MAIL_DEFAULT_SENDER': os.getenv('SMTP_FROM', 'noreply@localhost')
}

APP_URL = os.getenv('APP_URL', 'http://localhost:5000')
UPLOAD_MAX_SIZE = int(os.getenv('UPLOAD_MAX_SIZE', 10485760))  # 10MB default

from flask import (
    Flask, request, jsonify, send_file, send_from_directory,
    render_template_string, abort, g, make_response,
    redirect
)


from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import numpy as np
from PIL import Image, ImageOps, ImageFilter
import cv2

try:
    from flask_cors import CORS
    cors_available = True
except Exception:
    cors_available = False

try:
    from bm3d import bm3d
    _BM3D_OK = True
except Exception:
    _BM3D_OK = False

try:
    import pydicom
    _PYDICOM_OK = True
except Exception:
    _PYDICOM_OK = False

try:
    from skimage.metrics import structural_similarity as ssim_fn
    _SSIM_OK = True
except Exception:
    ssim_fn = None
    _SSIM_OK = False

try:
    import requests
except Exception:
    requests = None

PROJECT_ROOT = Path(__file__).parent.resolve()
FRONTEND_DIR = PROJECT_ROOT / "frontend"
# DB_PATH is replaced by DB_CONFIG below
WORK_DIR = PROJECT_ROOT / "work_dir"
PKL_DIR = WORK_DIR / "pkl"
COMPARE_ALL_DIR = WORK_DIR / "comparison" / "all"
UPLOADS_DIR = PROJECT_ROOT / "uploads"
DATASET_DIR = PROJECT_ROOT / "Dataset"
UNSEEN_DIR = PROJECT_ROOT / "unseen_demo_images"
DEFAULT_CSV = PROJECT_ROOT / "EHR.csv"
ICD_JSON = PROJECT_ROOT / "icd10_lookup.json"

for p in (WORK_DIR, PKL_DIR, COMPARE_ALL_DIR, UPLOADS_DIR):
    p.mkdir(parents=True, exist_ok=True)

TARGET_SIZE: Tuple[int, int] = (256, 256)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "text-bison-001")
GEMINI_AUTH_BEARER = os.getenv("GEMINI_AUTH_BEARER")
GEMINI_BASE = os.getenv("GEMINI_BASE", "https://generativelanguage.googleapis.com/v1beta2")

SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "change-this-in-prod")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "dcm"}

app = Flask(__name__, static_folder=None)
app.secret_key = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = UPLOAD_MAX_SIZE

# Configure email
app.config.update(MAIL_CONFIG)
mail = Mail(app)

# Configure CORS properly
if cors_available:
    allowed_origins = os.getenv('ALLOWED_ORIGINS', APP_URL).split(',')
    CORS(app, 
         origins=allowed_origins,
         supports_credentials=True,
         methods=["GET", "POST", "PUT", "DELETE"],
         allow_headers=["Content-Type", "Authorization", "X-Auth-Token"])

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"],
    storage_uri="memory://"
)

# Security headers
@app.after_request
def set_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://cdn.jsdelivr.net; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src 'self' https://fonts.gstatic.com;"
    
    # Only set HSTS in production
    if os.getenv('FLASK_ENV') == 'production':
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    
    return response



def get_db_connection(dictionary=False):
    """
    Creates a new database connection.
    dictionary=True returns a cursor that yields dicts (like sqlite3.Row).
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        return conn
    except Error as e:
        print(f"Error connecting to MySQL: {e}")
        return None

def ensure_db_and_tables():
    """
    Create DB and tables if they don't exist.
    """
    try:
        
        conn = mysql.connector.connect(
            host=DB_CONFIG['host'],
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password']
        )
        cur = conn.cursor()
        cur.execute(f"CREATE DATABASE IF NOT EXISTS {DB_CONFIG['database']}")
        conn.commit()
        cur.close()
        conn.close()

        
        conn = get_db_connection()
        if not conn:
            return
        cur = conn.cursor()

        
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                email VARCHAR(255) NOT NULL UNIQUE,
                password_hash VARCHAR(255) NOT NULL,
                role ENUM('user', 'admin') DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Reports table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INT AUTO_INCREMENT PRIMARY KEY,
                user_email VARCHAR(255),
                patient_id VARCHAR(100),
                age INT,
                sex VARCHAR(10),
                symptoms TEXT,
                diagnosis TEXT,
                icd10 VARCHAR(20),
                note TEXT,
                image_side_by_side TEXT,
                psnr FLOAT,
                ssim FLOAT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        
        try:
            cur.execute("CREATE INDEX idx_reports_created_at ON reports(created_at)")
        except Error:
            pass 

        conn.commit()
        cur.close()
        conn.close()
        print("MySQL database and tables verified.")

    except Error as e:
        print("ensure_db_and_tables failed:", e)
def create_jwt(email, role):
    payload = {
        "email": email,
        "role": role,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXP_HOURS),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)

def decode_jwt_from_request():
    """Extract and validate JWT from request - SINGLE SOURCE ONLY"""
    token = None
    
    # SECURITY: Only accept from Authorization header (standard)
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        token = auth.split(" ", 1)[1]
    
    # Fallback to cookie for browser-based access
    if not token:
        token = request.cookies.get("token")
    
    if not token:
        return None
    
    # Check if token is blacklisted
    if is_token_blacklisted(token):
        log_security_event("blacklisted_token_used", None, {"token_preview": token[:10]})
        return None
    
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return payload
    except jwt.ExpiredSignatureError:
        log_security_event("expired_token_used", None)
        return None
    except jwt.InvalidTokenError:
        log_security_event("invalid_token_used", None)
        return None

from functools import wraps

def jwt_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        payload = decode_jwt_from_request()
        if not payload:
            return jsonify({"error": "authentication required"}), 401
        g.user_email = payload["email"]
        g.user_role = payload.get("role", "user")
        return fn(*args, **kwargs)
    return wrapper


def admin_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        payload = decode_jwt_from_request()
        if not payload or payload.get("role") != "admin":
            return jsonify({"error": "admin required"}), 403
        g.user_email = payload["email"]
        g.user_role = "admin"
        return fn(*args, **kwargs)
    return wrapper



@app.teardown_appcontext
def close_connection(exception):

    db = getattr(g, "_database", None)
    if db is not None:
        db.close()

# Ensure DB exists at startup
ensure_db_and_tables()

def create_user(email: str, password: str, role: str = "user") -> bool:
    pw_hash = generate_password_hash(password)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # MySQL uses %s placeholders
        cur.execute("INSERT INTO users (email, password_hash, role) VALUES (%s, %s, %s)", (email, pw_hash, role))
        conn.commit()
        cur.close()
        conn.close()
        return True
    except mysql.connector.IntegrityError:
        return False
    except Exception as e:
        print("create_user error:", e)
        return False

def authenticate_user(email: str, password: str) -> bool:
    conn = get_db_connection()
    if not conn: return False
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT * FROM users WHERE email = %s", (email,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if not row:
        return False
    return check_password_hash(row["password_hash"], password)

def save_history(user_email: Optional[str],
                 patient_id: Optional[str],
                 age: Optional[int],
                 sex: Optional[str],
                 symptoms: str,
                 diagnosis: str,
                 icd10: str,
                 note: str,
                 image_side_by_side: Optional[str],
                 psnr_val: Optional[float],
                 ssim_val: Optional[float],
                 extra: Optional[Dict] = None) -> int:
    metadata = json.dumps(extra or {})
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO reports
            (user_email, patient_id, age, sex, symptoms, diagnosis, icd10, note, image_side_by_side, psnr, ssim, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (user_email, patient_id, age, sex, symptoms, diagnosis, icd10, note, image_side_by_side, psnr_val, ssim_val, metadata))
        conn.commit()
        rowid = cur.lastrowid
        cur.close()
        conn.close()
        return int(rowid)
    except Exception as e:
        print("save_history failed:", e)
        return -1

# ---------- ICD lookup utilities ----------
def load_icd_lookup() -> Dict[str, str]:
    if ICD_JSON.exists():
        try:
            data = json.loads(ICD_JSON.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception as e:
            print("Failed to load ICD JSON:", e)
    return {
        "pneumonia": "J18.9",
        "diabetes": "E11.9",
        "hypertension": "I10",
        "coronary artery disease": "I25",
        "cardiomegaly": "I51.7",
        "intracranial hemorrhage": "I62",
        "hydrocephalus": "G91.9",
        "space-occupying lesion": "C71"
    }
ICD_LOOKUP = load_icd_lookup()

def icd_lookup_by_diagnosis(diagnosis: str) -> Optional[str]:
    if not diagnosis:
        return None
    diag = diagnosis.lower().strip()
    for k, v in ICD_LOOKUP.items():
        if k.lower() == diag:
            return v
    for k, v in ICD_LOOKUP.items():
        if k.lower() in diag or diag in k.lower():
            return v
    return None
# ============ NEW SECURITY HELPER FUNCTIONS ============

def hash_token(token: str) -> str:
    """Create SHA-256 hash of token for storage"""
    return hashlib.sha256(token.encode()).hexdigest()

def is_token_blacklisted(token: str) -> bool:
    """Check if token has been revoked"""
    token_hash = hash_token(token)
    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT id FROM token_blacklist WHERE token_hash = %s",
            (token_hash,)
        )
        result = cur.fetchone()
        cur.close()
        conn.close()
        return result is not None
    except Exception:
        return False

def blacklist_token(token: str, reason: str = "logout"):
    """Add token to blacklist"""
    token_hash = hash_token(token)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO token_blacklist (token_hash, reason) VALUES (%s, %s)",
            (token_hash, reason)
        )
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Failed to blacklist token: {e}")

def log_security_event(event_type: str, email: Optional[str] = None, details: Optional[Dict] = None):
    """Log security-related events"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO security_events (event_type, email, ip_address, details)
            VALUES (%s, %s, %s, %s)
        """, (
            event_type,
            email,
            request.remote_addr,
            json.dumps(details or {})
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Failed to log security event: {e}")

def log_admin_action(action: str, admin_email: str, target_email: Optional[str] = None, details: Optional[Dict] = None):
    """Log all admin actions for audit trail"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO admin_audit_log 
            (admin_email, action, target_email, details, ip_address, user_agent)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (
            admin_email,
            action,
            target_email,
            json.dumps(details or {}),
            request.remote_addr,
            request.headers.get('User-Agent', '')[:500]
        ))
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Failed to log admin action: {e}")

def is_valid_email(email: str) -> bool:
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def is_strong_password(password: str) -> Tuple[bool, str]:
    """Check password strength"""
    if len(password) < 12:
        return False, "Password must be at least 12 characters"
    
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain uppercase letter"
    
    if not re.search(r'[a-z]', password):
        return False, "Password must contain lowercase letter"
    
    if not re.search(r'\d', password):
        return False, "Password must contain a number"
    
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return False, "Password must contain special character"
    
    # Check against common passwords
    common_passwords = ['password', '12345678', 'qwerty', 'admin123']
    if password.lower() in common_passwords:
        return False, "Password is too common"
    
    return True, "Strong password"

def send_email(to: str, subject: str, body: str):
    """Send email (implement with your email service)"""
    if not MAIL_CONFIG['MAIL_USERNAME']:
        print(f"EMAIL NOT CONFIGURED - Would send to {to}:")
        print(f"Subject: {subject}")
        print(f"Body: {body}")
        return
    
    try:
        msg = Message(
            subject=subject,
            recipients=[to],
            body=body
        )
        mail.send(msg)
        log_security_event("email_sent", to, {"subject": subject})
    except Exception as e:
        print(f"Failed to send email: {e}")
        log_security_event("email_failed", to, {"error": str(e)})

# ---------------- image helper functions ----------------
def _resize_pil(img: Image.Image, target: Tuple[int,int]) -> Image.Image:
    h, w = target
    return img.resize((w, h), Image.BICUBIC)

def read_image_to_float(path: Path, target: Tuple[int,int] = TARGET_SIZE) -> np.ndarray:
    suffix = path.suffix.lower()
    try:
        if suffix == ".dcm" and _PYDICOM_OK:
            ds = pydicom.dcmread(str(path))
            arr = ds.pixel_array.astype(np.float32)
            arr -= arr.min()
            if arr.max() > 0:
                arr /= arr.max()
            pil = Image.fromarray((arr * 255.0).astype("uint8")).convert("L")
        else:
            pil = Image.open(path).convert("L")
        pil = ImageOps.exif_transpose(pil)
        pil = _resize_pil(pil, target)
        return np.array(pil).astype(np.float32) / 255.0
    except Exception as e:
        raise RuntimeError(f"read_image_to_float failed for {path}: {e}")

def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate PSNR, capping at 100.0 to avoid infinity issues"""
    if a.shape != b.shape:
        return 0.0
    mse = float(np.mean((a - b) ** 2))
    if mse == 0 or mse < 1e-10:
        return 100.0  
    result = 20.0 * math.log10(1.0 / math.sqrt(mse))
    return min(result, 100.0)  


def compute_ssim(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    if not _SSIM_OK or a.size == 0 or b.size == 0:
        return None
    try:
        return float(ssim_fn(a, b, data_range=1.0))
    except Exception:
        return None

def simple_image_features(arr: np.ndarray) -> str:
    if arr.size == 0:
        return ""
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    try:
        edges = cv2.Canny((arr * 255).astype("uint8"), 50, 150)
        edge_density = float(edges.mean())
    except Exception:
        edge_density = 0.0
    return f"Mean intensity={mean:.2f}; contrast(std)={std:.3f}; edge_density={edge_density:.3f}"

def detect_modality_from_path_or_meta(path: Optional[Path], arr: Optional[np.ndarray] = None) -> str:
    if path:
        s = str(path).lower()
        if "ct" in s:
            return "CT"
        if "mri" in s or "mr" in s:
            return "MRI"
    if arr is not None:
        if arr.std() > 0.12:
            return "MRI"
    return "Other"

# ---------------- Brain-aware image analysis ----------------
def analyze_image(arr: np.ndarray) -> Dict[str, float]:
    if arr is None or arr.size == 0:
        return {"hyper_frac": 0.0, "symmetry_score": 0.0, "ventricle_fraction": 0.0, "contrast": 0.0}

    a = np.clip(arr.astype(np.float32), 0.0, 1.0)
    H, W = a.shape

    hyper_thr = 0.85
    hyper_frac = float((a >= hyper_thr).sum()) / (H * W)

    mid = W // 2
    left = a[:, :mid]
    right = a[:, -mid:]
    if left.shape != right.shape:
        try:
            right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)
        except Exception:
            pass
    try:
        if left.std() == 0 or right.std() == 0:
            symmetry_score = 0.0
        else:
            symmetry_score = float(np.corrcoef(left.flatten(), right.flatten())[0, 1])
            symmetry_score = max(0.0, min(1.0, (symmetry_score + 1.0) / 2.0))
    except Exception:
        symmetry_score = 0.0

    cw = max(1, int(W * 0.30))
    ch = max(1, int(H * 0.30))
    sx = (W - cw) // 2
    sy = (H - ch) // 2
    central = a[sy:sy+ch, sx:sx+cw]
    ventricle_thr = 0.25 
    ventricle_fraction = float((central <= ventricle_thr).sum()) / (central.size + 1e-9)

    eps = 1e-6
    contrast_metric = float((a.max() - a.min()) / (a.mean() + eps))

    return {
        "hyper_frac": round(hyper_frac, 6),
        "symmetry_score": round(symmetry_score, 4),
        "ventricle_fraction": round(ventricle_fraction, 6),
        "contrast": round(contrast_metric, 4)
    }

# ---------------- Enhancement pipeline ----------------
def enhance_pipeline(arr: np.ndarray) -> np.ndarray:
    den = None
    if _BM3D_OK:
        try:
            den = bm3d(arr, sigma_psd=0.03)
        except Exception:
            den = None

    if den is None:
        try:
            im8 = (arr * 255.0).astype("uint8")
            den8 = cv2.fastNlMeansDenoising(im8, None, h=10, templateWindowSize=7, searchWindowSize=21)
            den = den8.astype(np.float32) / 255.0
        except Exception:
            den = None

    if den is None:
        den = np.array(Image.fromarray((arr * 255.0).astype("uint8")).filter(ImageFilter.MedianFilter(size=3))).astype(np.float32) / 255.0

    try:
        den8 = (den * 255.0).astype("uint8")
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        den8 = clahe.apply(den8)
        den = den8.astype(np.float32) / 255.0
    except Exception:
        pass

    try:
        pil = Image.fromarray((den * 255.0).astype("uint8"))
        pil = pil.filter(ImageFilter.UnsharpMask(radius=1.5, percent=180, threshold=2))
        den = np.array(pil).astype(np.float32) / 255.0
    except Exception:
        pass

    return np.clip(den, 0.0, 1.0)

# ---------------- Report templating ----------------
def render_report_template(patient_id: Optional[str],
                           age: Optional[int],
                           sex: Optional[str],
                           modality: str,
                           diagnosis_text: str,
                           metrics: Dict[str, float],
                           automated_findings: str) -> str:
    report_lines: List[str] = []
    hdr = f"Patient ID: {patient_id or 'Unknown'}"
    if age:
        hdr += f" | Age: {age}"
    if sex:
        hdr += f" | Gender: {sex}"
    hdr += f"\nReport date: {time.strftime('%Y-%m-%d')}\n"
    report_lines.append(hdr)
    report_lines.append("Clinical history: " + (diagnosis_text or "Not provided."))
    report_lines.append("")
    report_lines.append(f"Study / Modality: {modality}")
    report_lines.append("")
    report_lines.append("Technique: Standard " + modality + " protocol")
    report_lines.append("")

    disease_simple = "None (No acute abnormality detected)"
    hyper = metrics.get("hyper_frac", 0.0)
    sym = metrics.get("symmetry_score", 0.0)
    vent = metrics.get("ventricle_fraction", 0.0)
    contrast_metric = metrics.get("contrast", 0.0)

    if hyper > 0.0015:
        disease_simple = "Suspected acute hemorrhage / hyperdense lesion"
    elif sym < 0.55:
        disease_simple = "Asymmetric mass effect â€” suspicious for space-occupying lesion"
    elif vent > 0.20:
        disease_simple = "Enlarged ventricles (ventriculomegaly / atrophy)"
    elif contrast_metric < 0.8:
        disease_simple = "Low contrast â€” possible diffuse abnormality or poor image quality"

    report_lines.append(f"Disease Detected (automated): {disease_simple}")
    report_lines.append("")
    report_lines.append("Findings:")
    if automated_findings:
        report_lines.append("  - " + automated_findings)
    else:
        report_lines.append("  - Automated analysis performed; see quantitative analysis below.")
    report_lines.append("")
    report_lines.append("Quantitative analysis:")
    report_lines.append(f"  - Hyperintense fraction (>=0.85): {metrics.get('hyper_frac', 0.0):.6f}")
    report_lines.append(f"  - Symmetry score (0-1): {metrics.get('symmetry_score', 0.0):.4f}")
    report_lines.append(f"  - Ventricle proxy fraction (central low-intensity): {metrics.get('ventricle_fraction', 0.0):.6f}")
    report_lines.append(f"  - Contrast metric: {metrics.get('contrast', 0.0):.4f}")
    report_lines.append("")

    impression_lines: List[str] = []
    suggested_diag = diagnosis_text or "unspecified"
    suggested_icd = icd_lookup_by_diagnosis(suggested_diag) or "UNK"

    if hyper > 0.0015:
        impression_lines.append("Acute-appearing focal hyperdensity suspicious for hemorrhage or calcified lesion.")
        suggested_diag = "intracranial hemorrhage"
        suggested_icd = icd_lookup_by_diagnosis(suggested_diag) or "I62"
    elif sym < 0.55:
        impression_lines.append("Marked left-right asymmetry suggesting mass effect; consider urgent neuroimaging correlation.")
        suggested_diag = "space-occupying lesion"
        suggested_icd = icd_lookup_by_diagnosis(suggested_diag) or "C71"
    elif vent > 0.20:
        impression_lines.append("Ventricular enlargement (ventriculomegaly) relative to expected for age.")
        suggested_diag = "hydrocephalus"
        suggested_icd = icd_lookup_by_diagnosis(suggested_diag) or "G91.9"
    else:
        impression_lines.append("No acute intracranial abnormality detected on automated analysis.")

    report_lines.append("Impression:")
    for i, line in enumerate(impression_lines, 1):
        report_lines.append(f"  {i}. {line}")
    report_lines.append(f"Suggested diagnosis (automated): {suggested_diag}")
    report_lines.append(f"ICD-10 code (suggested): {suggested_icd}")
    report_lines.append("")
    report_lines.append("Recommendations:")
    report_lines.append("  - Correlate with clinical exam and prior imaging.")
    report_lines.append("  - Recommend urgent radiologist review for suspected acute findings.")
    report_lines.append("")
    report_lines.append("Automated note: This report was generated by an automated pipeline and requires clinical correlation.")
    return "\n".join(report_lines)

def generate_clinical_note_local(age, sex, symptoms, diagnosis, image_desc) -> str:
    metrics = {"hyper_frac": 0.0, "symmetry_score": 0.0, "ventricle_fraction": 0.0, "contrast": 0.0}
    return render_report_template(None, age, sex, "Unknown", diagnosis or symptoms, metrics, image_desc or "")

# ---------------- Gemini helpers ----------------
def gemini_generate_text(prompt: str, max_output_tokens: int = 800, temperature: float = 0.0) -> Optional[str]:
    if not (GEMINI_API_KEY or GEMINI_AUTH_BEARER) or requests is None:
        return None
    url = f"{GEMINI_BASE}/models/{GEMINI_MODEL}:generateText"
    headers = {"Content-Type": "application/json"}
    params = {}
    if GEMINI_AUTH_BEARER:
        headers["Authorization"] = f"Bearer {GEMINI_AUTH_BEARER}"
    else:
        params["key"] = GEMINI_API_KEY
    body = {"prompt": {"text": prompt}, "maxOutputTokens": max_output_tokens, "temperature": temperature}
    try:
        resp = requests.post(url, headers=headers, params=params, json=body, timeout=30)
        resp.raise_for_status()
        j = resp.json()
        if isinstance(j, dict):
            if "candidates" in j and isinstance(j["candidates"], list) and j["candidates"]:
                cand = j["candidates"][0]
                if isinstance(cand, dict):
                    for k in ("content", "output", "text", "output_text"):
                        if k in cand and isinstance(cand[k], str):
                            return cand[k]
                    if "content" in cand and isinstance(cand["content"], list) and cand["content"]:
                        first = cand["content"][0]
                        if isinstance(first, dict) and "text" in first:
                            return first["text"]
            for k in ("output", "text", "output_text"):
                if k in j and isinstance(j[k], str):
                    return j[k]
        return None
    except Exception as e:
        print("Gemini call error:", e)
        return None

def gemini_suggest_icd(diagnosis: str) -> Optional[str]:
    if not diagnosis or not (GEMINI_API_KEY or GEMINI_AUTH_BEARER):
        return None
    prompt = (
        "You are a clinical coding assistant. Given the diagnosis text below, "
        "suggest the single most appropriate ICD-10 code and a brief rationale.\n\n"
        f"Diagnosis: \"{diagnosis}\"\n\nAnswer:"
    )
    resp = gemini_generate_text(prompt, max_output_tokens=80, temperature=0.0)
    if not resp:
        return None
    m = re.search(r"\b([A-Z][0-9]{1,2}\.?[0-9]{0,2})\b", resp)
    return m.group(1) if m else None

# ---------------- Admin helpers & endpoints ----------------

def get_user_role(email: str) -> str:
    if not email:
        return "user"
    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT role FROM users WHERE LOWER(email)=%s", (email.lower(),))
        r = cur.fetchone()
        cur.close()
        conn.close()
        return (r["role"] if r and r["role"] else "user") if r else "user"
    except Exception:
        return "user"

def is_admin() -> bool:
    payload = decode_jwt_from_request()
    if not payload:
        return False
    return payload.get("role") == "admin"


@app.route("/api/admin/list_users", methods=["GET"])
@admin_required
def admin_list_users():

    if not is_admin():
        return jsonify({"error": "admin required"}), 403
    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT id, email, role, created_at FROM users ORDER BY created_at DESC")
        rows = cur.fetchall()
        users = [{"id": r["id"], "email": r["email"], "role": r["role"] or "user", "created_at": r["created_at"]} for r in rows]
        cur.close()
        conn.close()
        return jsonify({"users": users}), 200
    except Exception as e:
        print("admin_list_users error:", e)
        return jsonify({"error": "failed to list users", "detail": str(e)}), 500
# ---------------- Set user role endpoint ----------------
@app.route("/api/admin/set_role", methods=["POST"])
@admin_required
@limiter.limit("20 per hour")
def admin_set_role():
    data = request.get_json(silent=True) or request.form.to_dict() or {}
    email = (data.get("email") or "").strip().lower()
    role = (data.get("role") or "user").strip().lower()
    
    if not email or role not in ("user", "admin"):
        return jsonify({"error": "email and role ('user'|'admin') required"}), 400
    
    # Prevent self-demotion if last admin
    if email == g.user_email.lower() and role == "user":
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT COUNT(*) as cnt FROM users WHERE role = 'admin'")
        admin_count = cur.fetchone()['cnt']
        cur.close()
        conn.close()
        
        if admin_count <= 1:
            return jsonify({"error": "Cannot demote last admin"}), 400
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "UPDATE users SET role = %s WHERE LOWER(email) = %s",
            (role, email)
        )
        changed = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        
        if changed == 0:
            return jsonify({"error": "user not found"}), 404
        
        # AUDIT LOG
        log_admin_action(
            action="role_change",
            admin_email=g.user_email,
            target_email=email,
            details={"new_role": role}
        )
        
        return jsonify({"ok": True, "email": email, "role": role}), 200
        
    except Exception as e:
        print("admin_set_role error:", e)
        return jsonify({"error": "failed to set role"}), 500
# ---------------- Report deletion endpoints ----------------
@app.route("/api/admin/delete_report/<int:rowid>", methods=["DELETE"])
def admin_delete_report(rowid):
    if not is_admin():
        return jsonify({"error": "admin required"}), 403
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM reports WHERE id = %s", (rowid,))
        conn.commit()
        deleted = cur.rowcount
        cur.close()
        conn.close()
        if deleted == 0:
            return jsonify({"error": "report not found"}), 404
        return jsonify({"ok": True, "deleted_report_id": rowid}), 200
    except Exception as e:
        print("admin_delete_report error:", e)
        return jsonify({"error": "failed to delete report", "detail": str(e)}), 500

@app.route("/api/admin/delete_user/<string:email>", methods=["DELETE"])
@admin_required
@limiter.limit("10 per hour")
def admin_delete_user_reports(email):
    target = email.strip().lower()
    if not target:
        return jsonify({"error": "email required"}), 400
    
    # Prevent deleting own reports
    if target == g.user_email.lower():
        return jsonify({"error": "Cannot delete your own reports"}), 400
    
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get count before deletion
        cur.execute("SELECT COUNT(*) as cnt FROM reports WHERE LOWER(user_email) = %s", (target,))
        count_before = cur.fetchone()[0]
        
        # Delete reports
        cur.execute("DELETE FROM reports WHERE LOWER(user_email) = %s", (target,))
        count = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        
        # AUDIT LOG
        log_admin_action(
            action="delete_user_reports",
            admin_email=g.user_email,
            target_email=target,
            details={"deleted_count": count, "count_before": count_before}
        )
        
        return jsonify({
            "ok": True, 
            "deleted_reports_count": count, 
            "user": target
        }), 200
        
    except Exception as e:
        print("admin_delete_user_reports error:", e)
        log_admin_action(
            action="delete_user_reports_failed",
            admin_email=g.user_email,
            target_email=target,
            details={"error": str(e)}
        )
        return jsonify({"error": "failed to delete user reports"}), 500

@app.route("/api/admin/clear_reports", methods=["DELETE"])
def admin_clear_reports():
    if not is_admin():
        return jsonify({"error": "admin required"}), 403
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM reports")
        count = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"ok": True, "deleted_reports_count": count}), 200
    except Exception as e:
        print("admin_clear_reports error:", e)
        return jsonify({"error": "failed to clear reports", "detail": str(e)}), 500

# ---------------- Frontend serving + auth routes ----------------
def _load_frontend_page(filename: str):
    HTML_PATH = FRONTEND_DIR / filename
    if not HTML_PATH.exists():
        return f"{filename} not found in {FRONTEND_DIR}", 404
    try:
        content = HTML_PATH.read_text(encoding="utf-8")
        return content, 200
    except Exception as e:
        return f"Error reading {filename}: {e}", 500

INDEX_HTML = """
<!doctype html>
<html>
<head><meta charset="utf-8"><title>EHR Assistant</title></head>
<body>
  <h2>EHR Clinical Assistant</h2>
  <ul>
    <li><a href="/frontend/index.html">Single Image Clinical Note Generator</a></li>
    <li><a href="/frontend/bulk.html">Bulk CSV Processor</a></li>
    <li><a href="/dashboard">Dashboard</a></li>
    <li><a href="/health">Health Check (JSON)</a></li>
  </ul>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def app_index():
    content, status = _load_frontend_page("home.html")
    if status == 200:
        return render_template_string(content, root=str(PROJECT_ROOT))
    return INDEX_HTML

@app.route("/frontend/<path:filename>")
def serve_frontend(filename):
    if not FRONTEND_DIR.exists():
        return ("Frontend folder not found. Create ./frontend and add required files.", 404)
    return send_from_directory(str(FRONTEND_DIR), filename)

@app.route("/favicon.ico")
def favicon():
    return ("", 204)

@app.route("/dashboard", methods=["GET"])
def dashboard():
    content, status = _load_frontend_page("dashboard.html")
    if status == 200:
        return render_template_string(content)
    return content, status
# ---------------- Registration route ----------------
@app.route("/register", methods=["GET", "POST"])
@limiter.limit("5 per hour")  # Prevent registration spam
def register_route():
    if request.method == "GET":
        content, status = _load_frontend_page("register.html")
        if status == 200:
            return render_template_string(content)
        return content, status

    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""
    password2 = request.form.get("password2") or ""

    # Validation
    if not email or not password:
        log_security_event("registration_failed", email, {"reason": "missing_fields"})
        return jsonify({"error": "Email and password required"}), 400
    
    if not is_valid_email(email):
        return jsonify({"error": "Invalid email format"}), 400

    if password != password2:
        return jsonify({"error": "Passwords do not match"}), 400
    
    # Check password strength
    is_strong, message = is_strong_password(password)
    if not is_strong:
        return jsonify({"error": message}), 400

    ok = create_user(email, password)
    if not ok:
        log_security_event("registration_failed", email, {"reason": "user_exists"})
        return jsonify({"error": "User already exists"}), 400

    log_security_event("user_registered", email)
    return jsonify({"ok": True, "message": "Registration successful"}), 200
# ---------------- Login / Logout routes ----------------
@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute")  # Prevent brute force
def login_route():
    if request.method == "GET":
        content, status = _load_frontend_page("login.html")
        if status == 200:
            return render_template_string(content)
        return content, status

    email = (request.form.get("email") or "").strip().lower()
    password = request.form.get("password") or ""

    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400

    # Check account lockout
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT failed_login_attempts, account_locked_until 
        FROM users WHERE LOWER(email) = %s
    """, (email,))
    user = cur.fetchone()
    
    if user:
        locked_until = user.get('account_locked_until')
        if locked_until and locked_until > datetime.now():
            cur.close()
            conn.close()
            log_security_event("login_attempt_locked", email)
            return jsonify({
                "error": f"Account locked until {locked_until.strftime('%H:%M:%S')}"
            }), 403
    
    cur.close()
    conn.close()

    # Authenticate
    if not authenticate_user(email, password):
        # Increment failed attempts
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            UPDATE users 
            SET failed_login_attempts = failed_login_attempts + 1,
                account_locked_until = CASE 
                    WHEN failed_login_attempts + 1 >= 5 
                    THEN NOW() + INTERVAL 15 MINUTE 
                    ELSE NULL 
                END
            WHERE LOWER(email) = %s
        """, (email,))
        conn.commit()
        cur.close()
        conn.close()
        
        log_security_event("login_failed", email)
        return jsonify({"error": "Invalid credentials"}), 401

    # Reset failed attempts on successful login
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        UPDATE users 
        SET failed_login_attempts = 0, 
            account_locked_until = NULL,
            last_login = NOW()
        WHERE LOWER(email) = %s
    """, (email,))
    conn.commit()
    cur.close()
    conn.close()

    role = get_user_role(email)
    token = create_jwt(email, role)

    log_security_event("login_success", email)

    # SECURITY: Set secure cookie
    resp = jsonify({
        "email": email,
        "role": role,
        "token": token  # Also return in response for mobile apps
    })
    
    # Secure cookie settings
    is_production = os.getenv('FLASK_ENV') == 'production'
    resp.set_cookie(
        'token', 
        token, 
        httponly=True, 
        secure=is_production,  # Only HTTPS in production
        samesite='Strict',  # Prevent CSRF
        max_age=JWT_EXP_HOURS * 3600
    )
    
    return resp, 200
# ---------------- Logout route ----------------

@app.route("/logout", methods=["POST"])
def logout_route():
    return jsonify({"ok": True, "message": "Logout handled client-side"}), 200

@app.route("/api/whoami")
@jwt_required
def whoami():
    return jsonify({
        "email": g.user_email,
        "role": g.user_role
    })
# ---------------- Serve work_dir and uploads ----------------
@app.route('/work_dir/<path:filename>')
def serve_work_dir(filename):
    try:
        full = (WORK_DIR / filename).resolve()
        if not str(full).startswith(str(WORK_DIR.resolve())):
            return abort(403)
        return send_file(full)
    except FileNotFoundError:
        return abort(404)
    except Exception:
        return abort(500)

@app.route("/uploads/<path:filename>", methods=["GET"])
def uploaded_file(filename):
    safe_name = os.path.normpath(filename)
    return send_from_directory(str(UPLOADS_DIR), safe_name)

# ---------------- Health endpoint ----------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "gemini_configured": bool(GEMINI_API_KEY or GEMINI_AUTH_BEARER),
        "bm3d_available": _BM3D_OK,
        "pydicom_available": _PYDICOM_OK,
        "ssim_available": _SSIM_OK,
        "dataset_dir_exists": DATASET_DIR.exists(),
        "unseen_dir_exists": UNSEEN_DIR.exists(),
        "default_csv_exists": DEFAULT_CSV.exists()
    })

# ---------------- API: history listing ----------------
@app.route("/api/history", methods=["GET"])
@jwt_required
def api_history():
    try:
        page = max(1, int(request.args.get("page", 1)))
        per_page = min(200, max(10, int(request.args.get("per_page", 20))))
    except Exception:
        page, per_page = 1, 20

    q = (request.args.get("q") or "").strip().lower()
    want_all = (request.args.get("all") or "").lower() in ("1", "true", "yes")
    user_filter_param = (request.args.get("user") or "").strip().lower()

    #  JWT identity
    user_email = g.user_email
    user_role = g.user_role

    #  USER FILTER LOGIC
    if user_role == "admin":
        if user_filter_param:
            user_filter = user_filter_param
        elif want_all:
            user_filter = ""
        else:
            user_filter = user_email.lower()
    else:
        # normal users can ONLY see their own data
        user_filter = user_email.lower()

    offset = (page - 1) * per_page
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)

    base_sql = """
        SELECT id, user_email, patient_id, created_at, age, sex,
               symptoms, diagnosis, icd10, note,
               image_side_by_side, psnr, ssim, metadata
        FROM reports
    """

    where_clauses = []
    params = []

    if q:
        where_clauses.append(
            "(LOWER(symptoms) LIKE %s OR LOWER(diagnosis) LIKE %s OR LOWER(note) LIKE %s)"
        )
        qparam = f"%{q}%"
        params.extend([qparam, qparam, qparam])

    if user_filter:
        where_clauses.append("LOWER(user_email) = %s")
        params.append(user_filter)

    if where_clauses:
        base_sql += " WHERE " + " AND ".join(where_clauses)

    # Count
    count_sql = f"SELECT COUNT(*) as cnt FROM ({base_sql}) as sub"
    cur.execute(count_sql, tuple(params))
    total = int(cur.fetchone()["cnt"] or 0)

    # Pagination
    base_sql += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
    params.extend([per_page, offset])
    cur.execute(base_sql, tuple(params))
    rows = cur.fetchall()

    items = []
    for r in rows:
        try:
            metadata = json.loads(r["metadata"]) if r["metadata"] else {}
        except Exception:
            metadata = {}

        items.append({
            "id": r["id"],
            "user_email": r["user_email"],
            "patient_id": r["patient_id"],
            "created_at": r["created_at"],
            "age": r["age"],
            "sex": r["sex"],
            "symptoms": r["symptoms"],
            "diagnosis": r["diagnosis"],
            "icd10": r["icd10"],
            "note": r["note"],
            "image_side_by_side": r["image_side_by_side"],
            "psnr": r["psnr"],
            "ssim": r["ssim"],
            "metadata": metadata
        })

    cur.close()
    conn.close()

    return jsonify({
        "page": page,
        "per_page": per_page,
        "total": total,
        "items": items
    }), 200


# ---------------- Bulk CSV processing ----------------
@app.route("/process_patient_csv", methods=["POST"])
@jwt_required
def process_patient_csv():
    user_email = g.user_email

    file = request.files.get("file")
    patients = []

    # Parse CSV file or use default
    if file:
        try:
            import pandas as pd
            df = pd.read_csv(file)
            for _, r in df.iterrows():
                # Handle age parsing - skip problematic values
                age_val = None
                try:
                    if not pd.isna(r.get("age")):
                        age_str = str(r.get("age")).strip()
                        # Remove non-numeric characters like '> 89'
                        age_str = ''.join(c for c in age_str if c.isdigit())
                        if age_str:
                            age_val = int(age_str)
                except:
                    age_val = None
                    
                patients.append({
                    "age": age_val,
                    "sex": str(r.get("sex")) if not pd.isna(r.get("sex")) else None,
                    "symptoms": str(r.get("symptoms")) if not pd.isna(r.get("symptoms")) else "",
                    "diagnosis": str(r.get("diagnosis")) if not pd.isna(r.get("diagnosis")) else "",
                    "image": r.get("image") if "image" in df.columns and not pd.isna(r.get("image")) else None
                })
        except Exception as e:
            print(f"Pandas failed, trying fallback CSV parser: {e}")
            try:
                stream = io.StringIO(file.stream.read().decode("utf-8", errors="replace"))
                reader = csv.DictReader(stream)
                for r in reader:
                    age_val = None
                    try:
                        age = r.get("age") or r.get("Age")
                        if age:
                            # Remove non-numeric characters
                            age_str = ''.join(c for c in str(age).strip() if c.isdigit())
                            if age_str:
                                age_val = int(age_str)
                    except:
                        age_val = None
                        
                    patients.append({
                        "age": age_val,
                        "sex": (r.get("sex") or r.get("Sex") or "").strip() or None,
                        "symptoms": (r.get("symptoms") or "").strip(),
                        "diagnosis": (r.get("diagnosis") or "").strip(),
                        "image": (r.get("image") or "").strip() or None
                    })
            except Exception as e2:
                return jsonify({"error": f"CSV parse failed: {e2}"}), 400
    else:
        # Use default CSV
        if DEFAULT_CSV.exists():
            try:
                import pandas as pd
                df = pd.read_csv(DEFAULT_CSV)
                for _, r in df.iterrows():
                    age_val = None
                    try:
                        if not pd.isna(r.get("age")):
                            age_str = str(r.get("age")).strip()
                            age_str = ''.join(c for c in age_str if c.isdigit())
                            if age_str:
                                age_val = int(age_str)
                    except:
                        age_val = None
                        
                    patients.append({
                        "age": age_val,
                        "sex": str(r.get("sex")) if not pd.isna(r.get("sex")) else None,
                        "symptoms": str(r.get("symptoms")) if not pd.isna(r.get("symptoms")) else "",
                        "diagnosis": str(r.get("diagnosis")) if not pd.isna(r.get("diagnosis")) else "",
                        "image": r.get("image") if "image" in df.columns and not pd.isna(r.get("image")) else None
                    })
            except Exception as e:
                return jsonify({"error": f"Failed to read default CSV: {e}"}), 400
        else:
            return jsonify({"error": "No CSV file provided and no default CSV found"}), 400

    print(f"Processing {len(patients)} patients")
    results = []
    H, W = TARGET_SIZE

    for idx, pat in enumerate(patients):
        print(f"Processing patient {idx}: {pat}")
        
        age = pat.get("age")
        sex = pat.get("sex")
        symptoms = pat.get("symptoms", "")
        diagnosis = pat.get("diagnosis", "")
        image_field = pat.get("image")

        # Initialize arrays
        arr = np.zeros((H, W), dtype=np.float32)
        enhanced = arr.copy()
        image_url = None
        image_desc = ""

        # Process image if provided
        if image_field and image_field.strip():
            try:
                img_path = Path(image_field.strip())
                
                paths_to_try = [
                    img_path,
                    PROJECT_ROOT / img_path,
                    DATASET_DIR / img_path.name,
                    UNSEEN_DIR / img_path.name,
                ]
                
                found_path = None
                for try_path in paths_to_try:
                    if try_path.exists():
                        found_path = try_path
                        break
                
                if found_path:
                    arr = read_image_to_float(found_path, TARGET_SIZE)
                    enhanced = enhance_pipeline(arr)
                    image_desc = simple_image_features(enhanced)

                    sbs = Image.new("L", (W * 2, H))
                    sbs.paste(Image.fromarray((arr * 255).astype("uint8")), (0, 0))
                    sbs.paste(Image.fromarray((enhanced * 255).astype("uint8")), (W, 0))

                    fname = f"bulk_{idx:06d}_{int(time.time())}.png"
                    sbs_path = COMPARE_ALL_DIR / fname
                    sbs.save(sbs_path)
                    image_url = f"/work_dir/comparison/all/{fname}"
                    
            except Exception as e:
                print(f"Error processing image for row {idx}: {e}")
        metrics = analyze_image(enhanced)
        patient_id = f"patient_{idx:06d}_{int(time.time())}"
        modality = detect_modality_from_path_or_meta(
            Path(image_field) if image_field else None, 
            enhanced
        )
        
        note = render_report_template(
            patient_id=patient_id,
            age=age,
            sex=sex,
            modality=modality,
            diagnosis_text=diagnosis or symptoms,
            metrics=metrics,
            automated_findings=image_desc
        )
        # Determine ICD-10 code
        icd = None
        if diagnosis:
            icd = icd_lookup_by_diagnosis(diagnosis)
        
        if not icd:
            hyper = metrics.get("hyper_frac", 0.0)
            sym = metrics.get("symmetry_score", 0.0)
            vent = metrics.get("ventricle_fraction", 0.0)
            
            if hyper > 0.0015:
                icd = "I62"
            elif sym < 0.55:
                icd = "C71"
            elif vent > 0.20:
                icd = "G91.9"
            else:
                icd = "Z00.00"
        p_val = None
        s_val = None
        if arr.size > 0 and enhanced.size > 0:
            try:
                p_val = psnr(arr, enhanced)  # Now capped at 100
                s_val = compute_ssim(arr, enhanced)
                
            except Exception as e:
                print(f"Error calculating metrics: {e}")
        hyper = metrics.get("hyper_frac", 0.0)
        sym = metrics.get("symmetry_score", 0.0)
        vent = metrics.get("ventricle_fraction", 0.0)
        
        disease_present = "No"
        disease_name = "None (No acute abnormality detected)"
        
        if hyper > 0.0015:
            disease_present = "Yes"
            disease_name = "Suspected acute hemorrhage / hyperdense lesion"
        elif sym < 0.55:
            disease_present = "Yes"
            disease_name = "Asymmetric mass effect â€“ suspicious for space-occupying lesion"
        elif vent > 0.20:
            disease_present = "Yes"
            disease_name = "Enlarged ventricles (ventriculomegaly / atrophy)"

        # Save to database
        try:
            rowid = save_history(
                user_email=user_email,
                patient_id=patient_id,
                age=age,
                sex=sex,
                symptoms=symptoms,
                diagnosis=diagnosis,
                icd10=icd,
                note=note,
                image_side_by_side=image_url,
                psnr_val=p_val,
                ssim_val=s_val,
                extra={"metrics": metrics, "disease_simple": disease_name}
            )
            
        except Exception as e:
            print(f"Error saving to history: {e}")
            rowid = -1

        # Build result
        results.append({
            "index": idx,
            "history_rowid": rowid,
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "symptoms": symptoms,
            "diagnosis": diagnosis,
            "icd10": icd,
            "clinical_note": note,
            "disease_name": disease_name,
            "disease_present": disease_present,
            "image_side_by_side": image_url,
            "psnr": p_val,
            "ssim": s_val,
            "metrics": metrics
        })
    
    # IMPORTANT: Return JSON response
    return jsonify({
        "count": len(results), 
        "results": results,
        "success": True
    }), 200

@app.route("/reports/pdf/<int:rowid>")
@jwt_required
def export_report_pdf(rowid):
    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM reports WHERE id = %s", (rowid,))
        r = cur.fetchone()
        cur.close()
        conn.close()

        if not r:
            return abort(404)

        #  Security check
        if g.user_role != "admin" and r["user_email"] != g.user_email:
            return abort(403)

        buffer = io.BytesIO()
        pdf = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        y = height - 40

        def draw(text):
            nonlocal y
            pdf.drawString(40, y, text)
            y -= 14

        pdf.setFont("Helvetica-Bold", 14)
        draw("EHR Clinical Report")
        y -= 10

        pdf.setFont("Helvetica", 10)
        draw(f"Patient ID: {r['patient_id']}")
        draw(f"Age: {r['age']} | Sex: {r['sex']}")
        draw(f"ICD-10: {r['icd10']}")
        draw(f"Created: {r['created_at']}")
        y -= 10

        draw("Clinical Note:")
        y -= 10
        for line in (r["note"] or "").split("\n"):
            if y < 40:
                pdf.showPage()
                y = height - 40
            draw(line)

        pdf.showPage()
        pdf.save()
        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=f"report_{rowid}.pdf"
        )

    except Exception as e:
        print("PDF export error:", e)
        return abort(500)

# ---------------- Single image processing ----------------
@app.route("/process_single", methods=["POST"])
@jwt_required
def process_single():
    user_email = g.user_email
    user_role = g.user_role

    data = request.form
    age = data.get("age")
    sex = data.get("sex")
    symptoms = data.get("symptoms", "")
    diagnosis = data.get("diagnosis", "")

    img_file = request.files.get("image")
    H, W = TARGET_SIZE

    arr = np.zeros((H, W), dtype=np.float32)
    enhanced = arr.copy()
    image_url = None
    image_desc = ""

    if img_file:
        try:
            # Read and process the image
            pil = Image.open(img_file.stream).convert("L")
            pil = ImageOps.exif_transpose(pil)
            pil = pil.resize((W, H))
            arr = np.array(pil).astype(np.float32) / 255.0
            
            # Apply enhancement pipeline
            enhanced = enhance_pipeline(arr)

            # Create side-by-side comparison
            sbs = Image.new("L", (W * 2, H))
            sbs.paste(Image.fromarray((arr * 255).astype("uint8")), (0, 0))
            sbs.paste(Image.fromarray((enhanced * 255).astype("uint8")), (W, 0))

            fname = f"{int(time.time())}.png"
            sbs.save(COMPARE_ALL_DIR / fname)
            image_url = f"/work_dir/comparison/all/{fname}"
            image_desc = simple_image_features(enhanced)
        except Exception as e:
            print(f"Image processing error: {e}")
            return jsonify({"error": f"Image processing failed: {str(e)}"}), 500

    # Analyze the enhanced image
    metrics = analyze_image(enhanced)
    patient_id = f"patient_{int(time.time())}"
    
    # Detect modality
    modality = detect_modality_from_path_or_meta(None, enhanced)

    # Generate clinical note with proper report template
    note = render_report_template(
        patient_id, age, sex, modality, diagnosis or symptoms, metrics, image_desc
    )

    # Get ICD-10 code - first try lookup, then try Gemini
    icd = icd_lookup_by_diagnosis(diagnosis) if diagnosis else None
    
    # If no ICD found and we have a diagnosis, try to extract from the note
    if not icd and diagnosis:
        # Try Gemini if available
        gemini_icd = gemini_suggest_icd(diagnosis)
        if gemini_icd:
            icd = gemini_icd
    
    # If still no ICD, check the automated findings from metrics
    if not icd:
        hyper = metrics.get("hyper_frac", 0.0)
        sym = metrics.get("symmetry_score", 0.0)
        vent = metrics.get("ventricle_fraction", 0.0)
        
        if hyper > 0.0015:
            icd = "I62" 
        elif sym < 0.55:
            icd = "C71"  
        elif vent > 0.20:
            icd = "G91.9"  
        else:
            icd = "Z00.00"  
    p_val = psnr(arr, enhanced) if arr.size > 0 else None
    s_val = compute_ssim(arr, enhanced) if arr.size > 0 else None
    rowid = save_history(
        user_email=user_email,
        patient_id=patient_id,
        age=age,
        sex=sex,
        symptoms=symptoms,
        diagnosis=diagnosis,
        icd10=icd,
        note=note,
        image_side_by_side=image_url,
        psnr_val=p_val,
        ssim_val=s_val,
        extra={"metrics": metrics}
    )

    return jsonify({
        "note": note,
        "icd10": icd,
        "image_side_by_side": image_url,
        "history_rowid": rowid,
        "psnr": p_val,
        "ssim": s_val,
        "metrics": metrics
    }), 200

# ---------------- Dashboard stats endpoint ----------------
@app.route("/api/dashboard_stats", methods=["GET"])
@jwt_required
def api_dashboard_stats():
    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)

        user_email = g.user_email
        user_role = g.user_role
        want_all = (request.args.get("all") or "").lower() in ("1", "true", "yes")

        params = []
        where_clause = ""

        # ðŸ” Only admin can see all data
        if user_role != "admin" or not want_all:
            where_clause = "WHERE LOWER(user_email) = %s"
            params.append(user_email.lower())

        # Total reports
        total_sql = f"SELECT COUNT(*) as cnt FROM reports {where_clause}"
        cur.execute(total_sql, tuple(params))
        total = int(cur.fetchone()["cnt"] or 0)

        # Reports in last 24 hours
        if where_clause:
            cur.execute(
                f"""
                SELECT COUNT(*) as cnt
                FROM reports
                {where_clause}
                AND created_at >= NOW() - INTERVAL 1 DAY
                """,
                tuple(params),
            )
        else:
            cur.execute(
                "SELECT COUNT(*) as cnt FROM reports WHERE created_at >= NOW() - INTERVAL 1 DAY"
            )
        last_24h = int(cur.fetchone()["cnt"] or 0)

        # Success rate
        if total > 0:
            if where_clause:
                cur.execute(
                    f"""
                    SELECT COUNT(*) as cnt
                    FROM reports
                    WHERE note IS NOT NULL
                      AND LENGTH(note) > 20
                      AND LOWER(user_email) = %s
                    """,
                    tuple(params),
                )
            else:
                cur.execute(
                    "SELECT COUNT(*) as cnt FROM reports WHERE note IS NOT NULL AND LENGTH(note) > 20"
                )
            success = int(cur.fetchone()["cnt"] or 0)
            success_rate = round(100.0 * success / total, 2)
        else:
            success_rate = 0.0

        # Failed reports
        if where_clause:
            cur.execute(
                f"""
                SELECT COUNT(*) as cnt
                FROM reports
                WHERE (note IS NULL OR LENGTH(note) < 20)
                  AND LOWER(user_email) = %s
                """,
                tuple(params),
            )
        else:
            cur.execute(
                "SELECT COUNT(*) as cnt FROM reports WHERE note IS NULL OR LENGTH(note) < 20"
            )
        failed_total = int(cur.fetchone()["cnt"] or 0)

        # Recent reports
        recent = []
        query_base = """
            SELECT id, patient_id, created_at, icd10, psnr, ssim, metadata
            FROM reports
        """
        if where_clause:
            query_base += f" {where_clause} ORDER BY created_at DESC LIMIT 10"
            cur.execute(query_base, tuple(params))
        else:
            query_base += " ORDER BY created_at DESC LIMIT 10"
            cur.execute(query_base)

        rows = cur.fetchall()
        for r in rows:
            disease_name = None
            try:
                md = json.loads(r["metadata"]) if r["metadata"] else {}
                disease_name = md.get("disease_simple")
            except Exception:
                pass

            recent.append({
                "id": r["id"],
                "patient_id": r["patient_id"],
                "created_at": r["created_at"],
                "icd10": r["icd10"],
                "psnr": r["psnr"],
                "ssim": r["ssim"],
                "disease_name": disease_name
            })

        cur.close()
        conn.close()

        return jsonify({
            "total": total,
            "last_24h": last_24h,
            "success_rate": success_rate,
            "failed_total": failed_total,
            "recent": recent
        }), 200

    except Exception as e:
        print("api_dashboard_stats error:", e)
        return jsonify({
            "error": "failed to compute dashboard stats",
            "detail": str(e)
        }), 500


# ---------------- Dashboard analytics endpoint ----------------
@app.route("/api/dashboard_analytics", methods=["GET"])
@jwt_required
def api_dashboard_analytics():
    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)

        user_email = g.user_email
        user_role = g.user_role
        want_all = (request.args.get("all") or "").lower() in ("1", "true", "yes")

        #  Only admin can see all data
        restrict_user = (user_role != "admin") or not want_all

        # -------- Timeseries (last 30 days) --------
        if restrict_user:
            cur.execute("""
                SELECT DATE(created_at) AS d, COUNT(*) AS cnt
                FROM reports
                WHERE LOWER(user_email) = %s
                  AND DATE(created_at) >= CURDATE() - INTERVAL 29 DAY
                GROUP BY d
                ORDER BY d ASC
            """, (user_email.lower(),))
        else:
            cur.execute("""
                SELECT DATE(created_at) AS d, COUNT(*) AS cnt
                FROM reports
                WHERE DATE(created_at) >= CURDATE() - INTERVAL 29 DAY
                GROUP BY d
                ORDER BY d ASC
            """)

        rows = cur.fetchall()

        from datetime import datetime, timedelta
        today = datetime.utcnow().date()
        start = today - timedelta(days=29)
        day_map = {str(r["d"]): r["cnt"] for r in rows}

        timeseries = []
        for i in range(30):
            d = start + timedelta(days=i)
            ds = d.isoformat()
            timeseries.append({"date": ds, "count": int(day_map.get(ds, 0))})

        # -------- Top ICD-10 --------
        if restrict_user:
            cur.execute("""
                SELECT icd10 AS icd, COUNT(*) AS cnt
                FROM reports
                WHERE LOWER(user_email) = %s
                  AND icd10 IS NOT NULL AND icd10 != ''
                GROUP BY icd10
                ORDER BY cnt DESC
                LIMIT 10
            """, (user_email.lower(),))
        else:
            cur.execute("""
                SELECT icd10 AS icd, COUNT(*) AS cnt
                FROM reports
                WHERE icd10 IS NOT NULL AND icd10 != ''
                GROUP BY icd10
                ORDER BY cnt DESC
                LIMIT 10
            """)

        icd_rows = cur.fetchall()
        icd_top = [{"icd": r["icd"] or "UNK", "count": int(r["cnt"])} for r in icd_rows]

        # -------- Disease Categories --------
        if restrict_user:
            cur.execute(
                "SELECT icd10, metadata, note FROM reports WHERE LOWER(user_email) = %s",
                (user_email.lower(),)
            )
        else:
            cur.execute("SELECT icd10, metadata, note FROM reports")

        rows = cur.fetchall()

        cat_counts = defaultdict(int)
        for r in rows:
            cat = "normal"
            try:
                md = json.loads(r["metadata"]) if r["metadata"] else {}
                ds = (md.get("disease_simple") or "").lower()
                note = (r["note"] or "").lower()
                icd = (r["icd10"] or "").upper()

                if "hemorrhage" in ds or "hyperdense" in ds or "hemorrhage" in note:
                    cat = "hemorrhage"
                elif "lesion" in ds or "mass effect" in ds or icd.startswith(("C", "I", "G")):
                    cat = "lesion"
                elif "ventric" in ds or "atrophy" in ds or "ventric" in note:
                    cat = "ventriculomegaly"
            except Exception:
                pass

            cat_counts[cat] += 1

        disease_categories = {
            "hemorrhage": int(cat_counts.get("hemorrhage", 0)),
            "lesion": int(cat_counts.get("lesion", 0)),
            "ventriculomegaly": int(cat_counts.get("ventriculomegaly", 0)),
            "normal": int(cat_counts.get("normal", 0)),
        }

        cur.close()
        conn.close()

        return jsonify({
            "timeseries": timeseries,
            "icd_top": icd_top,
            "disease_categories": disease_categories
        }), 200

    except Exception as e:
        print("api_dashboard_analytics error:", e)
        return jsonify({"error": "failed to compute analytics", "detail": str(e)}), 500


@app.route("/reports/<int:rowid>", methods=["GET"])
@jwt_required
def view_report(rowid):
    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT * FROM reports WHERE id = %s", (rowid,))
        r = cur.fetchone()
        cur.close()
        conn.close()

        if not r:
            return ("Report not found", 404)

        
        if g.user_role != "admin" and r["user_email"] != g.user_email:
            return ("Forbidden", 403)

        note = r["note"] or ""
        img = r["image_side_by_side"] or ""
        icd = r["icd10"] or "UNK"
        created = r["created_at"] or ""
        patient_id = r["patient_id"] or ""

        html = f"""
        <!doctype html>
        <html>
        <head><meta charset="utf-8"><title>Report {rowid}</title>
        <style>
        body{{font-family:system-ui; padding:20px; background:#f7fafc}}
        .card{{background:white; padding:18px; border-radius:10px; max-width:900px; margin:auto}}
        pre{{white-space:pre-wrap; background:#0b1220; color:#e6eef8; padding:12px; border-radius:8px}}
        img{{max-width:100%; border-radius:8px; margin-top:12px}}
        </style>
        </head>
        <body>
          <div class="card">
            <h2>Report ID: {rowid} â€” {patient_id}</h2>
            <p><strong>Created:</strong> {created} &nbsp;&nbsp; <strong>ICD-10:</strong> {icd}</p>
            <h3>Clinical Note</h3>
            <pre>{note}</pre>
            {"<h3>Image (original | enhanced)</h3><img src='" + img + "' />" if img else ""}
          </div>
        </body>
        </html>
        """
        return html

    except Exception as e:
        print("view_report error:", e)
        return ("Failed to render report", 500)

@app.route("/api/bulk_results_csv", methods=["GET"])
def bulk_results_csv():
    since = request.args.get("since")
    try:
        conn = get_db_connection()
        cur = conn.cursor(dictionary=True)
        if since:
            cur.execute("SELECT * FROM reports WHERE created_at >= %s ORDER BY created_at DESC", (since,))
        else:
            cur.execute("SELECT * FROM reports ORDER BY created_at DESC")
        rows = cur.fetchall()
        cur.close()
        conn.close()

        out = io.StringIO()
        writer = csv.writer(out)
        header = ["id", "user_email", "patient_id", "created_at", "age", "sex", "symptoms",
                  "diagnosis", "icd10", "note", "image_side_by_side", "psnr", "ssim", "metadata"]
        writer.writerow(header)
        for r in rows:
            writer.writerow([r["id"], r["user_email"], r["patient_id"], r["created_at"],
                             r["age"], r["sex"], r["symptoms"], r["diagnosis"],
                             r["icd10"], (r["note"] or "").replace("\n", " "), r["image_side_by_side"],
                             r["psnr"], r["ssim"], (r["metadata"] or "")])
        csv_data = out.getvalue()
        out.close()
        resp = make_response(csv_data)
        resp.headers["Content-Type"] = "text/csv"
        resp.headers["Content-Disposition"] = "attachment; filename=reports_export.csv"
        return resp
    except Exception as e:
        print("bulk_results_csv error:", e)
        return jsonify({"error": "failed to export CSV", "detail": str(e)}), 500

@app.route("/admin", methods=["GET"])
@admin_required
def admin_dashboard():
    if not is_admin():
        return redirect("/login")
    content, status = _load_frontend_page("admin_dashboard.html")
    if status == 200:
        return render_template_string(content)
    return content, status
RESET_TOKENS = {}
# ---------------- Password reset endpoints ----------------
@app.route("/forgot-password", methods=["POST"])
@limiter.limit("3 per hour")  # Strict rate limit
def forgot_password():
    email = (request.form.get("email") or "").strip().lower()
    
    if not email or not is_valid_email(email):
        return jsonify({"error": "Invalid email"}), 400
    
    # Constant-time response to prevent email enumeration
    time.sleep(0.5)
    
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    cur.execute("SELECT id FROM users WHERE LOWER(email) = %s", (email,))
    user = cur.fetchone()
    
    if user:
        # Generate cryptographically secure token
        token = secrets.token_urlsafe(32)
        
        # Store token with expiration
        cur.execute("""
            INSERT INTO password_resets (email, token)
            VALUES (%s, %s)
        """, (email, token))
        conn.commit()
        
        # Send email with reset link
        reset_url = f"{APP_URL}/reset-password?token={token}"
        email_body = f"""
You requested a password reset for your EHR Assistant account.

Click the link below to reset your password (valid for 1 hour):
{reset_url}

If you didn't request this, please ignore this email.

For security, this link will expire in 1 hour.
        """
        
        send_email(
            to=email,
            subject="Password Reset Request - EHR Assistant",
            body=email_body
        )
        
        log_security_event("password_reset_requested", email)
    
    cur.close()
    conn.close()
    
    # Always return success (prevent email enumeration)
    return jsonify({
        "message": "If an account exists with that email, a reset link has been sent."
    }), 200

# ---------------- Password reset page and submission ----------------
@app.route("/reset-password", methods=["GET", "POST"])
def reset_password_route():
    if request.method == "GET":
        content, status = _load_frontend_page("reset_password.html")
        if status == 200:
            return render_template_string(content)
        return content, status
    
    # POST - actually reset the password
    token = (request.form.get("token") or "").strip()
    password = request.form.get("password") or ""
    
    if not token or not password:
        return jsonify({"error": "Missing required fields"}), 400
    
    # Validate password strength
    is_strong, message = is_strong_password(password)
    if not is_strong:
        return jsonify({"error": message}), 400
    
    conn = get_db_connection()
    cur = conn.cursor(dictionary=True)
    
    # Find valid unused token (within 1 hour)
    cur.execute("""
        SELECT email FROM password_resets
        WHERE token = %s
          AND used = FALSE
          AND created_at >= NOW() - INTERVAL 1 HOUR
    """, (token,))
    
    reset = cur.fetchone()
    
    if not reset:
        cur.close()
        conn.close()
        log_security_event("password_reset_invalid_token", None, {"token_preview": token[:10]})
        return jsonify({"error": "Invalid or expired reset link"}), 400
    
    email = reset["email"]
    
    # Update password
    pw_hash = generate_password_hash(password)
    cur.execute(
        "UPDATE users SET password_hash = %s WHERE LOWER(email) = %s",
        (pw_hash, email)
    )
    
    # Mark token as used
    cur.execute(
        "UPDATE password_resets SET used = TRUE, used_at = NOW() WHERE token = %s",
        (token,)
    )
    
    conn.commit()
    cur.close()
    conn.close()
    
    # Notify user
    send_email(
        to=email,
        subject="Password Changed - EHR Assistant",
        body=f"""Your EHR Assistant password was successfully changed.

If you didn't make this change, please contact support immediately.

Login at: {APP_URL}/login
"""
    )
    
    log_security_event("password_reset_completed", email)
    
    return jsonify({"message": "Password updated successfully"}), 200
# ---------------- Logout endpoint ----------------
@app.route("/logout", methods=["GET", "POST"])
def logout_route():
    # Get token to blacklist it
    token = request.cookies.get("token")
    if token:
        blacklist_token(token, "user_logout")
    
    # Get email if available
    payload = decode_jwt_from_request()
    if payload:
        log_security_event("user_logout", payload.get("email"))
    
    resp = make_response(redirect("/login"))
    resp.set_cookie('token', '', expires=0)
    return resp

# ---------------- Error handlers ----------------
@app.errorhandler(404)
def log_404(e):
    return ("Not Found", 404)

if __name__ == "__main__":
    if ICD_JSON.exists():
        ICD_LOOKUP = load_icd_lookup()
    app.run(host="0.0.0.0", port=5000, debug=True)