# ============================================
# ECORIPE - AUTHENTICATION & API SERVER
# VERSION: 3.1 PRODUCTION READY - FULLY FIXED
# ============================================

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime, timedelta
from typing import Optional, List
import jwt
import sqlite3
import hashlib
import secrets
import pandas as pd
import os
import time
from contextlib import contextmanager
import logging

# Import your ML engine
from app import predict_crop_decision, r2, mae

# ============================================
# SETUP LOGGING
# ============================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ecor ripe-api")

# ============================================
# INITIALIZE FASTAPI
# ============================================
app = FastAPI(
    title="EcoRipe API",
    description="ML-Powered Apple Price Prediction",
    version="3.1.0"
)

# CORS configuration - Allow all origins for now
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# SECURITY CONFIGURATION
# ============================================
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# ============================================
# DATABASE SETUP (PERSISTENT STORAGE)
# ============================================
# Use Render disk if available
if os.path.exists('/data'):
    DB_PATH = '/data/ecor ripe_users.db'
    logger.info(f"✅ Using persistent storage: {DB_PATH}")
else:
    DB_PATH = 'ecor ripe_users.db'
    logger.info(f"📁 Using local storage: {DB_PATH}")

@contextmanager
def get_db():
    """Database connection with automatic cleanup"""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        yield conn
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def init_database():
    """Initialize all database tables"""
    try:
        with get_db() as conn:
            c = conn.cursor()
            
            # Users table
            c.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    email TEXT UNIQUE NOT NULL,
                    password_hash TEXT NOT NULL,
                    full_name TEXT,
                    phone TEXT,
                    location TEXT,
                    variety_preference TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    marketing_consent BOOLEAN DEFAULT 1
                )
            ''')
            
            # Create indexes for faster queries
            c.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')
            
            # Predictions table
            c.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    market TEXT,
                    variety TEXT,
                    arrival_qty REAL,
                    price_today REAL,
                    predicted_price REAL,
                    decision TEXT,
                    risk_ratio REAL,
                    confidence REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            c.execute('CREATE INDEX IF NOT EXISTS idx_predictions_user ON predictions(user_id)')
            
            # Visits table for analytics
            c.execute('''
                CREATE TABLE IF NOT EXISTS visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    page TEXT,
                    visited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.commit()
            
            # Check if admin user exists, create if not
            c.execute("SELECT COUNT(*) as count FROM users WHERE email = 'admin@ecor ripe.com'")
            if c.fetchone()['count'] == 0:
                # Create admin user with password: Admin@2026
                admin_hash = hash_password("Admin@2026")
                c.execute("""
                    INSERT INTO users (email, password_hash, full_name, is_active)
                    VALUES (?, ?, ?, ?)
                """, ('admin@ecor ripe.com', admin_hash, 'System Admin', 1))
                conn.commit()
                logger.info("✅ Admin user created")
            
            logger.info("✅ Database initialized successfully")
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

# Initialize database on startup
init_database()

# ============================================
# PYDANTIC MODELS
# ============================================
class UserRegister(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    variety_preference: Optional[str] = None
    marketing_consent: bool = True

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    email: str
    full_name: Optional[str] = None
    user_id: int

class PredictionRequest(BaseModel):
    market: str
    variety: str = "Gala"
    arrival_qty: float = Field(..., gt=0)
    price_today: float = Field(..., gt=0)
    price_1d_ago: float = Field(..., gt=0)
    price_3d_ago: float = Field(..., gt=0)
    days_since_harvest: int = Field(..., ge=0)
    max_safe_days: int = Field(..., gt=0)

class PredictionResponse(BaseModel):
    decision: str
    predicted_price: float
    risk_ratio: float

class UserProfileResponse(BaseModel):
    email: str
    full_name: Optional[str]
    location: Optional[str]
    variety_preference: Optional[str]
    member_since: str
    total_predictions: int
    recent_predictions: List[dict]

# ============================================
# HELPER FUNCTIONS
# ============================================
def hash_password(password: str) -> str:
    """Secure password hashing with salt"""
    salt = secrets.token_hex(16)
    hash_value = hashlib.sha256((salt + password).encode()).hexdigest()
    return f"{salt}:{hash_value}"

def verify_password(password: str, hash_str: str) -> bool:
    """Verify password against hash"""
    try:
        # Handle both old and new hash formats
        if ':' in hash_str:
            salt, hash_value = hash_str.split(':', 1)
            return hash_value == hashlib.sha256((salt + password).encode()).hexdigest()
        else:
            # Old format (unsalted) - for backward compatibility
            return hash_str == hashlib.sha256(password.encode()).hexdigest()
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

def create_access_token(data: dict) -> str:
    """Create JWT token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> Optional[dict]:
    """Verify JWT token"""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        logger.warning("Token expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        return None

# ============================================
# HEALTH CHECK ENDPOINTS
# ============================================
@app.get("/health")
@app.get("/healthz")
async def health_check():
    """Quick health check for Render"""
    start = time.time()
    try:
        with get_db() as conn:
            conn.execute("SELECT 1").fetchone()
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "database": db_status,
        "model_accuracy": f"{r2 * 100:.1f}%",
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================
# AUTHENTICATION ENDPOINTS - FIXED
# ============================================

@app.post("/api/register", response_model=TokenResponse)
async def register(user: UserRegister):
    """Register new user"""
    try:
        with get_db() as conn:
            c = conn.cursor()
            
            # Check if user exists
            c.execute("SELECT id FROM users WHERE email = ?", (user.email.lower().strip(),))
            if c.fetchone():
                raise HTTPException(status_code=400, detail="Email already registered")
            
            # Create user
            password_hash = hash_password(user.password)
            c.execute("""
                INSERT INTO users (email, password_hash, full_name, phone, location, 
                                 variety_preference, marketing_consent)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (user.email.lower().strip(), password_hash, user.full_name, user.phone, 
                  user.location, user.variety_preference, user.marketing_consent))
            
            user_id = c.lastrowid
            conn.commit()
            
            # Create token
            token = create_access_token({"sub": user.email.lower().strip(), "user_id": user_id})
            
            logger.info(f"✅ New user registered: {user.email}")
            
            return {
                "access_token": token,
                "token_type": "bearer",
                "email": user.email,
                "full_name": user.full_name,
                "user_id": user_id
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/api/login", response_model=TokenResponse)
async def login(user: UserLogin):
    """Login user - FIXED VERSION"""
    try:
        # Normalize email
        email = user.email.lower().strip()
        
        with get_db() as conn:
            c = conn.cursor()
            
            # Get user
            c.execute("SELECT id, email, full_name, password_hash FROM users WHERE email = ?", (email,))
            db_user = c.fetchone()
            
            # Debug logging
            logger.info(f"Login attempt for email: {email}")
            
            if not db_user:
                logger.warning(f"User not found: {email}")
                # Delay to prevent timing attacks
                time.sleep(1)
                raise HTTPException(status_code=401, detail="Invalid email or password")
            
            # Verify password
            if not verify_password(user.password, db_user['password_hash']):
                logger.warning(f"Invalid password for: {email}")
                time.sleep(1)
                raise HTTPException(status_code=401, detail="Invalid email or password")
            
            # Update last login
            c.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (db_user['id'],))
            conn.commit()
            
            # Create token
            token = create_access_token({"sub": db_user['email'], "user_id": db_user['id']})
            
            logger.info(f"✅ User logged in: {email}")
            
            return {
                "access_token": token,
                "token_type": "bearer",
                "email": db_user['email'],
                "full_name": db_user['full_name'],
                "user_id": db_user['id']
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.get("/api/verify")
async def verify_token_endpoint(request: Request):
    """Verify JWT token"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return {"valid": False, "error": "No token provided"}
    
    token = auth_header.replace("Bearer ", "")
    payload = verify_token(token)
    
    if payload:
        return {
            "valid": True,
            "email": payload.get("sub"),
            "user_id": payload.get("user_id")
        }
    return {"valid": False, "error": "Invalid or expired token"}

# ============================================
# PREDICTION ENDPOINTS - BOTH WORK!
# ============================================

# BACKWARD COMPATIBILITY - Frontend uses this!
@app.post("/predict")
async def predict_original(request: Request, prediction: PredictionRequest):
    """Original endpoint - kept so frontend never breaks"""
    try:
        # Get user if logged in
        user_id = None
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
            payload = verify_token(token)
            if payload:
                user_id = payload.get("user_id")
        
        # Create DataFrame for prediction
        input_df = pd.DataFrame([{
            "market": prediction.market,
            "arrival_qty": prediction.arrival_qty,
            "price_today": prediction.price_today,
            "price_1d_ago": prediction.price_1d_ago,
            "price_3d_ago": prediction.price_3d_ago,
            "days_since_harvest": prediction.days_since_harvest,
            "max_safe_days": prediction.max_safe_days
        }])
        
        # Get prediction from ML engine
        result = predict_crop_decision(input_df)
        
        # Save to database if user logged in
        if user_id and result.get("success", True):
            try:
                with get_db() as conn:
                    c = conn.cursor()
                    c.execute("""
                        INSERT INTO predictions 
                        (user_id, market, variety, arrival_qty, price_today, 
                         predicted_price, decision, risk_ratio, confidence)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (user_id, prediction.market, prediction.variety,
                          prediction.arrival_qty, prediction.price_today,
                          result["predicted_price"], result["decision"],
                          result["risk_ratio"], result.get("confidence", 95)))
                    conn.commit()
            except Exception as e:
                logger.error(f"Failed to save prediction: {e}")
        
        # Return in format frontend expects
        return {
            "decision": result["decision"],
            "predicted_price": result["predicted_price"],
            "risk_ratio": result["risk_ratio"]
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {
            "decision": "ERROR",
            "predicted_price": 0,
            "risk_ratio": 0
        }

# New enhanced endpoint
@app.post("/api/predict")
async def predict_enhanced(request: Request, prediction: PredictionRequest):
    """Enhanced prediction endpoint with more details"""
    return await predict_original(request, prediction)

# ============================================
# USER PROFILE ENDPOINT
# ============================================
@app.get("/api/user/profile", response_model=UserProfileResponse)
async def get_profile(request: Request):
    """Get user profile and prediction history"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = auth_header.replace("Bearer ", "")
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user_id = payload.get("user_id")
    
    with get_db() as conn:
        c = conn.cursor()
        
        # Get user details
        c.execute("""
            SELECT email, full_name, location, variety_preference, created_at
            FROM users WHERE id = ?
        """, (user_id,))
        user = c.fetchone()
        
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get prediction history
        c.execute("""
            SELECT market, variety, predicted_price, decision, confidence, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT 10
        """, (user_id,))
        predictions = c.fetchall()
        
        # Get total count
        c.execute("SELECT COUNT(*) as count FROM predictions WHERE user_id = ?", (user_id,))
        total = c.fetchone()['count']
    
    return {
        "email": user['email'],
        "full_name": user['full_name'],
        "location": user['location'],
        "variety_preference": user['variety_preference'],
        "member_since": user['created_at'],
        "total_predictions": total,
        "recent_predictions": [dict(p) for p in predictions]
    }

# ============================================
# ANALYTICS ENDPOINT
# ============================================
@app.post("/api/track-visit")
async def track_visit(request: Request):
    """Track page visits"""
    try:
        session_id = request.headers.get("X-Session-ID", "unknown")
        ip = request.client.host
        user_agent = request.headers.get("user-agent", "")
        page = request.headers.get("X-Page", "unknown")
        
        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO visits (session_id, ip_address, user_agent, page)
                VALUES (?, ?, ?, ?)
            """, (session_id, ip, user_agent, page))
            conn.commit()
        
        return {"status": "tracked"}
    except Exception as e:
        logger.error(f"Visit tracking error: {e}")
        return {"status": "error"}

# ============================================
# ADMIN ENDPOINTS
# ============================================
@app.get("/api/admin/stats")
async def admin_stats(admin_key: str):
    """Get system statistics (admin only)"""
    if admin_key != "ecor ripe-admin-2026":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    with get_db() as conn:
        c = conn.cursor()
        
        # User stats
        c.execute("SELECT COUNT(*) as count FROM users")
        total_users = c.fetchone()['count']
        
        c.execute("SELECT COUNT(*) as count FROM users WHERE date(created_at) = date('now')")
        today_users = c.fetchone()['count']
        
        # Prediction stats
        c.execute("SELECT COUNT(*) as count FROM predictions")
        total_predictions = c.fetchone()['count']
        
        # Email list
        c.execute("SELECT email FROM users WHERE marketing_consent = 1")
        emails = [row['email'] for row in c.fetchall()]
    
    return {
        "users": {
            "total": total_users,
            "today": today_users,
            "email_list": emails,
            "email_count": len(emails)
        },
        "predictions": {
            "total": total_predictions
        },
        "model": {
            "accuracy": f"{r2 * 100:.1f}%",
            "avg_error": f"₹{mae:.2f}"
        }
    }

@app.get("/api/admin/export-emails")
async def export_emails(admin_key: str):
    """Export emails as CSV (admin only)"""
    if admin_key != "ecor ripe-admin-2026":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT email, full_name, location, created_at 
            FROM users 
            WHERE marketing_consent = 1 
            ORDER BY created_at DESC
        """)
        users = c.fetchall()
    
    # Generate CSV
    csv_lines = ["Email,Full Name,Location,Registered Date"]
    for user in users:
        csv_lines.append(f"{user['email']},{user['full_name'] or ''},{user['location'] or ''},{user['created_at']}")
    
    return {
        "csv": "\n".join(csv_lines),
        "count": len(users),
        "filename": f"ecor ripe-emails-{datetime.now().strftime('%Y%m%d')}.csv"
    }

# ============================================
# ROOT ENDPOINT
# ============================================
@app.get("/")
async def root():
    """API root with documentation links"""
    return {
        "name": "EcoRipe API",
        "version": "3.1.0",
        "status": "running",
        "model_accuracy": f"{r2 * 100:.1f}%",
        "endpoints": {
            "health": "/health",
            "register": "/api/register",
            "login": "/api/login",
            "verify": "/api/verify",
            "predict": "/predict (legacy) or /api/predict (new)",
            "profile": "/api/user/profile",
            "docs": "/docs"
        },
        "documentation": "/docs"
    }

# ============================================
# DEBUG ENDPOINT (remove in production if desired)
# ============================================
@app.get("/debug/users")
async def debug_users(admin_key: str):
    """Debug: List all users (protected)"""
    if admin_key != "ecor ripe-debug-2026":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, email, created_at, last_login FROM users ORDER BY created_at DESC LIMIT 10")
        users = c.fetchall()
    
    return {
        "total": len(users),
        "users": [dict(u) for u in users]
    }

# ============================================
# STARTUP MESSAGE
# ============================================
print("\n" + "="*60)
print("🚀 ECORIPE API SERVER - FULLY FIXED")
print("="*60)
print(f"\n📊 Model Performance:")
print(f"   • Accuracy: {r2 * 100:.1f}%")
print(f"   • Avg Error: ₹{mae:.2f}")
print(f"\n🔐 Authentication: JWT with 7-day tokens")
print(f"💾 Database: {DB_PATH}")
print(f"\n📡 Endpoints:")
print(f"   • Login:     /api/login")
print(f"   • Register:  /api/register")
print(f"   • Predict:   /predict (legacy) & /api/predict")
print(f"\n✅ All systems go! Frontend will work without changes")
print("="*60)
