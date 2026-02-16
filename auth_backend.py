# ============================================
# ECORIPE - COMPLETE FIXED AUTH BACKEND
# WITH PERSISTENT STORAGE & ACCURATE PREDICTIONS
# ============================================

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from typing import Optional, List
import jwt
import sqlite3
import hashlib
import secrets
import pandas as pd
import asyncio
import os
import time
from contextlib import contextmanager

# ============================================
# IMPORT YOUR ENHANCED MODEL
# ============================================
from app import predict_crop_decision_enhanced, r2, model, scaler

# ============================================
# SETUP
# ============================================
app = FastAPI(title="EcoRipe API with Auth")

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT Configuration
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# ============================================
# PERSISTENT STORAGE ON RENDER DISK
# ============================================
# Use Render Disk if available, otherwise fallback to /tmp
if os.path.exists('/data'):
    DB_PATH = '/data/ecor ripe_users.db'
    print(f"✅ Using persistent storage: {DB_PATH}")
else:
    DB_PATH = '/tmp/ecor ripe_users.db'
    print(f"⚠️ Using temporary storage: {DB_PATH} (data will reset on restart)")

# ============================================
# DATABASE CONNECTION WITH CONTEXT MANAGER
# ============================================
@contextmanager
def get_db():
    """Get database connection with automatic closing"""
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        yield conn
    except Exception as e:
        print(f"❌ Database error: {e}")
        raise
    finally:
        if conn:
            conn.close()

def init_db_sync():
    """Initialize database tables"""
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
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_login TIMESTAMP,
                    is_active BOOLEAN DEFAULT 1,
                    marketing_consent BOOLEAN DEFAULT 1
                )
            ''')
            
            # Create index for faster email lookups
            c.execute('CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)')
            
            # Visits tracking table
            c.execute('''
                CREATE TABLE IF NOT EXISTS visits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    session_id TEXT,
                    ip_address TEXT,
                    user_agent TEXT,
                    page_visited TEXT,
                    visited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            # Predictions tracking table
            c.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER,
                    market TEXT,
                    arrival_qty REAL,
                    price_today REAL,
                    predicted_price REAL,
                    decision TEXT,
                    risk_ratio REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            ''')
            
            # Create index for predictions
            c.execute('CREATE INDEX IF NOT EXISTS idx_predictions_user_id ON predictions(user_id)')
            
            conn.commit()
            
            # Check if we have any users
            c.execute("SELECT COUNT(*) as count FROM users")
            count = c.fetchone()['count']
            print(f"✅ Database initialized with {count} existing users")
            
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

# ============================================
# STARTUP EVENT
# ============================================
@app.on_event("startup")
async def startup_event():
    """Initialize database AFTER server starts"""
    print("🚀 EcoRipe Auth Server starting...")
    print(f"📊 Model R² Score: {r2:.3f}")
    
    # Run database init in thread pool
    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(None, init_db_sync)
    
    if success:
        print("✅ Server ready to handle requests!")
    else:
        print("⚠️ Server started but database has issues")

# ============================================
# HEALTH CHECK
# ============================================
@app.get("/healthz")
@app.get("/health")
async def health_check():
    """Quick health check"""
    start_time = time.time()
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("SELECT 1")
            db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    response_time = (time.time() - start_time) * 1000
    
    return {
        "status": "healthy",
        "database": db_status,
        "response_time_ms": round(response_time, 2),
        "model_accuracy": round(r2 * 100, 1),
        "timestamp": datetime.utcnow().isoformat()
    }

# ============================================
# MODELS
# ============================================
class UserRegister(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    marketing_consent: bool = True

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    email: str
    full_name: Optional[str] = None
    user_id: int

class PredictionInput(BaseModel):
    market: str
    arrival_qty: float
    price_today: float
    price_1d_ago: float
    price_3d_ago: float
    days_since_harvest: int
    max_safe_days: int

class PredictionOutput(BaseModel):
    decision: str
    predicted_price: float
    risk_ratio: float
    price_trend_1d: float
    price_trend_3d: float
    warnings: List[str] = []
    confidence: float

# ============================================
# HELPER FUNCTIONS
# ============================================
def hash_password(password: str) -> str:
    """Hash password with salt"""
    salt = secrets.token_hex(8)
    return salt + ':' + hashlib.sha256((salt + password).encode()).hexdigest()

def verify_password(password: str, hash_str: str) -> bool:
    """Verify password against hash"""
    try:
        salt, hash_value = hash_str.split(':', 1)
        return hash_value == hashlib.sha256((salt + password).encode()).hexdigest()
    except:
        return False

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# ============================================
# AUTHENTICATION ENDPOINTS
# ============================================

@app.post("/api/register", response_model=Token)
async def register(user: UserRegister):
    """Register new user"""
    try:
        with get_db() as conn:
            c = conn.cursor()
            
            # Check if user exists
            c.execute("SELECT id FROM users WHERE email = ?", (user.email,))
            if c.fetchone():
                raise HTTPException(status_code=400, detail="Email already registered")
            
            # Create new user
            password_hash = hash_password(user.password)
            c.execute("""
                INSERT INTO users (email, password_hash, full_name, phone, location, marketing_consent)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user.email, password_hash, user.full_name, user.phone, user.location, user.marketing_consent))
            
            user_id = c.lastrowid
            conn.commit()
            
            # Create token
            access_token = create_access_token(
                data={"sub": user.email, "user_id": user_id},
                expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            )
            
            print(f"✅ New user registered: {user.email}")
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "email": user.email,
                "full_name": user.full_name,
                "user_id": user_id
            }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Registration error: {e}")
        raise HTTPException(status_code=500, detail="Registration failed")

@app.post("/api/login", response_model=Token)
async def login(user: UserLogin):
    """Login user"""
    try:
        with get_db() as conn:
            c = conn.cursor()
            
            # Get user
            c.execute("SELECT id, email, full_name, password_hash FROM users WHERE email = ?", (user.email,))
            db_user = c.fetchone()
            
            if not db_user or not verify_password(user.password, db_user['password_hash']):
                # Add small delay to prevent timing attacks
                await asyncio.sleep(0.5)
                raise HTTPException(status_code=401, detail="Invalid email or password")
            
            # Update last login
            c.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (db_user['id'],))
            conn.commit()
            
            # Create token
            access_token = create_access_token(
                data={"sub": db_user['email'], "user_id": db_user['id']},
                expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            )
            
            print(f"✅ User logged in: {user.email}")
            
            return {
                "access_token": access_token,
                "token_type": "bearer",
                "email": db_user['email'],
                "full_name": db_user['full_name'],
                "user_id": db_user['id']
            }
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@app.get("/api/verify-token")
async def verify_token(request: Request):
    """Verify JWT token"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return {"valid": False, "error": "No token provided"}
    
    token = auth_header.replace("Bearer ", "")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {
            "valid": True,
            "email": payload.get("sub"),
            "user_id": payload.get("user_id")
        }
    except jwt.ExpiredSignatureError:
        return {"valid": False, "error": "Token expired"}
    except jwt.InvalidTokenError:
        return {"valid": False, "error": "Invalid token"}

# ============================================
# PREDICTION ENDPOINT (UPDATED WITH ENHANCED MODEL)
# ============================================
@app.post("/predict", response_model=PredictionOutput)
async def predict(data: PredictionInput, request: Request):
    """Make price prediction using enhanced model"""
    try:
        # Get user if logged in
        user_id = None
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            try:
                token = auth_header.replace("Bearer ", "")
                payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
                user_id = payload.get("user_id")
            except:
                pass
        
        # Create DataFrame for prediction
        input_df = pd.DataFrame([{
            "market": data.market,
            "arrival_qty": data.arrival_qty,
            "price_today": data.price_today,
            "price_1d_ago": data.price_1d_ago,
            "price_3d_ago": data.price_3d_ago,
            "days_since_harvest": data.days_since_harvest,
            "max_safe_days": data.max_safe_days
        }])
        
        # Call enhanced prediction function
        result = predict_crop_decision_enhanced(input_df)
        
        # Save to database if logged in
        if user_id:
            try:
                with get_db() as conn:
                    c = conn.cursor()
                    c.execute("""
                        INSERT INTO predictions 
                        (user_id, market, arrival_qty, price_today, predicted_price, decision, risk_ratio)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (user_id, data.market, data.arrival_qty, data.price_today,
                          result['predicted_price'], result['decision'], result['risk_ratio']))
                    conn.commit()
            except Exception as e:
                print(f"⚠️ Failed to save prediction: {e}")
        
        return PredictionOutput(
            decision=result['decision'],
            predicted_price=result['predicted_price'],
            risk_ratio=result['risk_ratio'],
            price_trend_1d=result['price_trend_1d'],
            price_trend_3d=result['price_trend_3d'],
            warnings=result.get('warnings', []),
            confidence=result['confidence']
        )
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============================================
# TRACKING ENDPOINTS
# ============================================
@app.post("/api/track-visit")
async def track_visit(request: Request):
    """Track page visit"""
    try:
        session_id = request.headers.get("X-Session-ID", "unknown")
        ip = request.client.host
        user_agent = request.headers.get("user-agent", "")
        page = request.headers.get("X-Page", "unknown")
        
        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO visits (session_id, ip_address, user_agent, page_visited)
                VALUES (?, ?, ?, ?)
            """, (session_id, ip, user_agent, page))
            conn.commit()
        
        return {"status": "tracked"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# ============================================
# USER PROFILE ENDPOINT
# ============================================
@app.get("/api/user/profile")
async def get_profile(request: Request):
    """Get user profile"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    token = auth_header.replace("Bearer ", "")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("user_id")
        
        with get_db() as conn:
            c = conn.cursor()
            c.execute("""
                SELECT email, full_name, location, created_at, last_login, marketing_consent
                FROM users WHERE id = ?
            """, (user_id,))
            user = c.fetchone()
            
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            
            # Get user's prediction history
            c.execute("""
                SELECT market, predicted_price, decision, created_at
                FROM predictions
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT 10
            """, (user_id,))
            predictions = c.fetchall()
            
            return {
                "email": user['email'],
                "full_name": user['full_name'],
                "location": user['location'],
                "joined": user['created_at'],
                "last_login": user['last_login'],
                "marketing_consent": bool(user['marketing_consent']),
                "recent_predictions": [dict(p) for p in predictions]
            }
            
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ============================================
# ADMIN ENDPOINTS
# ============================================
@app.get("/api/admin/users")
async def get_users(admin_key: str):
    """Get all users (admin only)"""
    if admin_key != "ecor ripe-admin-2026":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""
            SELECT id, email, full_name, location, created_at, last_login, marketing_consent
            FROM users ORDER BY created_at DESC
        """)
        users = c.fetchall()
        
        return {
            "total": len(users),
            "users": [dict(u) for u in users]
        }

@app.get("/api/admin/stats")
async def get_stats(admin_key: str):
    """Get system statistics"""
    if admin_key != "ecor ripe-admin-2026":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    with get_db() as conn:
        c = conn.cursor()
        
        # User stats
        c.execute("SELECT COUNT(*) as count FROM users")
        total_users = c.fetchone()['count']
        
        c.execute("SELECT COUNT(*) as count FROM users WHERE date(created_at) = date('now')")
        today_users = c.fetchone()['count']
        
        # Visit stats
        c.execute("SELECT COUNT(*) as count FROM visits")
        total_visits = c.fetchone()['count']
        
        c.execute("SELECT COUNT(*) as count FROM visits WHERE date(visited_at) = date('now')")
        today_visits = c.fetchone()['count']
        
        # Prediction stats
        c.execute("SELECT COUNT(*) as count FROM predictions")
        total_predictions = c.fetchone()['count']
        
        c.execute("SELECT AVG(predicted_price) as avg FROM predictions")
        avg_price = c.fetchone()['avg']
        
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
            "visits": {
                "total": total_visits,
                "today": today_visits
            },
            "predictions": {
                "total": total_predictions,
                "average_price": round(avg_price, 2) if avg_price else None
            },
            "model": {
                "accuracy": round(r2 * 100, 1),
                "name": "Linear Regression"
            }
        }

@app.get("/api/admin/export-emails")
async def export_emails(admin_key: str):
    """Export emails as CSV"""
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
    import csv
    from io import StringIO
    
    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(['Email', 'Full Name', 'Location', 'Registered Date'])
    
    for user in users:
        writer.writerow([
            user['email'],
            user['full_name'] or '',
            user['location'] or '',
            user['created_at']
        ])
    
    return {
        "csv": output.getvalue(),
        "count": len(users),
        "filename": f"ecor ripe-emails-{datetime.now().strftime('%Y%m%d')}.csv"
    }

# ============================================
# DEBUG ENDPOINTS
# ============================================
@app.get("/debug/users")
async def debug_users():
    """Debug endpoint to see users (REMOVE IN PRODUCTION)"""
    try:
        with get_db() as conn:
            c = conn.cursor()
            c.execute("SELECT email, created_at FROM users ORDER BY created_at DESC")
            users = c.fetchall()
            return {
                "count": len(users),
                "users": [dict(u) for u in users],
                "db_path": DB_PATH
            }
    except Exception as e:
        return {"error": str(e), "db_path": DB_PATH}

@app.get("/debug/db-info")
async def db_info():
    """Database information"""
    return {
        "db_path": DB_PATH,
        "db_exists": os.path.exists(DB_PATH),
        "db_size": os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0,
        "writable": os.access(DB_PATH, os.W_OK) if os.path.exists(DB_PATH) else False,
        "model_accuracy": round(r2 * 100, 1)
    }

# ============================================
# ROOT ENDPOINT
# ============================================
@app.get("/")
async def root():
    return {
        "name": "EcoRipe API",
        "version": "2.0.0",
        "status": "running",
        "model_accuracy": f"{round(r2 * 100, 1)}%",
        "endpoints": {
            "health": "/healthz",
            "register": "/api/register",
            "login": "/api/login",
            "verify": "/api/verify-token",
            "predict": "/predict",
            "profile": "/api/user/profile",
            "docs": "/docs"
        },
        "documentation": "/docs"
    }
