# auth_backend.py - ADD THIS FILE (doesn't change your original code)

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta
from typing import Optional
import jwt
import sqlite3
import hashlib
import secrets
from your_original_file import predict_crop_decision  # YOUR original function!
import pandas as pd

# ============================================
# SETUP
# ============================================
app = FastAPI(title="EcoRipe API with Auth")

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT Configuration
SECRET_KEY = secrets.token_urlsafe(32)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 7 days

# ============================================
# DATABASE SETUP (FOR USER EMAILS)
# ============================================
def init_db():
    conn = sqlite3.connect('ecor ripe_users.db')
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
    
    # Email campaign tracking
    c.execute('''
        CREATE TABLE IF NOT EXISTS email_campaigns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            email_sent_at TIMESTAMP,
            email_type TEXT,
            opened BOOLEAN DEFAULT 0,
            clicked BOOLEAN DEFAULT 0,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database
init_db()

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

class PredictionInput(BaseModel):
    market: str
    arrival_qty: float
    price_today: float
    price_1d_ago: float
    price_3d_ago: float
    days_since_harvest: int
    max_safe_days: int

class VisitTrack(BaseModel):
    page: str
    session_id: Optional[str] = None

# ============================================
# HELPER FUNCTIONS
# ============================================
def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(password: str, hash: str) -> bool:
    """Verify password against hash"""
    return hash_password(password) == hash

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_db():
    """Get database connection"""
    conn = sqlite3.connect('ecor ripe_users.db')
    conn.row_factory = sqlite3.Row
    return conn

# ============================================
# AUTHENTICATION ENDPOINTS
# ============================================

@app.post("/api/register", response_model=Token)
async def register(user: UserRegister):
    """Register new user and collect email"""
    conn = get_db()
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
    conn.close()
    
    # Create token
    access_token = create_access_token(
        data={"sub": user.email, "user_id": user_id},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    # Track registration in your email list
    print(f"✅ NEW USER REGISTERED: {user.email} - Added to campaign list")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "email": user.email,
        "full_name": user.full_name
    }

@app.post("/api/login", response_model=Token)
async def login(user: UserLogin):
    """Login user and return token"""
    conn = get_db()
    c = conn.cursor()
    
    # Get user
    c.execute("SELECT id, email, full_name, password_hash FROM users WHERE email = ?", (user.email,))
    db_user = c.fetchone()
    
    if not db_user or not verify_password(user.password, db_user['password_hash']):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    
    # Update last login
    c.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (db_user['id'],))
    conn.commit()
    conn.close()
    
    # Create token
    access_token = create_access_token(
        data={"sub": db_user['email'], "user_id": db_user['id']},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    
    print(f"✅ USER LOGGED IN: {user.email}")
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "email": db_user['email'],
        "full_name": db_user['full_name']
    }

@app.get("/api/verify-token")
async def verify_token(request: Request):
    """Verify if token is valid"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return {"valid": False}
    
    token = auth_header.replace("Bearer ", "")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"valid": True, "email": payload.get("sub"), "user_id": payload.get("user_id")}
    except:
        return {"valid": False}

# ============================================
# TRACKING ENDPOINTS
# ============================================

@app.post("/api/track-visit")
async def track_visit(visit: VisitTrack, request: Request):
    """Track page visits (even without login)"""
    conn = get_db()
    c = conn.cursor()
    
    # Get client info
    ip = request.client.host
    user_agent = request.headers.get("user-agent", "")
    
    # Track visit
    c.execute("""
        INSERT INTO visits (session_id, ip_address, user_agent, page_visited)
        VALUES (?, ?, ?, ?)
    """, (visit.session_id, ip, user_agent, visit.page))
    
    conn.commit()
    conn.close()
    
    return {"status": "tracked"}

@app.post("/api/track-prediction")
async def track_prediction(prediction: PredictionInput, request: Request):
    """Track each prediction made"""
    auth_header = request.headers.get("Authorization")
    user_id = None
    
    # Try to get user if logged in
    if auth_header and auth_header.startswith("Bearer "):
        try:
            token = auth_header.replace("Bearer ", "")
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            user_id = payload.get("user_id")
        except:
            pass
    
    # Make prediction using YOUR function
    input_df = pd.DataFrame([{
        "market": prediction.market,
        "arrival_qty": prediction.arrival_qty,
        "price_today": prediction.price_today,
        "price_1d_ago": prediction.price_1d_ago,
        "price_3d_ago": prediction.price_3d_ago,
        "days_since_harvest": prediction.days_since_harvest,
        "max_safe_days": prediction.max_safe_days
    }])
    
    decision = predict_crop_decision(input_df)
    predicted_price = float(input_df["predicted_price"].iloc[0])
    risk_ratio = prediction.days_since_harvest / prediction.max_safe_days
    
    # Store in database
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        INSERT INTO predictions (user_id, market, arrival_qty, price_today, predicted_price, decision, risk_ratio)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, prediction.market, prediction.arrival_qty, prediction.price_today, 
          predicted_price, decision.iloc[0], risk_ratio))
    conn.commit()
    conn.close()
    
    print(f"✅ PREDICTION TRACKED: {prediction.market} - ₹{predicted_price:.2f}")
    
    return {
        "decision": decision.iloc[0],
        "predicted_price": predicted_price,
        "risk_ratio": risk_ratio,
        "tracked": True
    }

# ============================================
# ADMIN ENDPOINTS (FOR YOU TO SEE USER DATA)
# ============================================

@app.get("/api/admin/users")
async def get_users(password: str):
    """Secret admin endpoint - only YOU can access"""
    if password != "YOUR_SECRET_ADMIN_PASSWORD":  # Change this!
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT id, email, full_name, location, created_at, last_login, marketing_consent
        FROM users ORDER BY created_at DESC
    """)
    users = c.fetchall()
    conn.close()
    
    return {"users": [dict(u) for u in users]}

@app.get("/api/admin/stats")
async def get_stats(password: str):
    """Get visit and prediction stats"""
    if password != "YOUR_SECRET_ADMIN_PASSWORD":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    conn = get_db()
    c = conn.cursor()
    
    # Total users
    c.execute("SELECT COUNT(*) as count FROM users")
    total_users = c.fetchone()['count']
    
    # Total visits
    c.execute("SELECT COUNT(*) as count FROM visits")
    total_visits = c.fetchone()['count']
    
    # Total predictions
    c.execute("SELECT COUNT(*) as count FROM predictions")
    total_predictions = c.fetchone()['count']
    
    # Today's users
    c.execute("SELECT COUNT(*) as count FROM users WHERE date(created_at) = date('now')")
    today_users = c.fetchone()['count']
    
    # Email list for campaigns
    c.execute("SELECT email FROM users WHERE marketing_consent = 1")
    emails = [row['email'] for row in c.fetchall()]
    
    conn.close()
    
    return {
        "total_users": total_users,
        "total_visits": total_visits,
        "total_predictions": total_predictions,
        "today_users": today_users,
        "email_list": emails,
        "email_count": len(emails)
    }

@app.get("/api/admin/export-emails")
async def export_emails(password: str):
    """Export emails as CSV for campaigns"""
    if password != "YOUR_SECRET_ADMIN_PASSWORD":
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    conn = get_db()
    c = conn.cursor()
    c.execute("""
        SELECT email, full_name, location, created_at 
        FROM users 
        WHERE marketing_consent = 1 
        ORDER BY created_at DESC
    """)
    emails = c.fetchall()
    conn.close()
    
    # Format as CSV
    csv_data = "Email,Full Name,Location,Registered Date\n"
    for e in emails:
        csv_data += f"{e['email']},{e['full_name'] or ''},{e['location'] or ''},{e['created_at']}\n"
    
    return {"csv": csv_data, "count": len(emails)}

# ============================================
# YOUR ORIGINAL PREDICTION ENDPOINT (UPDATED WITH TRACKING)
# ============================================
@app.post("/predict")
async def predict(data: PredictionInput, request: Request):
    """YOUR original prediction function + tracking"""
    
    # Track this prediction
    track_result = await track_prediction(data, request)
    
    return {
        "decision": track_result["decision"],
        "predicted_price": track_result["predicted_price"],
        "risk_ratio": track_result["risk_ratio"]
    }

# ============================================
# START SERVER
# ============================================
if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*50)
    print("🚀 ECORIPE WITH AUTH IS RUNNING!")
    print("="*50)
    print("\n📧 Email collection is ACTIVE")
    print("👥 User tracking is ENABLED")
    print("📊 Admin panel at /api/admin/stats?password=YOUR_SECRET_ADMIN_PASSWORD")
    print("\n" + "="*50)
    uvicorn.run(app, host="0.0.0.0", port=8000)