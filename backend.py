import re
import secrets
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

import io
import os
import requests
import pandas as pd
import plotly.graph_objects as go
from fastapi import Depends, FastAPI, Form, HTTPException, Request, status, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from googleapiclient.discovery import build
from fastapi.templating import Jinja2Templates
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, String,
                        create_engine, func)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker
from starlette.middleware.sessions import SessionMiddleware
from pydantic import BaseModel

SECRET_KEY = "change-this-secret-for-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

engine = create_engine("sqlite:///./saas.db", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI(title="SentimentIQ SaaS Backend")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY, session_cookie="sentimentiq_session", max_age=86400)

@app.on_event("startup")
def startup_event():
    print("Starting FastAPI Backend (Using HuggingFace Cloud API)")

templates = Jinja2Templates(directory="templates")


class Tenant(Base):
    __tablename__ = "tenants"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    slug = Column(String, unique=True, nullable=False)
    users = relationship("User", back_populates="tenant")
    predictions = relationship("Prediction", back_populates="tenant")


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False)
    tenant = relationship("Tenant", back_populates="users")
    predictions = relationship("Prediction", back_populates="user")


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    text = Column(String, nullable=False)
    sentiment = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    tenant = relationship("Tenant", back_populates="predictions")
    user = relationship("User", back_populates="predictions")


Base.metadata.create_all(bind=engine, checkfirst=True)


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    sentiment: str
    confidence: float
    emoji: str
    tenant: str
    user: str


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")


def is_valid_text(text: str):
    t = text.strip()
    if len(t) < 5:
        return False, "Text is too short!", "Write at least a few meaningful words."
    if re.match(r"^[\d\s\W]+$", t):
        return False, "Only numbers or symbols detected!", "Please enter actual words."
    words = re.findall(r"[a-zA-Z]{3,}", t)
    if len(words) < 2:
        return False, "Not enough words to analyze!", "Try writing a proper sentence."
    if re.findall(r"[^aeiouAEIOU\s]{6,}", t):
        return False, "Gibberish detected!", "Please enter a meaningful English sentence."
    return True, "", ""

PBASE = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#4a5e7a', family='DM Sans'),
    margin=dict(l=0, r=0, t=10, b=0)
)

def sentiment_charts(pos, neu, neg, tot):
    fig = go.Figure(go.Pie(
        labels=["Positive","Neutral","Negative"], values=[pos, neu, neg], hole=0.60,
        marker=dict(colors=["#05f0a0","#ffc542","#ff3f5b"], line=dict(color="#070b14", width=3)),
        textinfo="percent", textfont=dict(size=12, color="white"),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>"
    ))
    fig.update_layout(**PBASE, height=240, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, font=dict(color="#e8edf5", size=11), bgcolor="rgba(0,0,0,0)"),
        annotations=[dict(text=f"<b>{tot}</b><br>texts", x=0.5, y=0.5, showarrow=False, font=dict(size=13, color="#e8edf5"))]
    )
    pie_html = fig.to_html(full_html=False, include_plotlyjs=False)
    
    fig2 = go.Figure(go.Bar(
        x=["Positive","Neutral","Negative"], y=[pos, neu, neg],
        marker=dict(color=["#05f0a0","#ffc542","#ff3f5b"], line=dict(color="rgba(0,0,0,0)", width=0)),
        text=[pos, neu, neg], textposition="outside", textfont=dict(color="#e8edf5", size=13),
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
    ))
    fig2.update_layout(**PBASE, height=240, bargap=0.38,
        xaxis=dict(showgrid=False, color="#4a5e7a", tickfont=dict(size=12)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#4a5e7a", showline=False))
    bar_html = fig2.to_html(full_html=False, include_plotlyjs=False)
    return pie_html, bar_html

TEXT_COLUMN_ALIASES = ["text", "tweet", "tweet_text", "content", "body", "post", "message", "comment"]

def normalize_text_column(df):
    lower_to_original = {str(column).lower().strip(): column for column in df.columns}
    for column in TEXT_COLUMN_ALIASES:
        original = lower_to_original.get(column)
        if original is None:
            continue
        if original == "text":
            return df
        return df.rename(columns={original: "text"})
    return None


HF_TOKEN = os.environ.get("HF_TOKEN")

def predict(text: str):
    # Use HuggingFace Inference API instead of local PyTorch model to save RAM
    API_URL = "https://api-inference.huggingface.co/models/cardiffnlp/twitter-roberta-base-sentiment"
    headers = {}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"
    
    try:
        response = requests.post(API_URL, headers=headers, json={"inputs": text}, timeout=10)
        # Handle cases where model is waking up on Hugging Face
        if response.status_code == 503:
            time.sleep(3)
            response = requests.post(API_URL, headers=headers, json={"inputs": text}, timeout=10)
            
        response.raise_for_status()
        data = response.json()
        
        # Format: [[{"label":"LABEL_2","score":0.95}, ...]]
        if not data or not isinstance(data, list) or not data[0]:
            raise ValueError("Invalid response from API")
            
        predictions = data[0]
        predictions.sort(key=lambda x: x.get("score", 0), reverse=True)
        top_pred = predictions[0]
        
        label_str = top_pred.get("label", "")
        conf = top_pred.get("score", 0) * 100
        
        # cardiffnlp/twitter-roberta-base-sentiment mapping:
        # LABEL_0 -> Negative, LABEL_1 -> Neutral, LABEL_2 -> Positive
        if conf < 90 or label_str == "LABEL_1":
            return "Neutral", conf, "😐"
        elif label_str == "LABEL_2":
            return "Positive", conf, "😊"
        else:
            return "Negative", conf, "😡"
            
    except Exception as e:
        print(f"HF API Error: {e}")
        return "Neutral", 0.0, "😐"


def get_dashboard_context(user: User, db: Session):
    tenant = db.query(Tenant).filter(Tenant.id == user.tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Tenant not found")

    total = db.query(func.count(Prediction.id)).filter(Prediction.tenant_id == tenant.id).scalar() or 0
    positive = db.query(func.count(Prediction.id)).filter(Prediction.tenant_id == tenant.id, Prediction.sentiment == "Positive").scalar() or 0
    neutral = db.query(func.count(Prediction.id)).filter(Prediction.tenant_id == tenant.id, Prediction.sentiment == "Neutral").scalar() or 0
    negative = db.query(func.count(Prediction.id)).filter(Prediction.tenant_id == tenant.id, Prediction.sentiment == "Negative").scalar() or 0
    recent = db.query(Prediction).filter(Prediction.tenant_id == tenant.id).order_by(Prediction.created_at.desc()).limit(10).all()

    return {
        "tenant": tenant,
        "metrics": {
            "total": total,
            "positive": positive,
            "neutral": neutral,
            "negative": negative,
            "positive_pct": f"{(positive / total * 100):.1f}%" if total else "0%",
            "neutral_pct": f"{(neutral / total * 100):.1f}%" if total else "0%",
            "negative_pct": f"{(negative / total * 100):.1f}%" if total else "0%",
        },
        "recent": recent,
    }


def get_session_user(request: Request, db: Session) -> Optional[User]:
    token = request.session.get("token")
    if not token:
        auth = request.headers.get("Authorization")
        if auth and auth.startswith("Bearer "):
            token = auth.split(" ", 1)[1]
    if not token:
        return None

    payload = decode_access_token(token)
    user_id = payload.get("sub")
    if not user_id:
        return None

    return db.query(User).filter(User.id == int(user_id)).first()


@app.get("/", response_class=RedirectResponse)
def home() -> RedirectResponse:
    return RedirectResponse(url="/login")


@app.get("/login", response_class=HTMLResponse)
def get_login(request: Request):
    return templates.TemplateResponse(request=request, name="login.html", context={"request": request, "error": None, "message": request.query_params.get("message", "")})


@app.post("/login", response_class=HTMLResponse)
def post_login(request: Request, email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email.lower().strip()).first()
    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse(request=request, name="login.html", context={"request": request, "error": "Invalid credentials.", "message": ""})

    token = create_access_token({"sub": str(user.id), "tenant_id": str(user.tenant_id)})
    request.session["token"] = token
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/register", response_class=HTMLResponse)
def get_register(request: Request):
    return templates.TemplateResponse(request=request, name="register.html", context={"request": request, "error": None})


@app.post("/register", response_class=HTMLResponse)
def post_register(
    request: Request,
    name: str = Form(...),
    email: str = Form(...),
    tenant_name: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    email_value = email.lower().strip()
    tenant_slug = slugify(tenant_name)
    if db.query(User).filter(User.email == email_value).first():
        return templates.TemplateResponse(request=request, name="register.html", context={"request": request, "error": "This email is already registered."})

    tenant = db.query(Tenant).filter(Tenant.slug == tenant_slug).first()
    if not tenant:
        tenant = Tenant(name=tenant_name.strip(), slug=tenant_slug)
        db.add(tenant)
        db.commit()
        db.refresh(tenant)

    user = User(
        email=email_value,
        name=name.strip(),
        hashed_password=get_password_hash(password),
        tenant_id=tenant.id,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return RedirectResponse(url="/login?message=Account created successfully. Please sign in.", status_code=status.HTTP_303_SEE_OTHER)


@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard(request: Request, db: Session = Depends(get_db)):
    user = get_session_user(request, db)
    if not user:
        return RedirectResponse(url="/login")

    context = get_dashboard_context(user, db)
    return templates.TemplateResponse(
        request=request, name="dashboard.html",
        context={
            "request": request,
            "user": user,
            "tenant": context["tenant"],
            "metrics": context["metrics"],
            "recent": context["recent"],
            "analysis": None,
            "active_tab": "text"
        },
    )


@app.post("/dashboard", response_class=HTMLResponse)
def post_dashboard(request: Request, text: str = Form(...), db: Session = Depends(get_db)):
    user = get_session_user(request, db)
    if not user:
        return RedirectResponse(url="/login")

    valid, error_title, error_hint = is_valid_text(text)
    analysis = None
    if valid:
        sentiment, confidence, emoji = predict(text)
        record = Prediction(
            tenant_id=user.tenant_id,
            user_id=user.id,
            text=text.strip(),
            sentiment=sentiment,
            confidence=round(confidence, 1),
        )
        db.add(record)
        db.commit()
        analysis = {
            "text": text,
            "sentiment": sentiment,
            "confidence": f"{confidence:.1f}%",
            "emoji": emoji,
            "message": "Prediction saved to your tenant dashboard.",
        }
    else:
        analysis = {"error": error_title, "hint": error_hint}

    context = get_dashboard_context(user, db)
    return templates.TemplateResponse(
        request=request, name="dashboard.html",
        context={
            "request": request,
            "user": user,
            "tenant": context["tenant"],
            "metrics": context["metrics"],
            "recent": context["recent"],
            "analysis": analysis,
            "active_tab": "text"
        },
    )

@app.post("/dashboard/csv", response_class=HTMLResponse)
async def post_dashboard_csv(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    user = get_session_user(request, db)
    if not user:
        return RedirectResponse(url="/login")
        
    analysis = None
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        df = normalize_text_column(df)
        if df is None:
            analysis = {"error": "Invalid CSV format", "hint": "CSV must include a text column named 'text', 'tweet', 'content', etc."}
        else:
            results, records = [], []
            for txt in df["text"].astype(str):
                valid, _, _ = is_valid_text(txt)
                if valid:
                    sentiment, conf, emoji = predict(txt)
                    records.append(Prediction(tenant_id=user.tenant_id, user_id=user.id, text=txt[:500], sentiment=sentiment, confidence=round(conf, 1)))
                else:
                    sentiment, conf, emoji = "Invalid", 0.0, "⚠️"
                results.append({"text": txt, "sentiment": sentiment, "confidence": round(conf,1), "emoji": emoji})
                
            if records:
                db.bulk_save_objects(records)
                db.commit()
            
            valid_results = [r for r in results if r["sentiment"] != "Invalid"]
            pos = sum(1 for r in valid_results if r["sentiment"] == "Positive")
            neu = sum(1 for r in valid_results if r["sentiment"] == "Neutral")
            neg = sum(1 for r in valid_results if r["sentiment"] == "Negative")
            tot = len(valid_results)
            
            pie_chart, bar_chart = sentiment_charts(pos, neu, neg, tot)
            
            analysis = {
                "type": "csv", "total": tot, "positive": pos, "neutral": neu, "negative": neg,
                "pie_chart": pie_chart, "bar_chart": bar_chart, "data": valid_results[:20]
            }
    except Exception as e:
        analysis = {"error": "Error processing CSV", "hint": str(e)}

    context = get_dashboard_context(user, db)
    return templates.TemplateResponse(request=request, name="dashboard.html", context={
        "request": request, "user": user, "tenant": context["tenant"], "metrics": context["metrics"], 
        "recent": context["recent"], "analysis": analysis, "active_tab": "csv"
    })

YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")

@app.post("/dashboard/youtube", response_class=HTMLResponse)
def post_dashboard_youtube(request: Request, yt_url: str = Form(...), comment_limit: int = Form(50), db: Session = Depends(get_db)):
    user = get_session_user(request, db)
    if not user:
        return RedirectResponse(url="/login")
        
    analysis = None
    if not YOUTUBE_API_KEY:
        analysis = {"error": "YouTube API Key missing", "hint": "Configure YOUTUBE_API_KEY in environment."}
    else:
        match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", yt_url)
        if not match:
            analysis = {"error": "Invalid YouTube URL", "hint": "Please paste a valid YouTube video link."}
        else:
            try:
                video_id = match.group(1)
                youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
                response = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=comment_limit, textFormat="plainText").execute()
                
                comments = []
                for item in response.get("items", []):
                    snippet = item["snippet"]["topLevelComment"]["snippet"]
                    comments.append({"text": snippet["textDisplay"], "likes": snippet["likeCount"]})
                
                if not comments:
                    analysis = {"error": "No comments found", "hint": "This video might have comments disabled."}
                else:
                    results, records = [], []
                    for c in comments:
                        txt = c["text"]
                        valid, _, _ = is_valid_text(txt)
                        if valid:
                            sentiment, conf, emoji = predict(txt)
                            records.append(Prediction(tenant_id=user.tenant_id, user_id=user.id, text=txt[:500], sentiment=sentiment, confidence=round(conf, 1)))
                        else:
                            sentiment, conf, emoji = "Neutral", 60.0, "😐"
                        results.append({"text": txt, "sentiment": sentiment, "confidence": round(conf, 1), "emoji": emoji, "likes": c["likes"]})
                    
                    if records:
                        db.bulk_save_objects(records)
                        db.commit()
                        
                    results.sort(key=lambda x: x.get("likes", 0), reverse=True)
                    
                    pos = sum(1 for r in results if r["sentiment"] == "Positive")
                    neu = sum(1 for r in results if r["sentiment"] == "Neutral")
                    neg = sum(1 for r in results if r["sentiment"] == "Negative")
                    tot = len(results)
                    
                    pie_chart, bar_chart = sentiment_charts(pos, neu, neg, tot)
                    
                    analysis = {
                        "type": "youtube", "total": tot, "positive": pos, "neutral": neu, "negative": neg,
                        "pie_chart": pie_chart, "bar_chart": bar_chart, "data": results[:20]
                    }
                    
            except Exception as e:
                analysis = {"error": "YouTube API Error", "hint": str(e)}

    context = get_dashboard_context(user, db)
    return templates.TemplateResponse(request=request, name="dashboard.html", context={
        "request": request, "user": user, "tenant": context["tenant"], "metrics": context["metrics"], 
        "recent": context["recent"], "analysis": analysis, "active_tab": "youtube"
    })


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/login")


@app.post("/api/login", response_model=LoginResponse)
def api_login(payload: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email.lower().strip()).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password")

    token = create_access_token({"sub": str(user.id), "tenant_id": str(user.tenant_id)})
    return {"access_token": token, "token_type": "bearer"}


@app.post("/api/register")
def api_register(payload: LoginRequest, tenant_name: str = Form(...), db: Session = Depends(get_db)):
    email_value = payload.email.lower().strip()
    if db.query(User).filter(User.email == email_value).first():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")

    tenant_slug = slugify(tenant_name)
    tenant = db.query(Tenant).filter(Tenant.slug == tenant_slug).first()
    if not tenant:
        tenant = Tenant(name=tenant_name.strip(), slug=tenant_slug)
        db.add(tenant)
        db.commit()
        db.refresh(tenant)

    user = User(
        email=email_value,
        name=payload.email.split("@")[0],
        hashed_password=get_password_hash(payload.password),
        tenant_id=tenant.id,
    )
    db.add(user)
    db.commit()
    return JSONResponse({"message": "Tenant registered successfully", "tenant": tenant.name})


def get_current_user_from_token(request: Request, db: Session = Depends(get_db)) -> User:
    auth = request.headers.get("Authorization")
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authorization token")
    token = auth.split(" ", 1)[1]
    payload = decode_access_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


@app.post("/api/predict", response_model=PredictResponse)
def api_predict(request: Request, payload: PredictRequest, user: User = Depends(get_current_user_from_token), db: Session = Depends(get_db)):
    valid, _, _ = is_valid_text(payload.text)
    if not valid:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid text")

    sentiment, confidence, emoji = predict(payload.text)
    record = Prediction(
        tenant_id=user.tenant_id,
        user_id=user.id,
        text=payload.text.strip(),
        sentiment=sentiment,
        confidence=round(confidence, 1),
    )
    db.add(record)
    db.commit()

    return PredictResponse(
        sentiment=sentiment,
        confidence=round(confidence, 1),
        emoji=emoji,
        tenant=user.tenant.name,
        user=user.email,
    )


@app.get("/api/stats")
def api_stats(user: User = Depends(get_current_user_from_token), db: Session = Depends(get_db)):
    total = db.query(func.count(Prediction.id)).filter(Prediction.tenant_id == user.tenant_id).scalar() or 0
    positive = db.query(func.count(Prediction.id)).filter(Prediction.tenant_id == user.tenant_id, Prediction.sentiment == "Positive").scalar() or 0
    neutral = db.query(func.count(Prediction.id)).filter(Prediction.tenant_id == user.tenant_id, Prediction.sentiment == "Neutral").scalar() or 0
    negative = db.query(func.count(Prediction.id)).filter(Prediction.tenant_id == user.tenant_id, Prediction.sentiment == "Negative").scalar() or 0
    return {
        "tenant": user.tenant.name,
        "total_predictions": total,
        "positive": positive,
        "neutral": neutral,
        "negative": negative,
    }
