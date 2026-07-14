import re
import secrets
import time
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Optional

import torch
import torch.nn.functional as F
from fastapi import Depends, FastAPI, Form, HTTPException, Request, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, String,
                        create_engine, func)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, relationship, sessionmaker
from starlette.middleware.sessions import SessionMiddleware
from transformers import AutoModelForSequenceClassification, AutoTokenizer
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


@lru_cache()
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return model, tokenizer


def predict(text: str):
    model, tokenizer = load_model()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()
    conf = probs[0][pred].item() * 100
    
    # cardiffnlp/twitter-roberta-base-sentiment mapping:
    # 0 -> Negative, 1 -> Neutral, 2 -> Positive
    if conf < 90 or pred == 1:
        return "Neutral", conf, "😐"
    elif pred == 2:
        return "Positive", conf, "😊"
    else:
        return "Negative", conf, "😡"


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
        },
    )


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
