from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database import get_db
from app.models import User, VerificationCode
from app.schemas import AuthResponse, LoginRequest, RegisterRequest, VerificationCodeRequest
from app.services.emailer import send_verification_email
from app.services.security import generate_verification_code, hash_secret, verify_secret


router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/request-code")
def request_code(payload: VerificationCodeRequest, db: Session = Depends(get_db)) -> dict[str, str]:
    code = generate_verification_code()
    record = VerificationCode(
        email=str(payload.email),
        code_hash=hash_secret(code),
        purpose=payload.purpose,
        expires_at=datetime.utcnow() + timedelta(minutes=settings.verification_code_ttl_minutes),
    )
    db.add(record)
    db.commit()
    sent = send_verification_email(email=str(payload.email), code=code, purpose=payload.purpose)
    message = "verification code sent" if sent else "verification code logged in backend console"
    return {"message": message}


@router.post("/register", response_model=AuthResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> AuthResponse:
    existing = db.scalar(select(User).where(User.email == str(payload.email)))
    if existing is not None:
        raise HTTPException(status_code=409, detail="email already registered")
    _consume_code(db=db, email=str(payload.email), code=payload.code, purpose="register")
    user = User(email=str(payload.email), password_hash=hash_secret(payload.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    return AuthResponse(user_id=user.id, email=user.email, message="registered")


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> AuthResponse:
    user = db.scalar(select(User).where(User.email == str(payload.email)))
    if user is None or not verify_secret(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="invalid email or password")
    _consume_code(db=db, email=str(payload.email), code=payload.code, purpose="login")
    return AuthResponse(user_id=user.id, email=user.email, message="logged in")


def _consume_code(*, db: Session, email: str, code: str, purpose: str) -> None:
    records = db.scalars(
        select(VerificationCode)
        .where(
            VerificationCode.email == email,
            VerificationCode.purpose == purpose,
            VerificationCode.consumed.is_(False),
            VerificationCode.expires_at >= datetime.utcnow(),
        )
        .order_by(VerificationCode.created_at.desc())
    ).all()
    for record in records:
        if verify_secret(code, record.code_hash):
            record.consumed = True
            db.commit()
            return
    raise HTTPException(status_code=400, detail="invalid or expired verification code")
