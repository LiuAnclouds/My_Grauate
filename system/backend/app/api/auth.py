from __future__ import annotations

from datetime import datetime, timedelta
import secrets

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database import get_db
from app.models import User, VerificationCode
from app.schemas import AuthResponse, LoginCaptchaResponse, LoginRequest, RegisterRequest, VerificationCodeRequest
from app.services.emailer import send_verification_email
from app.services.security import generate_verification_code, hash_secret, verify_secret


router = APIRouter(prefix="/auth", tags=["auth"])

_LOGIN_CAPTCHA_TTL_SECONDS = 300
_LOGIN_CAPTCHA_LENGTH = 5
_LOGIN_CAPTCHA_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"
_SESSION_TTL_DAYS = 7
_login_captcha_store: dict[str, dict[str, str | datetime | bool]] = {}


@router.get("/login-captcha", response_model=LoginCaptchaResponse)
def get_login_captcha() -> LoginCaptchaResponse:
    challenge_id = secrets.token_urlsafe(18)
    captcha_text = "".join(secrets.choice(_LOGIN_CAPTCHA_ALPHABET) for _ in range(_LOGIN_CAPTCHA_LENGTH))
    expires_at = datetime.utcnow() + timedelta(seconds=_LOGIN_CAPTCHA_TTL_SECONDS)
    _prune_captcha_store()
    _login_captcha_store[challenge_id] = {
        "code_hash": hash_secret(captcha_text),
        "expires_at": expires_at,
        "consumed": False,
    }
    return LoginCaptchaResponse(captcha_id=challenge_id, captcha_text=captcha_text, expires_at=expires_at)


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
    if settings.demo_mode:
        return {"message": "验证码已生成。", "code": code}
    message = "验证码已发送邮箱。" if sent else "邮件未发送，验证码已记录在后端日志。"
    return {"message": message}


@router.post("/register", response_model=AuthResponse)
def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> AuthResponse:
    existing = db.scalar(select(User).where(User.email == str(payload.email)))
    if existing is not None:
        raise HTTPException(status_code=409, detail="email already registered")
    _consume_code(db=db, email=str(payload.email), code=payload.code, purpose="register")
    user = User(email=str(payload.email), password_hash=hash_secret(payload.password), is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)
    return _auth_response(user=user, message="注册成功。")


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> AuthResponse:
    user = db.scalar(select(User).where(User.email == str(payload.email)))
    if user is None or not verify_secret(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="invalid email or password")
    _verify_login_captcha(captcha_id=payload.captcha_id, captcha_code=payload.captcha_code)
    return _auth_response(user=user, message="登录成功。" + ("已进入管理员模式。" if user.is_admin else ""))


def _auth_response(*, user: User, message: str) -> AuthResponse:
    return AuthResponse(
        user_id=user.id,
        email=user.email,
        message=message,
        is_admin=user.is_admin,
        session_expires_at=datetime.utcnow() + timedelta(days=_SESSION_TTL_DAYS),
    )


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


def _verify_login_captcha(*, captcha_id: str, captcha_code: str) -> None:
    _prune_captcha_store()
    stored = _login_captcha_store.get(captcha_id)
    normalized_code = captcha_code.strip().upper()
    if stored is None:
        raise HTTPException(status_code=400, detail="登录验证码已失效，请刷新后重试。")
    if bool(stored.get("consumed")):
        raise HTTPException(status_code=400, detail="登录验证码已使用，请刷新后重试。")
    expires_at = stored.get("expires_at")
    if not isinstance(expires_at, datetime) or expires_at < datetime.utcnow():
        _login_captcha_store.pop(captcha_id, None)
        raise HTTPException(status_code=400, detail="登录验证码已过期，请刷新后重试。")
    code_hash = stored.get("code_hash")
    if not isinstance(code_hash, str) or not verify_secret(normalized_code, code_hash):
        raise HTTPException(status_code=400, detail="登录验证码错误。")
    stored["consumed"] = True


def _prune_captcha_store() -> None:
    now = datetime.utcnow()
    expired_ids = [
        captcha_id
        for captcha_id, payload in _login_captcha_store.items()
        if not isinstance(payload.get("expires_at"), datetime) or payload["expires_at"] < now
    ]
    for captcha_id in expired_ids:
        _login_captcha_store.pop(captcha_id, None)
