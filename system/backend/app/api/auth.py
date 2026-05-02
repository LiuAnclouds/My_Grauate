from __future__ import annotations

from datetime import datetime, timedelta
import secrets

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import settings
from app.database import get_db
from app.models import DatasetUpload, GraphEdge, PersonNode, User, VerificationCode
from app.schemas import (
    AnalystNetworkSummary,
    AnalystSummary,
    AuthResponse,
    LoginCaptchaResponse,
    LoginRequest,
    RegisterRequest,
    VerificationCodeRequest,
)
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
    is_admin = _is_admin_registration(payload.admin_authorization_code)
    user = User(email=str(payload.email), password_hash=hash_secret(payload.password), is_admin=is_admin)
    db.add(user)
    db.commit()
    db.refresh(user)
    return _auth_response(user=user, message="注册成功。" + ("已开通管理员权限。" if user.is_admin else ""))


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)) -> AuthResponse:
    user = db.scalar(select(User).where(User.email == str(payload.email)))
    if user is None or not verify_secret(payload.password, user.password_hash):
        raise HTTPException(status_code=401, detail="invalid email or password")
    _verify_login_captcha(captcha_id=payload.captcha_id, captcha_code=payload.captcha_code)
    return _auth_response(user=user, message="登录成功。" + ("已进入管理员模式。" if user.is_admin else ""))


@router.get("/analysts", response_model=list[AnalystSummary])
def list_analysts(admin_user_id: int, db: Session = Depends(get_db)) -> list[AnalystSummary]:
    admin = db.get(User, admin_user_id)
    if admin is None or not admin.is_admin:
        raise HTTPException(status_code=403, detail="admin permission required")
    users = db.scalars(select(User).order_by(User.created_at.desc())).all()
    return [_analyst_summary(user=user, db=db) for user in users if not user.is_admin]


def _auth_response(*, user: User, message: str) -> AuthResponse:
    return AuthResponse(
        user_id=user.id,
        email=user.email,
        message=message,
        is_admin=user.is_admin,
        session_expires_at=datetime.utcnow() + timedelta(days=_SESSION_TTL_DAYS),
    )


def _is_admin_registration(admin_code: str | None) -> bool:
    code = (admin_code or "").strip()
    if not code:
        return False
    expected_code = settings.admin_registration_code.strip()
    if not expected_code or not secrets.compare_digest(code, expected_code):
        raise HTTPException(status_code=400, detail="invalid admin authorization code")
    return True


def _analyst_summary(*, user: User, db: Session) -> AnalystSummary:
    datasets = db.scalars(
        select(DatasetUpload)
        .where(DatasetUpload.owner_id == user.id)
        .order_by(DatasetUpload.created_at.desc())
    ).all()
    networks = [_network_summary(dataset=dataset, db=db) for dataset in datasets]
    latest = networks[0] if networks else None
    return AnalystSummary(
        user_id=user.id,
        email=user.email,
        is_admin=user.is_admin,
        is_active=user.is_active,
        created_at=user.created_at,
        network_count=len(networks),
        node_count=sum(item.node_count for item in networks),
        edge_count=sum(item.edge_count for item in networks),
        latest_network_name=latest.business_name if latest else None,
        latest_network_at=latest.created_at if latest else None,
        networks=networks,
    )


def _network_summary(*, dataset: DatasetUpload, db: Session) -> AnalystNetworkSummary:
    summary = dict(dataset.summary_json or {})
    business_id = str(summary.get("business_id") or f"BN-{dataset.id:04d}")
    business_name = str(summary.get("business_name") or dataset.name)
    node_count = int(
        summary.get("node_count")
        or db.scalar(select(func.count()).select_from(PersonNode).where(PersonNode.dataset_id == dataset.id))
        or 0
    )
    edge_count = int(
        summary.get("edge_count")
        or db.scalar(select(func.count()).select_from(GraphEdge).where(GraphEdge.dataset_id == dataset.id))
        or 0
    )
    return AnalystNetworkSummary(
        id=dataset.id,
        business_id=business_id,
        business_name=business_name,
        status=dataset.status,
        row_count=dataset.row_count,
        node_count=node_count,
        edge_count=edge_count,
        created_at=dataset.created_at,
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
