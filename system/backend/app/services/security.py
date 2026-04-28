from __future__ import annotations

import secrets

from passlib.context import CryptContext


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_secret(value: str) -> str:
    return pwd_context.hash(value)


def verify_secret(value: str, hashed_value: str) -> bool:
    return pwd_context.verify(value, hashed_value)


def generate_verification_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"
