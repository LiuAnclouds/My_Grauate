from __future__ import annotations

import smtplib
from email.message import EmailMessage

from app.core.config import settings


def send_verification_email(*, email: str, code: str, purpose: str) -> bool:
    subject = "DyRIFT system verification code"
    body = (
        f"Your verification code for {purpose} is {code}. "
        f"It expires in {settings.verification_code_ttl_minutes} minutes."
    )
    if not settings.smtp_host:
        print(f"[dev-email] to={email} purpose={purpose} code={code}")
        return False

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = settings.smtp_from or settings.smtp_username
    message["To"] = email
    message.set_content(body)

    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=20) as smtp:
        if settings.smtp_use_tls:
            smtp.starttls()
        if settings.smtp_username:
            smtp.login(settings.smtp_username, settings.smtp_password)
        smtp.send_message(message)
    return True
