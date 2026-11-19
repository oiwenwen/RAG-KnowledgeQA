import hashlib
import hmac
import secrets


def hash_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    if len(password) < 10:
        raise ValueError("密码至少需要 10 个字符")
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
    return salt, digest


def verify_password(password: str, salt: bytes, expected: bytes) -> bool:
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 310_000)
    return hmac.compare_digest(actual, expected)

