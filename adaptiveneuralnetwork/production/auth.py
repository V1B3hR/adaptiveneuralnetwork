"""
Authentication and authorization systems for production APIs.
"""

import hashlib
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import jwt

# Optional auth dependencies
try:
    from passlib.context import CryptContext
    from passlib.hash import bcrypt
    PASSLIB_AVAILABLE = True
except ImportError:
    PASSLIB_AVAILABLE = False
    class CryptContext:
        def __init__(self, *args, **kwargs):
            pass
        def hash(self, password):
            return password
        def verify(self, password, hashed):
            return password == hashed

try:
    from authlib.integrations.httpx_client import AsyncOAuth2Client
    from authlib.oauth2.rfc6749 import OAuth2Token
    AUTHLIB_AVAILABLE = True
except ImportError:
    AUTHLIB_AVAILABLE = False
    class AsyncOAuth2Client:
        pass
    class OAuth2Token:
        pass


@dataclass
class AuthConfig:
    """Configuration for authentication systems."""
    # JWT config
    jwt_secret_key: str = secrets.token_urlsafe(32)
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7

    # OAuth2 config
    oauth2_client_id: str | None = None
    oauth2_client_secret: str | None = None
    oauth2_server_url: str | None = None
    oauth2_redirect_uri: str | None = None

    # API key config
    api_key_length: int = 32
    api_key_prefix: str = "ann_"

    # Password config
    password_min_length: int = 8
    password_require_special: bool = True
    password_require_numbers: bool = True
    password_require_uppercase: bool = True

    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window_minutes: int = 15


@dataclass
class User:
    """User model for authentication."""
    id: str
    username: str
    email: str
    is_active: bool = True
    is_admin: bool = False
    scopes: list[str] = None
    created_at: datetime = None
    last_login: datetime = None

    def __post_init__(self):
        if self.scopes is None:
            self.scopes = ["read"]
        if self.created_at is None:
            self.created_at = datetime.utcnow()


@dataclass
class APIKey:
    """API key model."""
    key_id: str
    key_hash: str
    name: str
    user_id: str
    scopes: list[str]
    is_active: bool = True
    created_at: datetime = None
    last_used: datetime = None
    expires_at: datetime | None = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class AuthManager:
    """Base authentication manager."""

    def __init__(self, config: AuthConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize password context
        if PASSLIB_AVAILABLE:
            self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        else:
            self.pwd_context = None
            self.logger.warning("Passlib not available, password hashing disabled")

        # In-memory storage (replace with database in production)
        self.users = {}
        self.api_keys = {}
        self.rate_limits = {}

    def hash_password(self, password: str) -> str:
        """Hash a password."""
        if not self.pwd_context:
            return hashlib.sha256(password.encode()).hexdigest()
        return self.pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        if not self.pwd_context:
            return hashlib.sha256(plain_password.encode()).hexdigest() == hashed_password
        return self.pwd_context.verify(plain_password, hashed_password)

    def validate_password_strength(self, password: str) -> bool:
        """Validate password meets strength requirements."""
        if len(password) < self.config.password_min_length:
            return False

        if self.config.password_require_uppercase and not any(c.isupper() for c in password):
            return False

        if self.config.password_require_numbers and not any(c.isdigit() for c in password):
            return False

        if self.config.password_require_special and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            return False

        return True

    def generate_api_key(self) -> str:
        """Generate a new API key."""
        key = secrets.token_urlsafe(self.config.api_key_length)
        return f"{self.config.api_key_prefix}{key}"

    def hash_api_key(self, api_key: str) -> str:
        """Hash an API key for storage."""
        return hashlib.sha256(api_key.encode()).hexdigest()

    def create_user(self, username: str, email: str, password: str,
                   is_admin: bool = False, scopes: list[str] | None = None) -> User:
        """Create a new user."""
        if not self.validate_password_strength(password):
            raise ValueError("Password does not meet strength requirements")

        user_id = secrets.token_urlsafe(16)
        user = User(
            id=user_id,
            username=username,
            email=email,
            is_admin=is_admin,
            scopes=scopes or ["read"]
        )

        # Store user with hashed password
        self.users[user_id] = {
            "user": user,
            "password_hash": self.hash_password(password)
        }

        return user

    def authenticate_user(self, username: str, password: str) -> User | None:
        """Authenticate a user with username and password."""
        for user_data in self.users.values():
            user = user_data["user"]
            if user.username == username and user.is_active:
                if self.verify_password(password, user_data["password_hash"]):
                    user.last_login = datetime.utcnow()
                    return user
        return None

    def create_api_key(self, user_id: str, name: str, scopes: list[str],
                      expires_days: int | None = None) -> tuple[str, APIKey]:
        """Create a new API key for a user."""
        if user_id not in self.users:
            raise ValueError("User not found")

        api_key = self.generate_api_key()
        key_id = secrets.token_urlsafe(16)

        expires_at = None
        if expires_days:
            expires_at = datetime.utcnow() + timedelta(days=expires_days)

        api_key_obj = APIKey(
            key_id=key_id,
            key_hash=self.hash_api_key(api_key),
            name=name,
            user_id=user_id,
            scopes=scopes,
            expires_at=expires_at
        )

        self.api_keys[key_id] = api_key_obj

        return api_key, api_key_obj

    def authenticate_api_key(self, api_key: str) -> tuple[User, APIKey] | None:
        """Authenticate using an API key."""
        key_hash = self.hash_api_key(api_key)

        for api_key_obj in self.api_keys.values():
            if (api_key_obj.key_hash == key_hash and
                api_key_obj.is_active and
                (api_key_obj.expires_at is None or api_key_obj.expires_at > datetime.utcnow())):

                # Update last used
                api_key_obj.last_used = datetime.utcnow()

                # Get user
                user_data = self.users.get(api_key_obj.user_id)
                if user_data and user_data["user"].is_active:
                    return user_data["user"], api_key_obj

        return None

    def check_rate_limit(self, identifier: str) -> bool:
        """Check if request is within rate limits."""
        now = datetime.utcnow()
        window_start = now - timedelta(minutes=self.config.rate_limit_window_minutes)

        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = []

        # Clean old requests
        self.rate_limits[identifier] = [
            req_time for req_time in self.rate_limits[identifier]
            if req_time > window_start
        ]

        # Check limit
        if len(self.rate_limits[identifier]) >= self.config.rate_limit_requests:
            return False

        # Record this request
        self.rate_limits[identifier].append(now)
        return True

    def has_scope(self, user_scopes: list[str], required_scope: str) -> bool:
        """Check if user has required scope."""
        return required_scope in user_scopes or "admin" in user_scopes


class JWTAuth(AuthManager):
    """JWT-based authentication."""

    def create_access_token(self, user: User) -> str:
        """Create a JWT access token."""
        expire = datetime.utcnow() + timedelta(minutes=self.config.jwt_access_token_expire_minutes)

        payload = {
            "sub": user.id,
            "username": user.username,
            "scopes": user.scopes,
            "is_admin": user.is_admin,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "access"
        }

        return jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)

    def create_refresh_token(self, user: User) -> str:
        """Create a JWT refresh token."""
        expire = datetime.utcnow() + timedelta(days=self.config.jwt_refresh_token_expire_days)

        payload = {
            "sub": user.id,
            "exp": expire,
            "iat": datetime.utcnow(),
            "type": "refresh"
        }

        return jwt.encode(payload, self.config.jwt_secret_key, algorithm=self.config.jwt_algorithm)

    def verify_token(self, token: str) -> dict[str, Any] | None:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm]
            )

            # Check if token is expired
            if datetime.utcfromtimestamp(payload["exp"]) < datetime.utcnow():
                return None

            return payload

        except jwt.ExpiredSignatureError:
            self.logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError as e:
            self.logger.warning(f"Invalid token: {e}")
            return None

    def get_user_from_token(self, token: str) -> User | None:
        """Get user from JWT token."""
        payload = self.verify_token(token)
        if not payload or payload.get("type") != "access":
            return None

        user_id = payload.get("sub")
        if user_id in self.users:
            return self.users[user_id]["user"]

        return None

    def refresh_access_token(self, refresh_token: str) -> str | None:
        """Create new access token from refresh token."""
        payload = self.verify_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            return None

        user_id = payload.get("sub")
        if user_id in self.users:
            user = self.users[user_id]["user"]
            return self.create_access_token(user)

        return None


class OAuth2Auth(AuthManager):
    """OAuth2-based authentication."""

    def __init__(self, config: AuthConfig):
        if not AUTHLIB_AVAILABLE:
            raise ImportError("Authlib not available. Install with: pip install authlib httpx")

        super().__init__(config)

        if not all([config.oauth2_client_id, config.oauth2_client_secret, config.oauth2_server_url]):
            raise ValueError("OAuth2 configuration incomplete")

        self.client = AsyncOAuth2Client(
            client_id=config.oauth2_client_id,
            client_secret=config.oauth2_client_secret
        )

    def get_authorization_url(self, state: str | None = None) -> str:
        """Get OAuth2 authorization URL."""
        if not state:
            state = secrets.token_urlsafe(16)

        authorization_endpoint = f"{self.config.oauth2_server_url}/authorize"

        return self.client.create_authorization_url(
            authorization_endpoint,
            redirect_uri=self.config.oauth2_redirect_uri,
            state=state,
            scope="read write"
        )[0]

    async def exchange_code_for_token(self, code: str, state: str | None = None) -> OAuth2Token | None:
        """Exchange authorization code for access token."""
        try:
            token_endpoint = f"{self.config.oauth2_server_url}/token"

            token = await self.client.fetch_token(
                token_endpoint,
                code=code,
                redirect_uri=self.config.oauth2_redirect_uri
            )

            return token

        except Exception as e:
            self.logger.error(f"Token exchange failed: {e}")
            return None

    async def get_user_info(self, token: OAuth2Token) -> dict[str, Any] | None:
        """Get user information from OAuth2 provider."""
        try:
            userinfo_endpoint = f"{self.config.oauth2_server_url}/userinfo"

            response = await self.client.get(
                userinfo_endpoint,
                token=token
            )

            if response.status_code == 200:
                return response.json()

            return None

        except Exception as e:
            self.logger.error(f"Failed to get user info: {e}")
            return None

    async def authenticate_oauth2(self, code: str) -> User | None:
        """Authenticate user via OAuth2 flow."""
        token = await self.exchange_code_for_token(code)
        if not token:
            return None

        user_info = await self.get_user_info(token)
        if not user_info:
            return None

        # Create or update user
        username = user_info.get("username") or user_info.get("preferred_username")
        email = user_info.get("email")

        if not username or not email:
            return None

        # Check if user exists
        for user_data in self.users.values():
            user = user_data["user"]
            if user.email == email:
                user.last_login = datetime.utcnow()
                return user

        # Create new user
        user = User(
            id=secrets.token_urlsafe(16),
            username=username,
            email=email,
            scopes=["read", "write"]
        )

        self.users[user.id] = {
            "user": user,
            "password_hash": None,  # OAuth2 users don't have passwords
            "oauth2_token": token
        }

        return user


class MultiAuthManager:
    """Multi-method authentication manager."""

    def __init__(self, config: AuthConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize auth methods
        self.jwt_auth = JWTAuth(config)

        self.oauth2_auth = None
        if all([config.oauth2_client_id, config.oauth2_client_secret, config.oauth2_server_url]):
            try:
                self.oauth2_auth = OAuth2Auth(config)
            except Exception as e:
                self.logger.warning(f"OAuth2 initialization failed: {e}")

    def authenticate(self, auth_type: str, **kwargs) -> User | None:
        """Authenticate using specified method."""
        if auth_type == "password":
            return self.jwt_auth.authenticate_user(
                kwargs.get("username"),
                kwargs.get("password")
            )

        elif auth_type == "api_key":
            result = self.jwt_auth.authenticate_api_key(kwargs.get("api_key"))
            return result[0] if result else None

        elif auth_type == "jwt":
            return self.jwt_auth.get_user_from_token(kwargs.get("token"))

        elif auth_type == "oauth2" and self.oauth2_auth:
            # This would be called asynchronously
            return None

        return None

    def create_tokens(self, user: User) -> dict[str, str]:
        """Create both access and refresh tokens."""
        return {
            "access_token": self.jwt_auth.create_access_token(user),
            "refresh_token": self.jwt_auth.create_refresh_token(user),
            "token_type": "bearer"
        }
