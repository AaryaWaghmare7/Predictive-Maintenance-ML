"""Small single-process access/resource boundary; not a multi-user auth service."""

from collections import deque
import os
import secrets
import time

from fastapi.responses import JSONResponse


class AccessPolicy:
    def __init__(self):
        mode = os.getenv("EVISION_MODE", "local")
        if mode not in ("local", "production"):
            raise ValueError("EVISION_MODE must be local or production")
        self.production = mode == "production"
        self.token = os.getenv("EVISION_API_TOKEN", "")
        if self.token and (not 32 <= len(self.token) <= 256 or not self.token.isascii() or any(c.isspace() or ord(c) < 33 for c in self.token)):
            raise ValueError("EVISION_API_TOKEN must contain 32–256 non-whitespace printable ASCII characters")
        self.hosts = {h.strip().lower() for h in os.getenv("EVISION_ALLOWED_HOSTS", "localhost,127.0.0.1,[::1],testserver").split(",")}
        if self.production and (len(self.token) < 32 or not os.getenv("EVISION_ALLOWED_HOSTS")):
            raise ValueError("Production requires EVISION_API_TOKEN (at least 32 characters) and explicit EVISION_ALLOWED_HOSTS; secrets are never printed")
        self.window_seconds = 60
        self.manual_limit = 60
        self.upload_limit = 20
        self.requests = {}
        self.active_uploads = 0

    def check(self, scope):
        headers = dict(scope["headers"])
        host = headers.get(b"host", b"").decode("latin1").lower()
        hostname = host.rsplit(":", 1)[0] if ":" in host and not host.endswith("]") else host
        if hostname not in self.hosts:
            return 400, "Host is not allowed"
        if scope["method"] != "POST":
            return None
        origin = headers.get(b"origin")
        expected = f'{scope.get("scheme", "http")}://{host}'.encode()
        if origin and origin != expected:
            return 403, "Cross-origin requests are not allowed"
        if self.token:
            provided = headers.get(b"authorization", b"")
            if not secrets.compare_digest(provided, ("Bearer " + self.token).encode()):
                return 401, "A valid API bearer token is required"
        path = scope["path"]
        if path not in ("/predict-fault", "/analyze-csv", "/analyze-dataset"):
            return None
        content_type = headers.get(b"content-type", b"").split(b";")[0].strip().lower()
        wanted = b"application/json" if path == "/predict-fault" else b"multipart/form-data"
        # Let missing form/JSON bodies use the existing useful 422 response.
        if content_type and content_type != wanted:
            return 415, "Use JSON for manual prediction or multipart/form-data for uploads"
        now = time.monotonic()
        self.requests = {key: queue for key, queue in self.requests.items() if queue and now-queue[-1] < self.window_seconds}
        key = (scope.get("client", ("unknown",))[0], "manual" if path == "/predict-fault" else "upload")
        if key not in self.requests and len(self.requests) >= 1024:
            return 429, "Request capacity reached; retry later"
        queue = self.requests.setdefault(key, deque())
        while queue and now-queue[0] >= self.window_seconds:
            queue.popleft()
        limit = self.manual_limit if key[1] == "manual" else self.upload_limit
        if len(queue) >= limit:
            return 429, "Rate limit reached; retry after one minute"
        queue.append(now)
        return None

    @staticmethod
    def rejection(status, detail):
        return JSONResponse({"detail": detail}, status_code=status,
                            headers={"Retry-After": "60"} if status == 429 else {})
