"""Structured and validated RTSP endpoint configuration for Arnesis."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from urllib.parse import quote, urlsplit, urlunsplit


class RtspEndpointError(ValueError):
    """Raised when a camera RTSP endpoint is invalid."""


@dataclass(frozen=True, slots=True)
class RtspEndpoint:
    host: str
    port: int = 554
    username: str = "admin"
    stream_path: str = "/Streaming/Channels/101"
    secure: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "host", self._validate_host(self.host))
        if not 1 <= self.port <= 65535:
            raise RtspEndpointError("RTSP port must be between 1 and 65535.")
        if not self.username.strip():
            raise RtspEndpointError("RTSP username is required.")
        if any(character in self.username for character in "\r\n"):
            raise RtspEndpointError("RTSP username contains invalid characters.")
        object.__setattr__(self, "username", self.username.strip())
        object.__setattr__(self, "stream_path", self._validate_path(self.stream_path))

    @property
    def scheme(self) -> str:
        return "rtsps" if self.secure else "rtsp"

    def build_url(self, password: str) -> str:
        if not password:
            raise RtspEndpointError("RTSP password is required.")
        encoded_user = quote(self.username, safe="")
        encoded_password = quote(password, safe="")
        host = f"[{self.host}]" if ":" in self.host else self.host
        netloc = f"{encoded_user}:{encoded_password}@{host}:{self.port}"
        return urlunsplit((self.scheme, netloc, self.stream_path, "", ""))

    def masked_url(self) -> str:
        encoded_user = quote(self.username, safe="")
        host = f"[{self.host}]" if ":" in self.host else self.host
        netloc = f"{encoded_user}:*****@{host}:{self.port}"
        return urlunsplit((self.scheme, netloc, self.stream_path, "", ""))

    def database_uri(self) -> str:
        """Return a password-free URI safe for arn_camera.connection_uri."""
        encoded_user = quote(self.username, safe="")
        host = f"[{self.host}]" if ":" in self.host else self.host
        netloc = f"{encoded_user}@{host}:{self.port}"
        return urlunsplit((self.scheme, netloc, self.stream_path, "", ""))

    @classmethod
    def from_database_uri(cls, uri: str) -> RtspEndpoint:
        parsed = urlsplit(uri.strip())
        if parsed.scheme.lower() not in {"rtsp", "rtsps"}:
            raise RtspEndpointError("Camera URI must use rtsp:// or rtsps://.")
        if parsed.password is not None:
            raise RtspEndpointError(
                "Database camera URI must not contain a password. Store it with DPAPI."
            )
        if not parsed.hostname or not parsed.username:
            raise RtspEndpointError("Camera URI requires username and host.")
        return cls(
            host=parsed.hostname,
            port=parsed.port or 554,
            username=parsed.username,
            stream_path=parsed.path,
            secure=parsed.scheme.lower() == "rtsps",
        )

    @staticmethod
    def _validate_host(host: str) -> str:
        normalized = host.strip().strip("[]")
        if not normalized:
            raise RtspEndpointError("RTSP host is required.")
        try:
            return str(ipaddress.ip_address(normalized))
        except ValueError:
            hostname_pattern = re.compile(
                r"^(?=.{1,253}$)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)(?:\.(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?))*$"
            )
            if not hostname_pattern.fullmatch(normalized):
                raise RtspEndpointError("RTSP host is not a valid IP address or hostname.")
            return normalized.lower()

    @staticmethod
    def _validate_path(path: str) -> str:
        normalized = path.strip()
        if not normalized.startswith("/"):
            normalized = "/" + normalized
        if any(character in normalized for character in "\r\n?#"):
            raise RtspEndpointError("RTSP stream path contains invalid characters.")
        return normalized
