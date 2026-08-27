"""Reusable UX messages and confirmation helpers for Arnesis desktop views."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from tkinter import messagebox
from tkinter import Misc


class MessageLevel(str, Enum):
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(frozen=True, slots=True)
class UserMessage:
    level: MessageLevel
    title: str
    message: str
    details: str | None = None

    @property
    def display_text(self) -> str:
        return f"{self.message}\n\n{self.details}" if self.details else self.message


class DialogService:
    """Centralizes alerts and confirmations for consistent application UX."""

    @staticmethod
    def show(parent: Misc, message: UserMessage) -> None:
        if message.level == MessageLevel.ERROR:
            messagebox.showerror(message.title, message.display_text, parent=parent)
        elif message.level == MessageLevel.WARNING:
            messagebox.showwarning(message.title, message.display_text, parent=parent)
        else:
            messagebox.showinfo(message.title, message.display_text, parent=parent)

    @staticmethod
    def confirm(
        parent: Misc,
        *,
        title: str,
        message: str,
        destructive: bool = False,
    ) -> bool:
        prompt = message
        if destructive:
            prompt += "\n\nThis action cannot be undone."
        return bool(messagebox.askyesno(title, prompt, parent=parent, icon="warning"))
