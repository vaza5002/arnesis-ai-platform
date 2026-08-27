"""Central visual system for the Arnesis desktop Controller."""

from __future__ import annotations

import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk


@dataclass(frozen=True, slots=True)
class Colors:
    background: str = "#07152D"
    sidebar: str = "#081A38"
    sidebar_active: str = "#0E63D4"
    surface: str = "#0D2348"
    surface_alt: str = "#102B5A"
    surface_hover: str = "#15386F"
    border: str = "#1C467E"
    primary: str = "#018FFF"
    primary_hover: str = "#1BA6FF"
    accent: str = "#29E6FF"
    warning: str = "#FFC000"
    success: str = "#20C997"
    danger: str = "#FF5B69"
    text: str = "#F4F8FF"
    text_muted: str = "#9DB3D5"
    text_disabled: str = "#637AA1"
    input_background: str = "#F7FAFF"
    input_text: str = "#10213D"


class ArnesisTheme:
    """Applies one consistent blue dark-mode theme to Tk and ttk widgets."""

    colors = Colors()
    font_family = "Segoe UI"

    @classmethod
    def apply(cls, root: tk.Misc) -> None:
        root.option_add("*Font", (cls.font_family, 10))
        root.option_add("*tearOff", False)

        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        c = cls.colors
        style.configure(
            ".",
            background=c.background,
            foreground=c.text,
            font=(cls.font_family, 10),
            bordercolor=c.border,
            lightcolor=c.border,
            darkcolor=c.border,
        )
        style.configure(
            "Arnesis.Treeview",
            background=c.surface,
            fieldbackground=c.surface,
            foreground=c.text,
            borderwidth=0,
            relief="flat",
            rowheight=36,
            font=(cls.font_family, 10),
        )
        style.configure(
            "Arnesis.Treeview.Heading",
            background=c.surface_alt,
            foreground=c.text,
            borderwidth=0,
            relief="flat",
            padding=(12, 10),
            font=(cls.font_family, 10, "bold"),
        )
        style.map(
            "Arnesis.Treeview",
            background=[("selected", c.primary)],
            foreground=[("selected", "#FFFFFF")],
        )
        style.map(
            "Arnesis.Treeview.Heading",
            background=[("active", c.surface_hover)],
        )
        style.configure(
            "Arnesis.TEntry",
            fieldbackground=c.input_background,
            foreground=c.input_text,
            insertcolor=c.input_text,
            bordercolor=c.border,
            padding=(10, 8),
        )
        style.map(
            "Arnesis.TEntry",
            bordercolor=[("focus", c.primary)],
        )
        style.configure(
            "Arnesis.TCombobox",
            fieldbackground=c.input_background,
            background=c.input_background,
            foreground=c.input_text,
            arrowcolor=c.primary,
            padding=(10, 7),
        )
        style.map(
            "Arnesis.TCombobox",
            fieldbackground=[("readonly", c.input_background)],
            foreground=[("readonly", c.input_text)],
            bordercolor=[("focus", c.primary)],
        )
        style.configure(
            "Arnesis.TCheckbutton",
            background=c.surface,
            foreground=c.text,
            focuscolor=c.surface,
            padding=4,
        )
        style.map(
            "Arnesis.TCheckbutton",
            background=[("active", c.surface)],
            foreground=[("disabled", c.text_disabled)],
        )
        style.configure(
            "Arnesis.Vertical.TScrollbar",
            background=c.surface_alt,
            troughcolor=c.background,
            bordercolor=c.background,
            arrowcolor=c.text_muted,
            width=12,
        )
        style.configure(
            "Arnesis.Horizontal.TScrollbar",
            background=c.surface_alt,
            troughcolor=c.background,
            bordercolor=c.background,
            arrowcolor=c.text_muted,
        )
        style.configure(
            "Arnesis.TSeparator",
            background=c.border,
        )

    @classmethod
    def button(
        cls,
        parent: tk.Misc,
        *,
        text: str,
        command,
        variant: str = "secondary",
        width: int | None = None,
    ) -> tk.Button:
        c = cls.colors
        variants = {
            "primary": (c.primary, "#FFFFFF", c.primary_hover),
            "secondary": (c.surface_alt, c.text, c.surface_hover),
            "success": (c.success, "#051B16", "#47D7B0"),
            "warning": (c.warning, "#241B00", "#FFD34D"),
            "danger": (c.danger, "#FFFFFF", "#FF7A85"),
            "ghost": (c.surface, c.text_muted, c.surface_hover),
        }
        background, foreground, active = variants.get(variant, variants["secondary"])
        return tk.Button(
            parent,
            text=text,
            command=command,
            width=width,
            bg=background,
            fg=foreground,
            activebackground=active,
            activeforeground=foreground,
            disabledforeground=c.text_disabled,
            relief="flat",
            borderwidth=0,
            cursor="hand2",
            padx=15,
            pady=9,
            font=(cls.font_family, 10, "bold"),
        )
