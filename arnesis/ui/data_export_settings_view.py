"""Tkinter settings page for configurable Arnesis CSV output."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog

from arnesis.ui.theme import ArnesisTheme
from arnesis.ui.ux_messages import DialogService, MessageLevel, UserMessage


class DataExportSettingsView(tk.Frame):
    def __init__(self, parent: tk.Misc, settings_service, export_service) -> None:
        super().__init__(parent, bg=ArnesisTheme.colors.background)
        self.settings_service = settings_service
        self.export_service = export_service
        self.enabled_var = tk.BooleanVar()
        self.path_var = tk.StringVar()
        self.interval_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self._build()
        self.reload()

    def _build(self) -> None:
        colors = ArnesisTheme.colors
        card = tk.Frame(self, bg=colors.surface, padx=24, pady=24)
        card.pack(fill="x", padx=18, pady=18)
        tk.Label(card, text="DATA EXPORT", bg=colors.surface, fg=colors.accent,
                 font=(ArnesisTheme.font_family, 16, "bold")).grid(
                     row=0, column=0, columnspan=3, sticky="w", pady=(0, 18))
        tk.Checkbutton(card, text="Enable CSV export", variable=self.enabled_var,
                       bg=colors.surface, fg=colors.text,
                       selectcolor=colors.surface_alt).grid(row=1, column=0, columnspan=3, sticky="w")
        tk.Label(card, text="CSV output folder", bg=colors.surface,
                 fg=colors.text).grid(row=2, column=0, sticky="w", pady=(16, 5))
        tk.Entry(card, textvariable=self.path_var, bg="white", fg="#07152F").grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=(0, 8))
        tk.Button(card, text="Browse", command=self.browse,
                  bg="#018FFF", fg="white", relief="flat", padx=14).grid(row=3, column=2)
        tk.Label(card, text="Flush interval (seconds)", bg=colors.surface,
                 fg=colors.text).grid(row=4, column=0, sticky="w", pady=(16, 5))
        tk.Entry(card, textvariable=self.interval_var, width=12,
                 bg="white", fg="#07152F").grid(row=5, column=0, sticky="w")
        actions = tk.Frame(card, bg=colors.surface)
        actions.grid(row=6, column=0, columnspan=3, sticky="w", pady=(20, 0))
        tk.Button(actions, text="Test Folder", command=self.test_folder,
                  bg="#17203C", fg="white", relief="flat", padx=14, pady=7).pack(side="left")
        tk.Button(actions, text="Save", command=self.save,
                  bg="#018FFF", fg="white", relief="flat", padx=18, pady=7).pack(side="left", padx=8)
        tk.Button(actions, text="Flush Now", command=self.flush,
                  bg="#20C997", fg="#06142C", relief="flat", padx=14, pady=7).pack(side="left")
        tk.Label(card, textvariable=self.status_var, bg=colors.surface,
                 fg=colors.text_muted, justify="left").grid(
                     row=7, column=0, columnspan=3, sticky="w", pady=(18, 0))
        card.columnconfigure(0, weight=1)
        card.columnconfigure(1, weight=1)

    def reload(self) -> None:
        settings = self.settings_service.ensure_defaults()
        self.enabled_var.set(settings.enabled)
        self.path_var.set(settings.output_root)
        self.interval_var.set(str(settings.flush_interval_seconds))
        self._update_status()

    def browse(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.path_var.get() or str(Path.home()))
        if selected:
            self.path_var.set(selected)

    def test_folder(self) -> None:
        try:
            path = self.settings_service.validate_output_root(self.path_var.get())
            DialogService.show(self, UserMessage(MessageLevel.SUCCESS,
                "Folder available", "Arnesis can write CSV files to the selected folder.", str(path)))
        except Exception as exc:
            DialogService.show(self, UserMessage(MessageLevel.ERROR,
                "Folder unavailable", "The selected folder cannot be used.", str(exc)))

    def save(self) -> None:
        try:
            self.settings_service.save(enabled=self.enabled_var.get(),
                output_root=self.path_var.get(),
                flush_interval_seconds=int(self.interval_var.get()))
            self._update_status()
            DialogService.show(self, UserMessage(MessageLevel.SUCCESS,
                "Settings saved", "CSV export settings were saved successfully."))
        except Exception as exc:
            DialogService.show(self, UserMessage(MessageLevel.ERROR,
                "Settings not saved", "CSV export settings could not be saved.", str(exc)))

    def flush(self) -> None:
        rows = self.export_service.flush()
        self._update_status()
        DialogService.show(self, UserMessage(MessageLevel.SUCCESS,
            "CSV flush completed", f"{rows} pending station record(s) were written."))

    def _update_status(self) -> None:
        status = self.export_service.status()
        self.status_var.set(
            f"Status: {status.state}\nOutput: {status.output_root}\n"
            f"Pending rows: {status.pending_rows}\nLast write (UTC): {status.last_write_utc or '--'}\n"
            f"Last error: {status.last_error or '--'}"
        )
