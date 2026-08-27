"""Reusable searchable and sortable dark-mode data grid for Arnesis."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from tkinter import ttk
from typing import Any

from arnesis.ui.theme import ArnesisTheme


@dataclass(frozen=True, slots=True)
class GridColumn:
    key: str
    title: str
    width: int = 130
    anchor: str = "w"
    stretch: bool = True


class ModernDataGrid(tk.Frame):
    """Blue dark-mode grid with search, filtering, sorting and empty state."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        columns: Sequence[GridColumn],
        searchable_keys: Sequence[str] | None = None,
        on_open: Callable[[dict[str, Any]], None] | None = None,
        selection_mode: str = "browse",
        empty_title: str = "No records found",
        empty_message: str = "Create a record or change the current filters.",
    ) -> None:
        super().__init__(parent, bg=ArnesisTheme.colors.surface)
        self.columns = list(columns)
        self.searchable_keys = tuple(searchable_keys or [column.key for column in columns])
        self.on_open = on_open
        self.empty_title = empty_title
        self.empty_message = empty_message
        self._records: list[dict[str, Any]] = []
        self._record_by_iid: dict[str, dict[str, Any]] = {}
        self._filters: dict[str, Any] = {}
        self._sort_key: str | None = None
        self._sort_reverse = False
        self.search_var = tk.StringVar()
        self.count_var = tk.StringVar(value="0 records")
        self._build(selection_mode)
        self.search_var.trace_add("write", lambda *_: self.refresh())

    def _build(self, selection_mode: str) -> None:
        c = ArnesisTheme.colors
        toolbar = tk.Frame(self, bg=c.surface, padx=12, pady=10)
        toolbar.pack(fill="x")

        search_shell = tk.Frame(toolbar, bg=c.input_background, highlightthickness=1, highlightbackground=c.border)
        search_shell.pack(side="left", fill="x", expand=True)
        tk.Label(
            search_shell,
            text="Search",
            bg=c.input_background,
            fg=c.text_muted,
            font=(ArnesisTheme.font_family, 9, "bold"),
            padx=10,
        ).pack(side="left")
        ttk.Entry(
            search_shell,
            textvariable=self.search_var,
            style="Arnesis.TEntry",
        ).pack(side="left", fill="x", expand=True)
        ArnesisTheme.button(
            toolbar,
            text="Clear",
            command=lambda: self.search_var.set(""),
            variant="ghost",
        ).pack(side="left", padx=(8, 0))
        tk.Label(
            toolbar,
            textvariable=self.count_var,
            bg=c.surface,
            fg=c.text_muted,
            padx=10,
        ).pack(side="right")

        table_shell = tk.Frame(self, bg=c.surface, highlightthickness=1, highlightbackground=c.border)
        table_shell.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        keys = [column.key for column in self.columns]
        self.tree = ttk.Treeview(
            table_shell,
            columns=keys,
            show="headings",
            selectmode=selection_mode,
            style="Arnesis.Treeview",
        )
        for column in self.columns:
            self.tree.heading(
                column.key,
                text=column.title,
                command=lambda key=column.key: self.sort_by(key),
            )
            self.tree.column(
                column.key,
                width=column.width,
                minwidth=60,
                anchor=column.anchor,
                stretch=column.stretch,
            )
        vertical = ttk.Scrollbar(
            table_shell,
            orient="vertical",
            command=self.tree.yview,
            style="Arnesis.Vertical.TScrollbar",
        )
        horizontal = ttk.Scrollbar(
            table_shell,
            orient="horizontal",
            command=self.tree.xview,
            style="Arnesis.Horizontal.TScrollbar",
        )
        self.tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vertical.grid(row=0, column=1, sticky="ns")
        horizontal.grid(row=1, column=0, sticky="ew")
        table_shell.rowconfigure(0, weight=1)
        table_shell.columnconfigure(0, weight=1)
        self.tree.tag_configure("even", background=c.surface)
        self.tree.tag_configure("odd", background=c.surface_alt)
        self.tree.tag_configure("RUNNING", foreground=c.success)
        self.tree.tag_configure("PAUSED", foreground=c.warning)
        self.tree.tag_configure("ERROR", foreground=c.danger)
        self.tree.tag_configure("STOPPED", foreground=c.text_muted)
        self.tree.bind("<Double-1>", self._handle_open)
        self.tree.bind("<Return>", self._handle_open)

        self.empty_overlay = tk.Frame(table_shell, bg=c.surface)
        tk.Label(
            self.empty_overlay,
            text=self.empty_title,
            bg=c.surface,
            fg=c.text,
            font=(ArnesisTheme.font_family, 15, "bold"),
        ).pack(pady=(0, 6))
        tk.Label(
            self.empty_overlay,
            text=self.empty_message,
            bg=c.surface,
            fg=c.text_muted,
            font=(ArnesisTheme.font_family, 10),
        ).pack()

    def set_records(self, records: Iterable[Mapping[str, Any]]) -> None:
        self._records = [dict(record) for record in records]
        self.refresh()

    def set_filter(self, key: str, value: Any | None) -> None:
        if value in (None, "", "All"):
            self._filters.pop(key, None)
        else:
            self._filters[key] = value
        self.refresh()

    def clear_filters(self) -> None:
        self._filters.clear()
        self.search_var.set("")
        self.refresh()

    def selected_records(self) -> list[dict[str, Any]]:
        return [self._record_by_iid[iid] for iid in self.tree.selection() if iid in self._record_by_iid]

    def selected_record(self) -> dict[str, Any] | None:
        records = self.selected_records()
        return records[0] if records else None

    def select_by_id(self, record_id: Any) -> None:
        iid = str(record_id)
        if self.tree.exists(iid):
            self.tree.selection_set(iid)
            self.tree.focus(iid)
            self.tree.see(iid)

    def sort_by(self, key: str) -> None:
        if self._sort_key == key:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_key = key
            self._sort_reverse = False
        self.refresh()

    def refresh(self) -> None:
        selected_ids = set(self.tree.selection())
        search = self.search_var.get().strip().casefold()
        filtered = []
        for record in self._records:
            if search and not any(search in str(record.get(key, "")).casefold() for key in self.searchable_keys):
                continue
            if any(record.get(key) != value for key, value in self._filters.items()):
                continue
            filtered.append(record)

        if self._sort_key:
            filtered.sort(
                key=lambda item: self._sort_value(item.get(self._sort_key)),
                reverse=self._sort_reverse,
            )

        for item in self.tree.get_children():
            self.tree.delete(item)
        self._record_by_iid.clear()
        for index, record in enumerate(filtered):
            iid = str(record.get("id", index))
            if iid in self._record_by_iid:
                iid = f"{iid}-{index}"
            values = [record.get(column.key, "") for column in self.columns]
            status = str(record.get("status", "")).upper()
            tags = ["even" if index % 2 == 0 else "odd"]
            if status in {"RUNNING", "PAUSED", "ERROR", "STOPPED"}:
                tags.append(status)
            self.tree.insert("", "end", iid=iid, values=values, tags=tuple(tags))
            self._record_by_iid[iid] = record
        for iid in selected_ids:
            if self.tree.exists(iid):
                self.tree.selection_add(iid)
        total = len(self._records)
        visible = len(filtered)
        self.count_var.set(f"{visible} of {total} record(s)")
        if filtered:
            self.empty_overlay.place_forget()
        else:
            self.empty_overlay.place(relx=0.5, rely=0.5, anchor="center")

    def _handle_open(self, _event: object = None) -> None:
        record = self.selected_record()
        if record is not None and self.on_open is not None:
            self.on_open(record)

    @staticmethod
    def _sort_value(value: Any) -> tuple[int, Any]:
        if value is None:
            return (1, "")
        if isinstance(value, (int, float)):
            return (0, value)
        return (0, str(value).casefold())
