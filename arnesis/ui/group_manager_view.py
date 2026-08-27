"""Controller group-management view for Arnesis."""

from __future__ import annotations

import tkinter as tk
from tkinter import simpledialog, ttk
from typing import Callable

from arnesis.application.group_management_service import GroupManagementService
from arnesis.ui.ux_messages import DialogService, MessageLevel, UserMessage


class GroupManagerView(tk.Frame):
    """Dynamic group CRUD and lifecycle controls for the Controller."""

    NAVY = "#012059"
    PANEL = "#081F4E"
    CYAN = "#29E6FF"
    BLUE = "#018FFF"
    AMBER = "#FFC000"
    WHITE = "#F4F8FF"
    GREEN = "#167A5A"
    RED = "#A13A46"

    def __init__(
        self,
        parent: tk.Misc,
        service: GroupManagementService,
        open_cameras: Callable[[], None],
        open_rois: Callable[[int], None],
    ) -> None:
        super().__init__(parent, bg=self.NAVY)
        self.service = service
        self.open_cameras = open_cameras
        self.open_rois = open_rois
        self.selected_group_id: int | None = None
        self.gpu_map: dict[str, int | None] = {}
        self._dirty = False
        self._create_variables()
        self._build_ui()
        self._load_gpu_options()
        self.refresh()

    def _create_variables(self) -> None:
        self.status_var = tk.StringVar(value="Ready")
        self.code_var = tk.StringVar()
        self.name_var = tk.StringVar()
        self.description_var = tk.StringVar()
        self.enabled_var = tk.BooleanVar(value=True)
        self.gpu_var = tk.StringVar(value="Automatic")
        self.memory_var = tk.StringVar(value="8192")
        self.streams_var = tk.StringVar(value="10")
        for variable in (
            self.code_var,
            self.name_var,
            self.description_var,
            self.gpu_var,
            self.memory_var,
            self.streams_var,
        ):
            variable.trace_add("write", self._mark_dirty)

    def _build_ui(self) -> None:
        toolbar = tk.Frame(self, bg=self.NAVY, padx=18, pady=12)
        toolbar.pack(fill="x")
        tk.Label(
            toolbar,
            text="Groups",
            bg=self.NAVY,
            fg=self.WHITE,
            font=("Segoe UI", 20, "bold"),
        ).pack(side="left")
        for text, command, color in (
            ("New", self.new_group, self.BLUE),
            ("Duplicate", self.duplicate_group, "#345A91"),
            ("Cameras", self.open_cameras, "#345A91"),
            ("Manage ROIs", self.manage_rois, self.BLUE),
            ("Refresh", self.refresh, "#345A91"),
        ):
            tk.Button(
                toolbar,
                text=text,
                command=command,
                bg=color,
                fg="white",
                relief="flat",
                padx=14,
                pady=7,
            ).pack(side="right", padx=(8, 0))

        body = tk.PanedWindow(self, orient="horizontal", bg=self.NAVY, sashwidth=5)
        body.pack(fill="both", expand=True, padx=18, pady=(0, 12))
        listing = tk.Frame(body, bg=self.PANEL, padx=10, pady=10)
        editor = tk.Frame(body, bg=self.PANEL, padx=18, pady=14)
        body.add(listing, minsize=650)
        body.add(editor, minsize=430)

        columns = ("code", "name", "state", "cuda", "cameras", "rois")
        self.tree = ttk.Treeview(listing, columns=columns, show="headings")
        headings = {
            "code": "Code",
            "name": "Group",
            "state": "State",
            "cuda": "CUDA device",
            "cameras": "Cameras",
            "rois": "ROIs",
        }
        widths = {"code": 80, "name": 150, "state": 90, "cuda": 230, "cameras": 70, "rois": 60}
        for column in columns:
            self.tree.heading(column, text=headings[column])
            self.tree.column(column, width=widths[column], anchor="w")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_selection)

        controls = tk.Frame(listing, bg=self.PANEL)
        controls.pack(fill="x", pady=(10, 0))
        for text, command, color in (
            ("Start", self.start_group, self.GREEN),
            ("Pause", self.pause_group, self.AMBER),
            ("Resume", self.resume_group, self.BLUE),
            ("Stop", self.stop_group, self.RED),
        ):
            tk.Button(
                controls,
                text=text,
                command=command,
                bg=color,
                fg="white" if text != "Pause" else "black",
                relief="flat",
                padx=13,
                pady=7,
            ).pack(side="left", padx=(0, 7))

        tk.Label(
            editor,
            text="Group details",
            bg=self.PANEL,
            fg=self.CYAN,
            font=("Segoe UI", 14, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        fields = [
            ("Code *", self.code_var),
            ("Name *", self.name_var),
            ("Description", self.description_var),
            ("Maximum GPU memory (MiB) *", self.memory_var),
            ("Maximum concurrent streams *", self.streams_var),
        ]
        for row, (label, variable) in enumerate(fields, start=1):
            tk.Label(editor, text=label, bg=self.PANEL, fg=self.WHITE).grid(
                row=row, column=0, sticky="w", padx=(0, 12), pady=6
            )
            ttk.Entry(editor, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=6)

        tk.Label(editor, text="Preferred GPU", bg=self.PANEL, fg=self.WHITE).grid(
            row=6, column=0, sticky="w", padx=(0, 12), pady=6
        )
        self.gpu_combo = ttk.Combobox(editor, textvariable=self.gpu_var, state="readonly")
        self.gpu_combo.grid(row=6, column=1, sticky="ew", pady=6)
        ttk.Checkbutton(editor, text="Group enabled", variable=self.enabled_var).grid(
            row=7, column=1, sticky="w", pady=8
        )

        tk.Label(
            editor,
            text="CUDA assignment and capacity changes require the group to be stopped.",
            bg=self.PANEL,
            fg=self.AMBER,
            wraplength=390,
            justify="left",
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(4, 12))

        actions = tk.Frame(editor, bg=self.PANEL)
        actions.grid(row=9, column=0, columnspan=2, sticky="ew")
        for text, command, color in (
            ("Save", self.save_group, self.BLUE),
            ("Delete", self.delete_group, self.RED),
        ):
            tk.Button(
                actions,
                text=text,
                command=command,
                bg=color,
                fg="white",
                relief="flat",
                padx=15,
                pady=8,
            ).pack(side="left", padx=(0, 8))
        editor.columnconfigure(1, weight=1)

        tk.Label(
            self,
            textvariable=self.status_var,
            bg="#051835",
            fg=self.WHITE,
            anchor="w",
            padx=18,
            pady=7,
        ).pack(fill="x")

    def _load_gpu_options(self) -> None:
        options = self.service.list_gpu_options(include_disabled=True)
        self.gpu_map = {"Automatic": None}
        for option in options:
            self.gpu_map[option.label] = option.device_index
        self.gpu_combo["values"] = list(self.gpu_map)

    def refresh(self) -> None:
        if self._dirty and not DialogService.confirm(
            self,
            title="Discard changes",
            message="Refresh the group list and discard unsaved changes?",
        ):
            return
        try:
            selected_id = self.selected_group_id
            for item in self.tree.get_children():
                self.tree.delete(item)
            groups = self.service.list_groups()
            for group in groups:
                self.tree.insert(
                    "",
                    "end",
                    iid=str(group.id),
                    values=(
                        group.code,
                        group.name,
                        group.status,
                        group.gpu_label,
                        group.camera_count,
                        group.roi_count,
                    ),
                )
            self.status_var.set(f"{len(groups)} group(s) loaded.")
            self._dirty = False
            if selected_id is not None and self.tree.exists(str(selected_id)):
                self.tree.selection_set(str(selected_id))
        except Exception as exc:
            self._show_error("Unable to load groups", exc)

    def new_group(self) -> None:
        if not self._discard_changes():
            return
        self.selected_group_id = None
        self.code_var.set("")
        self.name_var.set("")
        self.description_var.set("")
        self.enabled_var.set(True)
        self.gpu_var.set("Automatic")
        self.memory_var.set("8192")
        self.streams_var.set("10")
        self._dirty = False
        self.status_var.set("New group. Required fields are marked with *.")

    def save_group(self) -> None:
        try:
            values = self._read_form()
            if self.selected_group_id is None:
                group = self.service.create_group(**values)
                message = "Group created successfully."
            else:
                group = self.service.update_group(self.selected_group_id, **values)
                message = "Group updated successfully."
            self.selected_group_id = group.id
            self._dirty = False
            self.refresh()
            self.tree.selection_set(str(group.id))
            DialogService.show(
                self,
                UserMessage(
                    MessageLevel.SUCCESS,
                    "Group saved",
                    message,
                    f"{group.code} | {group.gpu_label}",
                ),
            )
        except Exception as exc:
            self._show_error("Group could not be saved", exc)

    def duplicate_group(self) -> None:
        if self.selected_group_id is None:
            self._warning("No selection", "Select a group to duplicate.")
            return
        new_code = simpledialog.askstring(
            "Duplicate group",
            "Enter the new unique group code:",
            parent=self,
        )
        if not new_code:
            self.status_var.set("Group duplication canceled.")
            return
        try:
            group = self.service.duplicate_group(self.selected_group_id, new_code)
            self.refresh()
            self.tree.selection_set(str(group.id))
            DialogService.show(
                self,
                UserMessage(
                    MessageLevel.SUCCESS,
                    "Group duplicated",
                    "A disabled copy was created successfully.",
                    f"New group: {group.code}",
                ),
            )
        except Exception as exc:
            self._show_error("Group could not be duplicated", exc)

    def delete_group(self) -> None:
        if self.selected_group_id is None:
            self._warning("No selection", "Select a group to delete.")
            return
        group = self.service.get_group(self.selected_group_id)
        if not DialogService.confirm(
            self,
            title="Delete group",
            message=(
                f"Delete group {group.code}, its {group.camera_count} camera(s), "
                f"and its {group.roi_count} ROI(s)?"
            ),
            destructive=True,
        ):
            self.status_var.set("Group deletion canceled.")
            return
        try:
            self.service.delete_group(group.id)
            self.new_group()
            self.refresh()
            DialogService.show(
                self,
                UserMessage(MessageLevel.SUCCESS, "Group deleted", "The group was deleted successfully."),
            )
        except Exception as exc:
            self._show_error("Group could not be deleted", exc)

    def manage_rois(self) -> None:
        """Open ROI configuration for the explicitly selected saved group."""
        if self.selected_group_id is None:
            self._warning(
                "No selection",
                "Select a saved group before managing ROIs.",
            )
            return
        if self._dirty:
            self._warning(
                "Unsaved changes",
                "Save or discard group changes before managing ROIs.",
            )
            return
        self.open_rois(self.selected_group_id)

    def start_group(self) -> None:
        self._runtime_command("start")

    def pause_group(self) -> None:
        self._runtime_command("pause")

    def resume_group(self) -> None:
        self._runtime_command("resume")

    def stop_group(self) -> None:
        self._runtime_command("stop")

    def _runtime_command(self, command: str) -> None:
        if self.selected_group_id is None:
            self._warning("No selection", f"Select a group to {command}.")
            return
        if self._dirty:
            self._warning("Unsaved changes", "Save or discard group changes before sending runtime commands.")
            return
        group = self.service.get_group(self.selected_group_id)
        if command in {"start", "stop"} and not DialogService.confirm(
            self,
            title=f"{command.title()} group",
            message=f"{command.title()} real-time processing for group {group.code}?",
        ):
            self.status_var.set(f"{command.title()} command canceled.")
            return
        try:
            result = getattr(self.service, f"{command}_group")(group.id)
            self.refresh()
            DialogService.show(
                self,
                UserMessage(
                    MessageLevel.SUCCESS,
                    f"Group {command} command completed",
                    f"Group {group.code} is now {result['state']}.",
                    str(result.get("cuda_device", group.gpu_label)),
                ),
            )
        except Exception as exc:
            self._show_error(f"Unable to {command} group", exc)

    def _on_selection(self, _event: object = None) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        new_group_id = int(selection[0])
        if self.selected_group_id != new_group_id and not self._discard_changes():
            if self.selected_group_id is not None and self.tree.exists(str(self.selected_group_id)):
                self.tree.selection_set(str(self.selected_group_id))
            return
        try:
            group = self.service.get_group(new_group_id)
            self.selected_group_id = group.id
            self.code_var.set(group.code)
            self.name_var.set(group.name)
            self.description_var.set(group.description or "")
            self.enabled_var.set(group.enabled)
            matching_gpu = next(
                (label for label, value in self.gpu_map.items() if value == group.preferred_gpu_index),
                "Automatic",
            )
            self.gpu_var.set(matching_gpu)
            self.memory_var.set(str(group.max_gpu_memory_mb))
            self.streams_var.set(str(group.max_concurrent_streams))
            self._dirty = False
            self.status_var.set(f"Selected {group.code} | {group.status} | {group.gpu_label}")
        except Exception as exc:
            self._show_error("Unable to open group", exc)

    def _read_form(self) -> dict[str, object]:
        if self.gpu_var.get() not in self.gpu_map:
            raise ValueError("Select a valid CUDA assignment.")
        return {
            "code": self.code_var.get(),
            "name": self.name_var.get(),
            "description": self.description_var.get(),
            "enabled": self.enabled_var.get(),
            "preferred_gpu_index": self.gpu_map[self.gpu_var.get()],
            "max_gpu_memory_mb": int(self.memory_var.get()),
            "max_concurrent_streams": int(self.streams_var.get()),
        }

    def _discard_changes(self) -> bool:
        return not self._dirty or DialogService.confirm(
            self,
            title="Discard changes",
            message="Discard the unsaved group changes?",
        )

    def _mark_dirty(self, *_args: object) -> None:
        self._dirty = True

    def has_unsaved_changes(self) -> bool:
        return self._dirty

    def _warning(self, title: str, message: str) -> None:
        DialogService.show(self, UserMessage(MessageLevel.WARNING, title, message))

    def _show_error(self, title: str, exc: Exception) -> None:
        DialogService.show(
            self,
            UserMessage(MessageLevel.ERROR, title, str(exc) or type(exc).__name__),
        )
        self.status_var.set(title)
