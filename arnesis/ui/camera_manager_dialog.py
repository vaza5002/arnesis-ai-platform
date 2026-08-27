"""Modern blue dark-mode camera management dialog for Arnesis."""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk
from typing import Any

from arnesis.application.camera_management_service import CameraManagementService
from arnesis.ui.theme import ArnesisTheme
from arnesis.ui.ux_messages import DialogService, MessageLevel, UserMessage


class CameraManagerDialog(tk.Toplevel):
    """Manage secure RTSP cameras with a readable responsive dark-mode UI."""

    MAIN_STREAM = "Main stream (Channel 101)"
    SUB_STREAM = "Sub stream (Channel 102)"
    CUSTOM_STREAM = "Custom path"

    def __init__(self, parent: tk.Misc, service: CameraManagementService, on_manage_rois: Any | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self.on_manage_rois = on_manage_rois
        self.selected_camera_id: int | None = None
        self.group_map: dict[str, int] = {}
        self._all_cameras: list[Any] = []
        self._busy = False
        self._dirty = False

        self.title("Arnesis | Camera Management")
        self.geometry("1380x820")
        self.minsize(1120, 700)
        self.configure(bg=ArnesisTheme.colors.background)
        self.transient(parent)
        self.protocol("WM_DELETE_WINDOW", self._close)

        self._create_variables()
        self._configure_styles()
        self._build_ui()
        self._load_groups()
        self._refresh_cameras()
        self.grab_set()

    def _create_variables(self) -> None:
        self.status_var = tk.StringVar(value="Ready")
        self.search_var = tk.StringVar()
        self.group_filter_var = tk.StringVar(value="All groups")
        self.group_var = tk.StringVar()
        self.name_var = tk.StringVar()
        self.host_var = tk.StringVar()
        self.port_var = tk.StringVar(value="554")
        self.username_var = tk.StringVar(value="admin")
        self.password_var = tk.StringVar()
        self.stream_mode_var = tk.StringVar(value=self.MAIN_STREAM)
        self.path_var = tk.StringVar(value="/Streaming/Channels/101")
        self.fps_var = tk.StringVar(value="15")
        self.reconnect_var = tk.StringVar(value="5")
        self.width_var = tk.StringVar()
        self.height_var = tk.StringVar()
        self.enabled_var = tk.BooleanVar(value=True)
        self.show_password_var = tk.BooleanVar(value=False)
        self.credential_var = tk.StringVar(value="Credential not saved")
        self.connection_var = tk.StringVar(value="Not tested")
        self.summary_var = tk.StringVar(value="0 cameras")

        self.search_var.trace_add("write", lambda *_: self._render_cameras())
        self.group_filter_var.trace_add("write", lambda *_: self._render_cameras())
        self.stream_mode_var.trace_add("write", self._stream_mode_changed)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        c = ArnesisTheme.colors
        style.configure(
            "Camera.Treeview",
            background=c.surface,
            fieldbackground=c.surface,
            foreground=c.text,
            rowheight=38,
            borderwidth=0,
            font=(ArnesisTheme.font_family, 10),
        )
        style.configure(
            "Camera.Treeview.Heading",
            background=c.surface_alt,
            foreground=c.text,
            padding=(10, 10),
            borderwidth=0,
            font=(ArnesisTheme.font_family, 10, "bold"),
        )
        style.map(
            "Camera.Treeview",
            background=[("selected", c.primary)],
            foreground=[("selected", "#FFFFFF")],
        )
        style.configure(
            "Camera.TEntry",
            fieldbackground=c.input_background,
            foreground=c.input_text,
            insertcolor=c.input_text,
            padding=(10, 8),
        )
        style.configure(
            "Camera.TCombobox",
            fieldbackground=c.input_background,
            background=c.input_background,
            foreground=c.input_text,
            arrowcolor=c.input_text,
            padding=(10, 7),
        )
        style.map(
            "Camera.TCombobox",
            fieldbackground=[("readonly", c.input_background)],
            foreground=[("readonly", c.input_text)],
            selectbackground=[("readonly", c.input_background)],
            selectforeground=[("readonly", c.input_text)],
        )
        style.configure(
            "Camera.TCheckbutton",
            background=c.surface,
            foreground=c.text,
        )
        style.map("Camera.TCheckbutton", background=[("active", c.surface)])
        self.option_add("*TCombobox*Listbox.background", c.input_background)
        self.option_add("*TCombobox*Listbox.foreground", c.input_text)
        self.option_add("*TCombobox*Listbox.selectBackground", c.primary)
        self.option_add("*TCombobox*Listbox.selectForeground", "#FFFFFF")

    def _build_ui(self) -> None:
        c = ArnesisTheme.colors
        header = tk.Frame(self, bg=c.background, padx=24, pady=18)
        header.pack(fill="x")
        title = tk.Frame(header, bg=c.background)
        title.pack(side="left")
        tk.Label(
            title,
            text="Camera Management",
            bg=c.background,
            fg=c.text,
            font=(ArnesisTheme.font_family, 22, "bold"),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            title,
            text="Configure secure RTSP endpoints and validate camera connectivity",
            bg=c.background,
            fg=c.text_muted,
            font=(ArnesisTheme.font_family, 10),
            anchor="w",
        ).pack(fill="x", pady=(3, 0))
        ArnesisTheme.button(
            header,
            text="+ New camera",
            command=self._new_camera,
            variant="primary",
        ).pack(side="right")

        content = tk.PanedWindow(
            self,
            orient="horizontal",
            bg=c.background,
            sashwidth=7,
            borderwidth=0,
            sashrelief="flat",
        )
        content.pack(fill="both", expand=True, padx=20, pady=(0, 14))

        list_panel = tk.Frame(
            content,
            bg=c.surface,
            padx=16,
            pady=16,
            highlightthickness=1,
            highlightbackground=c.border,
        )
        editor_panel = tk.Frame(
            content,
            bg=c.surface,
            padx=20,
            pady=18,
            highlightthickness=1,
            highlightbackground=c.border,
        )
        content.add(list_panel, minsize=650, width=790)
        content.add(editor_panel, minsize=430, width=520)

        self._build_list_panel(list_panel)
        self._build_editor_panel(editor_panel)

        tk.Label(
            self,
            textvariable=self.status_var,
            bg="#05142C",
            fg=c.text,
            anchor="w",
            padx=20,
            pady=8,
            font=(ArnesisTheme.font_family, 9),
        ).pack(fill="x")

    def _build_list_panel(self, parent: tk.Frame) -> None:
        c = ArnesisTheme.colors
        top = tk.Frame(parent, bg=c.surface)
        top.pack(fill="x", pady=(0, 12))
        tk.Label(
            top,
            text="CAMERAS",
            bg=c.surface,
            fg=c.accent,
            font=(ArnesisTheme.font_family, 14, "bold"),
        ).pack(side="left")
        tk.Label(
            top,
            textvariable=self.summary_var,
            bg=c.surface_alt,
            fg=c.text,
            padx=12,
            pady=5,
            font=(ArnesisTheme.font_family, 9, "bold"),
        ).pack(side="right")

        filters = tk.Frame(parent, bg=c.surface)
        filters.pack(fill="x", pady=(0, 12))
        ttk.Entry(
            filters,
            textvariable=self.search_var,
            style="Camera.TEntry",
        ).pack(side="left", fill="x", expand=True)
        self.group_filter = ttk.Combobox(
            filters,
            textvariable=self.group_filter_var,
            state="readonly",
            style="Camera.TCombobox",
            width=25,
        )
        self.group_filter.pack(side="left", padx=(10, 0))
        ArnesisTheme.button(
            filters,
            text="Refresh",
            command=self._refresh_cameras,
            variant="secondary",
        ).pack(side="left", padx=(10, 0))

        columns = ("group", "name", "host", "enabled", "credential")
        table = tk.Frame(parent, bg=c.surface, highlightthickness=1, highlightbackground=c.border)
        table.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(
            table,
            columns=columns,
            show="headings",
            style="Camera.Treeview",
            selectmode="browse",
        )
        labels = {
            "group": "GROUP",
            "name": "CAMERA",
            "host": "IP / HOST",
            "enabled": "STATUS",
            "credential": "CREDENTIAL",
        }
        widths = {"group": 85, "name": 190, "host": 155, "enabled": 95, "credential": 110}
        for key in columns:
            self.tree.heading(key, text=labels[key])
            self.tree.column(key, width=widths[key], minwidth=70, anchor="w")
        scroll = ttk.Scrollbar(table, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        table.rowconfigure(0, weight=1)
        table.columnconfigure(0, weight=1)
        self.tree.tag_configure("even", background=c.surface)
        self.tree.tag_configure("odd", background=c.surface_alt)
        self.tree.tag_configure("disabled", foreground=c.text_disabled)
        self.tree.bind("<<TreeviewSelect>>", self._on_selection)
        self.tree.bind("<Double-1>", self._on_selection)

    def _build_editor_panel(self, parent: tk.Frame) -> None:
        c = ArnesisTheme.colors
        tk.Label(
            parent,
            text="CAMERA SETTINGS",
            bg=c.surface,
            fg=c.accent,
            font=(ArnesisTheme.font_family, 14, "bold"),
            anchor="w",
        ).pack(fill="x")
        self.editor_subtitle = tk.Label(
            parent,
            text="Create a new secure RTSP camera configuration",
            bg=c.surface,
            fg=c.text_muted,
            font=(ArnesisTheme.font_family, 9),
            anchor="w",
        )
        self.editor_subtitle.pack(fill="x", pady=(4, 14))

        canvas = tk.Canvas(parent, bg=c.surface, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        self.form = tk.Frame(canvas, bg=c.surface)
        self.form.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        window = canvas.create_window((0, 0), window=self.form, anchor="nw")
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._section("GENERAL")
        self.group_combo = self._field("Group *", self.group_var, combo=True)
        self._field("Camera name *", self.name_var)
        self._field("IP address / hostname *", self.host_var)

        self._section("AUTHENTICATION")
        self._field("Username *", self.username_var)
        password_row = self._field("Password", self.password_var, password=True)
        ttk.Checkbutton(
            self.form,
            text="Show password",
            variable=self.show_password_var,
            command=lambda: password_row.configure(show="" if self.show_password_var.get() else "*"),
            style="Camera.TCheckbutton",
        ).pack(anchor="w", pady=(2, 4))
        tk.Label(
            self.form,
            textvariable=self.credential_var,
            bg=c.surface,
            fg=c.success,
            font=(ArnesisTheme.font_family, 9, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(0, 8))

        self._section("VIDEO STREAM")
        self.stream_combo = self._field(
            "Stream quality",
            self.stream_mode_var,
            combo=True,
            values=(self.MAIN_STREAM, self.SUB_STREAM, self.CUSTOM_STREAM),
        )
        self.path_entry = self._field("Video stream path *", self.path_var)
        self._field("RTSP port *", self.port_var)

        self._section("PROCESSING")
        self._field("Target FPS *", self.fps_var)
        self._field("Reconnect interval (seconds) *", self.reconnect_var)
        self._field("Width (optional)", self.width_var)
        self._field("Height (optional)", self.height_var)
        ttk.Checkbutton(
            self.form,
            text="Camera enabled",
            variable=self.enabled_var,
            style="Camera.TCheckbutton",
        ).pack(anchor="w", pady=(8, 12))

        connection = tk.Frame(
            self.form,
            bg=c.surface_alt,
            padx=14,
            pady=12,
            highlightthickness=1,
            highlightbackground=c.border,
        )
        connection.pack(fill="x", pady=(2, 12))
        tk.Label(
            connection,
            text="CONNECTION STATUS",
            bg=c.surface_alt,
            fg=c.text_muted,
            font=(ArnesisTheme.font_family, 8, "bold"),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            connection,
            textvariable=self.connection_var,
            bg=c.surface_alt,
            fg=c.text,
            font=(ArnesisTheme.font_family, 10, "bold"),
            anchor="w",
            justify="left",
            wraplength=420,
        ).pack(fill="x", pady=(5, 0))

        actions = tk.Frame(self.form, bg=c.surface)
        actions.pack(fill="x", pady=(2, 14))
        self.save_button = ArnesisTheme.button(
            actions, text="Save camera", command=self._save, variant="primary"
        )
        self.save_button.pack(side="right")
        self.test_button = ArnesisTheme.button(
            actions, text="Test connection", command=self._test_connection, variant="success"
        )
        self.test_button.pack(side="right", padx=(0, 8))
        ArnesisTheme.button(
            actions, text="Manage ROIs", command=self._manage_rois, variant="secondary"
        ).pack(side="right", padx=(0, 8))
        ArnesisTheme.button(
            actions, text="Delete", command=self._delete, variant="danger"
        ).pack(side="left")

    def _section(self, text: str) -> None:
        c = ArnesisTheme.colors
        tk.Label(
            self.form,
            text=text,
            bg=c.surface,
            fg=c.text_muted,
            font=(ArnesisTheme.font_family, 8, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(12, 5))

    def _field(
        self,
        label: str,
        variable: tk.Variable,
        *,
        combo: bool = False,
        values: tuple[str, ...] = (),
        password: bool = False,
    ):
        c = ArnesisTheme.colors
        tk.Label(
            self.form,
            text=label,
            bg=c.surface,
            fg=c.text,
            font=(ArnesisTheme.font_family, 9, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(4, 4))
        if combo:
            widget = ttk.Combobox(
                self.form,
                textvariable=variable,
                values=values,
                state="readonly",
                style="Camera.TCombobox",
            )
        else:
            widget = ttk.Entry(
                self.form,
                textvariable=variable,
                show="*" if password else "",
                style="Camera.TEntry",
            )
        widget.pack(fill="x", pady=(0, 4))
        return widget

    def _load_groups(self) -> None:
        groups = self.service.list_groups()
        self.group_map = {
            f"{group['code']} - {group['name']}": int(group["id"])
            for group in groups
        }
        labels = list(self.group_map)
        self.group_combo.configure(values=labels)
        self.group_filter.configure(values=["All groups", *labels])
        if labels and not self.group_var.get():
            self.group_var.set(labels[0])
        if not groups:
            DialogService.show(
                self,
                UserMessage(
                    MessageLevel.WARNING,
                    "No groups available",
                    "Create at least one group before adding cameras.",
                ),
            )

    def _refresh_cameras(self) -> None:
        try:
            self._all_cameras = list(self.service.list_cameras())
            self._render_cameras()
            self.status_var.set(f"{len(self._all_cameras)} camera(s) loaded.")
        except Exception as exc:
            self._show_error("Unable to load cameras", exc)

    def _render_cameras(self) -> None:
        if not hasattr(self, "tree"):
            return
        search = self.search_var.get().strip().casefold()
        selected_group = self.group_filter_var.get()
        selected_group_id = self.group_map.get(selected_group)
        for item in self.tree.get_children():
            self.tree.delete(item)
        visible = []
        for camera in self._all_cameras:
            if selected_group_id is not None and camera.group_id != selected_group_id:
                continue
            text = f"{camera.group_code} {camera.name} {camera.host}".casefold()
            if search and search not in text:
                continue
            visible.append(camera)
        for index, camera in enumerate(visible):
            tags = ["even" if index % 2 == 0 else "odd"]
            if not camera.enabled:
                tags.append("disabled")
            self.tree.insert(
                "",
                "end",
                iid=str(camera.id),
                values=(
                    camera.group_code,
                    camera.name,
                    camera.host,
                    "Enabled" if camera.enabled else "Disabled",
                    "Protected" if camera.credential_configured else "Missing",
                ),
                tags=tuple(tags),
            )
        enabled = sum(1 for camera in self._all_cameras if camera.enabled)
        protected = sum(1 for camera in self._all_cameras if camera.credential_configured)
        self.summary_var.set(
            f"{len(self._all_cameras)} total  |  {enabled} enabled  |  {protected} protected"
        )

    def _on_selection(self, _event: object = None) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        try:
            camera = self.service.get_camera(int(selection[0]))
            self.selected_camera_id = camera.id
            self.group_var.set(next((k for k, v in self.group_map.items() if v == camera.group_id), ""))
            self.name_var.set(camera.name)
            self.host_var.set(camera.host)
            self.port_var.set(str(camera.port))
            self.username_var.set(camera.username)
            self.password_var.set("")
            self.path_var.set(camera.stream_path)
            self.fps_var.set(str(camera.target_fps))
            self.reconnect_var.set(str(camera.reconnect_seconds))
            self.width_var.set("" if camera.width is None else str(camera.width))
            self.height_var.set("" if camera.height is None else str(camera.height))
            self.enabled_var.set(camera.enabled)
            self.stream_mode_var.set(self._mode_from_path(camera.stream_path))
            self.credential_var.set(
                "Protected credential configured"
                if camera.credential_configured
                else "Credential missing"
            )
            self.connection_var.set("Not tested in this session")
            self.editor_subtitle.configure(text=f"Editing {camera.name}")
            self.status_var.set(f"Editing {camera.name}. Leave password blank to preserve it.")
            self._dirty = False
        except Exception as exc:
            self._show_error("Unable to open camera", exc)

    def _new_camera(self) -> None:
        self.selected_camera_id = None
        self.name_var.set("")
        self.host_var.set("")
        self.port_var.set("554")
        self.username_var.set("admin")
        self.password_var.set("")
        self.stream_mode_var.set(self.MAIN_STREAM)
        self.path_var.set("/Streaming/Channels/101")
        self.fps_var.set("15")
        self.reconnect_var.set("5")
        self.width_var.set("")
        self.height_var.set("")
        self.enabled_var.set(True)
        self.credential_var.set("Password required for a new camera")
        self.connection_var.set("Save the camera before testing")
        self.editor_subtitle.configure(text="Create a new secure RTSP camera configuration")
        self.tree.selection_remove(self.tree.selection())
        self.status_var.set("New camera. Required fields are marked with *.")
        self._dirty = False

    def _save(self) -> None:
        try:
            values = self._read_form()
            if self.selected_camera_id is None:
                if not values["password"]:
                    raise ValueError("Password is required for a new camera.")
                camera = self.service.create_camera(**values)
                message = "Camera created successfully."
            else:
                camera = self.service.update_camera(
                    self.selected_camera_id,
                    enabled=self.enabled_var.get(),
                    **values,
                )
                message = "Camera updated successfully."
            self.selected_camera_id = camera.id
            self.password_var.set("")
            self.credential_var.set("Protected credential configured")
            self._refresh_cameras()
            self.tree.selection_set(str(camera.id))
            DialogService.show(
                self,
                UserMessage(
                    MessageLevel.SUCCESS,
                    "Camera saved",
                    message,
                    f"Endpoint: {camera.masked_url}",
                ),
            )
        except Exception as exc:
            self._show_error("Camera could not be saved", exc)

    def _manage_rois(self) -> None:
        if self.selected_camera_id is None:
            DialogService.show(
                self,
                UserMessage(MessageLevel.WARNING, "No camera selected", "Select a saved camera first."),
            )
            return
        if not callable(self.on_manage_rois):
            raise RuntimeError("ROI navigation callback is unavailable.")
        camera_id = self.selected_camera_id
        self.grab_release()
        self.destroy()
        self.on_manage_rois(camera_id)

    def _test_connection(self) -> None:
        if self.selected_camera_id is None:
            DialogService.show(
                self,
                UserMessage(
                    MessageLevel.WARNING,
                    "Save required",
                    "Save the camera and protected credential before testing.",
                ),
            )
            return
        if self._busy:
            return
        self._set_busy(True)
        self.connection_var.set("Testing RTSP connection...")
        self.status_var.set("Testing RTSP connection. The interface remains available.")
        camera_id = self.selected_camera_id

        def worker() -> None:
            try:
                result = self.service.test_connection(camera_id)
                self.after(0, lambda: self._finish_connection_test(result))
            except Exception as exc:
                self.after(0, lambda: self._finish_connection_error(exc))

        threading.Thread(target=worker, name=f"camera-test-{camera_id}", daemon=True).start()

    def _finish_connection_test(self, result: Any) -> None:
        self._set_busy(False)
        if result.success:
            self.connection_var.set(
                f"Connected | {result.width} x {result.height} | {result.elapsed_ms} ms"
            )
            self.status_var.set("Connection test completed successfully.")
            DialogService.show(
                self,
                UserMessage(
                    MessageLevel.SUCCESS,
                    "Connection successful",
                    f"A valid frame was received from {result.camera_name}.",
                    f"Resolution: {result.width} x {result.height}\nElapsed: {result.elapsed_ms} ms",
                ),
            )
        else:
            friendly = self._friendly_error(result.error)
            self.connection_var.set(f"Connection failed | {friendly}")
            self.status_var.set("Connection test failed.")
            DialogService.show(
                self,
                UserMessage(
                    MessageLevel.ERROR,
                    "Connection failed",
                    "Arnesis could not read a valid frame from the camera.",
                    friendly,
                ),
            )

    def _finish_connection_error(self, exc: Exception) -> None:
        self._set_busy(False)
        self.connection_var.set(f"Connection error | {type(exc).__name__}")
        self._show_error("Connection test failed", exc)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.test_button.configure(state=state)
        self.save_button.configure(state=state)

    def _delete(self) -> None:
        if self.selected_camera_id is None:
            DialogService.show(
                self,
                UserMessage(MessageLevel.WARNING, "No selection", "Select a camera to delete."),
            )
            return
        if not DialogService.confirm(
            self,
            title="Delete camera",
            message="Delete this camera, its protected credential and all associated ROIs?",
            destructive=True,
        ):
            return
        try:
            self.service.delete_camera(self.selected_camera_id)
            self._new_camera()
            self._refresh_cameras()
            DialogService.show(
                self,
                UserMessage(MessageLevel.SUCCESS, "Camera deleted", "The camera was deleted successfully."),
            )
        except Exception as exc:
            self._show_error("Camera could not be deleted", exc)

    def _read_form(self) -> dict[str, object]:
        group_id = self.group_map.get(self.group_var.get())
        if group_id is None:
            raise ValueError("Select a valid group.")
        try:
            port = int(self.port_var.get())
            target_fps = float(self.fps_var.get())
            reconnect = int(self.reconnect_var.get())
            width = int(self.width_var.get()) if self.width_var.get().strip() else None
            height = int(self.height_var.get()) if self.height_var.get().strip() else None
        except ValueError as exc:
            raise ValueError("Port, FPS, reconnect interval and resolution must be numeric.") from exc
        return {
            "group_id": group_id,
            "name": self.name_var.get(),
            "host": self.host_var.get(),
            "port": port,
            "username": self.username_var.get(),
            "password": self.password_var.get() or None,
            "stream_path": self.path_var.get(),
            "target_fps": target_fps,
            "reconnect_seconds": reconnect,
            "width": width,
            "height": height,
        }

    def _stream_mode_changed(self, *_args: object) -> None:
        mode = self.stream_mode_var.get()
        if mode == self.MAIN_STREAM:
            self.path_var.set("/Streaming/Channels/101")
            if hasattr(self, "path_entry"):
                self.path_entry.configure(state="disabled")
        elif mode == self.SUB_STREAM:
            self.path_var.set("/Streaming/Channels/102")
            if hasattr(self, "path_entry"):
                self.path_entry.configure(state="disabled")
        elif hasattr(self, "path_entry"):
            self.path_entry.configure(state="normal")

    def _mode_from_path(self, path: str) -> str:
        if path == "/Streaming/Channels/101":
            return self.MAIN_STREAM
        if path == "/Streaming/Channels/102":
            return self.SUB_STREAM
        return self.CUSTOM_STREAM

    @staticmethod
    def _friendly_error(error: str | None) -> str:
        message = (error or "Unknown RTSP error.").strip()
        lowered = message.casefold()
        if "401" in lowered or "unauthorized" in lowered or "authentication" in lowered:
            return "Authentication failed. Re-enter username and password, save, and test again."
        if "404" in lowered or "not found" in lowered:
            return "The RTSP stream path was not found. Check stream quality and channel path."
        if "timeout" in lowered or "timed out" in lowered:
            return "The camera did not respond before the timeout. Check network connectivity."
        if "no valid frame" in lowered:
            return "The stream opened but did not return a valid video frame."
        return message[:450]

    def _show_error(self, title: str, exc: Exception) -> None:
        DialogService.show(
            self,
            UserMessage(MessageLevel.ERROR, title, str(exc) or type(exc).__name__),
        )
        self.status_var.set(title)

    def _close(self) -> None:
        if self._busy:
            DialogService.show(
                self,
                UserMessage(
                    MessageLevel.WARNING,
                    "Connection test running",
                    "Wait for the camera connection test to finish before closing.",
                ),
            )
            return
        if DialogService.confirm(
            self,
            title="Close camera management",
            message="Close Camera Management? Unsaved changes will be lost.",
        ):
            self.grab_release()
            self.destroy()
