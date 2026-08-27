"""Stable multi-group dashboard for Arnesis real-time processing.

The overview reuses group-card widgets instead of rebuilding them on every
refresh. Camera previews use a fixed-aspect canvas and preserve the previous
PhotoImage until a fully rendered replacement is ready.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

import cv2
from PIL import Image, ImageTk

from arnesis.ui.theme import ArnesisTheme
from arnesis.processing.visualization_renderer import InferenceVisualizationRenderer
from arnesis.ui.ux_messages import DialogService, MessageLevel, UserMessage


class RealTimeProcessingView(tk.Frame):
    """Show all groups together and open stable on-demand camera previews."""

    REFRESH_MS = 1000
    PREVIEW_MS = 150
    PREVIEW_ASPECT_RATIO = 16 / 9
    PREVIEW_MIN_WIDTH = 480
    PREVIEW_MIN_HEIGHT = 270

    STATE_COLORS = {
        "RUNNING": "#20C997",
        "STARTING": "#018FFF",
        "PAUSED": "#FFC000",
        "RECONNECTING": "#FF9F43",
        "STOPPING": "#018FFF",
        "STOPPED": "#53657C",
        "ERROR": "#FF5B69",
    }

    ACTIVE_CAMERA_STATES = {
        "RUNNING",
        "STARTING",
        "RECONNECTING",
        "PAUSED",
    }

    def __init__(
        self,
        parent: tk.Misc,
        group_service: Any,
        processing_service: Any,
        camera_service: Any,
    ) -> None:
        colors = ArnesisTheme.colors
        super().__init__(parent, bg=colors.background)

        self.group_service = group_service
        self.processing_service = processing_service
        self.camera_service = camera_service

        self._groups: list[Any] = []
        self._group_cards: dict[int, dict[str, Any]] = {}
        self._camera_cache: dict[int, list[Any]] = {}
        self._detail_group_id: int | None = None
        self._camera_widgets: dict[int, dict[str, Any]] = {}
        self._photos: dict[int, ImageTk.PhotoImage] = {}
        self._inference_results: dict[int, Any] = {}
        self._preview_subscriptions: set[tuple[int, int]] = set()
        self._overview_job: str | None = None
        self._preview_job: str | None = None
        self._destroyed = False
        self._focused_camera_id: int | None = None
        self.layout_var = tk.StringVar(value="Auto")

        self.search_var = tk.StringVar()
        self.filter_var = tk.StringVar(value="All states")
        self.total_var = tk.StringVar(value="0")
        self.running_var = tk.StringVar(value="0")
        self.stopped_var = tk.StringVar(value="0")
        self.error_var = tk.StringVar(value="0")
        self.cameras_var = tk.StringVar(value="0 / 0")

        self._build()
        self.search_var.trace_add("write", lambda *_: self._apply_card_visibility())
        self.reload_groups()
        self._schedule_overview_refresh()

    def destroy(self) -> None:
        self._destroyed = True
        self._release_preview_subscriptions()
        for job in (self._overview_job, self._preview_job):
            if job:
                try:
                    self.after_cancel(job)
                except tk.TclError:
                    pass
        self._overview_job = None
        self._preview_job = None
        super().destroy()

    def _build(self) -> None:
        colors = ArnesisTheme.colors
        self.overview = tk.Frame(self, bg=colors.background)
        self.detail = tk.Frame(self, bg=colors.background)

        header = tk.Frame(
            self.overview,
            bg=colors.surface,
            padx=18,
            pady=14,
            highlightthickness=1,
            highlightbackground=colors.border,
        )
        header.pack(fill="x", pady=(0, 12))
        tk.Label(
            header,
            text="MULTI-GROUP LIVE OPERATIONS",
            bg=colors.surface,
            fg=colors.accent,
            font=(ArnesisTheme.font_family, 15, "bold"),
        ).pack(side="left")
        ArnesisTheme.button(
            header,
            text="Refresh",
            command=self.reload_groups,
            variant="ghost",
        ).pack(side="right")

        kpis = tk.Frame(self.overview, bg=colors.background)
        kpis.pack(fill="x", pady=(0, 12))
        summary = (
            ("TOTAL GROUPS", self.total_var, colors.accent),
            ("RUNNING", self.running_var, colors.success),
            ("STOPPED", self.stopped_var, "#53657C"),
            ("ERRORS", self.error_var, colors.danger),
            ("ACTIVE CAMERAS", self.cameras_var, colors.warning),
        )
        for column, (label, variable, color) in enumerate(summary):
            kpis.grid_columnconfigure(column, weight=1, uniform="summary")
            self._create_kpi(kpis, column, label, variable, color)

        filters = tk.Frame(self.overview, bg=colors.surface, padx=14, pady=10)
        filters.pack(fill="x", pady=(0, 12))
        ttk.Entry(
            filters,
            textvariable=self.search_var,
            style="Arnesis.TEntry",
        ).pack(side="left", fill="x", expand=True)
        state_combo = ttk.Combobox(
            filters,
            textvariable=self.filter_var,
            state="readonly",
            values=("All states", "RUNNING", "PAUSED", "STOPPED", "ERROR"),
            style="Arnesis.TCombobox",
            width=18,
        )
        state_combo.pack(side="left", padx=(10, 0))
        state_combo.bind("<<ComboboxSelected>>", lambda _event: self._apply_card_visibility())

        shell = tk.Frame(
            self.overview,
            bg=colors.surface,
            highlightthickness=1,
            highlightbackground=colors.border,
        )
        shell.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(shell, bg=colors.surface, highlightthickness=0)
        scrollbar = ttk.Scrollbar(shell, orient="vertical", command=self.canvas.yview)
        self.cards_frame = tk.Frame(self.canvas, bg=colors.surface)
        self.cards_frame.bind(
            "<Configure>",
            lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.create_window(
            (0, 0),
            window=self.cards_frame,
            anchor="nw",
            tags="cards-body",
        )
        self.canvas.bind(
            "<Configure>",
            lambda event: self.canvas.itemconfigure("cards-body", width=event.width),
        )
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        for column in range(3):
            self.cards_frame.grid_columnconfigure(column, weight=1, uniform="groups")

        self.overview.pack(fill="both", expand=True)

    def _create_kpi(
        self,
        parent: tk.Misc,
        column: int,
        label: str,
        variable: tk.StringVar,
        color: str,
    ) -> None:
        colors = ArnesisTheme.colors
        card = tk.Frame(
            parent,
            bg=colors.surface,
            padx=14,
            pady=10,
            highlightthickness=1,
            highlightbackground=colors.border,
        )
        card.grid(row=0, column=column, sticky="nsew", padx=4)
        tk.Frame(card, bg=color, height=3).pack(fill="x", pady=(0, 7))
        tk.Label(
            card,
            text=label,
            bg=colors.surface,
            fg=colors.text_muted,
            font=(ArnesisTheme.font_family, 7, "bold"),
        ).pack(anchor="w")
        tk.Label(
            card,
            textvariable=variable,
            bg=colors.surface,
            fg=colors.text,
            font=(ArnesisTheme.font_family, 14, "bold"),
        ).pack(anchor="w", pady=(3, 0))

    def reload_groups(self) -> None:
        try:
            latest_groups = list(self.group_service.list_groups())
            self._groups = latest_groups
            current_ids = {int(self._value(group, "id")) for group in latest_groups}

            for group_id in tuple(self._group_cards):
                if group_id not in current_ids:
                    self._group_cards[group_id]["frame"].destroy()
                    del self._group_cards[group_id]
                    self._camera_cache.pop(group_id, None)

            for group in latest_groups:
                group_id = int(self._value(group, "id"))
                if group_id not in self._group_cards:
                    self._group_cards[group_id] = self._create_group_card(group)
                self._camera_cache[group_id] = list(
                    self.camera_service.list_cameras(group_id)
                )

            self._refresh_overview_values()
        except Exception as exc:
            self._show_error("Groups unavailable", exc)

    def _runtime_map(self) -> dict[int, dict[str, Any]]:
        try:
            return {
                int(item["group_id"]): item
                for item in self.processing_service.get_runtime_status()
            }
        except Exception:
            return {}

    def _refresh_overview_values(self) -> None:
        if self._detail_group_id is not None:
            return

        runtime = self._runtime_map()
        running = 0
        stopped = 0
        errors = 0
        active_cameras = 0
        configured_cameras = 0

        for group in self._groups:
            group_id = int(self._value(group, "id"))
            live = runtime.get(group_id)
            state = self._state_for(group, live)
            cameras = self._camera_cache.get(group_id, [])
            snapshots = live.get("cameras", []) if isinstance(live, dict) else []
            active = sum(
                str(self._value(item, "state", default="")).upper()
                in self.ACTIVE_CAMERA_STATES
                for item in snapshots
            )

            configured_cameras += len(cameras)
            active_cameras += active
            running += int(state == "RUNNING")
            stopped += int(state == "STOPPED")
            errors += int(state == "ERROR")

            self._update_group_card(
                self._group_cards[group_id],
                group,
                live,
                state,
                cameras,
                active,
            )

        self.total_var.set(str(len(self._groups)))
        self.running_var.set(str(running))
        self.stopped_var.set(str(stopped))
        self.error_var.set(str(errors))
        self.cameras_var.set(f"{active_cameras} / {configured_cameras}")
        self._apply_card_visibility()

    def _create_group_card(self, group: Any) -> dict[str, Any]:
        colors = ArnesisTheme.colors
        group_id = int(self._value(group, "id"))
        frame = tk.Frame(
            self.cards_frame,
            bg=colors.surface_alt,
            padx=16,
            pady=14,
            highlightthickness=2,
            highlightbackground=self.STATE_COLORS["STOPPED"],
            cursor="hand2",
        )

        header = tk.Frame(frame, bg=colors.surface_alt)
        header.pack(fill="x")
        title_var = tk.StringVar()
        tk.Label(
            header,
            textvariable=title_var,
            bg=colors.surface_alt,
            fg=colors.text,
            font=(ArnesisTheme.font_family, 12, "bold"),
        ).pack(side="left")
        state_var = tk.StringVar(value="STOPPED")
        state_label = tk.Label(
            header,
            textvariable=state_var,
            bg=self.STATE_COLORS["STOPPED"],
            fg="#FFFFFF",
            font=(ArnesisTheme.font_family, 8, "bold"),
            padx=10,
            pady=4,
        )
        state_label.pack(side="right")

        information = tk.Frame(frame, bg=colors.surface_alt)
        information.pack(fill="x", pady=(14, 10))
        camera_var, fps_var, device_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
        for label, variable in (
            ("CAMERAS", camera_var),
            ("TOTAL FPS", fps_var),
            ("DEVICE", device_var),
        ):
            box = tk.Frame(information, bg=colors.surface, padx=10, pady=8)
            box.pack(side="left", fill="x", expand=True, padx=3)
            tk.Label(
                box,
                text=label,
                bg=colors.surface,
                fg=colors.text_muted,
                font=(ArnesisTheme.font_family, 7, "bold"),
            ).pack(anchor="w")
            tk.Label(
                box,
                textvariable=variable,
                bg=colors.surface,
                fg=colors.text,
                font=(ArnesisTheme.font_family, 9, "bold"),
                wraplength=130,
            ).pack(anchor="w")

        actions = tk.Frame(frame, bg=colors.surface_alt)
        actions.pack(fill="x")
        primary_button = ArnesisTheme.button(
            actions,
            text="Start",
            command=lambda gid=group_id: self._handle_primary_action(gid),
            variant="success",
        )
        primary_button.pack(side="left")
        stop_button = ArnesisTheme.button(
            actions,
            text="Stop",
            command=lambda gid=group_id: self._execute_action(gid, "stop_group"),
            variant="danger",
        )
        stop_button.pack(side="right")

        for widget in (frame, header):
            widget.bind(
                "<Double-1>",
                lambda _event, gid=group_id: self._open_group(gid),
            )

        return {
            "frame": frame,
            "title_var": title_var,
            "state_var": state_var,
            "state_label": state_label,
            "camera_var": camera_var,
            "fps_var": fps_var,
            "device_var": device_var,
            "primary_button": primary_button,
            "stop_button": stop_button,
            "state": "STOPPED",
            "signature": None,
            "visible": False,
            "search_text": "",
        }

    def _update_group_card(
        self,
        card: dict[str, Any],
        group: Any,
        live: dict[str, Any] | None,
        state: str,
        cameras: list[Any],
        active: int,
    ) -> None:
        snapshots = live.get("cameras", []) if isinstance(live, dict) else []
        fps = sum(
            float(self._value(item, "measured_fps", "capture_fps", default=0) or 0)
            for item in snapshots
        )
        gpu = self._value(live, "cuda_device", "gpu_display", default=None)
        if not gpu:
            preferred = self._value(group, "preferred_gpu_index", default=None)
            gpu = f"CUDA:{preferred}" if preferred is not None else "Automatic CUDA"

        title = f"{self._value(group, 'code')}  |  {self._value(group, 'name')}"
        signature = (title, state, active, len(cameras), round(fps, 1), str(gpu))
        if card["signature"] == signature:
            return

        color = self.STATE_COLORS[state]
        foreground = "#06142C" if state in {"RUNNING", "PAUSED"} else "#FFFFFF"
        card["title_var"].set(title)
        card["state_var"].set(state)
        card["state_label"].configure(bg=color, fg=foreground)
        card["frame"].configure(highlightbackground=color)
        card["camera_var"].set(f"{active} / {len(cameras)}")
        card["fps_var"].set(f"{fps:.1f}")
        card["device_var"].set(str(gpu))
        card["state"] = state
        card["signature"] = signature
        card["search_text"] = (
            f"{title} "
            + " ".join(str(self._value(camera, "name", default="")) for camera in cameras)
        ).casefold()

        if state == "STOPPED":
            card["primary_button"].configure(text="Start", bg=ArnesisTheme.colors.success)
            card["stop_button"].pack_forget()
        else:
            card["primary_button"].configure(text="Open", bg=ArnesisTheme.colors.primary)
            if state in {"RUNNING", "PAUSED", "ERROR", "STARTING", "RECONNECTING"}:
                card["stop_button"].pack(side="right")
            else:
                card["stop_button"].pack_forget()

    def _apply_card_visibility(self) -> None:
        search = self.search_var.get().strip().casefold()
        state_filter = self.filter_var.get()
        visible_cards: list[dict[str, Any]] = []

        for card in self._group_cards.values():
            matches_search = not search or search in card["search_text"]
            matches_state = state_filter == "All states" or card["state"] == state_filter
            if matches_search and matches_state:
                visible_cards.append(card)
            elif card["visible"]:
                card["frame"].grid_remove()
                card["visible"] = False

        for index, card in enumerate(visible_cards):
            row, column = divmod(index, 3)
            desired = (row, column)
            current = card.get("grid_position")
            if not card["visible"] or current != desired:
                card["frame"].grid(
                    row=row,
                    column=column,
                    sticky="nsew",
                    padx=8,
                    pady=8,
                )
                card["grid_position"] = desired
                card["visible"] = True

    def _handle_primary_action(self, group_id: int) -> None:
        card = self._group_cards[group_id]
        if card["state"] == "STOPPED":
            self._execute_action(group_id, "start_group")
        else:
            self._open_group(group_id)

    def _execute_action(self, group_id: int, method_name: str) -> None:
        try:
            getattr(self.processing_service, method_name)(group_id)
            self.after(200, self._refresh_overview_values)
        except Exception as exc:
            self._show_error("Runtime action failed", exc)

    def _open_group(self, group_id: int) -> None:
        self._detail_group_id = group_id
        self.overview.pack_forget()
        self.detail.pack(fill="both", expand=True)
        self._build_detail()
        self._activate_preview_subscriptions()
        self._refresh_detail_metrics()
        self._schedule_preview_refresh()

    def _build_detail(self) -> None:
        for child in self.detail.winfo_children():
            child.destroy()

        colors = ArnesisTheme.colors
        group = next(
            group
            for group in self._groups
            if int(self._value(group, "id")) == self._detail_group_id
        )

        header = tk.Frame(self.detail, bg=colors.surface, padx=16, pady=12)
        header.pack(fill="x", pady=(0, 12))
        ArnesisTheme.button(
            header,
            text="Back to groups",
            command=self._close_detail,
            variant="ghost",
        ).pack(side="left")
        tk.Label(
            header,
            text=f"{self._value(group, 'code')} | {self._value(group, 'name')}",
            bg=colors.surface,
            fg=colors.accent,
            font=(ArnesisTheme.font_family, 15, "bold"),
        ).pack(side="left", padx=15)
        self.detail_state = tk.Label(
            header,
            text="STOPPED",
            bg=self.STATE_COLORS["STOPPED"],
            fg="#FFFFFF",
            padx=10,
            pady=5,
            font=(ArnesisTheme.font_family, 8, "bold"),
        )
        self.detail_state.pack(side="right")

        layout_bar = tk.Frame(self.detail, bg=colors.surface, padx=12, pady=8)
        layout_bar.pack(fill="x", pady=(0, 8))
        tk.Label(
            layout_bar,
            text="CAMERA LAYOUT",
            bg=colors.surface,
            fg=colors.text_muted,
            font=(ArnesisTheme.font_family, 8, "bold"),
        ).pack(side="left")
        layout_combo = ttk.Combobox(
            layout_bar,
            textvariable=self.layout_var,
            state="readonly",
            values=("Auto", "1 x 1", "2 x 2", "3 x 3", "Focus"),
            style="Arnesis.TCombobox",
            width=12,
        )
        layout_combo.pack(side="left", padx=(10, 0))
        layout_combo.bind(
            "<<ComboboxSelected>>",
            lambda _event: self._apply_camera_layout(),
        )
        ArnesisTheme.button(
            layout_bar,
            text="Reset focus",
            command=self._reset_camera_focus,
            variant="ghost",
        ).pack(side="right")
        tk.Label(
            layout_bar,
            text="Double-click video for fullscreen. Use Focus to enlarge one camera.",
            bg=colors.surface,
            fg=colors.text_muted,
            font=(ArnesisTheme.font_family, 8),
        ).pack(side="right", padx=(0, 12))

        detail_shell = tk.Frame(self.detail, bg=colors.surface)
        detail_shell.pack(fill="both", expand=True)
        detail_canvas = tk.Canvas(detail_shell, bg=colors.surface, highlightthickness=0)
        detail_scroll = ttk.Scrollbar(
            detail_shell,
            orient="vertical",
            command=detail_canvas.yview,
        )
        self.camera_grid = tk.Frame(detail_canvas, bg=colors.surface, padx=10, pady=10)
        self.camera_grid.bind(
            "<Configure>",
            lambda _event: detail_canvas.configure(scrollregion=detail_canvas.bbox("all")),
        )
        detail_canvas.create_window(
            (0, 0),
            window=self.camera_grid,
            anchor="nw",
            tags="camera-body",
        )
        detail_canvas.bind(
            "<Configure>",
            lambda event: detail_canvas.itemconfigure("camera-body", width=event.width),
        )
        detail_canvas.configure(yscrollcommand=detail_scroll.set)
        detail_canvas.pack(side="left", fill="both", expand=True)
        detail_scroll.pack(side="right", fill="y")

        self._camera_widgets.clear()
        self._photos.clear()
        cameras = self._camera_cache.get(self._detail_group_id, [])
        for column in range(3):
            self.camera_grid.grid_columnconfigure(column, weight=1, uniform="cameras")

        for index, camera in enumerate(cameras):
            camera_id = int(self._value(camera, "id"))
            card = tk.Frame(
                self.camera_grid,
                bg=colors.surface_alt,
                padx=10,
                pady=10,
                highlightthickness=1,
                highlightbackground=colors.border,
            )
            card.grid(
                row=index // 2,
                column=index % 2,
                sticky="nsew",
                padx=6,
                pady=6,
            )
            tk.Label(
                card,
                text=str(self._value(camera, "name")),
                bg=colors.surface_alt,
                fg=colors.text,
                font=(ArnesisTheme.font_family, 11, "bold"),
            ).pack(anchor="w")

            preview_shell = tk.Frame(
                card,
                bg="#030D1F",
                height=self.PREVIEW_MIN_HEIGHT,
                highlightthickness=1,
                highlightbackground="#102B5A",
            )
            preview_shell.pack(fill="x", pady=8)
            preview_shell.pack_propagate(False)

            preview_canvas = tk.Canvas(
                preview_shell,
                bg="#030D1F",
                highlightthickness=0,
                height=self.PREVIEW_MIN_HEIGHT,
            )
            preview_canvas.pack(fill="both", expand=True)
            text_id = preview_canvas.create_text(
                self.PREVIEW_MIN_WIDTH // 2,
                self.PREVIEW_MIN_HEIGHT // 2,
                text="Waiting for live frame",
                fill=colors.text_muted,
                font=(ArnesisTheme.font_family, 9),
                anchor="center",
            )
            image_id = preview_canvas.create_image(0, 0, anchor="center", state="hidden")
            preview_canvas.bind(
                "<Configure>",
                lambda event, cid=camera_id: self._on_preview_resize(cid, event.width, event.height),
            )
            preview_canvas.bind(
                "<Double-1>",
                lambda _event, cid=camera_id: self._fullscreen(cid),
            )

            metrics_var = tk.StringVar(value="STOPPED | 0.0 FPS | Resolution --")
            footer = tk.Frame(card, bg=colors.surface_alt)
            footer.pack(fill="x")
            tk.Label(
                footer,
                textvariable=metrics_var,
                bg=colors.surface_alt,
                fg=colors.accent,
                font=(ArnesisTheme.font_family, 8, "bold"),
            ).pack(side="left", anchor="w")
            ArnesisTheme.button(
                footer,
                text="Focus",
                command=lambda cid=camera_id: self._focus_camera(cid),
                variant="ghost",
            ).pack(side="right")

            self._camera_widgets[camera_id] = {
                "card": card,
                "preview_shell": preview_shell,
                "canvas": preview_canvas,
                "image_id": image_id,
                "text_id": text_id,
                "metrics_var": metrics_var,
                "last_frame": None,
                "last_captured_at": None,
                "last_render_key": None,
            }

        self._apply_camera_layout()

    def _on_preview_resize(self, camera_id: int, width: int, height: int) -> None:
        widgets = self._camera_widgets.get(camera_id)
        if widgets is None:
            return
        width = max(width, 1)
        height = max(height, 1)
        canvas: tk.Canvas = widgets["canvas"]
        canvas.coords(widgets["text_id"], width // 2, height // 2)
        canvas.coords(widgets["image_id"], width // 2, height // 2)
        frame = widgets.get("last_frame")
        if frame is not None and width >= 32 and height >= 32:
            self._render_preview(
                camera_id,
                frame,
                width,
                height,
                widgets.get("last_captured_at"),
            )

    def _refresh_detail_metrics(self) -> None:
        if self._detail_group_id is None:
            return
        live = self._runtime_map().get(self._detail_group_id)
        state = str(self._value(live, "state", default="STOPPED")).upper()
        color = self.STATE_COLORS.get(state, self.STATE_COLORS["STOPPED"])
        foreground = "#06142C" if state in {"RUNNING", "PAUSED"} else "#FFFFFF"
        self.detail_state.configure(text=state, bg=color, fg=foreground)

        snapshots = {
            int(self._value(item, "camera_id")): item
            for item in (live.get("cameras", []) if isinstance(live, dict) else [])
            if self._value(item, "camera_id", default=None) is not None
        }
        for camera_id, widgets in self._camera_widgets.items():
            snapshot = snapshots.get(camera_id)
            fps = float(
                self._value(snapshot, "measured_fps", "capture_fps", default=0) or 0
            )
            width = int(self._value(snapshot, "frame_width", default=0) or 0)
            height = int(self._value(snapshot, "frame_height", default=0) or 0)
            camera_state = str(self._value(snapshot, "state", default="STOPPED"))
            resolution = f"{width} x {height}" if width and height else "--"
            widgets["metrics_var"].set(
                f"{camera_state} | {fps:.1f} FPS | Resolution {resolution}"
            )

    def _activate_preview_subscriptions(self) -> None:
        if self._detail_group_id is None:
            return
        for camera_id in self._camera_widgets:
            key = (self._detail_group_id, camera_id)
            if key not in self._preview_subscriptions:
                self.processing_service.subscribe_preview(*key)
                self._preview_subscriptions.add(key)

    def _release_preview_subscriptions(self) -> None:
        subscriptions = tuple(self._preview_subscriptions)
        self._preview_subscriptions.clear()
        for group_id, camera_id in subscriptions:
            try:
                self.processing_service.unsubscribe_preview(group_id, camera_id)
            except Exception:
                pass
        self._photos.clear()

    def _schedule_preview_refresh(self) -> None:
        if self._destroyed or self._detail_group_id is None:
            return
        self._refresh_detail_metrics()
        for camera_id, widgets in self._camera_widgets.items():
            try:
                previous = self._inference_results.get(camera_id)
                latest = self.processing_service.latest_inference_result(
                    self._detail_group_id, camera_id,
                    None if previous is None else previous.sequence,
                )
                if latest is not None:
                    self._inference_results[camera_id] = latest
                packet = self.processing_service.preview_frame(
                    self._detail_group_id,
                    camera_id,
                )
                frame = getattr(packet, "frame", None) if packet is not None else None
                if frame is None:
                    continue
                captured_at = getattr(packet, "captured_at", None)
                if captured_at is None:
                    captured_at = id(frame)
                if widgets.get("last_captured_at") == captured_at:
                    continue
                widgets["last_frame"] = frame
                widgets["last_captured_at"] = captured_at
                canvas: tk.Canvas = widgets["canvas"]
                width = canvas.winfo_width()
                height = canvas.winfo_height()
                if width >= 32 and height >= 32:
                    self._render_preview(
                        camera_id,
                        frame,
                        width,
                        height,
                        captured_at,
                    )
            except Exception as exc:
                widgets["metrics_var"].set(
                    f"Preview error: {type(exc).__name__}: {exc}"
                )
        self._preview_job = self.after(self.PREVIEW_MS, self._schedule_preview_refresh)

    def _render_preview(
        self,
        camera_id: int,
        frame: Any,
        target_width: int,
        target_height: int,
        captured_at: Any = None,
    ) -> None:
        if target_width < 32 or target_height < 32:
            return
        frame_height, frame_width = frame.shape[:2]
        if frame_width <= 0 or frame_height <= 0:
            return

        scale = min(target_width / frame_width, target_height / frame_height)
        render_width = max(1, int(frame_width * scale))
        render_height = max(1, int(frame_height * scale))
        size = (render_width, render_height)

        widgets = self._camera_widgets[camera_id]
        render_key = (captured_at, size)
        if widgets.get("last_render_key") == render_key:
            return

        rendered = InferenceVisualizationRenderer.render(
            frame, self._inference_results.get(camera_id)
        )
        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize(size, resampling)
        photo = ImageTk.PhotoImage(image)

        canvas: tk.Canvas = widgets["canvas"]
        canvas.itemconfigure(widgets["image_id"], image=photo, state="normal")
        canvas.itemconfigure(widgets["text_id"], state="hidden")
        canvas.coords(widgets["image_id"], target_width // 2, target_height // 2)
        self._photos[camera_id] = photo
        widgets["photo"] = photo
        widgets["last_render_key"] = render_key

    def _layout_columns(self) -> int:
        mode = self.layout_var.get()
        if mode == "1 x 1" or mode == "Focus":
            return 1
        if mode == "2 x 2":
            return 2
        if mode == "3 x 3":
            return 3
        count = max(1, len(self._camera_widgets))
        return 1 if count == 1 else 2 if count <= 4 else 3

    def _layout_preview_height(self, columns: int) -> int:
        if columns == 1:
            return 480
        if columns == 2:
            return 300
        return 220

    def _apply_camera_layout(self) -> None:
        if not self._camera_widgets:
            return
        columns = self._layout_columns()
        focus_mode = self.layout_var.get() == "Focus"
        visible_ids = list(self._camera_widgets)
        if focus_mode and self._focused_camera_id in self._camera_widgets:
            visible_ids = [self._focused_camera_id]
        elif focus_mode:
            self._focused_camera_id = visible_ids[0]
            visible_ids = [self._focused_camera_id]

        for column in range(3):
            self.camera_grid.grid_columnconfigure(
                column,
                weight=1 if column < columns else 0,
                uniform="cameras" if column < columns else "",
            )

        preview_height = self._layout_preview_height(columns)
        for camera_id, widgets in self._camera_widgets.items():
            card: tk.Frame = widgets["card"]
            if camera_id not in visible_ids:
                card.grid_remove()
                continue
            index = visible_ids.index(camera_id)
            row, column = divmod(index, columns)
            card.grid(
                row=row,
                column=column,
                sticky="nsew",
                padx=6,
                pady=6,
            )
            widgets["preview_shell"].configure(height=preview_height)
            widgets["canvas"].configure(height=preview_height)
            widgets["last_render_key"] = None

    def _focus_camera(self, camera_id: int) -> None:
        self._focused_camera_id = camera_id
        self.layout_var.set("Focus")
        self._apply_camera_layout()

    def _reset_camera_focus(self) -> None:
        self._focused_camera_id = None
        self.layout_var.set("Auto")
        self._apply_camera_layout()

    def _fullscreen(self, camera_id: int) -> None:
        if self._detail_group_id is None:
            return
        packet = self.processing_service.preview_frame(
            self._detail_group_id,
            camera_id,
        )
        frame = getattr(packet, "frame", None) if packet is not None else None
        if frame is None:
            return

        window = tk.Toplevel(self)
        window.title("Arnesis Camera Preview")
        window.attributes("-fullscreen", True)
        window.configure(bg="#000000")
        label = tk.Label(window, bg="#000000")
        label.pack(fill="both", expand=True)
        window.bind("<Escape>", lambda _event: window.destroy())

        rendered = InferenceVisualizationRenderer.render(
            frame, self._inference_results.get(camera_id)
        )
        rgb = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        image.thumbnail((screen_width, screen_height))
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo)
        label.image = photo

    def _close_detail(self) -> None:
        self._release_preview_subscriptions()
        if self._preview_job:
            try:
                self.after_cancel(self._preview_job)
            except tk.TclError:
                pass
            self._preview_job = None
        self._focused_camera_id = None
        self.layout_var.set("Auto")
        self._detail_group_id = None
        self.detail.pack_forget()
        self.overview.pack(fill="both", expand=True)
        self._refresh_overview_values()

    def _schedule_overview_refresh(self) -> None:
        if self._destroyed or not self.winfo_exists():
            return
        if self._detail_group_id is None:
            self._refresh_overview_values()
        self._overview_job = self.after(
            self.REFRESH_MS,
            self._schedule_overview_refresh,
        )

    def _state_for(self, group: Any, live: dict[str, Any] | None) -> str:
        state = str(
            self._value(
                live,
                "state",
                "status",
                default=self._value(group, "status", default="STOPPED"),
            )
        ).upper()
        return state if state in self.STATE_COLORS else "STOPPED"

    def _show_error(self, title: str, exc: Exception) -> None:
        DialogService.show(
            self,
            UserMessage(
                MessageLevel.ERROR,
                title,
                "The requested operation could not be completed.",
                str(exc),
            ),
        )

    @staticmethod
    def _value(item: Any, *names: str, default: Any = None) -> Any:
        for name in names:
            if isinstance(item, dict) and name in item:
                return item[name]
            if item is not None and hasattr(item, name):
                return getattr(item, name)
        return default
