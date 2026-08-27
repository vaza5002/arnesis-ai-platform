"""Group-oriented polygon ROI editor for Arnesis.

The workspace follows the original Arnesis interaction model:
- group selection at the top;
- camera thumbnails in a left navigation rail;
- one complete camera frame in the right workspace;
- explicit frame refresh;
- polygon creation, editing, deletion, and persistence;
- normalized coordinates independent of camera resolution.
"""

from __future__ import annotations

import threading
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable

import cv2
from PIL import Image, ImageTk

from arnesis.ui.theme import ArnesisTheme




class RoiEditorView(tk.Frame):
    """Edit camera polygon ROIs from a group and camera preview workspace."""

    NAVY = "#012059"
    PANEL = "#081F4E"
    PANEL_ALT = "#0B2D67"
    CYAN = "#29E6FF"
    BLUE = "#018FFF"
    WHITE = "#F4F8FF"
    MUTED = "#9EB0CA"
    WARNING = "#FFC000"
    DANGER = "#D94E5D"
    CANVAS_BACKGROUND = "#020B1A"

    def __init__(
        self,
        parent: tk.Misc,
        group_service: Any,
        camera_service: Any,
        roi_service: Any,
        profile_service: Any | None = None,
        initial_group_id: int | None = None,
        initial_camera_id: int | None = None,
    ) -> None:
        super().__init__(parent, bg=self.NAVY)
        self.group_service = group_service
        self.camera_service = camera_service
        self.roi_service = roi_service
        self.profile_service = profile_service
        self._profile_ids_by_label: dict[str, int | None] = {"None": None}
        self.initial_group_id = initial_group_id
        self.initial_camera_id = initial_camera_id

        self._groups_by_label: dict[str, object] = {}
        self._cameras_by_id: dict[int, object] = {}
        self._camera_cards: dict[int, dict[str, tk.Widget]] = {}
        self._thumbnail_photos: dict[int, ImageTk.PhotoImage] = {}
        self._snapshot_frames: dict[int, Any] = {}
        self._snapshot_loading: set[int] = set()

        self.selected_group_id: int | None = None
        self.selected_camera_id: int | None = None
        self.selected_roi_id: int | None = None
        self.current_frame: Any | None = None
        self.current_image: Image.Image | None = None
        self.current_points: list[dict[str, float]] = []
        self.saved_rois: list[dict[str, Any]] = []
        self._main_photo: ImageTk.PhotoImage | None = None
        self._image_box = (0, 0, 1, 1)
        self._drag_point_index: int | None = None

        self.group_var = tk.StringVar()
        self.roi_name_var = tk.StringVar(value="ROI 1")
        self.enabled_var = tk.BooleanVar(value=True)
        self.profile_var = tk.StringVar(value="None")
        self.status_var = tk.StringVar(
            value="Select a group and camera, then refresh the frame."
        )
        self.camera_title_var = tk.StringVar(value="No camera selected")
        self.frame_info_var = tk.StringVar(value="Resolution: -- | Frame unavailable")

        self._build_ui()
        self._load_profile_options()
        self.reload_groups()

    def destroy(self) -> None:
        """Destroy the editor without live preview timers or subscriptions."""
        super().destroy()

    def _build_ui(self) -> None:
        header = tk.Frame(self, bg=self.PANEL, padx=18, pady=14)
        header.pack(fill="x", pady=(0, 10))

        title_stack = tk.Frame(header, bg=self.PANEL)
        title_stack.pack(side="left")
        tk.Label(
            title_stack,
            text="ROI CONFIGURATION",
            bg=self.PANEL,
            fg=self.CYAN,
            font=(ArnesisTheme.font_family, 16, "bold"),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            title_stack,
            text="Select a camera, refresh its complete frame, and define station polygons",
            bg=self.PANEL,
            fg=self.MUTED,
            font=(ArnesisTheme.font_family, 9),
            anchor="w",
        ).pack(fill="x", pady=(3, 0))

        group_controls = tk.Frame(header, bg=self.PANEL)
        group_controls.pack(side="right")
        tk.Label(
            group_controls,
            text="Processing group",
            bg=self.PANEL,
            fg=self.WHITE,
            font=(ArnesisTheme.font_family, 9, "bold"),
        ).pack(side="left", padx=(0, 8))
        self.group_combo = ttk.Combobox(
            group_controls,
            textvariable=self.group_var,
            state="readonly",
            width=30,
            style="Arnesis.TCombobox",
        )
        self.group_combo.pack(side="left")
        self.group_combo.bind("<<ComboboxSelected>>", self._on_group_selected)

        body = tk.PanedWindow(
            self,
            orient="horizontal",
            bg=self.NAVY,
            sashwidth=6,
            borderwidth=0,
            sashrelief="flat",
        )
        body.pack(fill="both", expand=True)

        camera_panel = tk.Frame(
            body,
            bg=self.PANEL,
            padx=12,
            pady=12,
            highlightthickness=1,
            highlightbackground="#173F78",
        )
        editor_panel = tk.Frame(
            body,
            bg=self.PANEL,
            padx=14,
            pady=12,
            highlightthickness=1,
            highlightbackground="#173F78",
        )
        body.add(camera_panel, minsize=300, width=340)
        body.add(editor_panel, minsize=760)

        self._build_camera_panel(camera_panel)
        self._build_editor_panel(editor_panel)

        status = tk.Label(
            self,
            textvariable=self.status_var,
            bg="#05142C",
            fg=self.WHITE,
            anchor="w",
            justify="left",
            padx=16,
            pady=8,
            font=(ArnesisTheme.font_family, 9),
        )
        status.pack(fill="x", pady=(10, 0))

    def _build_camera_panel(self, parent: tk.Frame) -> None:
        tk.Label(
            parent,
            text="GROUP CAMERAS",
            bg=self.PANEL,
            fg=self.CYAN,
            font=(ArnesisTheme.font_family, 12, "bold"),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            parent,
            text="Select a camera preview to edit its ROIs",
            bg=self.PANEL,
            fg=self.MUTED,
            font=(ArnesisTheme.font_family, 8),
            anchor="w",
        ).pack(fill="x", pady=(3, 10))

        shell = tk.Frame(parent, bg=self.PANEL)
        shell.pack(fill="both", expand=True)
        self.camera_canvas = tk.Canvas(
            shell,
            bg=self.PANEL,
            highlightthickness=0,
            width=300,
        )
        scrollbar = ttk.Scrollbar(
            shell,
            orient="vertical",
            command=self.camera_canvas.yview,
        )
        self.camera_list = tk.Frame(self.camera_canvas, bg=self.PANEL)
        self.camera_list.bind(
            "<Configure>",
            lambda _: self.camera_canvas.configure(
                scrollregion=self.camera_canvas.bbox("all")
            ),
        )
        self.camera_canvas.create_window(
            (0, 0),
            window=self.camera_list,
            anchor="nw",
            tags="camera_list",
        )
        self.camera_canvas.bind(
            "<Configure>",
            lambda event: self.camera_canvas.itemconfigure(
                "camera_list",
                width=event.width,
            ),
        )
        self.camera_canvas.configure(yscrollcommand=scrollbar.set)
        self.camera_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def _build_editor_panel(self, parent: tk.Frame) -> None:
        toolbar = tk.Frame(parent, bg=self.PANEL)
        toolbar.pack(fill="x", pady=(0, 10))

        title_stack = tk.Frame(toolbar, bg=self.PANEL)
        title_stack.pack(side="left", fill="x", expand=True)
        tk.Label(
            title_stack,
            textvariable=self.camera_title_var,
            bg=self.PANEL,
            fg=self.WHITE,
            font=(ArnesisTheme.font_family, 14, "bold"),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            title_stack,
            textvariable=self.frame_info_var,
            bg=self.PANEL,
            fg=self.MUTED,
            font=(ArnesisTheme.font_family, 8),
            anchor="w",
        ).pack(fill="x", pady=(2, 0))

        self._button(
            toolbar,
            "Refresh frame",
            self.refresh_current_frame,
            self.BLUE,
        ).pack(side="right")

        work_area = tk.Frame(parent, bg=self.PANEL)
        work_area.pack(fill="both", expand=True)

        canvas_shell = tk.Frame(
            work_area,
            bg=self.CANVAS_BACKGROUND,
            highlightthickness=1,
            highlightbackground="#173F78",
        )
        canvas_shell.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(
            canvas_shell,
            bg=self.CANVAS_BACKGROUND,
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)
        self.canvas.bind("<Configure>", lambda _: self._render_main_frame())

        controls = tk.Frame(
            work_area,
            bg=self.PANEL_ALT,
            padx=14,
            pady=14,
            width=260,
        )
        controls.pack(side="right", fill="y", padx=(12, 0))
        controls.pack_propagate(False)
        self._build_roi_controls(controls)

    def _build_roi_controls(self, parent: tk.Frame) -> None:
        tk.Label(
            parent,
            text="ROI TOOLS",
            bg=self.PANEL_ALT,
            fg=self.CYAN,
            font=(ArnesisTheme.font_family, 12, "bold"),
        ).pack(anchor="w")

        tk.Label(
            parent,
            text="ROI name / station",
            bg=self.PANEL_ALT,
            fg=self.WHITE,
            font=(ArnesisTheme.font_family, 9, "bold"),
        ).pack(anchor="w", pady=(16, 5))
        ttk.Entry(
            parent,
            textvariable=self.roi_name_var,
            style="Arnesis.TEntry",
        ).pack(fill="x")

        ttk.Checkbutton(
            parent,
            text="ROI enabled",
            variable=self.enabled_var,
            style="Arnesis.TCheckbutton",
        ).pack(anchor="w", pady=(10, 4))

        tk.Label(
            parent,
            text="Processing profile",
            bg=self.PANEL_ALT,
            fg=self.WHITE,
            font=(ArnesisTheme.font_family, 9, "bold"),
        ).pack(anchor="w", pady=(10, 5))
        self.profile_combo = ttk.Combobox(
            parent,
            textvariable=self.profile_var,
            state="readonly",
            style="Arnesis.TCombobox",
        )
        self.profile_combo.pack(fill="x")

        self._button(parent, "New polygon", self.new_polygon, self.BLUE).pack(
            fill="x", pady=(12, 0)
        )
        self._button(parent, "Undo point", self.undo_point, "#345A91").pack(
            fill="x", pady=(8, 0)
        )
        self._button(parent, "Clear polygon", self.clear_polygon, "#345A91").pack(
            fill="x", pady=(8, 0)
        )
        self._button(parent, "Save ROI", self.save_roi, "#1877D2").pack(
            fill="x", pady=(8, 0)
        )
        self._button(parent, "Delete ROI", self.delete_roi, self.DANGER).pack(
            fill="x", pady=(8, 0)
        )

        tk.Label(
            parent,
            text="SAVED ROIS",
            bg=self.PANEL_ALT,
            fg=self.CYAN,
            font=(ArnesisTheme.font_family, 10, "bold"),
        ).pack(anchor="w", pady=(20, 6))

        list_shell = tk.Frame(parent, bg=self.PANEL_ALT)
        list_shell.pack(fill="both", expand=True)
        self.roi_list = tk.Listbox(
            list_shell,
            bg="#061A3D",
            fg=self.WHITE,
            selectbackground=self.BLUE,
            selectforeground="#FFFFFF",
            activestyle="none",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground="#173F78",
            font=(ArnesisTheme.font_family, 9),
        )
        roi_scroll = ttk.Scrollbar(
            list_shell,
            orient="vertical",
            command=self.roi_list.yview,
        )
        self.roi_list.configure(yscrollcommand=roi_scroll.set)
        self.roi_list.pack(side="left", fill="both", expand=True)
        roi_scroll.pack(side="right", fill="y")
        self.roi_list.bind("<<ListboxSelect>>", self._on_roi_selected)

        tk.Label(
            parent,
            text=(
                "Click inside the frame to add polygon points. "
                "Drag a yellow point to adjust it. A minimum of three points is required."
            ),
            bg=self.PANEL_ALT,
            fg=self.MUTED,
            justify="left",
            wraplength=225,
            font=(ArnesisTheme.font_family, 8),
        ).pack(anchor="w", pady=(12, 0))

    def _load_profile_options(self) -> None:
        """Load enabled profiles for ROI assignment."""
        self._profile_ids_by_label = {"None": None}
        if self.profile_service is not None:
            for option in self.profile_service.list_options(enabled_only=True):
                self._profile_ids_by_label[option.label] = option.id
        if hasattr(self, "profile_combo"):
            self.profile_combo.configure(values=tuple(self._profile_ids_by_label))
        if self.profile_var.get() not in self._profile_ids_by_label:
            self.profile_var.set("None")

    def _profile_label(self, profile_id: int | None) -> str:
        return next(
            (label for label, value in self._profile_ids_by_label.items() if value == profile_id),
            "None",
        )

    def reload_groups(self) -> None:
        groups = list(self.group_service.list_groups())
        self._groups_by_label = {
            f"{self._value(group, 'code')} | {self._value(group, 'name')}": group
            for group in groups
        }
        labels = tuple(self._groups_by_label)
        self.group_combo.configure(values=labels)
        if not labels:
            self.group_var.set("")
            self.status_var.set("No processing groups are configured.")
            self._load_group_cameras(None)
            return
        preferred_label = next(
            (
                label
                for label, group in self._groups_by_label.items()
                if self.initial_group_id is not None
                and int(self._value(group, "id")) == int(self.initial_group_id)
            ),
            None,
        )
        if preferred_label is not None:
            self.group_var.set(preferred_label)
        elif self.group_var.get() not in self._groups_by_label:
            self.group_var.set(labels[0])
        self._on_group_selected()

    def _on_group_selected(self, _event: object | None = None) -> None:
        group = self._groups_by_label.get(self.group_var.get())
        group_id = self._value(group, "id", default=None)
        self.selected_group_id = int(group_id) if group_id is not None else None
        self._load_group_cameras(self.selected_group_id)

    def select_camera_context(
        self,
        group_id: int,
        camera_id: int | None = None,
    ) -> None:
        """Select a processing group and, optionally, one assigned camera."""
        matching_label = next(
            (
                label
                for label, group in self._groups_by_label.items()
                if int(self._value(group, "id")) == int(group_id)
            ),
            None,
        )
        if matching_label is None:
            raise ValueError(f"Group id {group_id} is not available in ROI Editor.")

        self.group_var.set(matching_label)
        self._on_group_selected()

        if camera_id is None:
            return
        if int(camera_id) not in self._cameras_by_id:
            raise ValueError(
                f"Camera id {camera_id} is not assigned to group id {group_id}."
            )
        self.select_camera(int(camera_id))

    def _load_group_cameras(self, group_id: int | None) -> None:
        for widget in self.camera_list.winfo_children():
            widget.destroy()
        self._cameras_by_id.clear()
        self._camera_cards.clear()
        self._thumbnail_photos.clear()
        self._snapshot_frames.clear()
        self._snapshot_loading.clear()
        self.selected_camera_id = None
        self.selected_roi_id = None
        self.current_frame = None
        self.current_image = None
        self.current_points.clear()
        self.saved_rois.clear()
        self._render_main_frame()
        self._refresh_roi_list()

        if group_id is None:
            return

        cameras = list(self.camera_service.list_cameras(group_id))
        self._cameras_by_id = {
            int(self._value(camera, "id")): camera for camera in cameras
        }
        for camera_id, camera in self._cameras_by_id.items():
            self._create_camera_card(camera_id, camera)

        if cameras:
            camera_ids = [int(self._value(camera, "id")) for camera in cameras]
            selected_id = (
                int(self.initial_camera_id)
                if self.initial_camera_id is not None
                and int(self.initial_camera_id) in camera_ids
                else camera_ids[0]
            )
            self.select_camera(selected_id)
            self._capture_group_thumbnails(camera_ids)
            self.initial_group_id = None
            self.initial_camera_id = None
        else:
            self.camera_title_var.set("No cameras configured")
            self.status_var.set("The selected group does not contain cameras.")

    def _create_camera_card(self, camera_id: int, camera: object) -> None:
        card = tk.Frame(
            self.camera_list,
            bg=self.PANEL_ALT,
            padx=8,
            pady=8,
            highlightthickness=2,
            highlightbackground="#173F78",
            cursor="hand2",
        )
        card.pack(fill="x", pady=(0, 10))

        preview = tk.Canvas(
            card,
            bg=self.CANVAS_BACKGROUND,
            height=130,
            highlightthickness=0,
            cursor="hand2",
        )
        preview.pack(fill="x")
        preview.create_text(
            145,
            65,
            text="Preview unavailable",
            fill=self.MUTED,
            font=(ArnesisTheme.font_family, 8),
            tags="placeholder",
        )

        name = tk.Label(
            card,
            text=str(self._value(camera, "name", default=f"Camera {camera_id}")),
            bg=self.PANEL_ALT,
            fg=self.WHITE,
            font=(ArnesisTheme.font_family, 10, "bold"),
            anchor="w",
        )
        name.pack(fill="x", pady=(7, 0))

        status = tk.Label(
            card,
            text="Click to select",
            bg=self.PANEL_ALT,
            fg=self.MUTED,
            font=(ArnesisTheme.font_family, 8),
            anchor="w",
        )
        status.pack(fill="x", pady=(2, 0))

        for widget in (card, preview, name, status):
            widget.bind(
                "<Button-1>",
                lambda _event, selected_id=camera_id: self.select_camera(selected_id),
            )

        self._camera_cards[camera_id] = {
            "frame": card,
            "preview": preview,
            "status": status,
        }

    def select_camera(self, camera_id: int) -> None:
        if camera_id not in self._cameras_by_id:
            return
        self.selected_camera_id = camera_id
        self.selected_roi_id = None
        self.current_points = []

        for item_id, widgets in self._camera_cards.items():
            widgets["frame"].configure(
                highlightbackground=self.CYAN if item_id == camera_id else "#173F78"
            )
            widgets["status"].configure(
                text="Selected" if item_id == camera_id else "Click to select",
                fg=self.CYAN if item_id == camera_id else self.MUTED,
            )

        camera = self._cameras_by_id[camera_id]
        self.camera_title_var.set(
            str(self._value(camera, "name", default=f"Camera {camera_id}"))
        )
        self.saved_rois = list(self.roi_service.list_by_camera(camera_id))
        self._refresh_roi_list()
        cached = self._snapshot_frames.get(camera_id)
        if cached is not None:
            self._apply_snapshot(camera_id, cached, update_main=True)
        else:
            self._capture_snapshot_async(camera_id, update_main=True)

    def refresh_current_frame(self) -> None:
        """Capture one new static RTSP frame for the selected camera."""
        if self.selected_camera_id is None:
            self.status_var.set("Select a camera first.")
            return
        self._capture_snapshot_async(self.selected_camera_id, update_main=True)

    def _capture_group_thumbnails(self, camera_ids: list[int]) -> None:
        """Capture one static thumbnail per camera without a refresh loop."""
        for camera_id in camera_ids:
            if camera_id != self.selected_camera_id:
                self._capture_snapshot_async(camera_id, update_main=False)

    def _capture_snapshot_async(self, camera_id: int, *, update_main: bool) -> None:
        if camera_id in self._snapshot_loading:
            return
        self._snapshot_loading.add(camera_id)
        widgets = self._camera_cards.get(camera_id)
        if widgets is not None:
            widgets["status"].configure(
                text="Capturing static frame...",
                fg=self.WARNING,
            )
        if update_main:
            self.status_var.set("Capturing one static frame from the camera...")

        def worker() -> None:
            result = self.camera_service.capture_snapshot(camera_id)
            self.after(
                0,
                lambda: self._finish_snapshot(camera_id, result, update_main),
            )

        threading.Thread(
            target=worker,
            name=f"roi-snapshot-{camera_id}",
            daemon=True,
        ).start()

    def _finish_snapshot(self, camera_id: int, result: Any, update_main: bool) -> None:
        self._snapshot_loading.discard(camera_id)
        widgets = self._camera_cards.get(camera_id)
        if result.success and result.frame is not None:
            frame = result.frame.copy()
            self._snapshot_frames[camera_id] = frame
            self._apply_snapshot(
                camera_id,
                frame,
                update_main=update_main and camera_id == self.selected_camera_id,
            )
            if widgets is not None:
                widgets["status"].configure(
                    text=f"Static frame | {result.width} x {result.height}",
                    fg=self.CYAN if camera_id == self.selected_camera_id else self.MUTED,
                )
            if update_main and camera_id == self.selected_camera_id:
                self.status_var.set(
                    f"Static frame captured in {result.elapsed_ms} ms. "
                    "The temporary camera connection is closed."
                )
            return
        error = result.error or "The camera returned no valid frame."
        if widgets is not None:
            widgets["status"].configure(text="Snapshot unavailable", fg=self.DANGER)
        if update_main and camera_id == self.selected_camera_id:
            self.status_var.set(f"Unable to capture static frame: {error}")

    def _apply_snapshot(self, camera_id: int, frame: Any, *, update_main: bool) -> None:
        self._render_thumbnail(camera_id, frame)
        if not update_main:
            return
        self.current_frame = frame.copy()
        self.current_image = Image.fromarray(
            cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
        )
        height, width = self.current_frame.shape[:2]
        self.frame_info_var.set(
            f"Resolution: {width} x {height} | Static configuration frame"
        )
        self._render_main_frame()

    def _render_thumbnail(self, camera_id: int, frame: Any) -> None:
        widgets = self._camera_cards.get(camera_id)
        if widgets is None:
            return
        canvas: tk.Canvas = widgets["preview"]
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        width = max(260, canvas.winfo_width())
        height = max(120, canvas.winfo_height())
        image.thumbnail((width, height), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self._thumbnail_photos[camera_id] = photo
        canvas.delete("all")
        canvas.create_image(width // 2, height // 2, image=photo, anchor="center")

    def new_polygon(self) -> None:
        if self.selected_camera_id is None:
            self.status_var.set("Select a camera first.")
            return
        self.selected_roi_id = None
        self.current_points = []
        self.roi_name_var.set(self._next_roi_name())
        self.enabled_var.set(True)
        self.profile_var.set("None")
        self.roi_list.selection_clear(0, tk.END)
        self.status_var.set("New polygon started. Add at least three points.")
        self._render_main_frame()

    def undo_point(self) -> None:
        if self.current_points:
            self.current_points.pop()
            self.status_var.set(
                f"Last point removed. Current polygon has {len(self.current_points)} point(s)."
            )
            self._render_main_frame()

    def clear_polygon(self) -> None:
        self.current_points = []
        self.status_var.set("Current polygon cleared.")
        self._render_main_frame()

    def save_roi(self) -> None:
        if self.selected_camera_id is None:
            self.status_var.set("Select a camera first.")
            return
        try:
            result = self.roi_service.save_polygon(
                camera_id=self.selected_camera_id,
                name=self.roi_name_var.get(),
                points=self.current_points,
                roi_id=self.selected_roi_id,
                enabled=self.enabled_var.get(),
            )
            self.selected_roi_id = int(result["id"])
            if self.profile_service is not None:
                self.profile_service.assign_roi_profile(
                    self.selected_roi_id,
                    self._profile_ids_by_label[self.profile_var.get()],
                )
            self.saved_rois = list(
                self.roi_service.list_by_camera(self.selected_camera_id)
            )
            self._refresh_roi_list(select_roi_id=self.selected_roi_id)
            self.status_var.set(
                f"ROI '{result['name']}' saved with {len(self.current_points)} points."
            )
            self._render_main_frame()
        except Exception as exc:
            self.status_var.set(f"Unable to save ROI: {type(exc).__name__}: {exc}")

    def delete_roi(self) -> None:
        if self.selected_roi_id is None:
            self.status_var.set("Select a saved ROI to delete.")
            return
        try:
            deleted_id = self.selected_roi_id
            self.roi_service.delete(deleted_id)
            self.selected_roi_id = None
            self.current_points = []
            if self.selected_camera_id is not None:
                self.saved_rois = list(
                    self.roi_service.list_by_camera(self.selected_camera_id)
                )
            self._refresh_roi_list()
            self.status_var.set(f"ROI id {deleted_id} deleted.")
            self._render_main_frame()
        except Exception as exc:
            self.status_var.set(f"Unable to delete ROI: {type(exc).__name__}: {exc}")

    def _on_roi_selected(self, _event: object | None = None) -> None:
        selection = self.roi_list.curselection()
        if not selection:
            return
        roi = self.saved_rois[selection[0]]
        self.selected_roi_id = int(roi["id"])
        self.roi_name_var.set(str(roi["name"]))
        self.enabled_var.set(bool(roi.get("enabled", True)))
        profile_id = roi.get("processing_profile_id", roi.get("profile_id"))
        self.profile_var.set(self._profile_label(None if profile_id is None else int(profile_id)))
        self.current_points = [dict(point) for point in roi.get("points", [])]
        self.status_var.set(
            f"Editing ROI '{roi['name']}'. Drag points or add new points."
        )
        self._render_main_frame()

    def _refresh_roi_list(self, select_roi_id: int | None = None) -> None:
        self.roi_list.delete(0, tk.END)
        selected_index: int | None = None
        for index, roi in enumerate(self.saved_rois):
            status = "" if roi.get("enabled", True) else " [Disabled]"
            self.roi_list.insert(tk.END, f"{roi['name']}{status}")
            if select_roi_id is not None and int(roi["id"]) == select_roi_id:
                selected_index = index
        if selected_index is not None:
            self.roi_list.selection_set(selected_index)
            self.roi_list.see(selected_index)

    def _on_canvas_press(self, event: tk.Event) -> None:
        normalized = self._canvas_to_normalized(event.x, event.y)
        if normalized is None:
            return
        nearest = self._nearest_point_index(event.x, event.y)
        if nearest is not None:
            self._drag_point_index = nearest
            return
        self.current_points.append(normalized)
        self.status_var.set(
            f"Point {len(self.current_points)} added to the current polygon."
        )
        self._render_main_frame()

    def _on_canvas_drag(self, event: tk.Event) -> None:
        if self._drag_point_index is None:
            return
        normalized = self._canvas_to_normalized(event.x, event.y)
        if normalized is None:
            return
        self.current_points[self._drag_point_index] = normalized
        self._render_main_frame()

    def _on_canvas_release(self, _event: tk.Event) -> None:
        self._drag_point_index = None

    def _render_main_frame(self) -> None:
        if not hasattr(self, "canvas"):
            return
        self.canvas.delete("all")
        if self.current_image is None:
            self.canvas.create_text(
                max(1, self.canvas.winfo_width()) // 2,
                max(1, self.canvas.winfo_height()) // 2,
                text="Select a camera and refresh the complete frame",
                fill=self.MUTED,
                font=(ArnesisTheme.font_family, 12),
            )
            return

        canvas_width = max(100, self.canvas.winfo_width())
        canvas_height = max(100, self.canvas.winfo_height())
        image_width, image_height = self.current_image.size
        scale = min(canvas_width / image_width, canvas_height / image_height)
        draw_width = max(1, int(image_width * scale))
        draw_height = max(1, int(image_height * scale))
        offset_x = (canvas_width - draw_width) // 2
        offset_y = (canvas_height - draw_height) // 2
        self._image_box = (offset_x, offset_y, draw_width, draw_height)

        resized = self.current_image.resize(
            (draw_width, draw_height),
            Image.Resampling.LANCZOS,
        )
        self._main_photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(
            offset_x,
            offset_y,
            anchor="nw",
            image=self._main_photo,
        )

        selected_id = self.selected_roi_id
        for roi in self.saved_rois:
            if int(roi["id"]) == selected_id:
                continue
            self._draw_polygon(
                roi.get("points", []),
                str(roi.get("color", self.CYAN)),
                str(roi["name"]),
                active=False,
            )
        self._draw_polygon(
            self.current_points,
            self.CYAN,
            self.roi_name_var.get().strip() or "Current ROI",
            active=True,
        )

    def _draw_polygon(
        self,
        points: list[dict[str, float]],
        color: str,
        name: str,
        *,
        active: bool,
    ) -> None:
        if not points:
            return
        coordinates: list[float] = []
        for point in points:
            x, y = self._normalized_to_canvas(point)
            coordinates.extend((x, y))
        if len(coordinates) >= 4:
            self.canvas.create_line(
                *coordinates,
                fill=color,
                width=3 if active else 2,
            )
        if len(coordinates) >= 6:
            self.canvas.create_line(
                coordinates[-2],
                coordinates[-1],
                coordinates[0],
                coordinates[1],
                fill=color,
                width=3 if active else 2,
            )
        first_x, first_y = coordinates[0], coordinates[1]
        self.canvas.create_text(
            first_x + 6,
            max(12, first_y - 8),
            text=name,
            fill=color,
            anchor="sw",
            font=(ArnesisTheme.font_family, 9, "bold"),
        )
        if active:
            for x, y in zip(coordinates[0::2], coordinates[1::2]):
                self.canvas.create_oval(
                    x - 6,
                    y - 6,
                    x + 6,
                    y + 6,
                    fill=self.WARNING,
                    outline="#FFFFFF",
                    width=1,
                )

    def _canvas_to_normalized(
        self,
        canvas_x: float,
        canvas_y: float,
    ) -> dict[str, float] | None:
        offset_x, offset_y, width, height = self._image_box
        if not (
            offset_x <= canvas_x <= offset_x + width
            and offset_y <= canvas_y <= offset_y + height
        ):
            return None
        return {
            "x": min(1.0, max(0.0, (canvas_x - offset_x) / width)),
            "y": min(1.0, max(0.0, (canvas_y - offset_y) / height)),
        }

    def _normalized_to_canvas(
        self,
        point: dict[str, float],
    ) -> tuple[float, float]:
        offset_x, offset_y, width, height = self._image_box
        return (
            offset_x + float(point["x"]) * width,
            offset_y + float(point["y"]) * height,
        )

    def _nearest_point_index(self, canvas_x: float, canvas_y: float) -> int | None:
        maximum_distance_squared = 12 * 12
        best_index: int | None = None
        best_distance = maximum_distance_squared
        for index, point in enumerate(self.current_points):
            point_x, point_y = self._normalized_to_canvas(point)
            distance = (canvas_x - point_x) ** 2 + (canvas_y - point_y) ** 2
            if distance <= best_distance:
                best_distance = distance
                best_index = index
        return best_index

    def _next_roi_name(self) -> str:
        existing = {str(roi["name"]).casefold() for roi in self.saved_rois}
        index = 1
        while f"roi {index}".casefold() in existing:
            index += 1
        return f"ROI {index}"

    @staticmethod
    def _button(
        parent: tk.Misc,
        text: str,
        command: Callable[[], None],
        background: str,
    ) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=background,
            fg="#FFFFFF",
            activebackground="#0E6AC7",
            activeforeground="#FFFFFF",
            relief="flat",
            borderwidth=0,
            padx=12,
            pady=8,
            cursor="hand2",
            font=(ArnesisTheme.font_family, 9, "bold"),
        )

    @staticmethod
    def _value(item: object, *names: str, default: object = None) -> object:
        for name in names:
            if isinstance(item, dict) and name in item:
                return item[name]
            if item is not None and hasattr(item, name):
                return getattr(item, name)
        return default
