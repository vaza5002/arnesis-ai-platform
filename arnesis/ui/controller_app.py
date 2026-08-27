"""Modern Arnesis desktop Controller with persistent navigation."""

from __future__ import annotations

import inspect

import tkinter as tk
from tkinter import ttk

from sqlalchemy import select

from arnesis.application.bootstrap import bootstrap_application
from arnesis.application.camera_management_service import CameraManagementService
from arnesis.application.group_management_service import GroupManagementService
from arnesis.application.processing_service import ProcessingService
from arnesis.application.processing_profile_service import ProcessingProfileService
from arnesis.application.roi_service import RoiService
from arnesis.domain.entities import GpuCapacity
from arnesis.ui.camera_manager_dialog import CameraManagerDialog
from arnesis.ui.group_manager_view import GroupManagerView
from arnesis.ui.realtime_processing_view import RealTimeProcessingView
from arnesis.ui.roi_editor_view import RoiEditorView
from arnesis.application.model_registry_service import ModelRegistryService
from arnesis.ui.model_registry_view import ModelRegistryView
from arnesis.ui.processing_profile_view import ProcessingProfileView
from arnesis.ui.theme import ArnesisTheme
from arnesis.ui.ux_messages import DialogService, MessageLevel, UserMessage


class ArnesisControllerApp(tk.Tk):
    """Main dark-mode shell for real-time Arnesis configuration and control."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Arnesis | Real-Time Controller")
        self.geometry("1500x900")
        self.minsize(1220, 740)
        self.configure(bg=ArnesisTheme.colors.background)
        self.protocol("WM_DELETE_WINDOW", self._close)

        self.context = bootstrap_application()
        self.processing = ProcessingService(self.context.database, self.context.gpu_capacity)
        self.group_service = GroupManagementService(self.context.database, self.processing)
        self.camera_service = CameraManagementService(self.context.database)
        self.model_service = ModelRegistryService(self.context.database)
        self.profile_service = ProcessingProfileService(self.context.database)
        self.roi_service = RoiService(self.context.database)
        self._camera_dialog: CameraManagerDialog | None = None
        self._current_view: tk.Widget | None = None
        self._roi_preview_subscriptions: set[tuple[int, int]] = set()
        self._nav_buttons: dict[str, tk.Button] = {}
        self.group_view: GroupManagerView | None = None

        ArnesisTheme.apply(self)
        self._configure_visible_input_styles()
        self._build_shell()
        self._navigate("Groups", self._show_groups)
        self.after(1000, self._refresh_status)

    def _configure_visible_input_styles(self) -> None:
        """Provide a global readable fallback for light input fields."""
        style = ttk.Style(self)
        for name in ("TEntry", "Arnesis.TEntry"):
            style.configure(name, fieldbackground="#FFFFFF", foreground="#07152F", insertcolor="#07152F")
            style.map(name, foreground=[("disabled", "#53657C"), ("readonly", "#07152F")],
                      fieldbackground=[("disabled", "#DCE3EC"), ("readonly", "#FFFFFF")])
        for name in ("TCombobox", "Arnesis.TCombobox"):
            style.configure(name, fieldbackground="#FFFFFF", foreground="#07152F", arrowcolor="#07152F")
            style.map(name, fieldbackground=[("readonly", "#FFFFFF"), ("focus", "#FFFFFF")],
                      foreground=[("readonly", "#07152F"), ("focus", "#07152F")],
                      selectbackground=[("readonly", "#FFFFFF")],
                      selectforeground=[("readonly", "#07152F")])

    def _build_shell(self) -> None:
        c = ArnesisTheme.colors
        self.sidebar = tk.Frame(self, bg=c.sidebar, width=250)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        brand = tk.Frame(self.sidebar, bg=c.sidebar, padx=22, pady=24)
        brand.pack(fill="x")
        tk.Label(
            brand,
            text="ARNESIS",
            bg=c.sidebar,
            fg=c.accent,
            font=(ArnesisTheme.font_family, 23, "bold"),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            brand,
            text="REAL-TIME CONTROL PLATFORM",
            bg=c.sidebar,
            fg=c.text_muted,
            font=(ArnesisTheme.font_family, 8, "bold"),
            anchor="w",
        ).pack(fill="x", pady=(3, 0))

        ttk.Separator(self.sidebar, orient="horizontal", style="Arnesis.TSeparator").pack(fill="x", padx=18)
        navigation = tk.Frame(self.sidebar, bg=c.sidebar, padx=10, pady=14)
        navigation.pack(fill="x")
        items = [
            ("Dashboard", lambda: self._show_placeholder("Dashboard", "Operational metrics and historical analysis will be integrated here.")),
            ("Real-Time Processing", self._show_realtime_processing),
            ("Groups", self._show_groups),
            ("Cameras", self._open_camera_manager),
            ("ROIs", self._show_rois),
            ("Models", self._show_models),
            ("Processing Profiles", self._show_processing_profiles),
            ("GPU Resources", lambda: self._show_placeholder("GPU Resources", "Configure memory, group and stream limits for each CUDA device.")),
            ("Logs", lambda: self._show_placeholder("Logs", "Review application, camera, database and CUDA events.")),
            ("Settings", lambda: self._show_placeholder("Settings", "Manage application defaults and advanced policies.")),
        ]
        for name, command in items:
            button = tk.Button(
                navigation,
                text=name,
                command=lambda item=name, action=command: self._navigate(item, action),
                bg=c.sidebar,
                fg=c.text_muted,
                activebackground=c.sidebar_active,
                activeforeground="#FFFFFF",
                relief="flat",
                borderwidth=0,
                anchor="w",
                font=(ArnesisTheme.font_family, 10, "bold"),
                padx=16,
                pady=11,
                cursor="hand2",
            )
            button.pack(fill="x", pady=2)
            self._nav_buttons[name] = button

        status = tk.Frame(self.sidebar, bg="#05142C", padx=16, pady=15)
        status.pack(side="bottom", fill="x")
        self.runtime_var = tk.StringVar(value="0 active groups")
        self.cuda_var = tk.StringVar(value="Validating CUDA devices...")
        tk.Label(
            status,
            text="SYSTEM STATUS",
            bg="#05142C",
            fg=c.text_muted,
            font=(ArnesisTheme.font_family, 8, "bold"),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            status,
            textvariable=self.runtime_var,
            bg="#05142C",
            fg=c.text,
            font=(ArnesisTheme.font_family, 9, "bold"),
            anchor="w",
            wraplength=210,
        ).pack(fill="x", pady=(8, 0))
        tk.Label(
            status,
            textvariable=self.cuda_var,
            bg="#05142C",
            fg=c.accent,
            font=(ArnesisTheme.font_family, 9),
            anchor="w",
            justify="left",
            wraplength=210,
        ).pack(fill="x", pady=(5, 0))

        self.main = tk.Frame(self, bg=c.background)
        self.main.pack(side="left", fill="both", expand=True)
        header = tk.Frame(self.main, bg=c.background, padx=26, pady=18)
        header.pack(fill="x")
        title_stack = tk.Frame(header, bg=c.background)
        title_stack.pack(side="left")
        self.page_title = tk.StringVar(value="Groups")
        self.page_subtitle = tk.StringVar(value="Configure dynamic real-time processing groups")
        tk.Label(
            title_stack,
            textvariable=self.page_title,
            bg=c.background,
            fg=c.text,
            font=(ArnesisTheme.font_family, 22, "bold"),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            title_stack,
            textvariable=self.page_subtitle,
            bg=c.background,
            fg=c.text_muted,
            font=(ArnesisTheme.font_family, 10),
            anchor="w",
        ).pack(fill="x", pady=(3, 0))

        self.header_status = tk.Label(
            header,
            text="CUDA required",
            bg=c.surface,
            fg=c.accent,
            font=(ArnesisTheme.font_family, 9, "bold"),
            padx=14,
            pady=8,
        )
        self.header_status.pack(side="right")

        self.content = tk.Frame(self.main, bg=c.background, padx=20, pady=0)
        self.content.pack(fill="both", expand=True, pady=(0, 18))

    def _navigate(self, name: str, command) -> None:
        subtitles = {
            "Dashboard": "Operational visibility and historical performance",
            "Real-Time Processing": "Control active groups and live camera pipelines",
            "Groups": "Configure dynamic real-time processing groups",
            "Cameras": "Manage secure RTSP endpoints and connection tests",
            "ROIs": "Define normalized regions and processing assignments",
            "Models": "Manage local CUDA model versions",
            "Processing Profiles": "Configure inference rules per ROI",
            "GPU Resources": "Monitor and limit CUDA device capacity",
            "Logs": "Review operational events and errors",
            "Settings": "Configure application-wide policies",
        }
        self.page_title.set(name)
        self.page_subtitle.set(subtitles.get(name, "Arnesis workspace"))
        c = ArnesisTheme.colors
        for key, button in self._nav_buttons.items():
            active = key == name
            button.configure(
                bg=c.sidebar_active if active else c.sidebar,
                fg="#FFFFFF" if active else c.text_muted,
            )
        command()

    def _clear_content(self) -> None:
        self._release_roi_preview_subscriptions()
        for child in self.content.winfo_children():
            child.destroy()
        self._current_view = None

    def _show_rois(
        self,
        camera_id: int | None = None,
        group_id: int | None = None,
    ) -> None:
        """Display the ROI editor using the constructor contract actually installed."""
        self._clear_content()

        view = self._create_roi_editor()
        view.pack(fill="both", expand=True)
        self._current_view = view

        if camera_id is not None:
            camera = self.camera_service.get_camera(int(camera_id))
            self._select_roi_editor_context(
                view=view,
                group_id=int(camera.group_id),
                camera_id=int(camera_id),
            )
        elif group_id is not None:
            self._select_roi_editor_context(
                view=view,
                group_id=int(group_id),
                camera_id=None,
            )

    def _create_roi_editor(self) -> RoiEditorView:
        """Create the installed RoiEditorView without assuming its revision."""
        parameters = inspect.signature(RoiEditorView.__init__).parameters
        modern_required = {"group_service", "camera_service", "roi_service"}

        if modern_required.issubset(parameters):
            arguments: dict[str, object] = {
                "parent": self.content,
                "group_service": self.group_service,
                "camera_service": self.camera_service,
                "roi_service": self.roi_service,
                "profile_service": self.profile_service,
            }

            # Runtime preview callbacks are optional. The current editor obtains
            # static configuration frames through CameraManagementService.
            if "frame_provider" in parameters:
                arguments["frame_provider"] = self._request_roi_frame
            elif "on_request_frame" in parameters:
                arguments["on_request_frame"] = self._request_roi_frame

            return RoiEditorView(**arguments)

        if "on_request_frame" in parameters:
            return RoiEditorView(
                self.content,
                self.roi_service,
                on_request_frame=self._request_roi_frame,
            )

        raise TypeError(
            "Unsupported RoiEditorView constructor: "
            f"{inspect.signature(RoiEditorView.__init__)}"
        )

    def _select_roi_editor_context(
        self,
        *,
        view: RoiEditorView,
        group_id: int,
        camera_id: int | None,
    ) -> None:
        """Select the requested group and optional camera in the ROI editor."""
        select_context = getattr(view, "select_camera_context", None)
        if callable(select_context):
            select_context(group_id, camera_id)
            return

        matching_label: str | None = None
        for label, group in view._groups_by_label.items():
            candidate_id = view._value(group, "id", default=None)
            if candidate_id is not None and int(candidate_id) == group_id:
                matching_label = label
                break

        if matching_label is None:
            raise ValueError(f"Group id {group_id} is not available in ROI Editor.")

        view.group_var.set(matching_label)
        view._on_group_selected()

        if camera_id is not None:
            if camera_id not in view._cameras_by_id:
                raise ValueError(
                    f"Camera id {camera_id} is not assigned to group id {group_id}."
                )
            view.select_camera(camera_id)

    def _open_camera_rois(self, camera_id: int) -> None:
        """Navigate from Camera Management to the selected camera ROI context."""
        self._camera_dialog = None
        self._navigate("ROIs", lambda: self._show_rois(camera_id=camera_id))

    def _request_roi_frame(self, group_id: int, camera_id: int):
        """Subscribe and return a non-destructive runtime preview packet."""
        key = (int(group_id), int(camera_id))
        if key not in self._roi_preview_subscriptions:
            self.processing.subscribe_preview(*key)
            self._roi_preview_subscriptions.add(key)
        return self.processing.preview_frame(*key)

    def _release_roi_preview_subscriptions(self) -> None:
        """Release ROI preview subscriptions when leaving the ROI workspace."""
        subscriptions = tuple(self._roi_preview_subscriptions)
        self._roi_preview_subscriptions.clear()
        for group_id, camera_id in subscriptions:
            try:
                self.processing.unsubscribe_preview(group_id, camera_id)
            except Exception:
                pass

    def _open_group_rois(self, group_id: int) -> None:
        """Open ROI configuration from the selected processing group."""
        self._navigate(
            "ROIs",
            lambda: self._show_rois(group_id=group_id),
        )

    def _show_processing_profiles(self) -> None:
        """Display processing profile configuration and model assignments."""
        self._clear_content()
        view = ProcessingProfileView(
            self.content,
            self.profile_service,
            self.model_service,
        )
        view.pack(fill="both", expand=True)
        self._current_view = view

    def _show_models(self) -> None:
        """Display the persistent CUDA model registry."""
        self._clear_content()
        view = ModelRegistryView(self.content, self.model_service)
        view.pack(fill="both", expand=True)
        self._current_view = view

    def _show_realtime_processing(self) -> None:
        """Display live group controls and camera monitoring cards."""
        self._clear_content()
        view = RealTimeProcessingView(
            self.content,
            self.group_service,
            self.processing,
            self.camera_service,
        )
        view.pack(fill="both", expand=True)
        self._current_view = view

    def _show_groups(self) -> None:
        self._clear_content()
        self.group_view = GroupManagerView(
            self.content,
            self.group_service,
            self._open_camera_manager,
            self._open_group_rois,
        )
        self.group_view.pack(fill="both", expand=True)
        self._current_view = self.group_view
        self.after_idle(self._apply_existing_view_styles)

    def _apply_existing_view_styles(self) -> None:
        """Improve existing group view widgets without changing its business logic."""
        if self.group_view is None or not self.group_view.winfo_exists():
            return
        c = ArnesisTheme.colors
        for widget in self._walk_widgets(self.group_view):
            if isinstance(widget, ttk.Treeview):
                widget.configure(style="Arnesis.Treeview")
                widget.tag_configure("even", background=c.surface)
                widget.tag_configure("odd", background=c.surface_alt)
            elif isinstance(widget, ttk.Entry):
                widget.configure(style="Arnesis.TEntry")
            elif isinstance(widget, ttk.Combobox):
                widget.configure(style="Arnesis.TCombobox")
            elif isinstance(widget, ttk.Checkbutton):
                widget.configure(style="Arnesis.TCheckbutton")

    def _show_placeholder(self, title: str, message: str) -> None:
        self._clear_content()
        c = ArnesisTheme.colors
        card = tk.Frame(
            self.content,
            bg=c.surface,
            padx=32,
            pady=32,
            highlightthickness=1,
            highlightbackground=c.border,
        )
        card.pack(fill="both", expand=True)
        tk.Label(
            card,
            text=title,
            bg=c.surface,
            fg=c.accent,
            font=(ArnesisTheme.font_family, 18, "bold"),
            anchor="w",
        ).pack(fill="x")
        tk.Label(
            card,
            text=message,
            bg=c.surface,
            fg=c.text,
            font=(ArnesisTheme.font_family, 11),
            anchor="w",
            justify="left",
            wraplength=820,
        ).pack(fill="x", pady=(12, 0))
        tk.Label(
            card,
            text="This workspace is prepared for a future implementation package.",
            bg=c.surface,
            fg=c.text_muted,
            font=(ArnesisTheme.font_family, 9),
            anchor="w",
        ).pack(fill="x", pady=(8, 0))

    def _open_camera_manager(self) -> None:
        if self._camera_dialog is not None and self._camera_dialog.winfo_exists():
            self._camera_dialog.lift()
            self._camera_dialog.focus_force()
            return
        if not self.group_service.list_groups():
            DialogService.show(
                self,
                UserMessage(
                    MessageLevel.WARNING,
                    "No groups available",
                    "Create and save a group before adding cameras.",
                ),
            )
            return
        self._camera_dialog = CameraManagerDialog(
            self,
            self.camera_service,
        )

    def _refresh_status(self) -> None:
        try:
            statuses = self.group_service.runtime_status()
            self.runtime_var.set(f"{len(statuses)} active group(s)")
            with self.context.database.session_scope() as session:
                capacities = session.scalars(
                    select(GpuCapacity).order_by(GpuCapacity.device_index)
                ).all()
            enabled = [
                f"CUDA:{gpu.device_index}  {gpu.device_name}"
                for gpu in capacities
                if gpu.enabled
            ]
            self.cuda_var.set("\n".join(enabled) if enabled else "No enabled CUDA devices")
            self.header_status.configure(
                text=f"{len(enabled)} CUDA device(s) online",
                fg=ArnesisTheme.colors.success if enabled else ArnesisTheme.colors.danger,
            )
        except Exception:
            self.runtime_var.set("Runtime status unavailable")
            self.cuda_var.set("CUDA status unavailable")
            self.header_status.configure(
                text="Status unavailable",
                fg=ArnesisTheme.colors.danger,
            )
        finally:
            if self.winfo_exists():
                self.after(1000, self._refresh_status)

    def _close(self) -> None:
        if (
            self.group_view is not None
            and self.group_view.winfo_exists()
            and self.group_view.has_unsaved_changes()
            and not DialogService.confirm(
                self,
                title="Unsaved changes",
                message="Close Arnesis and discard unsaved group changes?",
            )
        ):
            return
        active = self.group_service.runtime_status()
        if active and not DialogService.confirm(
            self,
            title="Stop active groups",
            message=f"Stop {len(active)} active group(s) and close Arnesis?",
            destructive=True,
        ):
            return
        try:
            if active:
                self.processing.stop_all()
        except Exception as exc:
            DialogService.show(
                self,
                UserMessage(
                    MessageLevel.ERROR,
                    "Shutdown incomplete",
                    "One or more groups could not be stopped safely.",
                    str(exc),
                ),
            )
            return
        self.context.close()
        self.destroy()

    @staticmethod
    def _walk_widgets(widget: tk.Misc):
        for child in widget.winfo_children():
            yield child
            yield from ArnesisControllerApp._walk_widgets(child)


def main() -> int:
    app = ArnesisControllerApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
