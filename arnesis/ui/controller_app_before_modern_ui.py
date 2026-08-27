"""Arnesis desktop Controller with persistent left navigation."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from arnesis.application.bootstrap import bootstrap_application
from arnesis.application.camera_management_service import CameraManagementService
from arnesis.application.group_management_service import GroupManagementService
from arnesis.application.processing_service import ProcessingService
from arnesis.ui.camera_manager_dialog import CameraManagerDialog
from arnesis.ui.group_manager_view import GroupManagerView
from arnesis.ui.ux_messages import DialogService, MessageLevel, UserMessage


class ArnesisControllerApp(tk.Tk):
    """Main desktop shell. Navigation is left; work and results are right."""

    NAVY = "#012059"
    PANEL = "#081F4E"
    SIDEBAR = "#061A3A"
    CARD = "#102B5A"
    BLUE = "#018FFF"
    CYAN = "#29E6FF"
    GOLD = "#FFC000"
    WHITE = "#F4F8FF"
    MUTED = "#9DB3D5"

    def __init__(self) -> None:
        super().__init__()
        self.title("Arnesis Desktop Controller")
        self.geometry("1440x860")
        self.minsize(1180, 720)
        self.configure(bg=self.NAVY)
        self.protocol("WM_DELETE_WINDOW", self._close)

        self.context = bootstrap_application()
        self.processing = ProcessingService(self.context.database, self.context.gpu_capacity)
        self.group_service = GroupManagementService(self.context.database, self.processing)
        self.camera_service = CameraManagementService(self.context.database)
        self._camera_dialog: CameraManagerDialog | None = None
        self._current_view: tk.Widget | None = None
        self._nav_buttons: dict[str, tk.Button] = {}
        self.group_view: GroupManagerView | None = None

        self._configure_styles()
        self._build_shell()
        self._show_groups()
        self.after(1000, self._refresh_runtime_banner)

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("Treeview", background=self.CARD, fieldbackground=self.CARD,
                        foreground=self.WHITE, rowheight=28, borderwidth=0)
        style.configure("Treeview.Heading", background="#173A70", foreground=self.WHITE,
                        font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", self.BLUE)])
        style.configure("TEntry", fieldbackground="white", foreground="black", padding=5)
        style.configure("TCombobox", padding=5)
        style.configure("TCheckbutton", background=self.PANEL, foreground=self.WHITE)

    def _build_shell(self) -> None:
        self.sidebar = tk.Frame(self, bg=self.SIDEBAR, width=238)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        brand = tk.Frame(self.sidebar, bg=self.SIDEBAR, padx=20, pady=22)
        brand.pack(fill="x")
        tk.Label(brand, text="ARNESIS", bg=self.SIDEBAR, fg=self.CYAN,
                 font=("Segoe UI", 22, "bold"), anchor="w").pack(fill="x")
        tk.Label(brand, text="Desktop Controller", bg=self.SIDEBAR, fg=self.WHITE,
                 font=("Segoe UI", 10), anchor="w").pack(fill="x")

        navigation = tk.Frame(self.sidebar, bg=self.SIDEBAR, padx=10)
        navigation.pack(fill="x", pady=(8, 0))
        items = [
            ("Dashboard", lambda: self._show_placeholder("Dashboard", "Operational and historical indicators will be integrated in the dashboard phase.")),
            ("Real-Time Processing", lambda: self._show_placeholder("Real-Time Processing", "Select and control dynamic groups from the Groups workspace during this phase.")),
            ("Groups", self._show_groups),
            ("Cameras", self._open_camera_manager),
            ("ROIs", lambda: self._show_placeholder("ROIs", "The visual ROI editor will use camera IDs and normalized coordinates.")),
            ("Models", lambda: self._show_placeholder("Models", "CUDA model catalog and version management are prepared for the inference phase.")),
            ("Processing Profiles", lambda: self._show_placeholder("Processing Profiles", "Reusable ROI processing profiles will be configured here.")),
            ("GPU Resources", lambda: self._show_placeholder("GPU Resources", "CUDA capacity is managed by the existing GPU capacity service.")),
            ("Logs", lambda: self._show_placeholder("Logs", "Application, database, camera and CUDA logs will be presented here.")),
            ("Settings", lambda: self._show_placeholder("Settings", "Application defaults and advanced camera policies will be configured here.")),
        ]
        for name, command in items:
            button = tk.Button(navigation, text=name, command=lambda n=name, c=command: self._navigate(n, c),
                               bg=self.SIDEBAR, fg=self.WHITE, activebackground=self.BLUE,
                               activeforeground="white", relief="flat", anchor="w",
                               font=("Segoe UI", 10), padx=14, pady=10, cursor="hand2")
            button.pack(fill="x", pady=2)
            self._nav_buttons[name] = button

        status = tk.Frame(self.sidebar, bg="#04132C", padx=16, pady=14)
        status.pack(side="bottom", fill="x")
        self.runtime_var = tk.StringVar(value="Runtime: 0 active groups")
        self.cuda_var = tk.StringVar(value="CUDA devices: validating...")
        tk.Label(status, textvariable=self.runtime_var, bg="#04132C", fg=self.WHITE,
                 font=("Segoe UI", 9), anchor="w", wraplength=198).pack(fill="x")
        tk.Label(status, textvariable=self.cuda_var, bg="#04132C", fg=self.CYAN,
                 font=("Segoe UI", 9), anchor="w", wraplength=198).pack(fill="x", pady=(6, 0))

        self.main = tk.Frame(self, bg=self.NAVY)
        self.main.pack(side="left", fill="both", expand=True)
        header = tk.Frame(self.main, bg=self.NAVY, padx=24, pady=16)
        header.pack(fill="x")
        self.page_title = tk.StringVar(value="Groups")
        tk.Label(header, textvariable=self.page_title, bg=self.NAVY, fg=self.WHITE,
                 font=("Segoe UI", 20, "bold"), anchor="w").pack(side="left")
        tk.Label(header, text="Actions and configuration on the left | Results and views on the right",
                 bg=self.NAVY, fg=self.MUTED, font=("Segoe UI", 9)).pack(side="right")
        self.content = tk.Frame(self.main, bg=self.NAVY, padx=18, pady=8)
        self.content.pack(fill="both", expand=True)

    def _navigate(self, name: str, command) -> None:
        self.page_title.set(name)
        for key, button in self._nav_buttons.items():
            button.configure(bg=self.BLUE if key == name else self.SIDEBAR)
        command()

    def _clear_content(self) -> None:
        for child in self.content.winfo_children():
            child.destroy()
        self._current_view = None

    def _show_groups(self) -> None:
        self.page_title.set("Groups")
        for key, button in self._nav_buttons.items():
            button.configure(bg=self.BLUE if key == "Groups" else self.SIDEBAR)
        self._clear_content()
        self.group_view = GroupManagerView(self.content, self.group_service, self._open_camera_manager)
        self.group_view.pack(fill="both", expand=True)
        self._current_view = self.group_view

    def _show_placeholder(self, title: str, message: str) -> None:
        self._clear_content()
        card = tk.Frame(self.content, bg=self.PANEL, padx=28, pady=28)
        card.pack(fill="both", expand=True)
        tk.Label(card, text=title, bg=self.PANEL, fg=self.CYAN,
                 font=("Segoe UI", 18, "bold"), anchor="w").pack(fill="x")
        tk.Label(card, text=message, bg=self.PANEL, fg=self.WHITE,
                 font=("Segoe UI", 11), anchor="w", justify="left", wraplength=760).pack(fill="x", pady=(12, 0))

    def _open_camera_manager(self) -> None:
        self.page_title.set("Cameras")
        for key, button in self._nav_buttons.items():
            button.configure(bg=self.BLUE if key == "Cameras" else self.SIDEBAR)
        if self._camera_dialog is not None and self._camera_dialog.winfo_exists():
            self._camera_dialog.lift()
            self._camera_dialog.focus_force()
            return
        if not self.group_service.list_groups():
            DialogService.show(self, UserMessage(MessageLevel.WARNING, "No groups available",
                                                  "Create and save a group before adding cameras."))
            return
        self._camera_dialog = CameraManagerDialog(self, self.camera_service)

    def _refresh_runtime_banner(self) -> None:
        try:
            statuses = self.group_service.runtime_status()
            devices = sorted({str(item["cuda_device"]) for item in statuses})
            suffix = " | " + ", ".join(devices) if devices else ""
            self.runtime_var.set(f"Runtime: {len(statuses)} active group(s){suffix}")
            capacities = self.context.gpu_capacity.list_capacities()
            enabled = [f"cuda:{c.device_index}" for c in capacities if c.enabled]
            self.cuda_var.set("CUDA: " + (", ".join(enabled) if enabled else "No enabled devices"))
        except Exception:
            self.runtime_var.set("Runtime status unavailable")
            self.cuda_var.set("CUDA status unavailable")
        finally:
            if self.winfo_exists():
                self.after(1000, self._refresh_runtime_banner)

    def _close(self) -> None:
        if self.group_view is not None and self.group_view.winfo_exists() and self.group_view.has_unsaved_changes():
            if not DialogService.confirm(self, title="Unsaved changes",
                                         message="Close Arnesis and discard unsaved group changes?"):
                return
        active = self.group_service.runtime_status()
        if active and not DialogService.confirm(self, title="Stop active groups",
                                                message=f"Stop {len(active)} active group(s) and close Arnesis?",
                                                destructive=True):
            return
        try:
            if active:
                self.processing.stop_all()
        except Exception as exc:
            DialogService.show(self, UserMessage(MessageLevel.ERROR, "Shutdown incomplete",
                                                  "One or more groups could not be stopped safely.", str(exc)))
            return
        self.context.close()
        self.destroy()


def main() -> int:
    app = ArnesisControllerApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
