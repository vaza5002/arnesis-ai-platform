"""Processing Profile management workspace for Arnesis."""
from __future__ import annotations

import json
import tkinter as tk
from tkinter import ttk
from typing import Any

from arnesis.application.processing_profile_service import ProcessingProfileService
from arnesis.ui.theme import ArnesisTheme
from arnesis.ui.ux_messages import DialogService, MessageLevel, UserMessage


class ProcessingProfileView(tk.Frame):
    """CRUD editor for dynamic detection, classification, and pose profiles."""

    def __init__(self, parent: tk.Misc, service: ProcessingProfileService, model_service: Any) -> None:
        super().__init__(parent, bg=ArnesisTheme.colors.background)
        self.service = service
        self.model_service = model_service
        self.selected_profile_id: int | None = None
        self.model_maps: dict[str, dict[str, int | None]] = {}
        self._create_variables()
        self._build_ui()
        self._load_models()
        self.refresh()

    def _create_variables(self) -> None:
        self.name_var = tk.StringVar()
        self.detector_var = tk.StringVar(value="None")
        self.classifier_var = tk.StringVar(value="None")
        self.pose_var = tk.StringVar(value="None")
        self.confidence_var = tk.StringVar(value="0.50")
        self.iou_var = tk.StringVar(value="0.45")
        self.frame_skip_var = tk.StringVar(value="0")
        self.debounce_var = tk.StringVar(value="0")
        self.duration_var = tk.StringVar(value="0")
        self.classes_var = tk.StringVar(value="person")
        self.enabled_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Ready")

    def _build_ui(self) -> None:
        c = ArnesisTheme.colors
        toolbar = tk.Frame(self, bg=c.background, padx=8, pady=10)
        toolbar.pack(fill="x")
        tk.Label(toolbar, text="PROCESSING PROFILES", bg=c.background, fg=c.accent,
                 font=(ArnesisTheme.font_family, 16, "bold")).pack(side="left")
        for text, command in (("New", self.new_profile), ("Refresh", self.refresh)):
            tk.Button(toolbar, text=text, command=command, bg="#018FFF", fg="white",
                      relief="flat", padx=14, pady=7).pack(side="right", padx=(8, 0))

        body = tk.PanedWindow(self, orient="horizontal", bg=c.background, sashwidth=6)
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        listing = tk.Frame(body, bg=c.surface, padx=12, pady=12)
        editor = tk.Frame(body, bg=c.surface, padx=18, pady=14)
        body.add(listing, minsize=520)
        body.add(editor, minsize=600)

        columns = ("name", "state", "detector", "rois")
        self.tree = ttk.Treeview(listing, columns=columns, show="headings", style="Arnesis.Treeview")
        for name, title, width in (("name", "Profile", 190), ("state", "State", 80),
                                   ("detector", "Detector ID", 100), ("rois", "ROIs", 60)):
            self.tree.heading(name, text=title)
            self.tree.column(name, width=width, anchor="w")
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self._on_selection)

        fields = [
            ("Profile name *", self.name_var),
            ("Confidence threshold *", self.confidence_var),
            ("IoU threshold *", self.iou_var),
            ("Frame skip *", self.frame_skip_var),
            ("Debounce (ms) *", self.debounce_var),
            ("Minimum event duration (ms) *", self.duration_var),
            ("Target classes (comma separated)", self.classes_var),
        ]
        row = 0
        for label, variable in fields:
            tk.Label(editor, text=label, bg=c.surface, fg=c.text).grid(row=row, column=0, sticky="w", pady=5, padx=(0, 12))
            ttk.Entry(editor, textvariable=variable, style="Arnesis.TEntry").grid(row=row, column=1, sticky="ew", pady=5)
            row += 1

        for label, variable, key in (("Detector model", self.detector_var, "DETECTION"),
                                     ("Classifier model", self.classifier_var, "CLASSIFICATION"),
                                     ("Pose model", self.pose_var, "POSE")):
            tk.Label(editor, text=label, bg=c.surface, fg=c.text).grid(row=row, column=0, sticky="w", pady=5, padx=(0, 12))
            combo = ttk.Combobox(editor, textvariable=variable, state="readonly", style="Arnesis.TCombobox")
            combo.grid(row=row, column=1, sticky="ew", pady=5)
            setattr(self, f"{key.lower()}_combo", combo)
            row += 1

        tk.Label(editor, text="Custom parameters (JSON object)", bg=c.surface, fg=c.text).grid(row=row, column=0, sticky="nw", pady=5, padx=(0, 12))
        self.parameters_text = tk.Text(editor, height=7, bg="white", fg="#07152F", insertbackground="#07152F", wrap="word")
        self.parameters_text.grid(row=row, column=1, sticky="nsew", pady=5)
        row += 1
        ttk.Checkbutton(editor, text="Profile enabled", variable=self.enabled_var,
                        style="Arnesis.TCheckbutton").grid(row=row, column=1, sticky="w", pady=8)
        row += 1
        actions = tk.Frame(editor, bg=c.surface)
        actions.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        tk.Button(actions, text="Save", command=self.save_profile, bg="#018FFF", fg="white",
                  relief="flat", padx=16, pady=8).pack(side="left")
        tk.Button(actions, text="Delete", command=self.delete_profile, bg="#D94E5D", fg="white",
                  relief="flat", padx=16, pady=8).pack(side="left", padx=(8, 0))
        editor.columnconfigure(1, weight=1)
        editor.rowconfigure(row - 2, weight=1)

        tk.Label(self, textvariable=self.status_var, bg="#05142C", fg=c.text,
                 anchor="w", padx=16, pady=8).pack(fill="x")

    def _load_models(self) -> None:
        maps = {"DETECTION": {"None": None}, "CLASSIFICATION": {"None": None}, "POSE": {"None": None}}
        for model in self.model_service.list_models():
            kind = str(model.model_type).upper()
            aliases = {"DETECTOR": "DETECTION", "CLASSIFIER": "CLASSIFICATION"}
            kind = aliases.get(kind, kind)
            if kind in maps and model.enabled:
                maps[kind][f"{model.name} | {model.version} | ID {model.id}"] = int(model.id)
        self.model_maps = maps
        self.detection_combo.configure(values=tuple(maps["DETECTION"]))
        self.classification_combo.configure(values=tuple(maps["CLASSIFICATION"]))
        self.pose_combo.configure(values=tuple(maps["POSE"]))

    def refresh(self) -> None:
        for item in self.tree.get_children(): self.tree.delete(item)
        profiles = self.service.list_profiles()
        for profile in profiles:
            self.tree.insert("", "end", iid=str(profile.id), values=(profile.name,
                "Enabled" if profile.enabled else "Disabled",
                profile.detector_model_id or "None", profile.roi_count))
        self.status_var.set(f"{len(profiles)} processing profile(s) loaded.")

    def new_profile(self) -> None:
        self.selected_profile_id = None
        self.name_var.set(""); self.detector_var.set("None"); self.classifier_var.set("None"); self.pose_var.set("None")
        self.confidence_var.set("0.50"); self.iou_var.set("0.45"); self.frame_skip_var.set("0")
        self.debounce_var.set("0"); self.duration_var.set("0"); self.classes_var.set("person"); self.enabled_var.set(True)
        self.parameters_text.delete("1.0", tk.END); self.parameters_text.insert("1.0", "{}")
        self.tree.selection_remove(self.tree.selection())
        self.status_var.set("New profile. Required fields are marked with *.")

    def _on_selection(self, _event: object = None) -> None:
        selection = self.tree.selection()
        if not selection: return
        profile = self.service.get_profile(int(selection[0]))
        self.selected_profile_id = profile.id
        self.name_var.set(profile.name); self.confidence_var.set(str(profile.confidence_threshold)); self.iou_var.set(str(profile.iou_threshold))
        self.frame_skip_var.set(str(profile.frame_skip)); self.debounce_var.set(str(profile.debounce_ms)); self.duration_var.set(str(profile.minimum_event_duration_ms))
        self.classes_var.set(", ".join(profile.target_classes)); self.enabled_var.set(profile.enabled)
        self.detector_var.set(self._label_for("DETECTION", profile.detector_model_id)); self.classifier_var.set(self._label_for("CLASSIFICATION", profile.classifier_model_id)); self.pose_var.set(self._label_for("POSE", profile.pose_model_id))
        self.parameters_text.delete("1.0", tk.END); self.parameters_text.insert("1.0", json.dumps(profile.custom_parameters, indent=2, sort_keys=True))
        self.status_var.set(f"Selected {profile.name} | {profile.roi_count} assigned ROI(s).")

    def save_profile(self) -> None:
        try:
            raw = self.parameters_text.get("1.0", tk.END).strip() or "{}"
            parameters = json.loads(raw)
            if not isinstance(parameters, dict): raise ValueError("Custom parameters must be a JSON object.")
            profile = self.service.save_profile(profile_id=self.selected_profile_id, name=self.name_var.get(),
                detector_model_id=self.model_maps["DETECTION"][self.detector_var.get()],
                classifier_model_id=self.model_maps["CLASSIFICATION"][self.classifier_var.get()],
                pose_model_id=self.model_maps["POSE"][self.pose_var.get()],
                confidence_threshold=float(self.confidence_var.get()), iou_threshold=float(self.iou_var.get()),
                frame_skip=int(self.frame_skip_var.get()), debounce_ms=int(self.debounce_var.get()),
                minimum_event_duration_ms=int(self.duration_var.get()),
                target_classes=self.classes_var.get().split(","), custom_parameters=parameters,
                enabled=self.enabled_var.get())
            self.selected_profile_id = profile.id; self.refresh(); self.tree.selection_set(str(profile.id))
            DialogService.show(self, UserMessage(MessageLevel.SUCCESS, "Profile saved", f"Processing profile '{profile.name}' was saved."))
        except Exception as exc:
            DialogService.show(self, UserMessage(MessageLevel.ERROR, "Profile could not be saved", str(exc) or type(exc).__name__))

    def delete_profile(self) -> None:
        if self.selected_profile_id is None:
            DialogService.show(self, UserMessage(MessageLevel.WARNING, "No selection", "Select a profile to delete.")); return
        profile = self.service.get_profile(self.selected_profile_id)
        if not DialogService.confirm(self, title="Delete processing profile", message=f"Delete profile '{profile.name}'?", destructive=True): return
        try:
            self.service.delete_profile(profile.id); self.new_profile(); self.refresh()
        except Exception as exc:
            DialogService.show(self, UserMessage(MessageLevel.ERROR, "Profile could not be deleted", str(exc) or type(exc).__name__))

    def _label_for(self, kind: str, model_id: int | None) -> str:
        return next((label for label, value in self.model_maps[kind].items() if value == model_id), "None")
