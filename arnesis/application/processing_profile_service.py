"""Application service for persistent Arnesis processing profiles."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select

from arnesis.domain.entities import ModelDefinition, ProcessingProfile, Roi


@dataclass(frozen=True, slots=True)
class ProcessingProfileRecord:
    id: int
    name: str
    detector_model_id: int | None
    classifier_model_id: int | None
    pose_model_id: int | None
    confidence_threshold: float
    iou_threshold: float
    frame_skip: int
    debounce_ms: int
    minimum_event_duration_ms: int
    target_classes: tuple[str, ...]
    custom_parameters: dict[str, Any]
    enabled: bool
    roi_count: int


@dataclass(frozen=True, slots=True)
class ProfileOption:
    id: int
    label: str


class ProcessingProfileService:
    """Create, update, delete, and assign dynamic processing profiles."""

    def __init__(self, database: Any) -> None:
        self.database = database

    def list_profiles(self) -> list[ProcessingProfileRecord]:
        with self.database.session_scope() as session:
            profiles = session.scalars(
                select(ProcessingProfile).order_by(ProcessingProfile.name)
            ).all()
            counts = dict(
                session.execute(
                    select(Roi.processing_profile_id, func.count(Roi.id))
                    .where(Roi.processing_profile_id.is_not(None))
                    .group_by(Roi.processing_profile_id)
                ).all()
            )
            return [
                self._record(profile, int(counts.get(profile.id, 0)))
                for profile in profiles
            ]

    def list_options(self, enabled_only: bool = True) -> list[ProfileOption]:
        statement = select(ProcessingProfile).order_by(ProcessingProfile.name)
        if enabled_only:
            statement = statement.where(ProcessingProfile.enabled.is_(True))
        with self.database.session_scope() as session:
            profiles = session.scalars(statement).all()
            return [ProfileOption(item.id, item.name) for item in profiles]

    def get_profile(self, profile_id: int) -> ProcessingProfileRecord:
        with self.database.session_scope() as session:
            profile = session.get(ProcessingProfile, int(profile_id))
            if profile is None:
                raise ValueError(f"Processing profile id {profile_id} was not found.")
            roi_count = session.scalar(
                select(func.count(Roi.id)).where(Roi.processing_profile_id == profile.id)
            ) or 0
            return self._record(profile, int(roi_count))

    def save_profile(
        self,
        *,
        profile_id: int | None,
        name: str,
        detector_model_id: int | None,
        classifier_model_id: int | None,
        pose_model_id: int | None,
        confidence_threshold: float,
        iou_threshold: float,
        frame_skip: int,
        debounce_ms: int,
        minimum_event_duration_ms: int,
        target_classes: list[str],
        custom_parameters: dict[str, Any],
        enabled: bool = True,
    ) -> ProcessingProfileRecord:
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Profile name is required.")
        confidence = float(confidence_threshold)
        iou = float(iou_threshold)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Confidence threshold must be between 0 and 1.")
        if not 0.0 <= iou <= 1.0:
            raise ValueError("IoU threshold must be between 0 and 1.")
        if int(frame_skip) < 0 or int(debounce_ms) < 0 or int(minimum_event_duration_ms) < 0:
            raise ValueError("Frame skip and event timing values cannot be negative.")

        with self.database.session_scope() as session:
            duplicate = session.scalar(
                select(ProcessingProfile.id).where(
                    func.lower(ProcessingProfile.name) == clean_name.lower(),
                    ProcessingProfile.id != (int(profile_id) if profile_id is not None else -1),
                )
            )
            if duplicate is not None:
                raise ValueError(f"A processing profile named '{clean_name}' already exists.")

            model_ids = [detector_model_id, classifier_model_id, pose_model_id]
            for model_id in (value for value in model_ids if value is not None):
                model = session.get(ModelDefinition, int(model_id))
                if model is None or not model.enabled:
                    raise ValueError(f"Model id {model_id} is missing or disabled.")

            if profile_id is None:
                profile = ProcessingProfile(name=clean_name)
                session.add(profile)
            else:
                profile = session.get(ProcessingProfile, int(profile_id))
                if profile is None:
                    raise ValueError(f"Processing profile id {profile_id} was not found.")

            profile.name = clean_name
            profile.detector_model_id = self._optional_int(detector_model_id)
            profile.classifier_model_id = self._optional_int(classifier_model_id)
            profile.pose_model_id = self._optional_int(pose_model_id)
            profile.confidence_threshold = confidence
            profile.iou_threshold = iou
            profile.frame_skip = int(frame_skip)
            profile.debounce_ms = int(debounce_ms)
            profile.minimum_event_duration_ms = int(minimum_event_duration_ms)
            profile.target_classes = self._clean_classes(target_classes)
            profile.custom_parameters = dict(custom_parameters)
            profile.enabled = bool(enabled)
            session.flush()
            saved_id = int(profile.id)

        return self.get_profile(saved_id)

    def delete_profile(self, profile_id: int) -> None:
        with self.database.session_scope() as session:
            profile = session.get(ProcessingProfile, int(profile_id))
            if profile is None:
                raise ValueError(f"Processing profile id {profile_id} was not found.")
            assigned = session.scalar(
                select(func.count(Roi.id)).where(Roi.processing_profile_id == profile.id)
            ) or 0
            if assigned:
                raise ValueError(
                    f"Profile '{profile.name}' is assigned to {assigned} ROI(s). "
                    "Reassign those ROIs before deleting it."
                )
            session.delete(profile)

    def assign_roi_profile(self, roi_id: int, profile_id: int | None) -> None:
        with self.database.session_scope() as session:
            roi = session.get(Roi, int(roi_id))
            if roi is None:
                raise ValueError(f"ROI id {roi_id} was not found.")
            if profile_id is not None:
                profile = session.get(ProcessingProfile, int(profile_id))
                if profile is None or not profile.enabled:
                    raise ValueError("Select an enabled processing profile.")
            roi.processing_profile_id = self._optional_int(profile_id)

    @staticmethod
    def _record(profile: ProcessingProfile, roi_count: int) -> ProcessingProfileRecord:
        return ProcessingProfileRecord(
            id=int(profile.id), name=profile.name,
            detector_model_id=profile.detector_model_id,
            classifier_model_id=profile.classifier_model_id,
            pose_model_id=profile.pose_model_id,
            confidence_threshold=float(profile.confidence_threshold),
            iou_threshold=float(profile.iou_threshold), frame_skip=int(profile.frame_skip),
            debounce_ms=int(profile.debounce_ms),
            minimum_event_duration_ms=int(profile.minimum_event_duration_ms),
            target_classes=tuple(profile.target_classes or []),
            custom_parameters=dict(profile.custom_parameters or {}),
            enabled=bool(profile.enabled), roi_count=roi_count,
        )

    @staticmethod
    def _clean_classes(values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            clean = value.strip()
            if clean and clean not in result:
                result.append(clean)
        return result

    @staticmethod
    def _optional_int(value: int | None) -> int | None:
        return None if value is None else int(value)
