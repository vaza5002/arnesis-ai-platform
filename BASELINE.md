# Arnesis v5.0.0 Validated Baseline

## Status

Functionally validated on August 27, 2026.

## Runtime architecture

- One canonical RTSP session per camera.
- Bounded latest-frame processing.
- CUDA-only model inference.
- CPU post-processing for ROI assignment, tracking, privacy blur, and UI rendering.
- CPU inference fallback is prohibited.

## Validated AI pipeline

- Head Counter processes the full frame for privacy.
- Detected heads are blurred in real-time previews.
- Productive head counts are restricted to enabled ROIs.
- Person detections outside enabled ROIs are discarded.
- Performance Classification runs only on accepted person crops.
- Expected classification labels are VA, NVA, and Neutral.
- Enabled ROIs are displayed in Real-Time Processing.

## External artifacts excluded from Git

- AI model binaries
- Camera passwords and credentials
- Local databases
- Virtual environments
- Videos and captures
- Local configuration overrides

## Baseline tag

v5.0.0-validated
