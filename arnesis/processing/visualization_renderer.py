"""Fail-closed privacy and ROI overlay renderer for Arnesis RT previews."""
from __future__ import annotations

import cv2
import numpy as np


class InferenceVisualizationRenderer:
    @staticmethod
    def render(frame: np.ndarray, result) -> np.ndarray:
        output = frame.copy()
        if result is None:
            # Privacy must fail closed while CUDA models are loading.
            return cv2.GaussianBlur(output, (51, 51), 0)

        InferenceVisualizationRenderer._blur_heads(output, result)
        station_by_roi = {station.roi_id: station for station in result.stations}
        for roi in result.rois:
            points = np.asarray(roi.points, dtype=np.int32).reshape((-1, 1, 2))
            color = InferenceVisualizationRenderer._hex_to_bgr(roi.color_hex)
            cv2.polylines(output, [points], True, color, 2, cv2.LINE_AA)
            station = station_by_roi.get(roi.roi_id)
            label = roi.station
            if station is not None:
                label = (f"{roi.station} | P:{station.people_count} H:{station.head_count} "
                         f"VA:{station.va_count} NVA:{station.nva_count} "
                         f"NEU:{station.neutral_count}")
            anchor = roi.points[0]
            cv2.putText(output, label, (anchor[0], max(18, anchor[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA)

        classification_by_track = {item.track_id: item for item in result.classifications}
        for detection in result.detections:
            color = (41, 230, 255)
            first = (int(detection.x1), int(detection.y1))
            second = (int(detection.x2), int(detection.y2))
            cv2.rectangle(output, first, second, color, 2, cv2.LINE_AA)
            label = f"ID {detection.track_id} | {detection.station}"
            classification = classification_by_track.get(detection.track_id)
            if classification is not None:
                label += f" | {classification.class_name} {classification.confidence:.2f}"
            cv2.putText(output, label, (first[0], max(18, first[1] - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2, cv2.LINE_AA)

        footer = (f"{result.cuda_device} | {result.processing_fps:.1f} FPS | "
                  f"Person {result.person_inference_ms:.1f} ms | "
                  f"Head {result.head_inference_ms:.1f} ms | "
                  f"Class {result.classification_inference_ms:.1f} ms")
        cv2.putText(output, footer, (12, max(24, output.shape[0] - 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (32, 201, 151), 2, cv2.LINE_AA)
        return output

    @staticmethod
    def _blur_heads(frame: np.ndarray, result) -> None:
        height, width = frame.shape[:2]
        for head in result.privacy_heads:
            box_width = head.x2 - head.x1
            box_height = head.y2 - head.y1
            expansion = result.privacy_box_expansion
            x1 = max(0, int(head.x1 - box_width * expansion))
            y1 = max(0, int(head.y1 - box_height * expansion))
            x2 = min(width, int(head.x2 + box_width * expansion))
            y2 = min(height, int(head.y2 + box_height * expansion))
            region_width = x2 - x1
            region_height = y2 - y1
            if region_width < result.privacy_minimum_region or region_height < result.privacy_minimum_region:
                continue
            kernel = min(result.privacy_blur_kernel, region_width, region_height)
            if kernel % 2 == 0:
                kernel -= 1
            if kernel >= 3:
                frame[y1:y2, x1:x2] = cv2.GaussianBlur(
                    frame[y1:y2, x1:x2], (kernel, kernel), 0)

    @staticmethod
    def _hex_to_bgr(value: str) -> tuple[int, int, int]:
        color = value.strip().lstrip("#")
        if len(color) != 6:
            return (255, 230, 41)
        return (int(color[4:6], 16), int(color[2:4], 16), int(color[0:2], 16))
