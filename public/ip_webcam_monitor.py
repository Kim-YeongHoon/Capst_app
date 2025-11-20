"""
IP 웹캠 실시간 스트림을 YOLO로 모니터링하며 헬멧 미착용 이벤트를 기록합니다.

- Android IP Webcam 앱 등의 MJPEG/RTSP URL을 VIDEO_SOURCE로 지정합니다.
- YOLO 모델이 사람(person)과 헬멧(helmet) 클래스를 동시에 출력한다고 가정합니다.
- 헬멧 미착용이 설정 시간 이상 지속되면 이벤트 메타데이터와 클립을 저장하고
  public/events/events.json에 누적하여 웹에서 묶음으로 볼 수 있습니다.

의존성: opencv-python, ultralytics, supervision
"""

from __future__ import annotations

import json
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

# -----------------------------
# 사용자 설정 (필요시 환경변수로 대체)
# -----------------------------
VIDEO_SOURCE = os.getenv("IP_CAM_URL", "http://192.168.0.2:8080/video")  # Android IP Webcam 기본 MJPEG URL 예시
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "./models/helmet_yolo.pt")
PERSON_CLASS_ID = int(os.getenv("PERSON_CLASS_ID", "0"))
HELMET_CLASS_ID = int(os.getenv("HELMET_CLASS_ID", "1"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.35"))
NO_HELMET_SECONDS_THRESHOLD = float(os.getenv("NO_HELMET_SECONDS", "3.0"))
IOU_HELMET_THRESHOLD = float(os.getenv("IOU_HELMET", "0.1"))
SHOW_WINDOW = os.getenv("SHOW_WINDOW", "false").lower() == "true"

# 이벤트 및 클립 저장 위치
PUBLIC_EVENTS_DIR = Path("public/events")
CLIPS_DIR = PUBLIC_EVENTS_DIR / "clips"
EVENT_LOG_JSON = PUBLIC_EVENTS_DIR / "events.json"
EVENT_LOG_CSV = Path("runs/event_log.csv")

# 이벤트 클립 생성 시 몇 초 전/후 프레임을 포함할지 설정
PRE_EVENT_SECONDS = float(os.getenv("PRE_EVENT_SECONDS", "2.0"))
POST_EVENT_SECONDS = float(os.getenv("POST_EVENT_SECONDS", "2.0"))

# -----------------------------
# 유틸 및 로깅 클래스
# -----------------------------

def ensure_event_outputs() -> None:
    """이벤트/클립 디렉터리와 이벤트 JSON 기본 구조를 준비합니다."""
    PUBLIC_EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    EVENT_LOG_JSON.touch(exist_ok=True)
    if EVENT_LOG_JSON.read_text(encoding="utf-8").strip() == "":
        EVENT_LOG_JSON.write_text("[]", encoding="utf-8")
    EVENT_LOG_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not EVENT_LOG_CSV.exists():
        EVENT_LOG_CSV.write_text("person_id,start_iso,end_iso,duration_seconds,clip_path\n", encoding="utf-8")


def load_events() -> List[Dict]:
    ensure_event_outputs()
    try:
        return json.loads(EVENT_LOG_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []


def save_events(events: List[Dict]) -> None:
    EVENT_LOG_JSON.write_text(json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8")


@dataclass
class EventLogEntry:
    person_id: int
    start_time: datetime
    end_time: datetime
    clip_path: str
    duration_seconds: float

    def to_dict(self) -> Dict:
        return {
            "person_id": self.person_id,
            "start": self.start_time.isoformat(),
            "end": self.end_time.isoformat(),
            "duration_seconds": round(self.duration_seconds, 2),
            "clip": self.clip_path,
        }


class EventLogger:
    """JSON/CSV 기반 이벤트 로거."""

    def __init__(self) -> None:
        ensure_event_outputs()
        self.events = load_events()

    def append(self, entry: EventLogEntry) -> None:
        self.events.append(entry.to_dict())
        save_events(self.events)
        csv_line = f"{entry.person_id},{entry.start_time.isoformat()},{entry.end_time.isoformat()},{entry.duration_seconds:.2f},{entry.clip_path}\n"
        with EVENT_LOG_CSV.open("a", encoding="utf-8") as f:
            f.write(csv_line)
        print(f"[LOG] 이벤트 기록: person={entry.person_id} dur={entry.duration_seconds:.1f}s clip={entry.clip_path}")

    @staticmethod
    def build_clip_path(timestamp: datetime) -> Path:
        filename = f"event_{timestamp.strftime('%Y%m%d_%H%M%S_%f')}.mp4"
        return CLIPS_DIR / filename


class EventClipWriter:
    """이벤트 발생 시 전/후 프레임을 포함한 클립을 저장합니다."""

    def __init__(self) -> None:
        self.writer: Optional[cv2.VideoWriter] = None
        self.frames_remaining: int = 0
        self.active_clip_path: Optional[Path] = None

    def start_clip(self, fps: float, frame_size: Tuple[int, int], pre_frames: List[np.ndarray]) -> Path:
        self.close()
        clip_path = EventLogger.build_clip_path(datetime.now())
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(str(clip_path), fourcc, fps, frame_size)
        self.frames_remaining = int(POST_EVENT_SECONDS * fps)
        self.active_clip_path = clip_path
        for frame in pre_frames:
            self.writer.write(frame)
        print(f"[LOG] 클립 시작: {clip_path}")
        return clip_path.relative_to(PUBLIC_EVENTS_DIR)

    def write_frame(self, frame: np.ndarray) -> None:
        if self.writer is None:
            return
        self.writer.write(frame)
        self.frames_remaining -= 1
        if self.frames_remaining <= 0:
            self.close()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
        self.writer = None
        self.frames_remaining = 0
        self.active_clip_path = None


@dataclass
class PersonState:
    no_helmet_started: Optional[datetime] = None
    reported: bool = False
    last_seen: datetime = field(default_factory=datetime.now)


# -----------------------------
# 메인 로직
# -----------------------------

def compute_iou_matrix(person_boxes: np.ndarray, helmet_boxes: np.ndarray) -> np.ndarray:
    if len(person_boxes) == 0 or len(helmet_boxes) == 0:
        return np.zeros((len(person_boxes), len(helmet_boxes)), dtype=float)
    return sv.box_iou(person_boxes, helmet_boxes)


def run_monitoring() -> None:
    ensure_event_outputs()

    print("[INFO] YOLO 가중치 로드 중...", YOLO_WEIGHTS)
    model = YOLO(YOLO_WEIGHTS)
    person_tracker = sv.ByteTrack()
    helmet_tracker = sv.ByteTrack()
    logger = EventLogger()
    clip_writer = EventClipWriter()

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise SystemExit(f"스트림을 열 수 없습니다: {VIDEO_SOURCE}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)
    frame_size = (frame_width, frame_height)
    pre_frame_buffer: Deque[np.ndarray] = deque(maxlen=int(PRE_EVENT_SECONDS * fps))

    person_states: Dict[int, PersonState] = {}
    print("[INFO] 실시간 모니터링 시작. 종료하려면 창에서 ESC를 누르세요.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            pre_frame_buffer.append(frame.copy())
            now = datetime.now()

            results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(results)

            if detections.class_id is None:
                person_det = sv.Detections.empty()
                helmet_det = sv.Detections.empty()
            else:
                person_indices = np.where(detections.class_id == PERSON_CLASS_ID)[0]
                helmet_indices = np.where(detections.class_id == HELMET_CLASS_ID)[0]
                person_det = detections[person_indices]
                helmet_det = detections[helmet_indices]

            tracked_persons = person_tracker.update_with_detections(person_det)
            tracked_helmets = helmet_tracker.update_with_detections(helmet_det)

            person_boxes = tracked_persons.xyxy.astype(int) if len(tracked_persons) else np.empty((0, 4), dtype=int)
            helmet_boxes = tracked_helmets.xyxy.astype(int) if len(tracked_helmets) else np.empty((0, 4), dtype=int)

            iou_matrix = compute_iou_matrix(person_boxes, helmet_boxes)

            for idx in range(len(tracked_persons)):
                pid = int(tracked_persons.tracker_id[idx]) if tracked_persons.tracker_id is not None else idx
                bbox = person_boxes[idx]
                helmet_overlap = iou_matrix[idx] if iou_matrix.size else np.array([])
                has_helmet = bool(len(helmet_overlap) and np.max(helmet_overlap) >= IOU_HELMET_THRESHOLD)

                state = person_states.get(pid, PersonState())
                state.last_seen = now

                if has_helmet:
                    if state.no_helmet_started:
                        state.no_helmet_started = None
                        state.reported = False
                else:
                    if state.no_helmet_started is None:
                        state.no_helmet_started = now
                    duration = (now - state.no_helmet_started).total_seconds()
                    if duration >= NO_HELMET_SECONDS_THRESHOLD and not state.reported:
                        clip_rel_path = clip_writer.start_clip(
                            fps=fps,
                            frame_size=frame_size,
                            pre_frames=list(pre_frame_buffer),
                        )
                        entry = EventLogEntry(
                            person_id=pid,
                            start_time=state.no_helmet_started,
                            end_time=now,
                            duration_seconds=duration,
                            clip_path=str(clip_rel_path),
                        )
                        logger.append(entry)
                        state.reported = True

                person_states[pid] = state

            clip_writer.write_frame(frame)

            if SHOW_WINDOW:
                vis = frame.copy()
                for idx in range(len(tracked_persons)):
                    bbox = person_boxes[idx]
                    pid = int(tracked_persons.tracker_id[idx]) if tracked_persons.tracker_id is not None else idx
                    state = person_states.get(pid)
                    color = (0, 255, 0) if state and state.no_helmet_started is None else (0, 0, 255)
                    cv2.rectangle(vis, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(
                        vis,
                        f"ID {pid}",
                        (bbox[0], max(20, bbox[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )
                cv2.imshow("IP Helmet Monitor", vis)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        clip_writer.close()
        if SHOW_WINDOW:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    run_monitoring()