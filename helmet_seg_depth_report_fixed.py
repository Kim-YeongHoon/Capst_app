import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import torch
import torch.nn.functional as F
from datetime import datetime
import csv
import os
import platform
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# ============================
# 0. 사용자 설정
# ============================
POSE_WEIGHTS   = r"C:/Users/qazdr/jupyyyttte/models/yolo11s-pose.pt"  # YOLO pose 모델(.pt 경로로 교체)
HELMET_WEIGHTS = r"C:/Users/qazdr/jupyyyttte/runs/detect/train20/weights/best.pt"  # 헬멧 bbox 모델
HELMET_CLASS_ID = 1  # train20: 0=person, 1=helmet

DEPTH_MODEL_DIR = r"C:/Users/qazdr/jupyyyttte/depth-anytv2s"  # DepthAnything v2 (HF 폴더)

VIDEO_SOURCE = r"C:/Users/qazdr/jupyyyttte/testmp4/KakaoTalk_20251030_224304616.mp4"  # 0이면 웹캠
OUTPUT_VIDEO = "./runs/output_helmet_report.mp4"
REPORT_CSV   = "./runs/helmet_violation_report.csv"

# 헬멧 미착용으로 간주할 최소 시간 (초)
NO_HELMET_SECONDS_THRESHOLD = 5.0

# 헬멧-사람 depth 유지 임계값 (상대값)
KEEP_DEPTH_THRESHOLD = 0.6

SHOW_WINDOW = True
WRITE_VIDEO = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================
# 1. 폴더/CSV 준비
# ============================
os.makedirs(os.path.dirname(OUTPUT_VIDEO) or ".", exist_ok=True)
os.makedirs(os.path.dirname(REPORT_CSV) or ".", exist_ok=True)

if not os.path.exists(REPORT_CSV):
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["track_id", "start_time_iso", "end_time_iso", "duration_seconds", "notes"])

# ============================
# 2. 모델 로드
# ============================
print("[INFO] Models loading...")
pose_model   = YOLO(POSE_WEIGHTS)      # 사람 포즈(스켈레톤) 모델
helmet_model = YOLO(HELMET_WEIGHTS)    # 헬멧 bbox 모델

depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_DIR)
depth_model     = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_DIR).to(DEVICE)
DEPTH_AVAILABLE = True
print("[INFO] DepthAnything V2 (local) loaded.")

# ============================
# 3. 트래커 & 상태 변수
# ============================
person_tracker = sv.ByteTrack()
helmet_tracker = sv.ByteTrack()

helmet_owner = {}      # {helmet_id: person_id}
helmet_last_depth = {} # {helmet_id: last_depth_value}

# person_state[pid] = {...}
person_state = {}

# ============================
# 4. 유틸 함수들
# ============================
def get_depth_map(frame):
    """DepthAnything V2로 frame → depth 맵 (H x W)"""
    if not DEPTH_AVAILABLE or depth_model is None or depth_processor is None:
        return None
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = depth_processor(images=rgb, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth  # [1, H', W']
    prediction = F.interpolate(
        predicted_depth.unsqueeze(1),          # [1,1,H',W']
        size=rgb.shape[:2],                    # (H,W)
        mode="bicubic",
        align_corners=False,
    ).squeeze(1)                               # [1,H,W]
    depth = prediction[0].cpu().numpy()
    return depth.astype(np.float32)

def calc_person_depth_from_keypoints(kpts_xy, depth_map):
    """키포인트 위치에서 depth 샘플링 후 평균값"""
    if depth_map is None or kpts_xy is None:
        return None
    h, w = depth_map.shape[:2]

    xs = np.round(kpts_xy[:, 0]).astype(int)
    ys = np.round(kpts_xy[:, 1]).astype(int)

    valid_mask = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    xs = xs[valid_mask]
    ys = ys[valid_mask]
    if xs.size == 0:
        return None

    vals = depth_map[ys, xs]
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return None
    return float(np.mean(vals))

def calc_helmet_depth(bbox, depth_map):
    """
    헬멧 bbox 중심 주변 patch의 평균 depth + 헬멧 최하단 중앙 좌표 반환
    depth 계산용 중심은 bbox 가운데, bottom_center는 시각화용.
    """
    if depth_map is None:
        return None, None
    x1, y1, x2, y2 = map(int, bbox)

    cx_center = int((x1 + x2) / 2)
    cy_center = int((y1 + y2) / 2)

    r = max(2, int(min(x2 - x1, y2 - y1) * 0.15))
    sx, ex = max(0, cx_center - r), min(depth_map.shape[1] - 1, cx_center + r)
    sy, ey = max(0, cy_center - r), min(depth_map.shape[0] - 1, cy_center + r)
    patch = depth_map[sy:ey+1, sx:ex+1]

    bottom_center = (int((x1 + x2) / 2), y2)

    if patch.size == 0:
        return None, bottom_center

    vals = patch.flatten()
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return None, bottom_center
    return float(np.mean(vals)), bottom_center

def append_event_csv(track_id, start_dt, end_dt, duration_s, notes=""):
    """CSV에 한 줄 기록"""
    with open(REPORT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([track_id, start_dt.isoformat(), end_dt.isoformat(), f"{duration_s:.2f}", notes])
    print(f"[LOG] event saved id={track_id} dur={duration_s:.1f}s notes={notes}")

def compute_head_box_from_keypoints(kpts_xy, person_bbox):
    """
    포즈 키포인트(코, 양 눈, 양 귀)를 이용해 머리 영역 bbox 계산.
    - 아래쪽 여유(margin_down)를 최소화하여 턱 아래·어깨가 들어가지 않도록 조정.
    """
    if kpts_xy is None:
        return None

    # COCO 17 keypoints: 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear
    head_indices = [0, 1, 2, 3, 4]
    if kpts_xy.shape[0] <= max(head_indices):
        return None

    head_points = kpts_xy[head_indices, :]  # (5, 2)
    xs = head_points[:, 0]
    ys = head_points[:, 1]

    valid = (~np.isnan(xs)) & (~np.isnan(ys))
    if not np.any(valid):
        return None

    xs = xs[valid]
    ys = ys[valid]
    if xs.size == 0:
        return None

    hx1, hy1 = float(xs.min()), float(ys.min())
    hx2, hy2 = float(xs.max()), float(ys.max())

    w = hx2 - hx1
    h = hy2 - hy1
    if w <= 0 or h <= 0:
        return None

    # 위/옆은 넉넉히, 아래는 거의 내리지 않음 (턱 정도까지만)
    margin_w    = 0.4 * w   # 좌우 확장
    margin_up   = 0.8 * h   # 위쪽 확장
    margin_down = 0.1 * h   # 아래쪽은 아주 조금만

    ex1 = hx1 - margin_w
    ex2 = hx2 + margin_w
    ey1 = hy1 - margin_up
    ey2 = hy2 + margin_down

    px1, py1, px2, py2 = person_bbox
    ex1 = max(ex1, px1)
    ex2 = min(ex2, px2)
    ey1 = max(ey1, py1)
    ey2 = min(ey2, py2)

    if ex2 <= ex1 or ey2 <= ey1:
        return None

    return [int(ex1), int(ey1), int(ex2), int(ey2)]

def is_helmet_on_head(helmet_bbox, person_bbox, head_box=None):
    """
    미착용 판단 기준:

      - 헬멧 바운딩박스 중점(cx)을 기준으로 좌우 30% 범위의 세로 띠(vertical band)를 만든다.
        * 헬멧 bbox 폭 = W
        * band_x1 = cx - 0.3*W
        * band_x2 = cx + 0.3*W
        * band_y1 = 헬멧 bbox 상단
        * band_y2 = 헬멧 bbox 하단

      - 이 세로 띠가 head_box와 전혀 겹치지 않을 때만 미착용으로 본다.
      - head_box가 없으면 person_bbox의 상단 40%만 헤드로 사용한다.
    """
    if helmet_bbox is None or person_bbox is None:
        return False

    hx1_h, hy1_h, hx2_h, hy2_h = map(float, helmet_bbox)
    if hx2_h <= hx1_h or hy2_h <= hy1_h:
        return False

    helmet_w = hx2_h - hx1_h
    cx = 0.5 * (hx1_h + hx2_h)

    band_x1 = cx - 0.3 * helmet_w
    band_x2 = cx + 0.3 * helmet_w
    band_y1 = hy1_h
    band_y2 = hy2_h

    # head_box가 없으면 person bbox의 상단 40%만 head로 취급
    if head_box is not None:
        hx1, hy1, hx2, hy2 = map(float, head_box)
    else:
        px1, py1, px2, py2 = map(float, person_bbox)
        if px2 <= px1 or py2 <= py1:
            return False
        pw = px2 - px1
        ph = py2 - py1
        hx1 = px1 + 0.15 * pw
        hx2 = px2 - 0.15 * pw
        hy1 = py1                      # 상단 시작
        hy2 = py1 + 0.40 * ph          # 상단 40%만

    if hx2 <= hx1 or hy2 <= hy1:
        return False

    ix1 = max(band_x1, hx1)
    iy1 = max(band_y1, hy1)
    ix2 = min(band_x2, hx2)
    iy2 = min(band_y2, hy2)

    # 세로 띠와 head 영역이 조금이라도 겹치면 "착용"
    if ix2 > ix1 and iy2 > iy1:
        return True
    else:
        # 전혀 겹치지 않을 때만 "미착용"
        return False

def box_iou_xyxy(box, boxes):
    """단일 box (4,)와 여러 boxes (N,4) 간 IoU 계산."""
    if boxes.size == 0:
        return np.array([])
    x1, y1, x2, y2 = box
    bx1 = boxes[:, 0]
    by1 = boxes[:, 1]
    bx2 = boxes[:, 2]
    by2 = boxes[:, 3]

    inter_x1 = np.maximum(x1, bx1)
    inter_y1 = np.maximum(y1, by1)
    inter_x2 = np.minimum(x2, bx2)
    inter_y2 = np.minimum(y2, by2)

    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (bx2 - bx1) * (by2 - by1)
    union = area1 + area2 - inter_area
    iou = inter_area / np.clip(union, 1e-6, None)
    return iou

# ============================
# 5. 헬멧-사람 매칭 (Depth + 2D 거리 + owner 유지)
# ============================
def match_helmet_to_person(helmet_id, helmet_depth, persons_info, helmet_bbox=None):
    """
    helmet_id: 추적된 헬멧 ID
    helmet_depth: 헬멧 depth
    helmet_bbox: 헬멧 bbox (중심점 계산용)
    persons_info: {pid: { "depth":..., "bbox":..., "head_box":..., "centroid":...}}

    - depth 차이가 작은 사람을 우선,
    - 동시에 2D 거리(헬멧 중심 vs head_box(또는 bbox) 중심)가 너무 크면 매칭하지 않음.
    """
    # 헬멧 중심 좌표
    if helmet_bbox is not None:
        hx1, hy1, hx2, hy2 = map(float, helmet_bbox)
        hcx = 0.5 * (hx1 + hx2)
        hcy = 0.5 * (hy1 + hy2)
    else:
        hcx = hcy = None

    # 1) 기존 owner 유지 시도 (depth + 2D 거리)
    if helmet_id in helmet_owner:
        owner = helmet_owner[helmet_id]
        if owner in persons_info and helmet_depth is not None:
            pd = persons_info[owner]["depth"]
            if pd is not None and abs(pd - helmet_depth) < KEEP_DEPTH_THRESHOLD:
                if hcx is not None and persons_info[owner]["bbox"] is not None:
                    px1, py1, px2, py2 = map(float, persons_info[owner]["bbox"])
                    pcx = 0.5 * (px1 + px2)
                    pcy = 0.5 * (py1 + py2)
                    dist2 = (pcx - hcx) ** 2 + (pcy - hcy) ** 2
                    # 너무 멀면 owner 유지 포기 (bbox 폭 기준)
                    if dist2 < (0.25 * max(px2 - px1, 1.0) ** 2):
                        return owner
                else:
                    return owner

    # 2) 새로 depth 기반 + 2D 거리로 최고 후보 찾기
    best_pid = None
    best_score = float("inf")

    for pid, info in persons_info.items():
        pd = info["depth"]
        if pd is None or helmet_depth is None:
            continue

        depth_diff = abs(pd - helmet_depth)

        # 2D 거리 (헬멧 중심 vs head_box 또는 bbox 중심)
        if hcx is not None:
            if info.get("head_box") is not None:
                bx1, by1, bx2, by2 = map(float, info["head_box"])
            else:
                bx1, by1, bx2, by2 = map(float, info["bbox"])
            pcx = 0.5 * (bx1 + bx2)
            pcy = 0.5 * (by1 + by2)
            dist = np.hypot(pcx - hcx, pcy - hcy)
        else:
            dist = 0.0

        # depth 차이 + 2D 거리의 가중 합으로 점수
        score = depth_diff + 0.002 * dist  # 거리 비중은 작게

        if score < best_score:
            best_score = score
            best_pid = pid

    if best_pid is None:
        return None

    # 너무 큰 score면 매칭하지 않음 (depth도 멀고 거리도 먼 경우)
    if best_score > KEEP_DEPTH_THRESHOLD * 3.0:
        return None

    return best_pid

# ============================
# 6. 메인 루프 시작
# ============================
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise SystemExit("카메라/비디오를 열 수 없습니다. VIDEO_SOURCE를 확인해 주셔야 합니다.")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

OFF_CONFIRM_FRAMES = int(NO_HELMET_SECONDS_THRESHOLD * fps)
ON_CONFIRM_FRAMES  = int(1.0 * fps)

if WRITE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_w, frame_h))

frame_idx = 0
print("[INFO] Start processing. ESC(ESC 키)로 종료.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        now = datetime.now()

        depth_map = get_depth_map(frame)

        # 2) 사람 포즈(스켈레톤) 추론 + bbox 추출 + 트래킹
        r_person = pose_model(frame, imgsz=640, conf=0.25, iou=0.45, verbose=False)[0]

        if r_person.boxes is not None and r_person.boxes.xyxy is not None:
            boxes_person = r_person.boxes.xyxy.cpu().numpy()
            conf_person  = r_person.boxes.conf.cpu().numpy()
            cls_person   = r_person.boxes.cls.cpu().numpy().astype(int)
        else:
            boxes_person = np.zeros((0, 4), dtype=float)
            conf_person  = np.zeros((0,), dtype=float)
            cls_person   = np.zeros((0,), dtype=int)

        if r_person.keypoints is not None and r_person.keypoints.xy is not None:
            kpts_all = r_person.keypoints.xy.cpu().numpy()
        else:
            kpts_all = None

        person_dets = sv.Detections(
            xyxy=boxes_person,
            confidence=conf_person,
            class_id=cls_person,
        )
        person_tracks = person_tracker.update_with_detections(person_dets)

        persons_info = {}
        n_person = len(person_tracks)

        for i in range(n_person):
            if person_tracks.tracker_id is not None:
                pid = int(person_tracks.tracker_id[i])
            else:
                pid = i

            xyxy = person_tracks.xyxy[i].astype(int)
            x1, y1, x2, y2 = xyxy.tolist()

            kp_xy = None
            head_box = None
            if boxes_person.shape[0] > 0 and kpts_all is not None:
                track_box = xyxy.astype(float)
                ious = box_iou_xyxy(track_box, boxes_person)
                if ious.size > 0:
                    best_idx = int(np.argmax(ious))
                    if ious[best_idx] > 0.1:
                        kp_xy = kpts_all[best_idx]
                        head_box = compute_head_box_from_keypoints(kp_xy, [x1, y1, x2, y2])

            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            p_depth = calc_person_depth_from_keypoints(kp_xy, depth_map) if depth_map is not None else None

            persons_info[pid] = {
                "depth": p_depth,
                "bbox": [x1, y1, x2, y2],
                "centroid": centroid,
                "keypoints": kp_xy,
                "head_box": head_box,
            }

            if pid not in person_state:
                person_state[pid] = {
                    "helmet_status": "UNKNOWN",
                    "on_count": 0,
                    "off_count": 0,
                    "nohelmet_start": None,
                    "alerted": False,
                    "events": []
                }

        # 3) 헬멧 탐지 + 트래킹
        r_helm = helmet_model(
            frame,
            imgsz=640,
            conf=0.25,
            iou=0.45,
            classes=[HELMET_CLASS_ID],
            verbose=False
        )[0]

        helm_dets = sv.Detections.from_ultralytics(r_helm)
        helm_tracks = helmet_tracker.update_with_detections(helm_dets)

        helmets_list = []
        n_helm = len(helm_tracks)
        for i in range(n_helm):
            if helm_tracks.tracker_id is not None:
                hid = int(helm_tracks.tracker_id[i])
            else:
                hid = i

            hb = helm_tracks.xyxy[i].astype(int).tolist()
            conf = float(helm_tracks.confidence[i]) if helm_tracks.confidence is not None else 0.0
            h_depth, bottom_center = calc_helmet_depth(hb, depth_map)

            helmets_list.append({
                "hid": hid,
                "bbox": hb,
                "depth": h_depth,
                "bottom_center": bottom_center,
                "conf": conf,
            })

        # 4) 헬멧 → 사람 매칭 + 세로 띠(중심±30%)가 헤드 박스와 겹치는 경우만 attached
        helmet_attached = {}

        for h in helmets_list:
            hid = h["hid"]
            h_depth = h["depth"]
            helmet_bbox = h["bbox"]

            candidate_pid = match_helmet_to_person(hid, h_depth, persons_info, helmet_bbox=helmet_bbox)

            if candidate_pid is not None:
                helmet_owner[hid] = candidate_pid
                helmet_last_depth[hid] = h_depth

                pbbox = persons_info[candidate_pid]["bbox"]
                head_box = persons_info[candidate_pid]["head_box"]

                if is_helmet_on_head(helmet_bbox, pbbox, head_box=head_box):
                    helmet_attached[hid] = candidate_pid

        # 5) 사람별 이번 프레임 헬멧 보유 여부
        person_has_helmet_obs = {pid: False for pid in persons_info.keys()}
        for hid, pid in helmet_attached.items():
            if pid in person_has_helmet_obs:
                person_has_helmet_obs[pid] = True

        # 6) 시간 기반 상태 업데이트
        for pid, info in persons_info.items():
            st = person_state[pid]
            obs = person_has_helmet_obs.get(pid, False)

            if obs:
                st["on_count"]  += 1
                st["off_count"]  = 0
            else:
                st["off_count"] += 1
                st["on_count"]   = 0

            if st["helmet_status"] != "HELMET_ON" and st["on_count"] >= ON_CONFIRM_FRAMES:
                st["helmet_status"] = "HELMET_ON"
                if st["nohelmet_start"] is not None:
                    end_dt = now
                    dur = (end_dt - st["nohelmet_start"]).total_seconds()
                    append_event_csv(pid, st["nohelmet_start"], end_dt, dur, notes="helmet_put_on")
                    st["events"].append((st["nohelmet_start"], end_dt, dur, "helmet_put_on"))
                    st["nohelmet_start"] = None
                    st["alerted"] = False

            if st["helmet_status"] != "NO_HELMET" and st["off_count"] >= OFF_CONFIRM_FRAMES:
                st["helmet_status"] = "NO_HELMET"
                st["nohelmet_start"] = now
                st["alerted"] = False

            if st["helmet_status"] == "NO_HELMET" and st["nohelmet_start"] is not None:
                dur = (now - st["nohelmet_start"]).total_seconds()
                if (not st["alerted"]) and dur >= NO_HELMET_SECONDS_THRESHOLD:
                    append_event_csv(pid, st["nohelmet_start"], now, dur, notes="ongoing_over_threshold")
                    st["alerted"] = True
                    try:
                        if platform.system() == "Windows":
                            import winsound
                            winsound.Beep(1000, 500)
                    except Exception:
                        pass

        # 7) 시각화
        vis = frame.copy()

        # 사람 박스/상태 + 머리 박스 표시
        for pid, info in persons_info.items():
            x1, y1, x2, y2 = info["bbox"]
            st = person_state[pid]
            status = st["helmet_status"]

            if status == "HELMET_ON":
                color = (0, 255, 0)
            elif status == "NO_HELMET":
                color = (0, 0, 255)
            else:
                color = (0, 255, 255)

            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label = f"ID:{pid} {status}"
            if info["depth"] is not None:
                label += f" D:{info['depth']:.2f}"
            cv2.putText(vis, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if info["head_box"] is not None:
                hx1, hy1, hx2, hy2 = info["head_box"]
                cv2.rectangle(vis, (hx1, hy1), (hx2, hy2), (255, 0, 0), 2)

            if status == "NO_HELMET" and st["nohelmet_start"] is not None:
                dur = (now - st["nohelmet_start"]).total_seconds()
                txt = f"NO HELMET {int(dur)}s"
                hx = int((x1 + x2) / 2)
                hy = max(0, y1 - 30)
                cv2.putText(vis, txt, (hx - 70, hy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if dur >= NO_HELMET_SECONDS_THRESHOLD:
                    cv2.rectangle(vis, (0, 0), (frame_w, 60), (0, 0, 255), -1)
                    cv2.putText(vis,
                                f"ALERT ID:{pid} NO HELMET {int(dur)}s",
                                (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                                (255, 255, 255), 2)

        # 헬멧 박스/매칭 상태 표시 + 중심±30% 세로 띠 표시
        for h in helmets_list:
            x1, y1, x2, y2 = h["bbox"]
            hid = h["hid"]
            depth_text = f"{h['depth']:.2f}" if h["depth"] is not None else "nan"

            if hid in helmet_attached:
                owner = helmet_attached[hid]
                label = f"H{hid}->ID{owner} d:{depth_text}"
                col = (255, 200, 0)
            elif hid in helmet_owner:
                owner = helmet_owner[hid]
                label = f"H{hid} (~ID{owner}) d:{depth_text}"
                col = (200, 200, 200)
            else:
                label = f"H{hid}->None d:{depth_text}"
                col = (0, 165, 255)

            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            cv2.putText(vis, label, (x1, y2 + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

            if h["bottom_center"] is not None:
                bc_x, bc_y = h["bottom_center"]
                cv2.circle(vis, (bc_x, bc_y), 3, (0, 0, 255), -1)

            helmet_w = x2 - x1
            cx = int(0.5 * (x1 + x2))
            band_x1 = int(cx - 0.3 * helmet_w)
            band_x2 = int(cx + 0.3 * helmet_w)
            band_x1 = max(band_x1, x1)
            band_x2 = min(band_x2, x2)
            cv2.rectangle(vis, (band_x1, y1), (band_x2, y2), (0, 255, 255), 1)

        # 8) 출력/저장
        if WRITE_VIDEO:
            writer.write(vis)
        if SHOW_WINDOW:
            cv2.imshow("Helmet-Pose-Depth System (time-smoothed)", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

finally:
    end_time = datetime.now()
    for pid, st in person_state.items():
        if st["helmet_status"] == "NO_HELMET" and st["nohelmet_start"] is not None:
            s = st["nohelmet_start"]
            dur = (end_time - s).total_seconds()
            append_event_csv(pid, s, end_time, dur, notes="end_of_stream")

    cap.release()
    if WRITE_VIDEO:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] 저장된 리포트:", REPORT_CSV)
    print("[INFO] 종료.")
