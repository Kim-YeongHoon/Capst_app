# helmet_seg_depth_report_fixed.py
# - YOLO(사람 seg / 헬멧 detector) + DepthAnything v2(Transformers) 결합
# - 헬멧 ↔ 사람 매칭을 엄격하게 하여 겹침/occlusion 문제 해결
# - 일정시간 미착용(설정값) 일 때 리포트(CSV)와 경고음 발생
# - 각 블록에 한국어 주석 자세히 포함

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
import time

# -----------------------------
# 사용자 설정 (필요시 수정)
# -----------------------------
PERSON_SEG_WEIGHTS = r"C:/Users/qazdr/jupyyyttte/yolo12l-person-seg-extended.pt"
HELMET_WEIGHTS     = r"C:/Users/qazdr/jupyyyttte/runs/detect/train20/weights/best.pt"

HELMET_CLASS_ID    = 1  # 헬멧 클래스 인덱스 (모델에 따라 다름)

DEPTH_MODEL_DIR    = r"C:/Users/qazdr/jupyyyttte/depth-anytv2s"  # Depth-Anything 모델 폴더
VIDEO_SOURCE = "C:/Users/qazdr/jupyyyttte/testmp4/KakaoTalk_20251030_224304616.mp4"  # 또는 0 (웹캠)
OUTPUT_VIDEO = "./runs/output_helmet_report_fixed.mp4"
REPORT_CSV   = "./runs/helmet_violation_report_fixed.csv"

NO_HELMET_SECONDS_THRESHOLD = 5.0   # "헬멧을 벗은 상태"로 간주할 최소 연속 시간 (초) - 너 요청: 5초
KEEP_DEPTH_THRESHOLD = 0.6         # depth 유지 허용치 (단위은 모델 출력 스케일에 따름)
WRITE_VIDEO = True
SHOW_WINDOW = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 디렉터리/파일 준비
# -----------------------------
os.makedirs(os.path.dirname(OUTPUT_VIDEO) or ".", exist_ok=True)
os.makedirs(os.path.dirname(REPORT_CSV) or ".", exist_ok=True)
if not os.path.exists(REPORT_CSV):
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["person_id", "start_time_iso", "end_time_iso", "duration_seconds", "notes"])

# -----------------------------
# 모델 로드
# -----------------------------
print("[INFO] 모델 로드 중...")
person_seg_model = YOLO(PERSON_SEG_WEIGHTS)   # 사람 세그 모델 (mask 제공)
helmet_model = YOLO(HELMET_WEIGHTS)           # 헬멧 탐지기 (bbox)

# DepthAnything (Transformers HF 형식) 로드
depth_processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_DIR)
depth_model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_DIR).to(DEVICE)
DEPTH_AVAILABLE = True
print("[INFO] DepthAnything 모델 로드 완료. DEVICE:", DEVICE)

# -----------------------------
# 트래커(간단) 초기화
# -----------------------------
person_tracker = sv.ByteTrack()
helmet_tracker = sv.ByteTrack()

# -----------------------------
# 상태 저장 구조
# -----------------------------
helmet_owner = {}        # {helmet_track_id: person_id}
helmet_last_depth = {}   # {helmet_track_id: last_depth}
person_state = {}        # {person_id: {"nohelmet_start": datetime or None, "alerted": bool, "events": []}}
# person_has_helmet 동적 계산 (프레임 단위) - 다른 로직에서 사용

# -----------------------------
# 유틸 함수들
# -----------------------------
def point_in_mask(mask, x, y):
    """픽셀 (x,y)가 mask 내부인지 확인 (mask: 2D numpy uint8)"""
    if mask is None:
        return False
    h, w = mask.shape
    if not (0 <= x < w and 0 <= y < h):
        return False
    return bool(mask[y, x])

def get_depth_map(frame):
    """
    DepthAnything HF 모델로 depth map 생성 (frame: BGR)
    반환: HxW float32 depth (모델 스케일)
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    inputs = depth_processor(images=rgb, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = depth_model(**inputs)
        predicted_depth = outputs.predicted_depth  # [1, H', W']
    # 원본 크기로 보간
    pred = F.interpolate(predicted_depth.unsqueeze(1), size=rgb.shape[:2], mode="bicubic", align_corners=False).squeeze(1)
    depth = pred[0].cpu().numpy()
    return depth.astype(np.float32)

def calc_person_depth(mask, depth_map):
    """사람 mask 내부 픽셀의 평균 depth (nan 제외)"""
    if depth_map is None or mask is None:
        return None
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    vals = depth_map[ys, xs]
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return None
    return float(np.mean(vals))

def calc_helmet_depth(bbox, depth_map):
    """헬멧 bbox 중심 근처 패치의 평균 depth와 중심 좌표 반환"""
    if depth_map is None:
        return None, None
    x1, y1, x2, y2 = map(int, bbox)
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    r = max(2, int(min(x2 - x1, y2 - y1) * 0.15))
    sx, ex = max(0, cx - r), min(depth_map.shape[1] - 1, cx + r)
    sy, ey = max(0, cy - r), min(depth_map.shape[0] - 1, cy + r)
    patch = depth_map[sy:ey+1, sx:ex+1]
    if patch.size == 0:
        return None, (cx, cy)
    vals = patch.flatten()
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return None, (cx, cy)
    return float(np.mean(vals)), (cx, cy)

def append_event_csv(person_id, start_dt, end_dt, duration_s, notes=""):
    """CSV에 위반/복구 이벤트 기록"""
    with open(REPORT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([person_id, start_dt.isoformat(), end_dt.isoformat(), f"{duration_s:.2f}", notes])
    print(f"[LOG] 저장: id={person_id} dur={duration_s:.1f}s notes={notes}")

# 헬멧이 사람 머리(상단) 영역에 있는지 판단 함수 (엄격하게)
def is_helmet_on_head(helmet_center, person_bbox, person_mask=None):
    """
    - helmet_center: (cx,cy)
    - person_bbox: [x1,y1,x2,y2]
    - person_mask: optional, mask가 있으면 마스크 내부 확인 우선
    규칙: mask가 있으면 mask 내부 + head 영역 근접(상단 35%) 우선.
          mask 없으면 bbox의 상단 35% 영역(머리영역) 내부여야 True.
    """
    cx, cy = helmet_center
    x1, y1, x2, y2 = person_bbox
    head_y2 = int(y1 + 0.35 * (y2 - y1))  # bbox 상단 35% 영역을 머리로 간주

    # mask가 있으면 mask 내부인지 확인 (더 신뢰)
    if person_mask is not None:
        # mask는 전체 프레임 크기 기준으로 들어온다고 가정
        if point_in_mask(person_mask, cx, cy):
            return True
        else:
            # 마스크가 있지만 중심이 mask 밖이라면 False
            return False

    # mask가 없으면 bbox 기반 판단 (머리 영역 안에 중심이 있어야 함)
    if (x1 <= cx <= x2) and (y1 <= cy <= head_y2):
        return True
    return False

# -----------------------------
# 엄격한 헬멧-사람 매칭 함수 (수정된 핵심)
# -----------------------------
def match_helmet_to_person(helmet_id, helmet_depth, helmet_center, persons_info, assigned_persons_frame):
    """
    - persons_info: dict person_id -> {"mask","depth","bbox","centroid"}
    - assigned_persons_frame: set -> 같은 프레임에서 이미 새로운 헬멧으로 할당된 사람을 피하기 위해 사용
    반환: matched person_id or None
    규칙 요약:
     1) 기존 owner 유지(있으면 우선) - depth와 head 위치 체크
     2) 새 매칭: depth 차이로 후보 선택하되 해당 후보가 이미 헬멧 보유중이면 제외
     3) 후보는 반드시 head 위치(또는 mask 내부)에 헤멧 중심이 있어야 확정
    """
    # 1) 이전 owner가 있으면 우선 유지(occlusion 대비)
    if helmet_id in helmet_owner:
        prev_owner = helmet_owner[helmet_id]
        # 이전 owner가 현재 persons_info에 있으면 depth/머리 위치로 유지 여부 검사
        if prev_owner in persons_info:
            pd = persons_info[prev_owner]["depth"]
            # depth 값이 양쪽 다 있으면 depth 기준 유지 확인
            if pd is not None and helmet_depth is not None and abs(pd - helmet_depth) < KEEP_DEPTH_THRESHOLD:
                # 추가로 머리 위치(엄격)도 확인 — 가려져도 prev bbox/마스크 정보가 있다면 머리 check 시도
                if is_helmet_on_head(helmet_center, persons_info[prev_owner]["bbox"], persons_info[prev_owner].get("mask")):
                    return prev_owner
        # 유지 조건 실패면 이전 owner 해제 (다음 단계에서 재매칭 가능)
        try:
            del helmet_owner[helmet_id]
        except KeyError:
            pass

    # 2) depth 기준으로 후보 선택 (가장 가까운 사람). 단, 이미 '현재 프레임에 헬멧 가진 사람'은 후보에서 제외
    best_pid = None
    best_dd = float("inf")
    for pid, info in persons_info.items():
        # depth가 둘 다 있어야 비교 가능
        pd = info["depth"]
        if pd is None or helmet_depth is None:
            continue
        # 이미 헬멧을 가지고 있는 사람(프레임 레벨)이라면 후보에서 제외 -> 중복 착용 방지
        # 현재 helmet_owner에서 이 pid가 value로 존재하면 '이미 누군가의 헬멧을 보유 중'으로 판단
        if pid in assigned_persons_frame:
            continue
        # depth 차이 판단
        dd = abs(pd - helmet_depth)
        if dd < best_dd:
            best_dd = dd
            best_pid = pid

    # 3) 후보 기반 최종 판정: 후보가 없으면 fallback으로 mask 내부 검사(마스크 가진 경우만)
    if best_pid is None:
        # fallback: 만약 mask가 있고 헬멧 중심이 mask 내부면 매칭
        for pid, info in persons_info.items():
            if info.get("mask") is None:
                continue
            cx, cy = helmet_center
            if point_in_mask(info["mask"], cx, cy):
                # 추가로 해당 사람이 현재 다른 헬멧에 할당되었는지 확인
                if pid in assigned_persons_frame:
                    continue
                return pid
        return None

    # 4) 후보가 있으면 반드시 head 영역 내(또는 mask 내부)여야지 최종 매칭
    if is_helmet_on_head(helmet_center, persons_info[best_pid]["bbox"], persons_info[best_pid].get("mask")):
        return best_pid

    # 5) 마지막 여지: mask가 없고 depth 차이가 아주 작으면 허용 (낮은 신뢰)
    if best_dd < (KEEP_DEPTH_THRESHOLD * 0.5):  # 엄격하게 낮춤
        # 단, 이 경우에도 해당 사람이 다른 헬멧에 이미 할당되었는지 확인
        if best_pid not in assigned_persons_frame:
            return best_pid

    return None

# -----------------------------
# 메인 루프
# -----------------------------
cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    raise SystemExit("VIDEO_SOURCE를 열 수 없습니다. 경로/장치 확인 필요.")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1280)
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720)

if WRITE_VIDEO:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_w, frame_h))

frame_idx = 0
print("[INFO] 처리 시작. ESC로 종료.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        now = datetime.now()

        # (A) Depth map 생성 (시간 걸릴 수 있음)
        depth_map = get_depth_map(frame) if DEPTH_AVAILABLE else None

        # (B) 사람 세그멘테이션 및 추적
        r_person = person_seg_model(frame, imgsz=640, conf=0.25, iou=0.45, verbose=False)[0]
        person_dets = sv.Detections.from_ultralytics(r_person)
        person_tracks = person_tracker.update_with_detections(person_dets)

        # persons_info 구성
        persons_info = {}
        for i in range(len(person_tracks)):
            pid = int(person_tracks.tracker_id[i]) if person_tracks.tracker_id is not None else i
            xyxy = person_tracks.xyxy[i].astype(int)
            x1, y1, x2, y2 = xyxy.tolist()
            mask = person_tracks.mask[i].astype(np.uint8) if hasattr(person_tracks, "mask") and person_tracks.mask is not None else None
            centroid = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            p_depth = calc_person_depth(mask, depth_map) if depth_map is not None else None
            persons_info[pid] = {"mask": mask, "depth": p_depth, "bbox": [x1, y1, x2, y2], "centroid": centroid}
            if pid not in person_state:
                person_state[pid] = {"nohelmet_start": None, "alerted": False, "events": []}

        # (C) 헬멧 detection + 추적
        r_helm = helmet_model(frame, imgsz=640, conf=0.25, iou=0.45, classes=[HELMET_CLASS_ID], verbose=False)[0]
        helm_dets = sv.Detections.from_ultralytics(r_helm)
        helm_tracks = helmet_tracker.update_with_detections(helm_dets)

        # (D) 프레임 레벨 초기화: 이 프레임에서 새로 '헬멧 소유'로 할당된 사람 집합
        assigned_persons_frame = set()
        # (E) 헬멧 리스트 수집
        helmets_list = []
        for i in range(len(helm_tracks)):
            hid = int(helm_tracks.tracker_id[i]) if helm_tracks.tracker_id is not None else i
            hb = helm_tracks.xyxy[i].astype(int).tolist()
            conf = float(helm_tracks.confidence[i]) if helm_tracks.confidence is not None else 0.0
            h_depth, center = calc_helmet_depth(hb, depth_map)
            helmets_list.append({"hid": hid, "bbox": hb, "depth": h_depth, "center": center, "conf": conf})

        # (F) 헬멧별로 새 매칭 시도 — 순서는 단순히 나열 순서 (필요시 confidence 우선 정렬 가능)
        # 중요한 점: match_helmet_to_person 내부에서 assigned_persons_frame를 보면서 같은 프레임 중복 할당 차단
        for h in helmets_list:
            hid = h["hid"]
            owner = match_helmet_to_person(hid, h["depth"], h["center"], persons_info, assigned_persons_frame)
            if owner is not None:
                # 이미 다른 헬멧이 그 사람에 할당되어있는지(프레임) 다시 체크 (중복 방지)
                if owner in assigned_persons_frame:
                    # skip: 다른 헬멧이 먼저 점유
                    continue
                helmet_owner[hid] = owner
                helmet_last_depth[hid] = h["depth"]
                assigned_persons_frame.add(owner)
            else:
                # 매칭 실패한 헬멧은 owner 해제 (있는 경우)
                if hid in helmet_owner:
                    del helmet_owner[hid]
                if hid in helmet_last_depth:
                    del helmet_last_depth[hid]

        # (G) 사람별 헬멧 보유 상태 정리 (프레임 단위)
        person_has_helmet = {pid: False for pid in persons_info.keys()}
        for hid, pid in helmet_owner.items():
            if pid in person_has_helmet:
                person_has_helmet[pid] = True

        # (H) 사람별 상태 업데이트: 헬멧 없으면 타이머 시작, 특정 시간 초과 시 리포트/경고
        for pid, state in person_state.items():
            if pid not in persons_info:
                # 현재 프레임에 사람이 없으면(occlusion/탐지누락) 상태 유지하되,
                # 만약 이전에 nohelmet_start가 있고 오래됐으면 기록하도록 둠
                continue
            has_helmet = person_has_helmet.get(pid, False)
            if not has_helmet:
                # 헬멧이 없는 상태이면 시작 타임을 기록
                if state["nohelmet_start"] is None:
                    state["nohelmet_start"] = datetime.now()
                    state["alerted"] = False
                else:
                    dur = (datetime.now() - state["nohelmet_start"]).total_seconds()
                    # NO_HELMET_SECONDS_THRESHOLD 초를 넘으면 이벤트 기록 및 경고 (한번만)
                    if (not state["alerted"]) and dur >= NO_HELMET_SECONDS_THRESHOLD:
                        append_event_csv(pid, state["nohelmet_start"], datetime.now(), dur, notes="no_helmet_over_threshold")
                        state["alerted"] = True
                        # 윈도우즈 경보음(선택)
                        try:
                            if platform.system() == "Windows":
                                import winsound
                                winsound.Beep(1000, 500)
                        except Exception:
                            pass
            else:
                # 헬멧을 착용한 경우, 만약 이전에 nohelmet_start가 있으면 종료 기록
                if state["nohelmet_start"] is not None:
                    end_dt = datetime.now()
                    dur = (end_dt - state["nohelmet_start"]).total_seconds()
                    append_event_csv(pid, state["nohelmet_start"], end_dt, dur, notes="helmet_put_on")
                    state["events"].append((state["nohelmet_start"], end_dt, dur, "helmet_put_on"))
                    state["nohelmet_start"] = None
                    state["alerted"] = False

        # (I) 시각화: 사람 박스/마스크, 헬멧 박스, 경고 배너
        vis = frame.copy()
        # 사람 표시
        for pid, info in persons_info.items():
            x1, y1, x2, y2 = info["bbox"]
            color = (0,255,0) if person_has_helmet.get(pid, False) else (0,0,255)
            cv2.rectangle(vis, (x1,y1),(x2,y2), color, 2)
            label = f"ID:{pid}"
            if info["depth"] is not None:
                label += f" D:{info['depth']:.2f}"
            cv2.putText(vis, label, (x1, max(0,y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            # 마스크가 있으면 반투명으로 표시 (디버그용)
            if info.get("mask") is not None:
                mask = info["mask"]
                colored_mask = np.zeros_like(vis)
                colored_mask[:,:,2] = (mask>0).astype(np.uint8) * 150  # red overlay
                vis = cv2.addWeighted(vis, 1.0, colored_mask, 0.25, 0)

            # no-helmet 타이머 표시
            st = person_state.get(pid, {}).get("nohelmet_start", None)
            if st is not None:
                dur = int((datetime.now() - st).total_seconds())
                txt = f"NO HELMET {dur}s"
                hx = int((x1 + x2)/2)
                hy = max(0, y1-30)
                if dur >= NO_HELMET_SECONDS_THRESHOLD:
                    cv2.rectangle(vis, (0,0), (frame_w, 60), (0,0,255), -1)
                    cv2.putText(vis, f"ALERT ID:{pid} NO HELMET {dur}s", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
                cv2.putText(vis, txt, (hx-60, hy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        # 헬멧 표시
        for h in helmets_list:
            x1,y1,x2,y2 = h["bbox"]
            hid = h["hid"]
            depth_text = f"{h['depth']:.2f}" if h["depth"] is not None else "nan"
            if hid in helmet_owner:
                owner = helmet_owner[hid]
                label = f"H{hid}->ID{owner} D:{depth_text}"
                col = (255,200,0)
            else:
                label = f"H{hid}->None D:{depth_text}"
                col = (0,165,255)
            cv2.rectangle(vis, (x1,y1),(x2,y2), col, 2)
            cv2.putText(vis, label, (x1, y2+16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        # 쓰기/표시
        if WRITE_VIDEO:
            writer.write(vis)
        if SHOW_WINDOW:
            cv2.imshow("Helmet-Seg-Depth-Fixed", vis)
            if cv2.waitKey(1) & 0xFF == 27:
                break

finally:
    # 스트림 종료 시 열려있는 no-helmet 이벤트 마감
    end_time = datetime.now()
    for pid, state in person_state.items():
        if state.get("nohelmet_start") is not None:
            s = state["nohelmet_start"]
            dur = (end_time - s).total_seconds()
            append_event_csv(pid, s, end_time, dur, notes="end_of_stream")
    cap.release()
    if WRITE_VIDEO:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] 리포트 저장 위치:", REPORT_CSV)
    print("[INFO] 종료.")
