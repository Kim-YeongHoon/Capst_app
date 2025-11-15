// app.js (local preview + sample events only)

console.log("[helmet-demo] app.js loaded");

// 1) DOM 요소 캐싱
const videoInput = document.getElementById("videoInput");
const uploadBtn = document.getElementById("uploadBtn");
const statusEl = document.getElementById("status");
const previewVideo = document.getElementById("previewVideo");
const loadSampleEventsBtn = document.getElementById("loadSampleEventsBtn");
const eventsContainer = document.getElementById("eventsContainer");

// 방어 코드: 필수 요소 존재 여부 확인
if (!videoInput || !uploadBtn || !statusEl || !previewVideo || !loadSampleEventsBtn || !eventsContainer) {
  console.error("[helmet-demo] One or more expected DOM elements are missing.");
}

// 2) 샘플 YOLO 이벤트 (데모용 하드코딩 데이터)
const SAMPLE_EVENTS = [
  {
    id: "evt-001",
    timestamp_sec: 12.34,
    frame_index: 370,
    class_name: "no_helmet",
    confidence: 0.932,
    bbox: [422, 188, 560, 416],
    worker_id: "worker-7",
  },
  {
    id: "evt-002",
    timestamp_sec: 47.92,
    frame_index: 1438,
    class_name: "no_helmet",
    confidence: 0.876,
    bbox: [218, 140, 332, 364],
    worker_id: "worker-2",
  },
  {
    id: "evt-003",
    timestamp_sec: 88.11,
    frame_index: 2643,
    class_name: "no_helmet",
    confidence: 0.803,
    bbox: [506, 172, 640, 420],
    worker_id: "worker-11",
  },
];

// 현재 미리보기용 object URL (메모리 관리용)
let currentObjectUrl = null;

// 상태 메시지 업데이트
const setStatus = (message) => {
  if (!statusEl) return;
  statusEl.textContent = message;
};

// 기존 object URL 해제 (메모리 누수 방지)
const revokeCurrentObjectUrl = () => {
  if (!currentObjectUrl) return;
  URL.revokeObjectURL(currentObjectUrl);
  currentObjectUrl = null;
};

// 초 단위를 보기 좋은 문자열로
const formatTimestamp = (seconds) => {
  if (!Number.isFinite(seconds)) return "-";
  return seconds.toFixed(2) + "초";
};

// 신뢰도(0~1)를 퍼센트 문자열로
const formatConfidence = (confidence) => {
  if (!Number.isFinite(confidence)) return "-";
  return `${(confidence * 100).toFixed(1)}%`;
};

// YOLO 이벤트 카드 렌더링
const renderEvents = (events = []) => {
  if (!eventsContainer) return;

  eventsContainer.innerHTML = "";

  if (!Array.isArray(events) || events.length === 0) {
    const emptyState = document.createElement("div");
    emptyState.className = "empty-events";
    emptyState.textContent =
      "표시할 이벤트가 없습니다. 샘플 데이터를 불러오거나 분석 결과를 연결해 주세요.";
    eventsContainer.appendChild(emptyState);
    console.log("[helmet-demo] Rendered empty events state");
    return;
  }

  events.forEach((evt) => {
    const card = document.createElement("div");
    card.className = "event-item";

    const header = document.createElement("div");
    header.className = "event-header";
    header.textContent = `이벤트 ID: ${evt.id || "(알 수 없음)"}`;

    const body = document.createElement("div");
    body.className = "event-body";

    const classRow = document.createElement("div");
    classRow.textContent = `클래스: ${evt.class_name || "(알 수 없음)"}`;

    const timeRow = document.createElement("div");
    timeRow.textContent = `탐지 시각: ${formatTimestamp(
      evt.timestamp_sec
    )} (프레임 ${
      Number.isInteger(evt.frame_index) ? evt.frame_index : "-"
    })`;

    const confidenceRow = document.createElement("div");
    confidenceRow.textContent = `신뢰도: ${formatConfidence(
      evt.confidence
    )}`;

    const bboxRow = document.createElement("div");
    const bboxLabel = document.createElement("span");
    bboxLabel.textContent = "bbox: ";
    const bboxValue = document.createElement("code");
    bboxValue.textContent = Array.isArray(evt.bbox)
      ? `[${evt.bbox.join(", ")}]`
      : "[?, ?, ?, ?]";
    bboxRow.append(bboxLabel, bboxValue);

    const workerRow = document.createElement("div");
    const workerLabel = evt.worker_id ?? "미확인 작업자";
    workerRow.textContent = `작업자 ID: ${workerLabel}`;

    body.append(classRow, timeRow, confidenceRow, bboxRow, workerRow);
    card.append(header, body);
    eventsContainer.appendChild(card);
  });

  console.log(`[helmet-demo] Rendered ${events.length} event cards`);
};

// 선택한 파일을 로컬에서 미리보기
const previewLocalVideo = (file) => {
  if (!file) {
    window.alert("mp4 파일을 선택해 주세요.");
    return;
  }

  if (!file.type.startsWith("video/")) {
    window.alert("영상 파일만 업로드할 수 있습니다.");
    return;
  }

  if (!previewVideo) {
    console.warn("[helmet-demo] previewVideo element is missing");
    return;
  }

  revokeCurrentObjectUrl();

  currentObjectUrl = URL.createObjectURL(file);
  previewVideo.src = currentObjectUrl;
  previewVideo.load();

  const playPromise = previewVideo.play();
  if (playPromise && typeof playPromise.catch === "function") {
    playPromise.catch(() => {
      console.warn(
        "[helmet-demo] Autoplay was prevented; user interaction may be required."
      );
    });
  }

  setStatus("로컬 파일 미리보기 중 (Firebase 업로드는 비활성화 상태)");
  console.log("[helmet-demo] Previewing local video", {
    name: file.name,
    size: file.size,
  });
};

// 4) 이벤트 리스너 설정

if (videoInput) {
  videoInput.addEventListener("change", (event) => {
    const file = event.target.files?.[0];

    if (!file) {
      setStatus("선택된 파일이 없습니다. mp4 파일을 선택해 주세요.");
      return;
    }

    setStatus(`선택된 파일: ${file.name}`);
    // 새로운 영상을 선택하면 이전 탐지 결과는 초기화
    renderEvents([]);
  });
}

if (uploadBtn) {
  uploadBtn.addEventListener("click", () => {
    const file = videoInput?.files?.[0];
    if (!file) {
      window.alert("mp4 파일을 먼저 선택해 주세요.");
      return;
    }

    previewLocalVideo(file);
  });
}

if (loadSampleEventsBtn) {
  loadSampleEventsBtn.addEventListener("click", () => {
    renderEvents(SAMPLE_EVENTS);
  });
}

// 페이지 이탈 시 object URL 정리
window.addEventListener("beforeunload", () => {
  revokeCurrentObjectUrl();
});

// 5) 초기 상태 세팅
setStatus('영상 파일을 선택한 뒤 "업로드 / 미리보기" 버튼을 눌러주세요.');
renderEvents([]);

console.log("[helmet-demo] app.js init complete");
