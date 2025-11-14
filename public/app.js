// app.js

// 1) Firebase SDK import (모듈 방식)
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.14.0/firebase-app.js";
import {
  getStorage,
  ref,
  uploadBytesResumable,
  getDownloadURL,
} from "https://www.gstatic.com/firebasejs/10.14.0/firebase-storage.js";

console.log("[helmet-demo] app.js loaded");

const firebaseConfig = {
  apiKey: "AIzaSyBQ0wEdKnCwzPOWGJa3VAzq6GT_ebJl8ms",
  authDomain: "helmet-demo.firebaseapp.com",
  projectId: "helmet-demo",
  storageBucket: "helmet-demo.firebasestorage.app",
  messagingSenderId: "1037642230282",
  appId: "1:1037642230282:web:5f992777cb2ee822e98df3",
  measurementId: "G-HCQPN61BT5"
};

const app = initializeApp(firebaseConfig);
const storage = getStorage(app);

console.log("[helmet-demo] Firebase initialized");

// 3) DOM 요소
const videoInput = document.getElementById("videoInput");
const uploadBtn = document.getElementById("uploadBtn");
const statusEl = document.getElementById("status");
const previewVideo = document.getElementById("previewVideo");

const loadSampleEventsBtn = document.getElementById("loadSampleEventsBtn");
const eventsContainer = document.getElementById("eventsContainer");

// 방어코드: 요소 못 찾으면 바로 콘솔에 표시
if (!videoInput || !uploadBtn || !statusEl || !previewVideo) {
  console.error("[helmet-demo] DOM 요소를 찾지 못했습니다. index.html과 id를 확인하세요.");
}

// 4) 업로드 로직
uploadBtn.addEventListener("click", () => {
  console.log("[helmet-demo] 업로드 버튼 클릭됨");

  const file = videoInput.files?.[0];
  console.log("[helmet-demo] 선택된 파일:", file);

  if (!file) {
    alert("먼저 업로드할 영상 파일(mp4)을 선택해 주세요.");
    return;
  }

  uploadBtn.disabled = true;
  statusEl.textContent = "업로드 시작...";
  console.log("[helmet-demo] 업로드 시작");

  const filePath = `videos/${Date.now()}_${file.name}`;
  const storageRef = ref(storage, filePath);

  const uploadTask = uploadBytesResumable(storageRef, file);
  console.log("[helmet-demo] uploadTask 생성됨:", uploadTask);

  uploadTask.on(
    "state_changed",
    (snapshot) => {
      const { bytesTransferred, totalBytes, state } = snapshot;
      const percent = totalBytes
        ? Math.round((bytesTransferred / totalBytes) * 100)
        : 0;

      console.log(
        "[helmet-demo] state_changed:",
        "bytesTransferred =", bytesTransferred,
        "totalBytes =", totalBytes,
        "state =", state,
        "percent =", percent
      );

      statusEl.textContent = `업로드 중... ${percent}%`;
    },
    (error) => {
      console.error("[helmet-demo] 업로드 오류:", error.code, error.message);
      statusEl.textContent = `업로드 중 오류 발생: ${error.code}`;
      uploadBtn.disabled = false;
    },
    async () => {
      console.log("[helmet-demo] 업로드 완료, 다운로드 URL 요청");

      try {
        const downloadURL = await getDownloadURL(uploadTask.snapshot.ref);
        console.log("[helmet-demo] downloadURL:", downloadURL);

        statusEl.textContent = "업로드 완료! 영상 미리보기를 불러옵니다.";

        previewVideo.src = downloadURL;
        previewVideo.load();
        previewVideo
          .play()
          .catch(() => {
            statusEl.textContent +=
              " (브라우저 자동재생이 막혀 있으면 play 버튼을 눌러주세요)";
          });
      } catch (e) {
        console.error("[helmet-demo] URL 가져오기 오류:", e);
        statusEl.textContent = "업로드는 완료됐지만, 다운로드 URL을 가져오는데 실패했습니다.";
      } finally {
        uploadBtn.disabled = false;
      }
    }
  );
});

// 5) 샘플 YOLO 이벤트 JSON

const SAMPLE_EVENTS = [
  {
    id: "evt_001",
    timestamp_sec: 2.87,
    frame_index: 69,
    class_name: "no_helmet",
    confidence: 0.93,
    bbox: [320, 180, 410, 350],
    worker_id: "worker_A",
  },
  {
    id: "evt_002",
    timestamp_sec: 11.42,
    frame_index: 274,
    class_name: "no_helmet",
    confidence: 0.88,
    bbox: [140, 210, 230, 360],
    worker_id: "worker_B",
  },
];

function renderEvents(events) {
  eventsContainer.innerHTML = "";

  if (!events || events.length === 0) {
    eventsContainer.textContent = "표시할 이벤트가 없습니다.";
    return;
  }

  for (const evt of events) {
    const item = document.createElement("div");
    item.className = "event-item";

    const header = document.createElement("div");
    header.className = "event-header";
    header.textContent = `이벤트 ID: ${evt.id} ｜ 클래스: ${evt.class_name}`;

    const body = document.createElement("div");
    body.innerHTML = `
      <div>시간: <strong>${evt.timestamp_sec.toFixed(2)}s</strong> (프레임: ${evt.frame_index})</div>
      <div>신뢰도: <strong>${(evt.confidence * 100).toFixed(1)}%</strong></div>
      <div>Bounding Box: <code>[${evt.bbox.join(", ")}]</code></div>
      <div>작업자 ID: ${evt.worker_id}</div>
    `;

    item.appendChild(header);
    item.appendChild(body);
    eventsContainer.appendChild(item);
  }
}

loadSampleEventsBtn.addEventListener("click", () => {
  console.log("[helmet-demo] 샘플 이벤트 렌더링");
  renderEvents(SAMPLE_EVENTS);
});

