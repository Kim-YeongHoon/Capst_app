// app.js

// 1) Firebase 모듈 import (CDN, ES module)
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.14.0/firebase-app.js";
import {
  getStorage,
  ref,
  uploadBytesResumable,
  getDownloadURL,
} from "https://www.gstatic.com/firebasejs/10.14.0/firebase-storage.js";

// 2) Firebase 초기화 ---------------------------------------
// ⚠️ 이 부분은 네 Firebase 콘솔에서 받은 값으로 채워 넣기
// 프로젝트 ID: helmet-demo (이미 알고 있는 값)
// 나머지는 콘솔 → 프로젝트 설정 → 내 앱(firebaseConfig)에서 복붙
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

// 3) DOM 요소 캐싱 -----------------------------------------
const videoInput = document.getElementById("videoInput");
const uploadBtn = document.getElementById("uploadBtn");
const statusEl = document.getElementById("status");
const previewVideo = document.getElementById("previewVideo");

const loadSampleEventsBtn = document.getElementById("loadSampleEventsBtn");
const eventsContainer = document.getElementById("eventsContainer");

// 4) mp4 업로드 → 퍼센트 표시 → getDownloadURL → <video> 재생 -----

uploadBtn.addEventListener("click", () => {
  const file = videoInput.files?.[0];

  if (!file) {
    alert("먼저 업로드할 영상 파일(mp4)을 선택해 주세요.");
    return;
  }

  // 버튼 잠그고 상태 초기화
  uploadBtn.disabled = true;
  statusEl.textContent = "업로드 시작...";

  // Storage 경로: videos/타임스탬프_원본파일명
  const filePath = `videos/${Date.now()}_${file.name}`;
  const storageRef = ref(storage, filePath);

  const uploadTask = uploadBytesResumable(storageRef, file);

  uploadTask.on(
    "state_changed",
    (snapshot) => {
      const percent =
        (snapshot.bytesTransferred / snapshot.totalBytes) * 100;
      const rounded = Math.round(percent);
      statusEl.textContent = `업로드 중... ${rounded}%`;
    },
    (error) => {
      console.error("업로드 오류:", error);
      statusEl.textContent = "업로드 중 오류 발생. 콘솔을 확인해 주세요.";
      uploadBtn.disabled = false;
    },
    async () => {
      // 업로드 완료
      try {
        const downloadURL = await getDownloadURL(uploadTask.snapshot.ref);
        statusEl.textContent = "업로드 완료! 영상 미리보기를 불러옵니다.";

        // <video>에 다운로드 URL 세팅
        previewVideo.src = downloadURL;
        previewVideo.load();
        previewVideo
          .play()
          .catch(() => {
            statusEl.textContent +=
              " (브라우저 자동재생이 막혀 있으면 play 버튼을 눌러주세요)";
          });

        // 여기서 나중에: YOLO 서버에 downloadURL을 보내서 분석 요청 → JSON 수신 → renderEvents 호출
      } catch (e) {
        console.error("URL 가져오기 오류:", e);
        statusEl.textContent =
          "업로드는 완료됐지만, 다운로드 URL을 가져오는데 실패했습니다.";
      } finally {
        uploadBtn.disabled = false;
      }
    }
  );
});

// 5) YOLO 이벤트 JSON 샘플 + 렌더링 ---------------------------

// 나중에 실제 YOLO inference 결과 JSON 구조를 이런 느낌으로 맞추면 됨
const SAMPLE_EVENTS = [
  {
    id: "evt_001",
    timestamp_sec: 2.87,
    frame_index: 69,
    class_name: "no_helmet",
    confidence: 0.93,
    bbox: [320, 180, 410, 350], // [x1, y1, x2, y2]
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
  eventsContainer.innerHTML = ""; // 초기화

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
  renderEvents(SAMPLE_EVENTS);
});

