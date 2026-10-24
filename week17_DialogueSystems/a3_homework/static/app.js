const messagesEl = document.getElementById("messages");
const chatForm = document.getElementById("chatForm");
const userInput = document.getElementById("userInput");
const btnSend = document.getElementById("btnSend");
const btnNewChat = document.getElementById("btnNewChat");
const statusEl = document.getElementById("status");

const GPT_LOGO = `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 2a7 7 0 0 1 7 7c0 2.6-1.4 4.9-3.5 6.1L19 22h-3.2l-2.2-4.4h-3.2L8.2 22H5l3.5-6.9A7 7 0 0 1 12 2z"/></svg>`;

const USER_AVATAR = `<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 12c2.7 0 4.8-2.1 4.8-4.8S14.7 2.4 12 2.4 7.2 4.5 7.2 7.2 9.3 12 12 12zm0 2.4c-3.2 0-9.6 1.6-9.6 4.8v2.4h19.2v-2.4c0-3.2-6.4-4.8-9.6-4.8z"/></svg>`;

let sessionId = null;
let isLoading = false;

const SUGGESTIONS = [
  "我要看电影",
  "我要买衣服",
  "我要买衣服 长袖 红色 L",
];

function setLoading(loading) {
  isLoading = loading;
  btnSend.disabled = loading || !userInput.value.trim();
  userInput.disabled = loading;
  statusEl.textContent = loading
    ? "正在处理…"
    : "输入后由任务型对话引擎（NLU · DST · DPO · NLG）实时回复";
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  });
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

function formatAssistantHtml(content) {
  return content
    .split("\n")
    .map((line) => {
      const safe = escapeHtml(line);
      if (line.startsWith("【系统动作】")) {
        return `<span class="msg-action">${safe}</span>`;
      }
      return safe;
    })
    .join("<br>");
}

function renderWelcome() {
  messagesEl.innerHTML = `
    <div class="welcome-block">
      <h2>今天我能帮你什么？</h2>
      <div class="chips">
        ${SUGGESTIONS.map(
          (s) =>
            `<button type="button" class="chip" data-text="${escapeHtml(s)}">${escapeHtml(s)}</button>`
        ).join("")}
      </div>
    </div>`;
  messagesEl.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      userInput.value = chip.dataset.text;
      userInput.dispatchEvent(new Event("input"));
      userInput.focus();
    });
  });
}

function appendMessage(role, content) {
  if (role === "user") {
    messagesEl.querySelector(".welcome-block")?.remove();
  }

  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  if (role === "user") {
    row.innerHTML = `
      <div class="message-inner">
        <div class="avatar avatar-user" role="img" aria-label="你">${USER_AVATAR}</div>
        <div class="message-content">${escapeHtml(content)}</div>
      </div>`;
  } else {
    row.innerHTML = `
      <div class="message-inner">
        <div class="avatar" role="img" aria-label="助手">${GPT_LOGO}</div>
        <div class="message-content">${formatAssistantHtml(content)}</div>
      </div>`;
  }

  messagesEl.appendChild(row);
  scrollToBottom();
  return row;
}

function showTyping() {
  const row = document.createElement("div");
  row.className = "message-row assistant typing";
  row.id = "typingIndicator";
  row.innerHTML = `
    <div class="message-inner">
      <div class="avatar" role="img" aria-label="助手">${GPT_LOGO}</div>
      <div class="message-content typing-dots"><span></span><span></span><span></span></div>
    </div>`;
  messagesEl.appendChild(row);
  scrollToBottom();
}

function hideTyping() {
  document.getElementById("typingIndicator")?.remove();
}

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `请求失败 (${res.status})`);
  }
  return res.json();
}

async function initSession(showGreeting = true) {
  setLoading(true);
  try {
    const data = await api("/api/session", {});
    sessionId = data.session_id;
    renderWelcome();
    if (showGreeting && data.greeting) {
      messagesEl.querySelector(".welcome-block")?.remove();
      appendMessage("assistant", data.greeting);
    }
  } finally {
    setLoading(false);
  }
}

async function sendMessage(text) {
  const trimmed = text.trim();
  if (!trimmed || isLoading) return;

  appendMessage("user", trimmed);
  userInput.value = "";
  resizeInput();

  setLoading(true);
  showTyping();
  try {
    if (!sessionId) await initSession(false);
    const data = await api("/api/chat", {
      session_id: sessionId,
      message: trimmed,
    });
    sessionId = data.session_id;
    hideTyping();
    appendMessage("assistant", data.reply || "");
  } catch (e) {
    hideTyping();
    appendMessage("assistant", `抱歉，服务出错：${e.message}`);
  } finally {
    setLoading(false);
    userInput.focus();
  }
}

function resizeInput() {
  userInput.style.height = "auto";
  userInput.style.height = Math.min(userInput.scrollHeight, 200) + "px";
  btnSend.disabled = isLoading || !userInput.value.trim();
}

chatForm.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage(userInput.value);
});

userInput.addEventListener("input", resizeInput);

userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!btnSend.disabled) chatForm.requestSubmit();
  }
});

btnNewChat.addEventListener("click", async () => {
  if (sessionId) {
    try {
      await api("/api/reset", { session_id: sessionId });
    } catch {
      sessionId = null;
    }
  }
  messagesEl.innerHTML = "";
  await initSession(true);
  userInput.focus();
});

initSession(true);
userInput.focus();
resizeInput();
