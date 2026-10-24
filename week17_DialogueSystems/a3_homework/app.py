#!/usr/bin/env python3
"""任务型对话 Web 服务：前端 ChatGPT 风格，后端调用 dl.DialogueSystem。"""

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dl import DialogueSystem

BASE_DIR = Path(__file__).resolve().parent
os.chdir(BASE_DIR)

STATIC_DIR = BASE_DIR / "static"
GREETING = (
    "您好！我是任务型对话助手。\n"
    "您可以说「我要看电影」或「我要买衣服」开始；"
    "多轮追问时会自动保留上下文（重听策略）。"
)

app = FastAPI(title="任务型多轮对话", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_dialogue_system = DialogueSystem()
_sessions: dict[str, dict] = {}


def _initial_memory() -> dict:
    return {
        "available_nodes": [
            "scenario-买衣服_node1",
            "scenario-看电影_node1",
        ]
    }


def _get_or_create_session(session_id: str | None) -> tuple[str, dict, bool]:
    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id], False
    new_id = str(uuid.uuid4())
    memory = _initial_memory()
    _sessions[new_id] = memory
    return new_id, memory, True


def _memory_debug(memory: dict) -> dict:
    return {
        "hit_node": memory.get("hit_node"),
        "active_node": memory.get("active_node"),
        "require_slot": memory.get("require_slot"),
        "require_repeat": memory.get("require_repeat"),
        "policy": memory.get("policy"),
        "filled_slots": {k: memory[k] for k in memory if k.startswith("#")},
    }


class SessionResponse(BaseModel):
    session_id: str
    greeting: str


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    reset: bool = False
    debug: dict | None = None


class ResetRequest(BaseModel):
    session_id: str


class ResetResponse(BaseModel):
    session_id: str
    reply: str


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/session", response_model=SessionResponse)
async def create_session() -> SessionResponse:
    session_id, _, _ = _get_or_create_session(None)
    return SessionResponse(session_id=session_id, greeting=GREETING)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    session_id, memory, session_new = _get_or_create_session(req.session_id)
    if req.session_id and session_new:
        raise HTTPException(
            status_code=404,
            detail="会话已失效（服务可能重启过），请点击「新对话」重新开始。",
        )

    if message.lower() in {"reset", "重新开始", "新对话"}:
        memory.clear()
        memory.update(_initial_memory())
        return ChatResponse(
            session_id=session_id,
            reply="【已重置对话】\n\n" + GREETING,
            reset=True,
            debug=_memory_debug(memory),
        )

    memory = _dialogue_system.run(message, memory)
    _sessions[session_id] = memory
    reply = memory.get("response", "")
    if memory.get("pending_action"):
        actions = memory["pending_action"]
        if isinstance(actions, list):
            action_text = "；".join(str(a) for a in actions)
        else:
            action_text = str(actions)
        reply = f"{reply}\n【系统动作】{action_text}"
    return ChatResponse(
        session_id=session_id,
        reply=reply,
        debug=_memory_debug(memory),
    )


@app.post("/api/reset", response_model=ResetResponse)
async def reset_session(req: ResetRequest) -> ResetResponse:
    if req.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    memory = _initial_memory()
    _sessions[req.session_id] = memory
    return ResetResponse(session_id=req.session_id, reply=GREETING)


if __name__ == "__main__":
    import uvicorn

    # reload=True 会清空内存会话，导致多轮上下文丢失；本地调试可改为 reload=True
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
