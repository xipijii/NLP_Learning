#!/usr/bin/env python3
"""任务型对话 Web 服务（FastAPI）。"""

import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from dialogue_manager import DialogueManager

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="任务型多轮对话", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_sessions: dict[str, DialogueManager] = {}


def _get_or_create_session(session_id: str | None) -> tuple[str, DialogueManager]:
    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]
    new_id = str(uuid.uuid4())
    dm = DialogueManager(scenario_dir=BASE_DIR)
    _sessions[new_id] = dm
    return new_id, dm


class SessionResponse(BaseModel):
    session_id: str
    greeting: str


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(..., min_length=0)


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    reset: bool = False


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
    session_id, dm = _get_or_create_session(None)
    greeting = dm.turn("")
    return SessionResponse(session_id=session_id, greeting=greeting)


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="消息不能为空")

    session_id, dm = _get_or_create_session(req.session_id)

    if message.lower() == "reset":
        dm.reset()
        reply = "【已重置对话】\n\n" + dm.turn("")
        return ChatResponse(session_id=session_id, reply=reply, reset=True)

    reply = dm.turn(message)
    return ChatResponse(session_id=session_id, reply=reply)


@app.post("/api/reset", response_model=ResetResponse)
async def reset_session(req: ResetRequest) -> ResetResponse:
    if req.session_id not in _sessions:
        raise HTTPException(status_code=404, detail="会话不存在或已过期")
    dm = _sessions[req.session_id]
    dm.reset()
    return ResetResponse(session_id=req.session_id, reply=dm.turn(""))
