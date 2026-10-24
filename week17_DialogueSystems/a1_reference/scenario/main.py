#!/usr/bin/env python3
"""任务型多轮对话：CLI 或 Web 服务。"""

import argparse

from dialogue_manager import DialogueManager


def run_web(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    print(f"Web 界面: http://{host}:{port}/")
    uvicorn.run("app:app", host=host, port=port, reload=False)


def run_cli() -> None:
    dm = DialogueManager()
    print("=" * 50)
    print("任务型多轮对话系统（NLU / DST / PM / NLG）")
    print("输入 quit 退出，输入 reset 重新开始")
    print("示例：我要看电影 | 8点 流浪地球 | 来个爆米花")
    print("示例：我要买衣服 | 短袖 | 红色 | L | 可以分期付款吗")
    print("=" * 50)
    print(dm.turn(""))
    while True:
        try:
            user = input("\n用户> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not user:
            continue
        if user.lower() in {"quit", "exit", "q"}:
            print("再见。")
            break
        if user.lower() == "reset":
            dm.reset()
            print("【已重置对话状态】")
            print(dm.turn(""))
            continue
        bot = dm.turn(user)
        print(f"助手> {bot}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="任务型多轮对话系统")
    parser.add_argument(
        "--web",
        action="store_true",
        help="启动网页端服务（ChatGPT 风格界面）",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Web 服务地址")
    parser.add_argument("--port", type=int, default=8000, help="Web 服务端口")
    args = parser.parse_args()
    if args.web:
        run_web(args.host, args.port)
    else:
        run_cli()
