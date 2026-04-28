"""
一键启动 星枢反欺诈平台 开发环境
用法：python start_dev.py
按 Ctrl+C 同时关闭前后端。
"""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

SYSTEM_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = SYSTEM_ROOT / "backend"
FRONTEND_DIR = SYSTEM_ROOT / "frontend"

BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = 8001
FRONTEND_PORT = 5173


def check_port(port: int) -> bool:
    """检查端口是否已被占用。"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((BACKEND_HOST, port)) == 0


def main() -> None:
    procs: list[subprocess.Popen] = []

    # ---------- 后端 ----------
    backend_port = BACKEND_PORT
    if check_port(backend_port):
        print(f"[信息] 端口 {backend_port} 已被占用，尝试 {backend_port + 1}")
        backend_port = BACKEND_PORT + 1
        if check_port(backend_port):
            print(f"[错误] 端口 {backend_port} 也被占用，请手动释放后重试。")
            sys.exit(1)

    backend_cmd = [
        sys.executable, "-m", "uvicorn", "app.main:app",
        "--app-dir", str(BACKEND_DIR),
        "--host", BACKEND_HOST,
        "--port", str(backend_port),
        "--reload",
    ]

    # ---------- 前端 ----------
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    frontend_cmd = [npm, "run", "dev"]
    frontend_env = os.environ.copy()
    frontend_env["VITE_API_TARGET"] = f"http://{BACKEND_HOST}:{backend_port}"

    print("=" * 56)
    print("  星枢反欺诈平台 · 开发环境启动")
    print("=" * 56)
    print(f"  后端  →  http://{BACKEND_HOST}:{backend_port}")
    print(f"  前端  →  http://{BACKEND_HOST}:{FRONTEND_PORT}  (若被占用 Vite 会自动切换)")
    print(f"  按 Ctrl+C 同时关闭前后端")
    print("=" * 56)
    print()

    try:
        backend_proc = subprocess.Popen(backend_cmd)
        procs.append(backend_proc)

        time.sleep(1)

        frontend_proc = subprocess.Popen(frontend_cmd, cwd=str(FRONTEND_DIR), env=frontend_env)
        procs.append(frontend_proc)

        for proc in procs:
            proc.wait()

    except KeyboardInterrupt:
        print("\n[信息] 正在关闭所有服务...")

    finally:
        for proc in procs:
            try:
                if sys.platform == "win32":
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    proc.terminate()
            except OSError:
                pass

        for proc in procs:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()

        print("[完成] 开发环境已关闭。")


if __name__ == "__main__":
    main()
