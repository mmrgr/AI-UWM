from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
from threading import Timer
import webbrowser


ROOT = Path(__file__).resolve().parents[2]
FRONTEND = ROOT / "frontend"
DIST_INDEX = FRONTEND / "dist" / "index.html"


def _latest_source_time() -> float:
    candidates = [
        FRONTEND / "index.html",
        FRONTEND / "package.json",
        FRONTEND / "package-lock.json",
        FRONTEND / "vite.config.ts",
        *list((FRONTEND / "src").rglob("*")),
    ]
    files = [path for path in candidates if path.is_file()]
    return max((path.stat().st_mtime for path in files), default=0.0)


def _run_npm(command: str) -> None:
    npm = shutil.which("npm")
    if npm is None:
        raise RuntimeError(
            "未找到 Node.js/npm。请安装 Node.js 20 或更高版本后重新双击启动。"
        )
    subprocess.run(
        f'"{npm}" {command}',
        cwd=FRONTEND,
        shell=True,
        check=True,
    )


def _ensure_frontend() -> None:
    dependencies = FRONTEND / "node_modules"
    if not dependencies.is_dir():
        print("[1/3] 首次运行：正在安装前端依赖……", flush=True)
        _run_npm("install")
    if (
        not DIST_INDEX.is_file()
        or DIST_INDEX.stat().st_mtime < _latest_source_time()
    ):
        print("[2/3] 正在构建前端……", flush=True)
        _run_npm("run build")
    else:
        print("[1/3] 前端已经是最新版本。", flush=True)


def _ensure_python_dependencies() -> None:
    missing = [
        name
        for name in ("fastapi", "uvicorn")
        if importlib.util.find_spec(name) is None
    ]
    if not missing:
        return
    print("[2/3] 首次运行：正在安装 Python 服务依赖……", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", f"{ROOT}[studio]"],
        cwd=ROOT,
        check=True,
    )


def main() -> None:
    try:
        _ensure_frontend()
        _ensure_python_dependencies()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"\n启动准备失败：{exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    import uvicorn

    from .api import create_app

    host = "127.0.0.1"
    port = int(os.environ.get("WATERMET2_STUDIO_PORT", "8000"))
    url = f"http://{host}:{port}"
    print(f"[3/3] WaterMet2 Studio 已启动：{url}", flush=True)
    print("关闭此窗口即可停止系统。", flush=True)
    if os.environ.get("WATERMET2_STUDIO_NO_BROWSER") != "1":
        Timer(1.0, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(serve_frontend=True), host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
