#!/bin/zsh
set -e
cd "$(dirname "$0")"

if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "未找到 FFmpeg。请先在终端运行：brew install ffmpeg"
  echo "安装完成后，再双击启动.command。"
  read -r "?按回车退出…"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "未找到 uv（Python 环境管理工具）。请先安装 uv，然后重新双击启动。"
  echo "安装说明：https://docs.astral.sh/uv/getting-started/installation/"
  read -r "?按回车退出…"
  exit 1
fi

if [ ! -f .venv/.setup-complete ]; then
  echo "正在准备声音生成环境（首次启动需要联网下载依赖）…"
  if ! uv sync --project . --python 3.12 --locked --no-install-project; then
    echo "声音生成环境安装失败。请检查上方提示、网络和磁盘空间，然后重新启动。"
    read -r "?按回车退出…"
    exit 1
  fi
  touch .venv/.setup-complete
fi

if [ ! -f asr/.venv/.setup-complete ]; then
  echo "正在准备本机转写环境（首次启动需要联网下载依赖）…"
  if ! uv sync --project asr --python 3.12 --locked --no-install-project; then
    echo "转写环境安装失败。请检查上方提示、网络和磁盘空间，然后重新启动。"
    read -r "?按回车退出…"
    exit 1
  fi
  touch asr/.venv/.setup-complete
fi

exec .venv/bin/python app.py
