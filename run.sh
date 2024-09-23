#!/bin/bash

# 输出操作信息
echo "Updating package list and installing dependencies..."
# 更新系统包并安装 ffmpeg
sudo apt update && sudo apt install -y ffmpeg

# 安装 torch 和 pyannote.audio 的依赖
echo "Installing torch and pyannote.audio..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pyannote.audio

# 安装 faster-whisper
echo "Installing faster-whisper from GitHub..."
pip install git+https://github.com/openai/whisper.git

# 检查是否存在 audio 文件夹
if [ ! -d "audio" ]; then
    echo "The 'audio' folder does not exist. Please create an 'audio' folder and add your audio files."
    exit 1
fi

# 检查是否有 Python 脚本
if [ ! -f "transcription_script.py" ]; then
    echo "Python script 'transcription_script.py' not found!"
    exit 1
fi

# 确保脚本文件有执行权限
chmod +x transcription_script.py

# 运行 Python 转录脚本
echo "Running transcription script..."
python transcription_script.py

# 完成
echo "All tasks completed successfully."
