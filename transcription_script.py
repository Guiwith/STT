import os
from pyannote.audio import Pipeline
from faster_whisper import WhisperModel
from pydub import AudioSegment

# 函数：提取音频片段
def extract_audio_segment(audio_path, start_time, end_time):
    audio = AudioSegment.from_wav(audio_path)
    return audio[start_time * 1000:end_time * 1000]  # pydub使用毫秒

# 获取脚本所在的目录路径
current_dir = os.path.dirname(os.path.abspath(__file__))

# 定义音频文件夹路径
audio_dir = os.path.join(current_dir, 'audio')

# 检查音频文件夹是否存在
if not os.path.exists(audio_dir):
    print(f"Audio folder not found: {audio_dir}")
    exit(1)

# 获取音频文件夹中的所有音频文件
audio_files = [f for f in os.listdir(audio_dir) if f.endswith('.wav') or f.endswith('.mp3')]

if not audio_files:
    print(f"No audio files found in {audio_dir}")
    exit(1)

# Step 1: 初始化 pyannote 语者分离 Pipeline
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

# Step 2: 初始化 faster-whisper 模型
model = WhisperModel("large-v2")

# Step 3: 遍历 audio 文件夹中的每个音频文件
for audio_file in audio_files:
    audio_path = os.path.join(audio_dir, audio_file)
    print(f"Processing file: {audio_path}")
    
    # 执行语者分离
    diarization = pipeline(audio_path)
    
    # 保存转录结果的字典
    transcription_results = {}

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        # 提取每个发言人的音频片段
        speaker_segment = extract_audio_segment(audio_path, turn.start, turn.end)
        temp_segment_path = os.path.join(current_dir, "temp_speaker_segment.wav")
        speaker_segment.export(temp_segment_path, format="wav")

        # 使用 faster-whisper 转录该片段
        segments, info = model.transcribe(temp_segment_path)

        # 将转录结果保存到字典
        if speaker not in transcription_results:
            transcription_results[speaker] = []
        for segment in segments:
            transcription_results[speaker].append(segment.text)

    # Step 4: 输出或保存转录结果
    output_file = os.path.join(current_dir, f"{os.path.splitext(audio_file)[0]}_transcription.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        for speaker, texts in transcription_results.items():
            f.write(f"Speaker {speaker}:\n")
            for text in texts:
                f.write(f"{text}\n")
            f.write("\n")  # 每个发言人分隔

    print(f"Transcription for {audio_file} saved to {output_file}")

print("所有音频文件处理完成。")
