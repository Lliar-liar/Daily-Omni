import os
import subprocess
import torchaudio
from variables import base_path,csv_path
max_segments=3
base_dir=base_path
def get_video_duration(file_path):
    command = [
        "ffprobe",
        "-i", file_path,
        "-show_entries", "format=duration",
        "-v", "quiet",
        "-of", "csv=p=0"
    ]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return float(result.stdout.strip())

def segment_audio(file_path, output_dir, segment_duration=10, max_segments=3):
    """
    将音频文件分段并保存到指定目录，最多保留 max_segments 个分段。

    :param file_path: 原始音频文件路径
    :param output_dir: 保存分段音频的目录
    :param segment_duration: 每段音频的时长（秒）
    :param max_segments: 最大分段数量
    """
    # 获取原始文件名（不带路径和扩展名）
    base_name = os.path.basename(file_path)
    file_name_without_ext = os.path.splitext(base_name)[0]

    # 加载音频
    waveform, sample_rate = torchaudio.load(file_path)

    # 每段的采样点数
    segment_samples = int(segment_duration * sample_rate)

    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 分段处理
    num_segments = 0
    for start_sample in range(0, waveform.size(1), segment_samples):
        if num_segments >= max_segments:
            break

        end_sample = min(start_sample + segment_samples, waveform.size(1))
        segment = waveform[:, start_sample:end_sample]

        # 保存分段文件，命名为原文件名_编号.wav
        segment_file = os.path.join(output_dir, f"{file_name_without_ext}_{num_segments}.wav")
        torchaudio.save(segment_file, segment, sample_rate)
        print(f"Saved segment: {segment_file}")
        num_segments += 1

    print(f"Completed segmentation for {file_path}: {num_segments} segments.")



def segment_video_ffmpeg(file_path, output_dir, segment_duration=10, max_segments=3):
    """
    使用 ffmpeg 分段视频文件并保存到指定目录，最多保留 max_segments 个分段。

    :param file_path: 原始视频文件路径
    :param output_dir: 保存分段视频的目录
    :param segment_duration: 每段视频的时长（秒）
    :param max_segments: 最大分段数量
    """
    base_name = os.path.basename(file_path)
    file_name_without_ext = os.path.splitext(base_name)[0]

    # 获取视频总时长
    try:
        total_duration = get_video_duration(file_path)
    except Exception as e:
        print(f"Error getting duration for video {file_path}: {e}")
        return
    print(total_duration)
    os.makedirs(output_dir, exist_ok=True)

    num_segments = 0
    for start_time in range(0, int(total_duration), segment_duration):
        if num_segments >= max_segments:
            break

        end_time = min(start_time + segment_duration, total_duration)
        output_file = os.path.join(output_dir, f"{file_name_without_ext}_{num_segments}.mp4")

        try:
            
                
            result=subprocess.run(
                [
                    "ffmpeg",
                    "-y",  
                    # "-hwaccel cuvid", 
                    "-i", file_path,
                    "-ss", str(start_time),
                    "-to", str(end_time),
                    "-c:v", "libx264",   
                    "-preset", "slow",    
                    "-crf", "22",
                    "-pix_fmt", "yuv420p",   
                    "-c:a", "copy",   
                    output_file
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            
            print(result.stderr)
            print(result.stdout)
            print(f"Saved segment: {output_file}")
        except Exception as e:
            print(f"Error saving segment {output_file}: {e}")

        num_segments += 1

    print(f"Completed segmentation for {file_path}: {num_segments} segments.")


def process_csv_and_segment_videos_ffmpeg(csv_path,base_dir, max_segments=3):
    import csv

    with open(csv_path, mode="r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            video_path=os.path.join(base_dir,row['video_id'],f"{row['video_id']}_video.mp4")
            audio_path=os.path.join(base_dir,row['video_id'],f"{row['video_id']}_audio.wav")
            output_dir=os.path.join(base_dir,row['video_id'])
            if row['duration']=='30':
                segment_duration=10
            else:
                segment_duration=20
            print(segment_duration)
            segment_video_ffmpeg(file_path=video_path,output_dir=output_dir,segment_duration=segment_duration,max_segments=max_segments)
            segment_audio(file_path=audio_path,output_dir=output_dir,segment_duration=segment_duration,max_segments=max_segments)

if __name__ == "__main__":
    process_csv_and_segment_videos_ffmpeg(csv_path, base_dir=base_dir, max_segments=max_segments)
