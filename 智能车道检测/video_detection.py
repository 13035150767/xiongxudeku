import cv2
import numpy as np
import os
from moviepy import VideoFileClip
from lane_detection import detect_lane_lines

TEST_VIDEOS_DIR = r"E:\PythonProject\计算机视觉\智能车道检测\test_videos"
TEST_OUTPUT_DIR = r"E:\PythonProject\计算机视觉\智能车道检测\结果"


def process_video_frame(frame):
    result, left_slope, right_slope = detect_lane_lines(frame)
    return result


def process_video(input_path, output_path, target_resolution=None):
    print(f"正在加载视频: {input_path}")

    clip = VideoFileClip(input_path)

    if target_resolution is not None:
        clip = clip.resize(target_resolution)
        print(f"视频分辨率已调整为: {target_resolution}")

    print("正在处理视频帧...")

    processed_clip = clip.image_transform(process_video_frame)

    processed_clip.write_videofile(output_path, codec='libx264', audio=False)

    print(f"视频处理完成! 输出文件: {output_path}")

    clip.close()
    processed_clip.close()


def process_video_with_progress(input_path, output_path, target_resolution=None):
    print(f"正在加载视频: {input_path}")

    clip = VideoFileClip(input_path)

    if target_resolution is not None:
        clip = clip.resize(target_resolution)
        print(f"视频分辨率已调整为: {target_resolution}")

    print(f"视频时长: {clip.duration}秒")
    print(f"视频帧率: {clip.fps}fps")
    print(f"视频分辨率: {clip.size}")

    from moviepy.editor import VideoFileClip
    import progressbar

    total_frames = int(clip.duration * clip.fps)

    print("正在处理视频帧...")

    widgets = ['处理中: ', progressbar.Percentage(), ' ',
               progressbar.Bar(marker='=', left='[', right=']'),
               ' ', progressbar.ETA()]

    pbar = progressbar.ProgressBar(widgets=widgets, maxval=total_frames).start()

    def process_frame_with_progress(frame, pbar):
        result = process_video_frame(frame)
        pbar.update(pbar.currval + 1)
        return result

    processed_clip = clip.fl(lambda gf: (gf(pbar) for gf in [lambda t: process_frame_with_progress(clip.get_frame(t), pbar)]),)
    processed_clip = clip.image_transform(process_video_frame)

    pbar.finish()

    processed_clip.write_videofile(output_path, codec='libx264', audio=False)

    print(f"视频处理完成! 输出文件: {output_path}")

    clip.close()
    processed_clip.close()


def process_video_simple(input_path, output_path, target_resolution=None):
    print(f"正在加载视频: {input_path}")

    clip = VideoFileClip(input_path)

    if target_resolution is not None:
        clip = clip.resize(target_resolution)
        print(f"视频分辨率已调整为: {target_resolution}")

    print(f"视频时长: {clip.duration}秒")
    print(f"视频帧率: {clip.fps}fps")
    print(f"视频分辨率: {clip.size}")

    print("正在处理视频帧，请稍候...")

    processed_clip = clip.image_transform(process_video_frame)

    processed_clip.write_videofile(output_path, codec='libx264', audio=False)

    print(f"视频处理完成! 输出文件: {output_path}")

    clip.close()
    processed_clip.close()


def process_all_test_videos():
    if not os.path.exists(TEST_VIDEOS_DIR):
        print(f"测试视频目录不存在: {TEST_VIDEOS_DIR}")
        return

    if not os.path.exists(TEST_OUTPUT_DIR):
        os.makedirs(TEST_OUTPUT_DIR)
        print(f"创建输出目录: {TEST_OUTPUT_DIR}")

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv')
    video_files = [f for f in os.listdir(TEST_VIDEOS_DIR) if f.lower().endswith(video_extensions)]

    if not video_files:
        print(f"测试视频目录中没有找到视频文件: {TEST_VIDEOS_DIR}")
        return

    print(f"找到 {len(video_files)} 个视频文件，开始处理...")

    for filename in video_files:
        input_path = os.path.join(TEST_VIDEOS_DIR, filename)
        output_path = os.path.join(TEST_OUTPUT_DIR, f"result_{filename}")

        try:
            clip = VideoFileClip(input_path)
            processed_clip = clip.image_transform(process_video_frame)
            processed_clip.write_videofile(output_path, codec='libx264', audio=False)
            clip.close()
            processed_clip.close()
            print(f"已处理: {filename} -> {os.path.basename(output_path)}")
        except Exception as e:
            print(f"处理视频失败 {filename}: {str(e)}")

    print("所有视频处理完成!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--batch":
            process_all_test_videos()
        else:
            input_path = sys.argv[1]
            output_path = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"

            target_resolution = None
            if len(sys.argv) > 3:
                resolution_str = sys.argv[3]
                if 'x' in resolution_str:
                    width, height = map(int, resolution_str.split('x'))
                    target_resolution = (height, width)

            process_video_simple(input_path, output_path, target_resolution)
    else:
        print("使用方法:")
        print("  python video_detection.py <输入视频路径> [输出视频路径] [分辨率]")
        print("  python video_detection.py --batch")
        print("\n示例:")
        print("  python video_detection.py input.mp4 output.mp4")
        print("  python video_detection.py input.mp4 output.mp4 720x1280")
        print("  python video_detection.py --batch")