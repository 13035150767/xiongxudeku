import cv2
import numpy as np
import os
import sys
import logging
import traceback
from datetime import datetime
from moviepy import VideoFileClip
from lane_detection import detect_lane_lines

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from utils.deduplicator import BatchDeduplicator, DedupStrategy
TEST_VIDEOS_DIR = os.path.join(SCRIPT_DIR, "test_videos")
RESULT_DIR = os.path.join(SCRIPT_DIR, "结果")

video_detection_counter = 0

logger = logging.getLogger(__name__)


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


def generate_video_output_filename(original_filename, prefix="result", include_timestamp=True, include_counter=True):
    global video_detection_counter
    
    name, ext = os.path.splitext(original_filename)
    parts = [prefix]
    
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(timestamp)
    
    if include_counter:
        video_detection_counter += 1
        parts.append(f"ID{video_detection_counter:04d}")
    
    parts.append(name)
    return "_".join(parts) + ext


def ensure_result_directory():
    try:
        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)
            logger.info(f"✓ 创建结果目录成功: {RESULT_DIR}")
        return True
    except PermissionError as e:
        logger.error(f"✗ 权限不足，无法创建结果目录: {RESULT_DIR}")
        logger.error(f"  错误详情: {str(e)}")
        return False
    except OSError as e:
        logger.error(f"✗ 创建结果目录失败: {RESULT_DIR}")
        logger.error(f"  错误详情: {str(e)}")
        return False


def process_all_test_videos():
    global video_detection_counter
    
    logger.info("=" * 60)
    logger.info("开始批量处理测试视频")
    logger.info(f"脚本目录: {SCRIPT_DIR}")
    logger.info(f"测试视频目录: {TEST_VIDEOS_DIR}")
    logger.info(f"结果保存目录: {RESULT_DIR}")
    
    if not os.path.exists(TEST_VIDEOS_DIR):
        logger.error(f"✗ 测试视频目录不存在: {TEST_VIDEOS_DIR}")
        return False
    
    if not ensure_result_directory():
        logger.error("✗ 无法创建结果目录，批量处理终止")
        return False
    
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')
    try:
        all_files = os.listdir(TEST_VIDEOS_DIR)
        video_files = [f for f in all_files if f.lower().endswith(video_extensions)]
    except PermissionError as e:
        logger.error(f"✗ 权限不足，无法访问测试视频目录: {TEST_VIDEOS_DIR}")
        logger.error(f"  错误详情: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"✗ 读取测试视频目录失败: {TEST_VIDEOS_DIR}")
        logger.error(f"  错误详情: {str(e)}")
        return False
    
    if not video_files:
        logger.warning(f"⚠ 测试视频目录中没有找到视频文件: {TEST_VIDEOS_DIR}")
        logger.info(f"  支持的视频格式: {', '.join(video_extensions)}")
        return False
    
    logger.info(f"✓ 找到 {len(video_files)} 个视频文件")
    logger.info("-" * 60)
    
    dedup = BatchDeduplicator(strategy=DedupStrategy.FILENAME)
    
    video_detection_counter = 0
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    for idx, filename in enumerate(video_files, 1):
        logger.info(f"\n[{idx}/{len(video_files)}] 处理文件: {filename}")
        
        should_process, _ = dedup.should_process_file(filename)
        if not should_process:
            skipped_count += 1
            logger.info(f"  ⚠ 跳过重复文件: {filename}")
            continue
        
        input_path = os.path.join(TEST_VIDEOS_DIR, filename)
        output_filename = generate_video_output_filename(filename)
        output_path = os.path.join(RESULT_DIR, output_filename)
        
        if dedup.is_output_duplicate(output_path):
            skipped_count += 1
            continue
        
        try:
            clip = VideoFileClip(input_path)
            logger.info(f"  视频时长: {clip.duration:.1f}秒, 帧率: {clip.fps}fps, 分辨率: {clip.size}")
            
            processed_clip = clip.image_transform(process_video_frame)
            processed_clip.write_videofile(output_path, codec='libx264', audio=False, logger=None)
            
            clip.close()
            processed_clip.close()
            
            if os.path.exists(output_path):
                file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                logger.info(f"✓ 保存成功: {output_path}")
                logger.info(f"  文件大小: {file_size_mb:.2f} MB")
                success_count += 1
            else:
                logger.error(f"✗ 保存失败: 文件未创建 - {output_path}")
                fail_count += 1
                
        except PermissionError as e:
            fail_count += 1
            logger.error(f"  ✗ 权限不足: {str(e)}")
        except Exception as e:
            fail_count += 1
            logger.error(f"  ✗ 处理失败: {str(e)}")
            logger.error(f"  堆栈跟踪:\n{traceback.format_exc()}")
    
    dedup.log_stats()
    
    logger.info("\n" + "=" * 60)
    logger.info("批量处理完成!")
    logger.info(f"  总计: {len(video_files)} 个文件")
    logger.info(f"  成功: {success_count} 个")
    logger.info(f"  失败: {fail_count} 个")
    logger.info(f"  跳过(重复): {skipped_count} 个")
    logger.info(f"  结果保存位置: {RESULT_DIR}")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("智能车道检测系统 - 视频处理模块")
    logger.info("=" * 60)
    
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
        logger.info("\n使用方法:")
        logger.info("  python video_detection.py <输入视频路径> [输出视频路径] [分辨率]")
        logger.info("  python video_detection.py --batch")
        logger.info("\n参数说明:")
        logger.info("  <输入视频路径>  输入视频文件的路径")
        logger.info("  [输出视频路径]  可选，指定输出路径（默认: output.mp4）")
        logger.info("  [分辨率]        可选，格式: 宽x高，例如 720x1280")
        logger.info("  --batch         批量处理test_videos目录下的所有视频")
        logger.info("\n示例:")
        logger.info("  python video_detection.py input.mp4 output.mp4")
        logger.info("  python video_detection.py input.mp4 output.mp4 720x1280")
        logger.info("  python video_detection.py --batch")
        logger.info("\n目录结构:")
        logger.info(f"  脚本目录:       {SCRIPT_DIR}")
        logger.info(f"  测试视频目录:   {TEST_VIDEOS_DIR}")
        logger.info(f"  结果保存目录:   {RESULT_DIR}")
        logger.info("\n输出文件命名规则:")
        logger.info("  result_<时间戳>_ID<序号>_<原文件名>.<扩展名>")
        logger.info("  例如: result_20260530_143025_ID0001_solidWhiteRight.mp4")
        logger.info("=" * 60)