"""
车道线检测系统主程序

提供命令行接口和批处理功能。
"""
import os
import sys
import argparse
import time
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import Config, get_config, reset_config
from utils.logger import setup_logger, get_logger, DetectionLogger, PerformanceLogger
from utils.visualization import Visualizer
from handlers.input_handler import ImageReader, VideoReader
from handlers.output_handler import ImageWriter, VideoWriter, DataWriter
from core.preprocessor import Preprocessor
from core.detector import LaneDetector


class LaneDetectionPipeline:
    """车道线检测流水线"""
    
    def __init__(self, config: Optional[Config] = None):
        if config is None:
            config = get_config()
        self.config = config
        
        log_file = os.path.join(config.path.log_dir, "lane_detection.log")
        self.logger = setup_logger(
            name="lane_detection",
            level=config.log.level,
            log_file=log_file,
            console_enabled=config.log.console_enabled,
            file_enabled=config.log.file_enabled
        )
        
        self.preprocessor = Preprocessor()
        self.detector = LaneDetector()
        self.visualizer = Visualizer()
        
        self.detection_logger = DetectionLogger(self.logger)
        self.perf_logger = PerformanceLogger(self.logger)
        
        self.logger.info("车道线检测系统初始化完成")
    
    def process_image(self, image, draw_info: bool = True) -> tuple:
        """
        处理单张图像
        
        Args:
            image: 输入图像
            draw_info: 是否绘制信息
            
        Returns:
            tuple: (处理后的图像, 检测结果)
        """
        self.perf_logger.start("图像处理")
        
        enhanced, color_filtered, edges = self.preprocessor.preprocess(image)
        
        if edges is None:
            return image, {'left_line': None, 'right_line': None}
        
        result = self.detector.detect(edges, image.shape)
        
        output = self.visualizer.draw_lane_lines(
            image, result['left_line'], result['right_line']
        )
        
        if draw_info:
            output = self.visualizer.draw_detection_status(
                output, result['left_detected'], result['right_detected']
            )
        
        elapsed = self.perf_logger.end()
        result['processing_time_ms'] = elapsed
        
        self.detection_logger.log_frame(result['left_detected'], result['right_detected'])
        
        return output, result
    
    def process_image_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        show: bool = False
    ) -> bool:
        """
        处理图像文件
        
        Args:
            input_path: 输入路径
            output_path: 输出路径
            show: 是否显示结果
            
        Returns:
            bool: 是否成功
        """
        self.logger.info(f"处理图像: {input_path}")
        
        reader = ImageReader()
        image = reader.read(input_path)
        
        if image is None:
            return False
        
        self.detector.reset()
        output, result = self.process_image(image)
        
        if output_path:
            writer = ImageWriter(os.path.dirname(output_path), prefix="")
            filename = os.path.basename(output_path)
            writer.write(output, filename)
        
        if show:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
            plt.title('车道线检测结果')
            plt.axis('off')
            plt.show()
        
        return True
    
    def process_video_file(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        show_progress: bool = True
    ) -> bool:
        """
        处理视频文件
        
        Args:
            input_path: 输入路径
            output_path: 输出路径
            show_progress: 是否显示进度
            
        Returns:
            bool: 是否成功
        """
        self.logger.info(f"处理视频: {input_path}")
        
        reader = VideoReader()
        video_info = reader.get_info(input_path)
        
        if video_info is None:
            return False
        
        self.logger.info(f"视频信息: {video_info}")
        
        width = video_info['width']
        height = video_info['height']
        fps = video_info['fps']
        frame_count = video_info['frame_count']
        
        writer = None
        if output_path:
            output_dir = os.path.dirname(output_path)
            filename = os.path.basename(output_path)
            writer = VideoWriter(output_dir, fps=fps, prefix="")
            writer.open(filename, width, height)
        
        self.detector.reset()
        self.detection_logger.reset()
        
        processed_frames = 0
        start_time = time.time()
        
        for frame_idx, frame in reader.read_frames(input_path):
            output, result = self.process_image(frame, draw_info=True)
            
            if writer:
                writer.write(output)
            
            processed_frames += 1
            
            if show_progress and processed_frames % 10 == 0:
                elapsed = time.time() - start_time
                current_fps = processed_frames / max(elapsed, 0.001)
                progress = processed_frames / frame_count * 100
                print(f"\r处理进度: {progress:.1f}% ({processed_frames}/{frame_count}) FPS: {current_fps:.1f}", end="")
        
        if show_progress:
            print()
        
        if writer:
            writer.close()
        
        reader.close()
        
        elapsed = time.time() - start_time
        avg_fps = processed_frames / max(elapsed, 0.001)
        
        self.logger.info(f"视频处理完成: {processed_frames}帧, 耗时{elapsed:.2f}秒, 平均FPS: {avg_fps:.1f}")
        self.detection_logger.log_statistics()
        
        return True
    
    def batch_process_images(
        self,
        input_dir: str,
        output_dir: str
    ) -> dict:
        """
        批量处理图像
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            
        Returns:
            dict: 处理统计
        """
        self.logger.info(f"批量处理图像: {input_dir}")
        
        reader = ImageReader()
        writer = ImageWriter(output_dir)
        
        files = reader.get_image_files(input_dir)
        
        if not files:
            self.logger.warning(f"未找到图像文件: {input_dir}")
            return {'total': 0, 'success': 0, 'failed': 0}
        
        self.detection_logger.reset()
        success_count = 0
        failed_count = 0
        
        for filename in files:
            input_path = os.path.join(input_dir, filename)
            image = reader.read(input_path)
            
            if image is None:
                failed_count += 1
                continue
            
            self.detector.reset()
            output, result = self.process_image(image)
            writer.write(output, filename)
            
            success_count += 1
            print(f"\r已处理: {success_count}/{len(files)}", end="")
        
        print()
        
        stats = {
            'total': len(files),
            'success': success_count,
            'failed': failed_count
        }
        
        self.logger.info(f"批量处理完成: {stats}")
        self.detection_logger.log_statistics()
        
        return stats
    
    def batch_process_videos(
        self,
        input_dir: str,
        output_dir: str
    ) -> dict:
        """
        批量处理视频
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            
        Returns:
            dict: 处理统计
        """
        self.logger.info(f"批量处理视频: {input_dir}")
        
        reader = VideoReader()
        files = reader.get_video_files(input_dir)
        
        if not files:
            self.logger.warning(f"未找到视频文件: {input_dir}")
            return {'total': 0, 'success': 0, 'failed': 0}
        
        success_count = 0
        failed_count = 0
        
        for filename in files:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, f"result_{filename}")
            
            if self.process_video_file(input_path, output_path):
                success_count += 1
            else:
                failed_count += 1
        
        stats = {
            'total': len(files),
            'success': success_count,
            'failed': failed_count
        }
        
        self.logger.info(f"批量处理完成: {stats}")
        
        return stats


import cv2


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="车道线检测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --image test.jpg
  python main.py --image test.jpg --output result.jpg
  python main.py --video test.mp4
  python main.py --batch-images --input-dir ./images --output-dir ./results
  python main.py --batch-videos --input-dir ./videos --output-dir ./results
  python main.py --config config.json
        """
    )
    
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--image', type=str, help='输入图像路径')
    parser.add_argument('--video', type=str, help='输入视频路径')
    parser.add_argument('--output', type=str, help='输出路径')
    parser.add_argument('--input-dir', type=str, help='输入目录')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    parser.add_argument('--batch-images', action='store_true', help='批量处理图像')
    parser.add_argument('--batch-videos', action='store_true', help='批量处理视频')
    parser.add_argument('--show', action='store_true', help='显示结果')
    parser.add_argument('--log-level', type=str, default='INFO', help='日志级别')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    reset_config()
    config = get_config(args=args)
    config.log.level = args.log_level
    
    pipeline = LaneDetectionPipeline(config)
    
    try:
        if args.image:
            output_path = args.output
            if output_path is None and args.output_dir:
                output_path = os.path.join(args.output_dir, f"result_{os.path.basename(args.image)}")
            
            pipeline.process_image_file(args.image, output_path, args.show)
        
        elif args.video:
            output_path = args.output
            if output_path is None and args.output_dir:
                output_path = os.path.join(args.output_dir, f"result_{os.path.basename(args.video)}")
            
            pipeline.process_video_file(args.video, output_path)
        
        elif args.batch_images:
            input_dir = args.input_dir or config.path.test_images_dir
            output_dir = args.output_dir or config.path.output_dir
            pipeline.batch_process_images(input_dir, output_dir)
        
        elif args.batch_videos:
            input_dir = args.input_dir or config.path.test_videos_dir
            output_dir = args.output_dir or config.path.output_dir
            pipeline.batch_process_videos(input_dir, output_dir)
        
        else:
            print("请指定输入文件或使用批量处理模式")
            print("使用 --help 查看帮助信息")
    
    except KeyboardInterrupt:
        print("\n用户中断")
    except Exception as e:
        pipeline.logger.error(f"程序异常: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
