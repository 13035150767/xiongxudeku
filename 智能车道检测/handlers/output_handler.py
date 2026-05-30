"""
数据输出处理模块

提供图像和视频的保存功能，支持结果去重。
"""
import os
import cv2
import numpy as np
from typing import Optional, List, Dict, Any
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger
from utils.deduplicator import ResultDeduplicator, DedupStrategy, DedupStats


class OutputHandler:
    """输出处理基类"""
    
    def __init__(self, output_dir: str, enable_dedup: bool = True,
                 dedup_strategy: DedupStrategy = DedupStrategy.COMBINED):
        self.output_dir = output_dir
        self.logger = get_logger()
        self.enable_dedup = enable_dedup
        self.deduplicator = ResultDeduplicator(strategy=dedup_strategy) if enable_dedup else None
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            self.logger.info(f"创建输出目录: {output_dir}")
    
    def get_dedup_stats(self) -> Optional[DedupStats]:
        if self.deduplicator:
            return self.deduplicator.get_stats()
        return None
    
    def log_dedup_stats(self) -> None:
        if self.deduplicator:
            self.deduplicator.log_stats()


class ImageWriter(OutputHandler):
    """图像保存器"""
    
    def __init__(self, output_dir: str, prefix: str = "result_",
                 enable_dedup: bool = True,
                 dedup_strategy: DedupStrategy = DedupStrategy.COMBINED):
        super().__init__(output_dir, enable_dedup, dedup_strategy)
        self.prefix = prefix
    
    def write(self, image: np.ndarray, filename: str,
              detection_result: Optional[Dict[str, Any]] = None) -> bool:
        """
        保存图像（支持去重）
        
        Args:
            image: 图像数据
            filename: 文件名
            detection_result: 检测结果（用于去重判断）
            
        Returns:
            bool: 是否成功保存（重复记录返回False但不代表错误）
        """
        if self.enable_dedup and self.deduplicator:
            is_dup, hash_key = self.deduplicator.is_duplicate(
                filename, image, detection_result
            )
            if is_dup:
                self.logger.info(f"跳过重复结果: {filename} (hash: {hash_key[:8]}...)")
                return False
        
        output_path = os.path.join(self.output_dir, f"{self.prefix}{filename}")
        
        try:
            ext = os.path.splitext(filename)[1]
            cv2.imencode(ext, image)[1].tofile(output_path)
            self.logger.debug(f"保存图像: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"保存图像失败: {output_path}, 错误: {e}")
            return False
    
    def write_with_timestamp(self, image: np.ndarray, name: str = "image") -> str:
        """
        使用时间戳保存图像
        
        Args:
            image: 图像数据
            name: 文件名前缀
            
        Returns:
            str: 保存的文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{name}_{timestamp}.jpg"
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            cv2.imencode('.jpg', image)[1].tofile(output_path)
            self.logger.debug(f"保存图像: {output_path}")
            return output_path
        except Exception as e:
            self.logger.error(f"保存图像失败: {output_path}, 错误: {e}")
            return ""


class VideoWriter(OutputHandler):
    """视频保存器"""
    
    def __init__(self, output_dir: str, fps: float = 25.0, prefix: str = "result_",
                 enable_dedup: bool = True,
                 dedup_strategy: DedupStrategy = DedupStrategy.FILENAME):
        super().__init__(output_dir, enable_dedup, dedup_strategy)
        self.fps = fps
        self.prefix = prefix
        self.writer: Optional[cv2.VideoWriter] = None
        self.current_path: Optional[str] = None
    
    def open(self, filename: str, width: int, height: int, codec: str = 'mp4v') -> bool:
        """
        打开视频文件准备写入
        
        Args:
            filename: 文件名
            width: 视频宽度
            height: 视频高度
            codec: 编码器
            
        Returns:
            bool: 是否成功
        """
        output_path = os.path.join(self.output_dir, f"{self.prefix}{filename}")
        
        if self.enable_dedup and self.deduplicator:
            is_dup, _ = self.deduplicator.is_duplicate(filename)
            if is_dup:
                self.logger.info(f"跳过重复视频输出: {filename}")
                return False
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
            
            if not self.writer.isOpened():
                self.logger.error(f"无法创建视频写入器: {output_path}")
                return False
            
            self.current_path = output_path
            self.logger.info(f"创建视频: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"创建视频失败: {output_path}, 错误: {e}")
            return False
    
    def write(self, frame: np.ndarray) -> bool:
        """
        写入一帧
        
        Args:
            frame: 帧图像
            
        Returns:
            bool: 是否成功
        """
        if self.writer is None:
            self.logger.error("视频写入器未打开")
            return False
        
        try:
            self.writer.write(frame)
            return True
        except Exception as e:
            self.logger.error(f"写入帧失败: {e}")
            return False
    
    def close(self) -> None:
        """关闭视频写入器"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
            self.logger.info(f"视频保存完成: {self.current_path}")
            self.current_path = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class DataWriter(OutputHandler):
    """数据保存器"""
    
    def __init__(self, output_dir: str, enable_dedup: bool = True,
                 dedup_strategy: DedupStrategy = DedupStrategy.DETECTION_RESULT):
        super().__init__(output_dir, enable_dedup, dedup_strategy)
    
    def save_lane_data(self, data: List[dict], filename: str = "lane_data.txt") -> bool:
        """
        保存车道线数据（自动去重）
        
        Args:
            data: 车道线数据列表
            filename: 文件名
            
        Returns:
            bool: 是否成功
        """
        original_count = len(data)
        
        if self.enable_dedup and self.deduplicator:
            unique_data = []
            for item in data:
                is_dup, _ = self.deduplicator.is_duplicate(
                    filename, detection_result=item
                )
                if not is_dup:
                    unique_data.append(item)
            data = unique_data
        
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 50 + "\n")
                f.write("车道线检测数据报告\n")
                f.write(f"原始记录数: {original_count}\n")
                f.write(f"去重后记录数: {len(data)}\n")
                f.write(f"去除重复: {original_count - len(data)} 条\n")
                f.write("=" * 50 + "\n\n")
                
                for i, item in enumerate(data):
                    f.write(f"Frame {i}:\n")
                    if 'left_line' in item and item['left_line']:
                        f.write(f"  Left: {item['left_line']}\n")
                    if 'right_line' in item and item['right_line']:
                        f.write(f"  Right: {item['right_line']}\n")
                    if 'left_slope' in item:
                        f.write(f"  Left slope: {item['left_slope']:.4f}\n")
                    if 'right_slope' in item:
                        f.write(f"  Right slope: {item['right_slope']:.4f}\n")
                    f.write("\n")
            
            self.logger.info(f"保存车道线数据: {output_path}")
            self.logger.info(f"  去重前: {original_count} 条, 去重后: {len(data)} 条")
            return True
        except Exception as e:
            self.logger.error(f"保存数据失败: {output_path}, 错误: {e}")
            return False
    
    def save_statistics(self, stats: dict, filename: str = "statistics.txt") -> bool:
        """
        保存统计信息
        
        Args:
            stats: 统计信息
            filename: 文件名
            
        Returns:
            bool: 是否成功
        """
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("=" * 50 + "\n")
                f.write("车道线检测统计报告\n")
                f.write("=" * 50 + "\n\n")
                
                for key, value in stats.items():
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.2f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            
            self.logger.info(f"保存统计信息: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"保存统计失败: {output_path}, 错误: {e}")
            return False
