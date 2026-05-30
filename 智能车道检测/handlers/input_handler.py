"""
数据输入处理模块

提供图像和视频的读取功能。
"""
import os
import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Generator, Tuple, List
from pathlib import Path

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger


class InputHandler(ABC):
    """输入处理基类"""
    
    def __init__(self):
        self.logger = get_logger()
    
    @abstractmethod
    def read(self, path: str) -> Optional[np.ndarray]:
        """读取数据"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """关闭资源"""
        pass


class ImageReader(InputHandler):
    """图像读取器"""
    
    SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    def __init__(self):
        super().__init__()
    
    def read(self, path: str) -> Optional[np.ndarray]:
        """
        读取图像文件
        
        Args:
            path: 图像文件路径
            
        Returns:
            Optional[np.ndarray]: 图像数据，失败返回None
        """
        if not os.path.exists(path):
            self.logger.error(f"图像文件不存在: {path}")
            return None
        
        try:
            image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                self.logger.error(f"无法解码图像: {path}")
                return None
            return image
        except Exception as e:
            self.logger.error(f"读取图像失败: {path}, 错误: {e}")
            return None
    
    def read_batch(self, directory: str) -> Generator[Tuple[str, np.ndarray], None, None]:
        """
        批量读取目录下的图像
        
        Args:
            directory: 图像目录
            
        Yields:
            Tuple[str, np.ndarray]: (文件名, 图像数据)
        """
        if not os.path.exists(directory):
            self.logger.error(f"图像目录不存在: {directory}")
            return
        
        files = sorted([
            f for f in os.listdir(directory)
            if f.lower().endswith(self.SUPPORTED_EXTENSIONS)
        ])
        
        self.logger.info(f"找到 {len(files)} 个图像文件")
        
        for filename in files:
            path = os.path.join(directory, filename)
            image = self.read(path)
            if image is not None:
                yield filename, image
    
    def get_image_files(self, directory: str) -> List[str]:
        """
        获取目录下的图像文件列表
        
        Args:
            directory: 图像目录
            
        Returns:
            List[str]: 图像文件列表
        """
        if not os.path.exists(directory):
            return []
        
        return sorted([
            f for f in os.listdir(directory)
            if f.lower().endswith(self.SUPPORTED_EXTENSIONS)
        ])
    
    def close(self) -> None:
        pass


class VideoReader(InputHandler):
    """视频读取器"""
    
    SUPPORTED_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')
    
    def __init__(self):
        super().__init__()
        self.cap: Optional[cv2.VideoCapture] = None
        self.current_path: Optional[str] = None
    
    def open(self, path: str) -> bool:
        """
        打开视频文件
        
        Args:
            path: 视频文件路径
            
        Returns:
            bool: 是否成功打开
        """
        if not os.path.exists(path):
            self.logger.error(f"视频文件不存在: {path}")
            return False
        
        try:
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                self.logger.error(f"无法打开视频: {path}")
                return False
            self.current_path = path
            return True
        except Exception as e:
            self.logger.error(f"打开视频失败: {path}, 错误: {e}")
            return False
    
    def read(self, path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        读取下一帧
        
        Args:
            path: 视频文件路径（首次读取时需要指定）
            
        Returns:
            Optional[np.ndarray]: 帧图像，失败返回None
        """
        if path is not None and path != self.current_path:
            if not self.open(path):
                return None
        
        if self.cap is None or not self.cap.isOpened():
            self.logger.error("视频未打开")
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def read_frames(self, path: str) -> Generator[Tuple[int, np.ndarray], None, None]:
        """
        读取所有帧
        
        Args:
            path: 视频文件路径
            
        Yields:
            Tuple[int, np.ndarray]: (帧索引, 帧图像)
        """
        if not self.open(path):
            return
        
        frame_index = 0
        while True:
            frame = self.read()
            if frame is None:
                break
            yield frame_index, frame
            frame_index += 1
        
        self.logger.info(f"视频读取完成，共 {frame_index} 帧")
    
    def get_info(self, path: str) -> Optional[dict]:
        """
        获取视频信息
        
        Args:
            path: 视频文件路径
            
        Returns:
            Optional[dict]: 视频信息
        """
        if not self.open(path):
            return None
        
        info = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'frame_count': int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            'duration': self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(self.cap.get(cv2.CAP_PROP_FPS), 1)
        }
        return info
    
    def get_video_files(self, directory: str) -> List[str]:
        """
        获取目录下的视频文件列表
        
        Args:
            directory: 视频目录
            
        Returns:
            List[str]: 视频文件列表
        """
        if not os.path.exists(directory):
            return []
        
        return sorted([
            f for f in os.listdir(directory)
            if f.lower().endswith(self.SUPPORTED_EXTENSIONS)
        ])
    
    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            self.current_path = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class CameraReader(InputHandler):
    """摄像头读取器"""
    
    def __init__(self, camera_id: int = 0):
        super().__init__()
        self.camera_id = camera_id
        self.cap: Optional[cv2.VideoCapture] = None
    
    def open(self) -> bool:
        """打开摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                self.logger.error(f"无法打开摄像头: {self.camera_id}")
                return False
            return True
        except Exception as e:
            self.logger.error(f"打开摄像头失败: {e}")
            return False
    
    def read(self, path: Optional[str] = None) -> Optional[np.ndarray]:
        """读取一帧"""
        if self.cap is None or not self.cap.isOpened():
            if not self.open():
                return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame
    
    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
