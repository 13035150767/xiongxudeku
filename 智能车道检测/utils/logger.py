"""
日志系统模块

提供统一的日志记录功能，支持文件和终端输出。
"""
import os
import logging
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional
from datetime import datetime


_loggers: dict = {}


def setup_logger(
    name: str = "lane_detection",
    level: str = "INFO",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    console_enabled: bool = True,
    file_enabled: bool = True
) -> logging.Logger:
    """
    设置日志记录器
    
    Args:
        name: 日志记录器名称
        level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: 日志格式
        log_file: 日志文件路径
        max_file_size: 最大文件大小（字节）
        backup_count: 备份文件数量
        console_enabled: 是否启用终端输出
        file_enabled: 是否启用文件输出
        
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    if name in _loggers:
        return _loggers[name]
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    
    formatter = logging.Formatter(log_format)
    
    if console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if file_enabled and log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    _loggers[name] = logger
    return logger


def get_logger(name: str = "lane_detection") -> logging.Logger:
    """
    获取日志记录器
    
    Args:
        name: 日志记录器名称
        
    Returns:
        logging.Logger: 日志记录器
    """
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)


class PerformanceLogger:
    """性能日志记录器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_time: Optional[datetime] = None
        self.operation_name: str = ""
    
    def start(self, operation_name: str) -> None:
        """开始计时"""
        self.operation_name = operation_name
        self.start_time = datetime.now()
        self.logger.debug(f"开始执行: {operation_name}")
    
    def end(self) -> float:
        """结束计时并返回耗时（毫秒）"""
        if self.start_time is None:
            return 0.0
        
        elapsed = (datetime.now() - self.start_time).total_seconds() * 1000
        self.logger.debug(f"完成执行: {self.operation_name}, 耗时: {elapsed:.2f}ms")
        self.start_time = None
        return elapsed


class DetectionLogger:
    """检测日志记录器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.frame_count: int = 0
        self.detection_count: int = 0
        self.left_detected: int = 0
        self.right_detected: int = 0
    
    def log_frame(self, left_detected: bool, right_detected: bool) -> None:
        """记录帧检测结果"""
        self.frame_count += 1
        if left_detected:
            self.left_detected += 1
        if right_detected:
            self.right_detected += 1
        if left_detected or right_detected:
            self.detection_count += 1
    
    def get_statistics(self) -> dict:
        """获取统计信息"""
        return {
            "total_frames": self.frame_count,
            "detection_frames": self.detection_count,
            "left_detected": self.left_detected,
            "right_detected": self.right_detected,
            "detection_rate": self.detection_count / max(self.frame_count, 1) * 100,
            "left_rate": self.left_detected / max(self.frame_count, 1) * 100,
            "right_rate": self.right_detected / max(self.frame_count, 1) * 100
        }
    
    def log_statistics(self) -> None:
        """记录统计信息"""
        stats = self.get_statistics()
        self.logger.info(f"检测统计: 总帧数={stats['total_frames']}, "
                        f"检测帧数={stats['detection_frames']}, "
                        f"检测率={stats['detection_rate']:.2f}%, "
                        f"左侧检测率={stats['left_rate']:.2f}%, "
                        f"右侧检测率={stats['right_rate']:.2f}%")
    
    def reset(self) -> None:
        """重置统计"""
        self.frame_count = 0
        self.detection_count = 0
        self.left_detected = 0
        self.right_detected = 0
