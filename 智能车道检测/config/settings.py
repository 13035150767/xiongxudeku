"""
车道线检测系统配置管理模块

该模块提供统一的配置管理功能，支持从配置文件和命令行参数加载配置。
"""
import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class ColorThresholdConfig:
    """颜色阈值配置"""
    white_lower: List[int] = field(default_factory=lambda: [0, 200, 0])
    white_upper: List[int] = field(default_factory=lambda: [180, 255, 255])
    yellow_lower: List[int] = field(default_factory=lambda: [10, 50, 80])
    yellow_upper: List[int] = field(default_factory=lambda: [40, 255, 255])


@dataclass
class ROIConfig:
    """感兴趣区域配置"""
    left_start_x: float = 0.05
    left_end_x: float = 0.45
    right_start_x: float = 0.55
    right_end_x: float = 0.95
    top_y: float = 0.55
    bottom_y: float = 1.0


@dataclass
class HoughConfig:
    """霍夫变换配置"""
    rho: float = 1.0
    theta: float = 3.14159 / 180
    threshold: int = 25
    min_line_length: int = 20
    max_line_gap: int = 20


@dataclass
class LineValidationConfig:
    """直线验证配置"""
    left_slope_min: float = -1.5
    left_slope_max: float = -0.4
    right_slope_min: float = 0.4
    right_slope_max: float = 1.5
    min_line_length: int = 30
    cluster_slope_threshold: float = 0.15
    cluster_position_threshold: int = 50


@dataclass
class SmoothConfig:
    """平滑配置"""
    alpha: float = 0.4
    enable_temporal_smooth: bool = True


@dataclass
class PerformanceConfig:
    """性能配置"""
    target_fps: int = 25
    max_frame_queue_size: int = 10
    enable_multithreading: bool = False
    num_workers: int = 2


@dataclass
class PathConfig:
    """路径配置"""
    test_images_dir: str = r"E:\PythonProject\计算机视觉\智能车道检测\test_images"
    test_videos_dir: str = r"E:\PythonProject\计算机视觉\智能车道检测\test_videos"
    output_dir: str = r"E:\PythonProject\计算机视觉\智能车道检测\结果"
    log_dir: str = r"E:\PythonProject\计算机视觉\智能车道检测\logs"
    config_file: str = ""


@dataclass
class LogConfig:
    """日志配置"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    console_enabled: bool = True
    max_file_size: int = 10 * 1024 * 1024
    backup_count: int = 5


@dataclass
class Config:
    """主配置类"""
    color_threshold: ColorThresholdConfig = field(default_factory=ColorThresholdConfig)
    roi: ROIConfig = field(default_factory=ROIConfig)
    hough: HoughConfig = field(default_factory=HoughConfig)
    line_validation: LineValidationConfig = field(default_factory=LineValidationConfig)
    smooth: SmoothConfig = field(default_factory=SmoothConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    path: PathConfig = field(default_factory=PathConfig)
    log: LogConfig = field(default_factory=LogConfig)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """从JSON文件加载配置"""
        config = cls()
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            config._update_from_dict(data)
        return config
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> 'Config':
        """从命令行参数加载配置"""
        config = cls()
        if hasattr(args, 'config') and args.config:
            config = cls.from_file(args.config)
        if hasattr(args, 'images_dir') and args.images_dir:
            config.path.test_images_dir = args.images_dir
        if hasattr(args, 'videos_dir') and args.videos_dir:
            config.path.test_videos_dir = args.videos_dir
        if hasattr(args, 'output_dir') and args.output_dir:
            config.path.output_dir = args.output_dir
        return config
    
    def _update_from_dict(self, data: Dict[str, Any]) -> None:
        """从字典更新配置"""
        for key, value in data.items():
            if hasattr(self, key):
                attr = getattr(self, key)
                if isinstance(attr, object) and isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if hasattr(attr, sub_key):
                            setattr(attr, sub_key, sub_value)
                else:
                    setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        import dataclasses
        return dataclasses.asdict(self)
    
    def save(self, path: str) -> None:
        """保存配置到文件"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


_config_instance: Optional[Config] = None


def get_config(config_path: Optional[str] = None, args: Optional[argparse.Namespace] = None) -> Config:
    """
    获取配置实例（单例模式）
    
    Args:
        config_path: 配置文件路径
        args: 命令行参数
        
    Returns:
        Config: 配置实例
    """
    global _config_instance
    
    if _config_instance is None:
        if args is not None:
            _config_instance = Config.from_args(args)
        elif config_path is not None:
            _config_instance = Config.from_file(config_path)
        else:
            _config_instance = Config()
    
    return _config_instance


def reset_config() -> None:
    """重置配置实例"""
    global _config_instance
    _config_instance = None
