"""
图像预处理模块

提供图像增强、颜色过滤、边缘检测等预处理功能。
"""
import cv2
import numpy as np
from typing import Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import get_config
from utils.logger import get_logger


class Preprocessor:
    """图像预处理器"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
    
    def enhance_image(self, image: np.ndarray, clip_limit: float = 2.5) -> np.ndarray:
        """
        使用CLAHE增强图像
        
        Args:
            image: 输入图像
            clip_limit: 对比度限制
            
        Returns:
            np.ndarray: 增强后的图像
        """
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def filter_white_yellow_colors(
        self,
        image: np.ndarray,
        white_lower: Optional[np.ndarray] = None,
        white_upper: Optional[np.ndarray] = None,
        yellow_lower: Optional[np.ndarray] = None,
        yellow_upper: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        过滤白色和黄色车道线
        
        Args:
            image: 输入图像
            white_lower: 白色下限阈值
            white_upper: 白色上限阈值
            yellow_lower: 黄色下限阈值
            yellow_upper: 黄色上限阈值
            
        Returns:
            np.ndarray: 过滤后的图像
        """
        hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        
        if white_lower is None:
            white_lower = np.array(self.config.color_threshold.white_lower, dtype=np.uint8)
        if white_upper is None:
            white_upper = np.array(self.config.color_threshold.white_upper, dtype=np.uint8)
        if yellow_lower is None:
            yellow_lower = np.array(self.config.color_threshold.yellow_lower, dtype=np.uint8)
        if yellow_upper is None:
            yellow_upper = np.array(self.config.color_threshold.yellow_upper, dtype=np.uint8)
        
        white_mask = cv2.inRange(hsl, white_lower, white_upper)
        yellow_mask = cv2.inRange(hsl, yellow_lower, yellow_upper)
        
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        return cv2.bitwise_and(image, image, mask=mask)
    
    def detect_edges(
        self,
        image: np.ndarray,
        low_threshold: int = 50,
        high_threshold: int = 150,
        gaussian_kernel: Tuple[int, int] = (5, 5)
    ) -> np.ndarray:
        """
        Canny边缘检测
        
        Args:
            image: 输入图像
            low_threshold: 低阈值
            high_threshold: 高阈值
            gaussian_kernel: 高斯核大小
            
        Returns:
            np.ndarray: 边缘图像
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, gaussian_kernel, 0)
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        return edges
    
    def adaptive_canny_edges(self, image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
        """
        自适应Canny边缘检测
        
        Args:
            image: 输入图像
            sigma: 阈值计算参数
            
        Returns:
            np.ndarray: 边缘图像
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        median = np.median(blurred)
        
        low_threshold = int(max(0, (1.0 - sigma) * median))
        high_threshold = int(min(255, (1.0 + sigma) * median))
        
        low_threshold = max(30, low_threshold)
        high_threshold = max(90, high_threshold)
        
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        return edges
    
    def create_roi_mask(
        self,
        image_shape: Tuple[int, int, int],
        vertices: np.ndarray
    ) -> np.ndarray:
        """
        创建感兴趣区域掩码
        
        Args:
            image_shape: 图像形状
            vertices: ROI顶点
            
        Returns:
            np.ndarray: ROI掩码
        """
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [vertices], 255)
        return mask
    
    def apply_roi(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        应用感兴趣区域
        
        Args:
            image: 输入图像
            mask: ROI掩码
            
        Returns:
            np.ndarray: 应用ROI后的图像
        """
        return cv2.bitwise_and(image, image, mask=mask)
    
    def create_left_right_roi(
        self,
        image_shape: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建左右分离的ROI
        
        Args:
            image_shape: 图像形状
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (左ROI掩码, 右ROI掩码)
        """
        height, width = image_shape[:2]
        roi_config = self.config.roi
        
        left_vertices = np.array([[
            [width * roi_config.left_start_x, height],
            [width * roi_config.left_start_x, height * 0.95],
            [width * 0.25, height * roi_config.top_y],
            [width * roi_config.left_end_x, height * roi_config.top_y],
            [width * roi_config.left_end_x, height * 0.95],
            [width * roi_config.left_end_x, height]
        ]], dtype=np.int32)
        
        right_vertices = np.array([[
            [width * roi_config.right_start_x, height],
            [width * roi_config.right_start_x, height * 0.95],
            [width * roi_config.right_start_x, height * roi_config.top_y],
            [width * 0.75, height * roi_config.top_y],
            [width * roi_config.right_end_x, height * 0.95],
            [width * roi_config.right_end_x, height]
        ]], dtype=np.int32)
        
        left_mask = self.create_roi_mask(image_shape, left_vertices)
        right_mask = self.create_roi_mask(image_shape, right_vertices)
        
        return left_mask, right_mask
    
    def preprocess(
        self,
        image: np.ndarray,
        enhance: bool = True,
        filter_colors: bool = True,
        detect_edges: bool = True,
        apply_roi: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        完整预处理流程
        
        Args:
            image: 输入图像
            enhance: 是否增强
            filter_colors: 是否过滤颜色
            detect_edges: 是否检测边缘
            apply_roi: 是否应用ROI
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (增强图像, 颜色过滤图像, 边缘图像)
        """
        enhanced = image.copy()
        if enhance:
            enhanced = self.enhance_image(enhanced)
        
        color_filtered = enhanced.copy()
        if filter_colors:
            color_filtered = self.filter_white_yellow_colors(enhanced)
        
        edges = None
        if detect_edges:
            edges = self.adaptive_canny_edges(color_filtered)
            
            if apply_roi:
                left_mask, right_mask = self.create_left_right_roi(image.shape)
                left_edges = self.apply_roi(edges, left_mask)
                right_edges = self.apply_roi(edges, right_mask)
                edges = cv2.bitwise_or(left_edges, right_edges)
        
        return enhanced, color_filtered, edges
