"""
可视化模块

提供结果可视化功能。
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List
import matplotlib.pyplot as plt


class Visualizer:
    """可视化工具类"""
    
    COLORS = {
        'left_lane': (0, 255, 0),
        'right_lane': (0, 255, 0),
        'roi': (255, 0, 0),
        'detected_lines': (0, 0, 255),
        'text': (255, 255, 255),
        'warning': (0, 255, 255)
    }
    
    def __init__(self, font_scale: float = 0.6, thickness: int = 2):
        self.font_scale = font_scale
        self.thickness = thickness
    
    def draw_lane_lines(
        self,
        image: np.ndarray,
        left_line: Optional[Tuple[int, int, int, int]],
        right_line: Optional[Tuple[int, int, int, int]],
        thickness: int = 8
    ) -> np.ndarray:
        """
        绘制车道线
        
        Args:
            image: 输入图像
            left_line: 左车道线 (x1, y1, x2, y2)
            right_line: 右车道线 (x1, y1, x2, y2)
            thickness: 线宽
            
        Returns:
            np.ndarray: 绘制后的图像
        """
        output = image.copy()
        
        if left_line is not None:
            x1, y1, x2, y2 = left_line
            cv2.line(output, (x1, y1), (x2, y2), self.COLORS['left_lane'], thickness)
        
        if right_line is not None:
            x1, y1, x2, y2 = right_line
            cv2.line(output, (x1, y1), (x2, y2), self.COLORS['right_lane'], thickness)
        
        return output
    
    def draw_lane_area(
        self,
        image: np.ndarray,
        left_line: Optional[Tuple[int, int, int, int]],
        right_line: Optional[Tuple[int, int, int, int]],
        color: Tuple[int, int, int] = (0, 255, 0),
        alpha: float = 0.3
    ) -> np.ndarray:
        """
        绘制车道区域
        
        Args:
            image: 输入图像
            left_line: 左车道线
            right_line: 右车道线
            color: 填充颜色
            alpha: 透明度
            
        Returns:
            np.ndarray: 绘制后的图像
        """
        if left_line is None or right_line is None:
            return image
        
        output = image.copy()
        
        pts = np.array([
            [left_line[0], left_line[1]],
            [left_line[2], left_line[3]],
            [right_line[2], right_line[3]],
            [right_line[0], right_line[1]]
        ], dtype=np.int32)
        
        overlay = output.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
        
        return output
    
    def draw_roi(
        self,
        image: np.ndarray,
        vertices: np.ndarray,
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        绘制感兴趣区域
        
        Args:
            image: 输入图像
            vertices: ROI顶点
            color: 颜色
            thickness: 线宽
            
        Returns:
            np.ndarray: 绘制后的图像
        """
        output = image.copy()
        cv2.polylines(output, [vertices], True, color, thickness)
        return output
    
    def draw_info(
        self,
        image: np.ndarray,
        info: dict,
        position: Tuple[int, int] = (10, 30)
    ) -> np.ndarray:
        """
        绘制信息文本
        
        Args:
            image: 输入图像
            info: 信息字典
            position: 起始位置
            
        Returns:
            np.ndarray: 绘制后的图像
        """
        output = image.copy()
        y = position[1]
        
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(output, text, (position[0], y),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                       self.COLORS['text'], self.thickness)
            y += 25
        
        return output
    
    def draw_fps(
        self,
        image: np.ndarray,
        fps: float,
        position: Tuple[int, int] = (10, 30)
    ) -> np.ndarray:
        """
        绘制FPS
        
        Args:
            image: 输入图像
            fps: 帧率
            position: 位置
            
        Returns:
            np.ndarray: 绘制后的图像
        """
        output = image.copy()
        text = f"FPS: {fps:.1f}"
        cv2.putText(output, text, position,
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                   self.COLORS['text'], self.thickness)
        return output
    
    def draw_detection_status(
        self,
        image: np.ndarray,
        left_detected: bool,
        right_detected: bool,
        position: Tuple[int, int] = (10, 60)
    ) -> np.ndarray:
        """
        绘制检测状态
        
        Args:
            image: 输入图像
            left_detected: 左车道线是否检测到
            right_detected: 右车道线是否检测到
            position: 位置
            
        Returns:
            np.ndarray: 绘制后的图像
        """
        output = image.copy()
        
        left_status = "Left: OK" if left_detected else "Left: --"
        right_status = "Right: OK" if right_detected else "Right: --"
        
        left_color = self.COLORS['text'] if left_detected else self.COLORS['warning']
        right_color = self.COLORS['text'] if right_detected else self.COLORS['warning']
        
        cv2.putText(output, left_status, position,
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                   left_color, self.thickness)
        cv2.putText(output, right_status, (position[0], position[1] + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                   right_color, self.thickness)
        
        return output
    
    def create_comparison_view(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        titles: Tuple[str, str] = ("Original", "Processed")
    ) -> np.ndarray:
        """
        创建对比视图
        
        Args:
            original: 原始图像
            processed: 处理后图像
            titles: 标题
            
        Returns:
            np.ndarray: 对比图像
        """
        h1, w1 = original.shape[:2]
        h2, w2 = processed.shape[:2]
        
        max_h = max(h1, h2)
        max_w = max(w1, w2)
        
        canvas = np.zeros((max_h, max_w * 2 + 10, 3), dtype=np.uint8)
        canvas[:h1, :w1] = original
        canvas[:h2, max_w + 10:] = processed
        
        cv2.putText(canvas, titles[0], (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas, titles[1], (max_w + 20, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return canvas
    
    def create_detection_steps_view(
        self,
        images: List[np.ndarray],
        titles: List[str],
        cols: int = 3
    ) -> np.ndarray:
        """
        创建检测步骤视图
        
        Args:
            images: 图像列表
            titles: 标题列表
            cols: 列数
            
        Returns:
            np.ndarray: 步骤视图
        """
        n = len(images)
        rows = (n + cols - 1) // cols
        
        h, w = images[0].shape[:2]
        canvas = np.zeros((rows * (h + 40), cols * w, 3), dtype=np.uint8)
        
        for i, (img, title) in enumerate(zip(images, titles)):
            row = i // cols
            col = i % cols
            
            y_start = row * (h + 40)
            x_start = col * w
            
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            canvas[y_start:y_start + h, x_start:x_start + w] = img
            
            cv2.putText(canvas, title, (x_start + 10, y_start + h + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        return canvas
