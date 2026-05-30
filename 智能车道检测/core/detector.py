"""
车道线检测模块

提供车道线检测、拟合、平滑等核心算法。
"""
import cv2
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import get_config
from utils.logger import get_logger


class LaneDetector:
    """车道线检测器"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = get_logger()
        
        self.prev_left_line: Optional[Tuple[int, int, int, int]] = None
        self.prev_right_line: Optional[Tuple[int, int, int, int]] = None
    
    def detect_lines(
        self,
        edges: np.ndarray,
        rho: Optional[float] = None,
        theta: Optional[float] = None,
        threshold: Optional[int] = None,
        min_line_length: Optional[int] = None,
        max_line_gap: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        霍夫变换检测直线
        
        Args:
            edges: 边缘图像
            rho: 距离分辨率
            theta: 角度分辨率
            threshold: 累加器阈值
            min_line_length: 最小线长
            max_line_gap: 最大线间隙
            
        Returns:
            Optional[np.ndarray]: 检测到的直线
        """
        hough_config = self.config.hough
        
        if rho is None:
            rho = hough_config.rho
        if theta is None:
            theta = hough_config.theta
        if threshold is None:
            threshold = hough_config.threshold
        if min_line_length is None:
            min_line_length = hough_config.min_line_length
        if max_line_gap is None:
            max_line_gap = hough_config.max_line_gap
        
        lines = cv2.HoughLinesP(
            edges, rho, theta, threshold, np.array([]),
            minLineLength=min_line_length, maxLineGap=max_line_gap
        )
        return lines
    
    def validate_line(
        self,
        line: Tuple[int, int, int, int],
        image_shape: Tuple[int, int, int],
        side: str
    ) -> bool:
        """
        验证直线属性
        
        Args:
            line: 直线坐标 (x1, y1, x2, y2)
            image_shape: 图像形状
            side: 'left' 或 'right'
            
        Returns:
            bool: 是否有效
        """
        height, width = image_shape[:2]
        x1, y1, x2, y2 = line
        validation_config = self.config.line_validation
        
        if x2 == x1:
            return False
        
        slope = (y2 - y1) / (x2 - x1)
        
        if side == 'left':
            if slope > validation_config.left_slope_max or slope < validation_config.left_slope_min:
                return False
            mid_x = (x1 + x2) / 2
            if mid_x > width * 0.5:
                return False
        else:
            if slope < validation_config.right_slope_min or slope > validation_config.right_slope_max:
                return False
            mid_x = (x1 + x2) / 2
            if mid_x < width * 0.5:
                return False
        
        line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_length < validation_config.min_line_length:
            return False
        
        return True
    
    def cluster_lines(
        self,
        lines: Optional[np.ndarray],
        image_shape: Tuple[int, int, int],
        side: str
    ) -> List[List[Tuple[int, int, int, int]]]:
        """
        聚类直线
        
        Args:
            lines: 检测到的直线
            image_shape: 图像形状
            side: 'left' 或 'right'
            
        Returns:
            List[List[Tuple]]: 聚类结果
        """
        if lines is None or len(lines) == 0:
            return []
        
        validation_config = self.config.line_validation
        
        valid_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if self.validate_line((x1, y1, x2, y2), image_shape, side):
                valid_lines.append((x1, y1, x2, y2))
        
        if len(valid_lines) == 0:
            return []
        
        clusters = []
        used = [False] * len(valid_lines)
        
        for i, line1 in enumerate(valid_lines):
            if used[i]:
                continue
            
            cluster = [line1]
            used[i] = True
            
            x1_i, y1_i, x2_i, y2_i = line1
            slope_i = (y2_i - y1_i) / (x2_i - x1_i) if x2_i != x1_i else 0
            mid_x_i = (x1_i + x2_i) / 2
            
            for j, line2 in enumerate(valid_lines):
                if used[j]:
                    continue
                
                x1_j, y1_j, x2_j, y2_j = line2
                slope_j = (y2_j - y1_j) / (x2_j - x1_j) if x2_j != x1_j else 0
                mid_x_j = (x1_j + x2_j) / 2
                
                slope_diff = abs(slope_i - slope_j)
                pos_diff = abs(mid_x_i - mid_x_j)
                
                if (slope_diff < validation_config.cluster_slope_threshold and 
                    pos_diff < validation_config.cluster_position_threshold):
                    cluster.append(line2)
                    used[j] = True
            
            clusters.append(cluster)
        
        return clusters
    
    def select_best_cluster(
        self,
        clusters: List[List[Tuple[int, int, int, int]]]
    ) -> Optional[List[Tuple[int, int, int, int]]]:
        """
        选择最佳聚类
        
        Args:
            clusters: 聚类列表
            
        Returns:
            Optional[List[Tuple]]: 最佳聚类
        """
        if not clusters:
            return None
        
        best_cluster = None
        best_score = 0
        
        for cluster in clusters:
            total_length = 0
            for line in cluster:
                x1, y1, x2, y2 = line
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                total_length += length
            
            score = total_length * len(cluster)
            
            if score > best_score:
                best_score = score
                best_cluster = cluster
        
        return best_cluster
    
    def fit_line(
        self,
        cluster: Optional[List[Tuple[int, int, int, int]]],
        image_shape: Tuple[int, int, int]
    ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[float]]:
        """
        拟合车道线
        
        Args:
            cluster: 直线聚类
            image_shape: 图像形状
            
        Returns:
            Tuple[Optional[Tuple], Optional[float]]: (拟合线, 斜率)
        """
        if not cluster:
            return None, None
        
        height = image_shape[0]
        width = image_shape[1]
        
        x_coords = []
        y_coords = []
        
        for line in cluster:
            x1, y1, x2, y2 = line
            x_coords.extend([x1, x2])
            y_coords.extend([y1, y2])
        
        if len(x_coords) < 2:
            return None, None
        
        try:
            poly_coeffs = np.polyfit(x_coords, y_coords, 1)
        except:
            return None, None
        
        slope = poly_coeffs[0]
        intercept = poly_coeffs[1]
        
        y1 = height
        y2 = int(height * 0.58)
        
        if abs(slope) < 0.1:
            return None, None
        
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        
        if x1 < 0 or x1 > width or x2 < 0 or x2 > width:
            return None, None
        
        return (x1, y1, x2, y2), slope
    
    def check_crossing(
        self,
        left_line: Optional[Tuple[int, int, int, int]],
        right_line: Optional[Tuple[int, int, int, int]]
    ) -> bool:
        """
        检查左右车道线是否交叉
        
        Args:
            left_line: 左车道线
            right_line: 右车道线
            
        Returns:
            bool: 是否交叉
        """
        if left_line is None or right_line is None:
            return False
        
        x1_l, y1_l, x2_l, y2_l = left_line
        x1_r, y1_r, x2_r, y2_r = right_line
        
        if x2_l == x1_l or x2_r == x1_r:
            return False
        
        slope_l = (y2_l - y1_l) / (x2_l - x1_l)
        slope_r = (y2_r - y1_r) / (x2_r - x1_r)
        
        intercept_l = y1_l - slope_l * x1_l
        intercept_r = y1_r - slope_r * x1_r
        
        if abs(slope_l - slope_r) < 0.01:
            return False
        
        x_cross = (intercept_r - intercept_l) / (slope_l - slope_r)
        y_cross = slope_l * x_cross + intercept_l
        
        min_y = min(y1_l, y2_l, y1_r, y2_r)
        max_y = max(y1_l, y2_l, y1_r, y2_r)
        
        if min_y < y_cross < max_y:
            return True
        
        return False
    
    def smooth_line(
        self,
        current_line: Optional[Tuple[int, int, int, int]],
        prev_line: Optional[Tuple[int, int, int, int]],
        alpha: Optional[float] = None
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        平滑车道线
        
        Args:
            current_line: 当前帧车道线
            prev_line: 上一帧车道线
            alpha: 平滑系数
            
        Returns:
            Optional[Tuple]: 平滑后的车道线
        """
        if alpha is None:
            alpha = self.config.smooth.alpha
        
        if current_line is None and prev_line is None:
            return None
        if current_line is None:
            return prev_line
        if prev_line is None:
            return current_line
        
        x1 = int(alpha * current_line[0] + (1 - alpha) * prev_line[0])
        y1 = int(alpha * current_line[1] + (1 - alpha) * prev_line[1])
        x2 = int(alpha * current_line[2] + (1 - alpha) * prev_line[2])
        y2 = int(alpha * current_line[3] + (1 - alpha) * prev_line[3])
        
        return (x1, y1, x2, y2)
    
    def detect(
        self,
        edges: np.ndarray,
        image_shape: Tuple[int, int, int]
    ) -> Dict[str, Any]:
        """
        完整检测流程
        
        Args:
            edges: 边缘图像
            image_shape: 图像形状
            
        Returns:
            Dict[str, Any]: 检测结果
        """
        height, width = image_shape[:2]
        
        left_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        right_mask = np.zeros(image_shape[:2], dtype=np.uint8)
        
        left_vertices = np.array([[
            [width * 0.05, height],
            [width * 0.25, height * 0.55],
            [width * 0.45, height * 0.55],
            [width * 0.45, height]
        ]], dtype=np.int32)
        
        right_vertices = np.array([[
            [width * 0.55, height],
            [width * 0.55, height * 0.55],
            [width * 0.75, height * 0.55],
            [width * 0.95, height]
        ]], dtype=np.int32)
        
        cv2.fillPoly(left_mask, [left_vertices], 255)
        cv2.fillPoly(right_mask, [right_vertices], 255)
        
        left_edges = cv2.bitwise_and(edges, edges, mask=left_mask)
        right_edges = cv2.bitwise_and(edges, edges, mask=right_mask)
        
        left_lines = self.detect_lines(left_edges)
        right_lines = self.detect_lines(right_edges)
        
        left_clusters = self.cluster_lines(left_lines, image_shape, 'left')
        right_clusters = self.cluster_lines(right_lines, image_shape, 'right')
        
        best_left_cluster = self.select_best_cluster(left_clusters)
        best_right_cluster = self.select_best_cluster(right_clusters)
        
        left_line, left_slope = self.fit_line(best_left_cluster, image_shape)
        right_line, right_slope = self.fit_line(best_right_cluster, image_shape)
        
        if self.check_crossing(left_line, right_line):
            left_line = None
            right_line = None
        
        if self.config.smooth.enable_temporal_smooth:
            left_line = self.smooth_line(left_line, self.prev_left_line)
            right_line = self.smooth_line(right_line, self.prev_right_line)
        
        self.prev_left_line = left_line
        self.prev_right_line = right_line
        
        return {
            'left_line': left_line,
            'right_line': right_line,
            'left_slope': left_slope,
            'right_slope': right_slope,
            'left_detected': left_line is not None,
            'right_detected': right_line is not None
        }
    
    def reset(self) -> None:
        """重置检测器状态"""
        self.prev_left_line = None
        self.prev_right_line = None
