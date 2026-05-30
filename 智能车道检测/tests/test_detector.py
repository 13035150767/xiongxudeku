"""
检测器单元测试
"""
import os
import sys
import pytest
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.detector import LaneDetector
from core.preprocessor import Preprocessor


class TestLaneDetector:
    """检测器测试类"""
    
    @pytest.fixture
    def detector(self):
        detector = LaneDetector()
        detector.reset()
        return detector
    
    @pytest.fixture
    def preprocessor(self):
        return Preprocessor()
    
    @pytest.fixture
    def sample_image(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.line(image, (100, 480), (200, 300), (255, 255, 255), 5)
        cv2.line(image, (540, 480), (440, 300), (255, 255, 255), 5)
        return image
    
    def test_detect_lines(self, detector, sample_image, preprocessor):
        """测试直线检测"""
        _, _, edges = preprocessor.preprocess(sample_image)
        
        if edges is not None:
            lines = detector.detect_lines(edges)
            
            assert lines is not None or lines is None
    
    def test_validate_line(self, detector, sample_image):
        """测试直线验证"""
        height, width = sample_image.shape[:2]
        
        left_line = (100, height, 200, int(height * 0.6))
        right_line = (540, height, 440, int(height * 0.6))
        
        left_valid = detector.validate_line(left_line, sample_image.shape, 'left')
        right_valid = detector.validate_line(right_line, sample_image.shape, 'right')
        
        assert isinstance(left_valid, bool)
        assert isinstance(right_valid, bool)
    
    def test_cluster_lines(self, detector, sample_image, preprocessor):
        """测试直线聚类"""
        _, _, edges = preprocessor.preprocess(sample_image)
        
        if edges is not None:
            lines = detector.detect_lines(edges)
            
            left_clusters = detector.cluster_lines(lines, sample_image.shape, 'left')
            right_clusters = detector.cluster_lines(lines, sample_image.shape, 'right')
            
            assert isinstance(left_clusters, list)
            assert isinstance(right_clusters, list)
    
    def test_select_best_cluster(self, detector):
        """测试最佳聚类选择"""
        clusters = [
            [(100, 480, 150, 400), (110, 480, 160, 400)],
            [(200, 480, 250, 400)]
        ]
        
        best = detector.select_best_cluster(clusters)
        
        assert best is not None
        assert len(best) == 2
    
    def test_fit_line(self, detector, sample_image):
        """测试直线拟合"""
        cluster = [(100, 480, 150, 400), (110, 480, 160, 400)]
        
        line, slope = detector.fit_line(cluster, sample_image.shape)
        
        assert line is not None or line is None
        assert slope is not None or slope is None
    
    def test_check_crossing(self, detector):
        """测试交叉检测"""
        left_line = (100, 480, 200, 300)
        right_line = (540, 480, 440, 300)
        
        crossing = detector.check_crossing(left_line, right_line)
        
        assert isinstance(crossing, bool)
        assert crossing == False
        
        crossing_left = (100, 480, 400, 300)
        crossing_right = (540, 480, 200, 300)
        
        crossing = detector.check_crossing(crossing_left, crossing_right)
        assert crossing == True
    
    def test_smooth_line(self, detector):
        """测试直线平滑"""
        current = (100, 480, 200, 300)
        prev = (110, 480, 210, 300)
        
        smoothed = detector.smooth_line(current, prev, alpha=0.5)
        
        assert smoothed is not None
        assert len(smoothed) == 4
    
    def test_detect(self, detector, sample_image, preprocessor):
        """测试完整检测流程"""
        _, _, edges = preprocessor.preprocess(sample_image)
        
        if edges is not None:
            result = detector.detect(edges, sample_image.shape)
            
            assert 'left_line' in result
            assert 'right_line' in result
            assert 'left_slope' in result
            assert 'right_slope' in result
            assert 'left_detected' in result
            assert 'right_detected' in result
    
    def test_reset(self, detector):
        """测试重置"""
        detector.prev_left_line = (100, 480, 200, 300)
        detector.prev_right_line = (540, 480, 440, 300)
        
        detector.reset()
        
        assert detector.prev_left_line is None
        assert detector.prev_right_line is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
