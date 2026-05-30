"""
预处理器单元测试
"""
import os
import sys
import pytest
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.preprocessor import Preprocessor
from config.settings import Config


class TestPreprocessor:
    """预处理器测试类"""
    
    @pytest.fixture
    def preprocessor(self):
        return Preprocessor()
    
    @pytest.fixture
    def sample_image(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.line(image, (100, 480), (200, 300), (255, 255, 255), 5)
        cv2.line(image, (540, 480), (440, 300), (255, 255, 255), 5)
        return image
    
    def test_enhance_image(self, preprocessor, sample_image):
        """测试图像增强"""
        enhanced = preprocessor.enhance_image(sample_image)
        
        assert enhanced is not None
        assert enhanced.shape == sample_image.shape
        assert enhanced.dtype == np.uint8
    
    def test_filter_white_yellow_colors(self, preprocessor, sample_image):
        """测试颜色过滤"""
        filtered = preprocessor.filter_white_yellow_colors(sample_image)
        
        assert filtered is not None
        assert filtered.shape == sample_image.shape
    
    def test_detect_edges(self, preprocessor, sample_image):
        """测试边缘检测"""
        edges = preprocessor.detect_edges(sample_image)
        
        assert edges is not None
        assert len(edges.shape) == 2
        assert edges.dtype == np.uint8
    
    def test_adaptive_canny_edges(self, preprocessor, sample_image):
        """测试自适应边缘检测"""
        edges = preprocessor.adaptive_canny_edges(sample_image)
        
        assert edges is not None
        assert len(edges.shape) == 2
    
    def test_create_roi_mask(self, preprocessor, sample_image):
        """测试ROI掩码创建"""
        height, width = sample_image.shape[:2]
        vertices = np.array([[
            [0, height],
            [width * 0.45, height * 0.6],
            [width * 0.55, height * 0.6],
            [width, height]
        ]], dtype=np.int32)
        
        mask = preprocessor.create_roi_mask(sample_image.shape, vertices)
        
        assert mask is not None
        assert mask.shape == sample_image.shape[:2]
        assert mask.max() == 255
    
    def test_create_left_right_roi(self, preprocessor, sample_image):
        """测试左右分离ROI"""
        left_mask, right_mask = preprocessor.create_left_right_roi(sample_image.shape)
        
        assert left_mask is not None
        assert right_mask is not None
        assert left_mask.shape == sample_image.shape[:2]
        assert right_mask.shape == sample_image.shape[:2]
    
    def test_preprocess(self, preprocessor, sample_image):
        """测试完整预处理流程"""
        enhanced, color_filtered, edges = preprocessor.preprocess(sample_image)
        
        assert enhanced is not None
        assert color_filtered is not None
        assert edges is not None
        assert enhanced.shape == sample_image.shape
        assert color_filtered.shape == sample_image.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
