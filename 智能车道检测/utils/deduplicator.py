"""
数据去重模块

提供检测结果去重功能，支持基于内容哈希和字段组合的去重策略。
"""
import hashlib
import numpy as np
from typing import Optional, List, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger


class DedupStrategy(Enum):
    CONTENT_HASH = "content_hash"
    DETECTION_RESULT = "detection_result"
    FILENAME = "filename"
    COMBINED = "combined"


@dataclass
class DedupStats:
    total_records: int = 0
    unique_records: int = 0
    duplicate_records: int = 0
    skipped_files: int = 0

    @property
    def duplicate_rate(self) -> float:
        if self.total_records == 0:
            return 0.0
        return self.duplicate_records / self.total_records * 100

    def to_dict(self) -> Dict[str, Any]:
        return {
            "去重前记录数": self.total_records,
            "去重后记录数": self.unique_records,
            "重复记录数": self.duplicate_records,
            "跳过文件数": self.skipped_files,
            "重复率": f"{self.duplicate_rate:.2f}%"
        }


class ResultDeduplicator:
    """检测结果去重器"""

    def __init__(self, strategy: DedupStrategy = DedupStrategy.COMBINED):
        self.strategy = strategy
        self.logger = get_logger()
        self._seen_hashes: Set[str] = set()
        self._seen_filenames: Set[str] = set()
        self._seen_results: Set[str] = set()
        self.stats = DedupStats()

    def reset(self) -> None:
        self._seen_hashes.clear()
        self._seen_filenames.clear()
        self._seen_results.clear()
        self.stats = DedupStats()

    def _compute_image_hash(self, image: np.ndarray) -> str:
        if image is None:
            return ""
        try:
            if image.size > 1024 * 1024:
                step = max(1, image.shape[0] // 256)
                sample = image[::step, ::step]
            else:
                sample = image
            return hashlib.md5(sample.tobytes()).hexdigest()
        except Exception:
            return hashlib.md5(np.ascontiguousarray(image).tobytes()).hexdigest()

    def _compute_detection_hash(self, detection_result: Dict[str, Any]) -> str:
        parts = []
        for key in ['left_line', 'right_line', 'left_slope', 'right_slope',
                     'left_detected', 'right_detected']:
            if key in detection_result:
                val = detection_result[key]
                if val is not None:
                    if isinstance(val, float):
                        parts.append(f"{key}:{val:.4f}")
                    else:
                        parts.append(f"{key}:{val}")
        return hashlib.md5("|".join(parts).encode()).hexdigest()

    def _compute_filename_hash(self, filename: str) -> str:
        base_name = os.path.splitext(filename)[0]
        return hashlib.md5(base_name.encode()).hexdigest()

    def _compute_combined_hash(self, filename: str, image: Optional[np.ndarray],
                                detection_result: Optional[Dict[str, Any]]) -> str:
        parts = [self._compute_filename_hash(filename)]
        if image is not None:
            parts.append(self._compute_image_hash(image))
        if detection_result is not None:
            parts.append(self._compute_detection_hash(detection_result))
        return hashlib.md5("|".join(parts).encode()).hexdigest()

    def is_duplicate(self, filename: str,
                     image: Optional[np.ndarray] = None,
                     detection_result: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
        self.stats.total_records += 1

        if self.strategy == DedupStrategy.CONTENT_HASH:
            if image is None:
                self.stats.unique_records += 1
                return False, ""
            hash_key = self._compute_image_hash(image)

        elif self.strategy == DedupStrategy.DETECTION_RESULT:
            if detection_result is None:
                self.stats.unique_records += 1
                return False, ""
            hash_key = self._compute_detection_hash(detection_result)

        elif self.strategy == DedupStrategy.FILENAME:
            hash_key = self._compute_filename_hash(filename)

        elif self.strategy == DedupStrategy.COMBINED:
            hash_key = self._compute_combined_hash(filename, image, detection_result)

        else:
            self.stats.unique_records += 1
            return False, ""

        if hash_key in self._seen_hashes:
            self.stats.duplicate_records += 1
            return True, hash_key

        self._seen_hashes.add(hash_key)
        self.stats.unique_records += 1
        return False, hash_key

    def is_file_processed(self, filename: str) -> bool:
        if filename in self._seen_filenames:
            self.stats.skipped_files += 1
            return True
        self._seen_filenames.add(filename)
        return False

    def get_stats(self) -> DedupStats:
        return self.stats

    def log_stats(self) -> None:
        stats_dict = self.stats.to_dict()
        self.logger.info("去重统计:")
        for key, value in stats_dict.items():
            self.logger.info(f"  {key}: {value}")


class BatchDeduplicator:
    """批量处理去重器"""

    def __init__(self, strategy: DedupStrategy = DedupStrategy.FILENAME):
        self.strategy = strategy
        self.logger = get_logger()
        self._processed_files: Set[str] = set()
        self._output_hashes: Set[str] = set()
        self.stats = DedupStats()

    def reset(self) -> None:
        self._processed_files.clear()
        self._output_hashes.clear()
        self.stats = DedupStats()

    def should_process_file(self, filename: str) -> Tuple[bool, str]:
        self.stats.total_records += 1

        base_name = os.path.splitext(filename)[0].lower()
        file_hash = hashlib.md5(base_name.encode()).hexdigest()

        if file_hash in self._processed_files:
            self.stats.duplicate_records += 1
            self.logger.info(f"  ⚠ 跳过重复文件: {filename}")
            return False, file_hash

        self._processed_files.add(file_hash)
        self.stats.unique_records += 1
        return True, file_hash

    def is_output_duplicate(self, output_path: str, image: Optional[np.ndarray] = None) -> bool:
        if os.path.exists(output_path):
            self.logger.info(f"  ⚠ 输出文件已存在，跳过: {os.path.basename(output_path)}")
            return True
        return False

    def get_stats(self) -> DedupStats:
        return self.stats

    def log_stats(self) -> None:
        stats_dict = self.stats.to_dict()
        self.logger.info("批量处理去重统计:")
        for key, value in stats_dict.items():
            self.logger.info(f"  {key}: {value}")
