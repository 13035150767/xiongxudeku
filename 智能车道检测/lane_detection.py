import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import logging
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from utils.deduplicator import BatchDeduplicator, DedupStrategy
TEST_IMAGES_DIR = os.path.join(SCRIPT_DIR, "test_images")
TEST_VIDEOS_DIR = os.path.join(SCRIPT_DIR, "test_videos")
RESULT_DIR = os.path.join(SCRIPT_DIR, "结果")

detection_counter = 0

prev_left_line = None
prev_right_line = None

def setup_logging():
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

def ensure_result_directory():
    try:
        if not os.path.exists(RESULT_DIR):
            os.makedirs(RESULT_DIR)
            logger.info(f"✓ 创建结果目录成功: {RESULT_DIR}")
        else:
            logger.info(f"✓ 结果目录已存在: {RESULT_DIR}")
        return True
    except PermissionError as e:
        logger.error(f"✗ 权限不足，无法创建结果目录: {RESULT_DIR}")
        logger.error(f"  错误详情: {str(e)}")
        return False
    except OSError as e:
        logger.error(f"✗ 创建结果目录失败: {RESULT_DIR}")
        logger.error(f"  错误详情: {str(e)}")
        return False

def generate_output_filename(original_filename, prefix="result", include_timestamp=True, include_counter=True):
    global detection_counter
    
    name, ext = os.path.splitext(original_filename)
    
    parts = [prefix]
    
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parts.append(timestamp)
    
    if include_counter:
        detection_counter += 1
        parts.append(f"ID{detection_counter:04d}")
    
    parts.append(name)
    
    new_filename = "_".join(parts) + ext
    return new_filename


def enhance_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    
    lab_enhanced = cv2.merge([l_enhanced, a, b])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    
    return enhanced


def select_white_yellow_colors_hsl(image):
    hsl = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    
    white_lower = np.array([0, 200, 0], dtype=np.uint8)
    white_upper = np.array([180, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsl, white_lower, white_upper)
    
    yellow_lower = np.array([10, 50, 80], dtype=np.uint8)
    yellow_upper = np.array([40, 255, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hsl, yellow_lower, yellow_upper)
    
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask=mask)


def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges


def create_roi_mask(image_shape, vertices):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [vertices], 255)
    return mask


def apply_roi(image, mask):
    return cv2.bitwise_and(image, image, mask=mask)


def create_left_right_roi(image_shape):
    height, width = image_shape[:2]
    
    left_vertices = np.array([[
        [width * 0.05, height],
        [width * 0.05, height * 0.95],
        [width * 0.25, height * 0.55],
        [width * 0.45, height * 0.55],
        [width * 0.45, height * 0.95],
        [width * 0.45, height]
    ]], dtype=np.int32)
    
    right_vertices = np.array([[
        [width * 0.55, height],
        [width * 0.55, height * 0.95],
        [width * 0.55, height * 0.55],
        [width * 0.75, height * 0.55],
        [width * 0.95, height * 0.95],
        [width * 0.95, height]
    ]], dtype=np.int32)
    
    left_mask = create_roi_mask(image_shape, left_vertices)
    right_mask = create_roi_mask(image_shape, right_vertices)
    
    return left_mask, right_mask


def hough_lines(image, rho=1, theta=np.pi/180, threshold=25, min_line_len=20, max_line_gap=20):
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]),
                              minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines


def validate_line_properties(line, image_shape, side):
    height, width = image_shape[:2]
    x1, y1, x2, y2 = line
    
    if x2 == x1:
        return False
    
    slope = (y2 - y1) / (x2 - x1)
    
    if side == 'left':
        if slope > -0.4 or slope < -1.5:
            return False
        mid_x = (x1 + x2) / 2
        if mid_x > width * 0.5:
            return False
    else:
        if slope < 0.4 or slope > 1.5:
            return False
        mid_x = (x1 + x2) / 2
        if mid_x < width * 0.5:
            return False
    
    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if line_length < 30:
        return False
    
    return True


def cluster_lines_by_slope_position(lines, image_shape, side):
    if lines is None or len(lines) == 0:
        return []
    
    valid_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if validate_line_properties((x1, y1, x2, y2), image_shape, side):
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
            
            if slope_diff < 0.15 and pos_diff < 50:
                cluster.append(line2)
                used[j] = True
        
        clusters.append(cluster)
    
    return clusters


def select_best_cluster(clusters):
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


def fit_lane_line(cluster, image_shape):
    if not cluster:
        return None, None
    
    height = image_shape[0]
    
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
    
    if x1 < 0 or x1 > image_shape[1] or x2 < 0 or x2 > image_shape[1]:
        return None, None
    
    return (x1, y1, x2, y2), slope


def check_lines_cross(left_line, right_line):
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


def smooth_lane_line(current_line, prev_line, alpha=0.4):
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


def draw_lane_lines(image, left_line, right_line, thickness=8):
    output = np.copy(image)

    if left_line is not None:
        x1, y1, x2, y2 = left_line
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), thickness)

    if right_line is not None:
        x1, y1, x2, y2 = right_line
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 0), thickness)

    return output


def detect_lane_lines(image, prev_left_slope=None, prev_right_slope=None):
    global prev_left_line, prev_right_line
    
    height, width = image.shape[:2]

    enhanced = enhance_image(image)

    color_filtered = select_white_yellow_colors_hsl(enhanced)

    edges = canny_edge_detection(color_filtered)

    left_mask, right_mask = create_left_right_roi(image.shape)
    left_edges = apply_roi(edges, left_mask)
    right_edges = apply_roi(edges, right_mask)

    left_lines = hough_lines(left_edges)
    right_lines = hough_lines(right_edges)

    left_clusters = cluster_lines_by_slope_position(left_lines, image.shape, 'left')
    right_clusters = cluster_lines_by_slope_position(right_lines, image.shape, 'right')

    best_left_cluster = select_best_cluster(left_clusters)
    best_right_cluster = select_best_cluster(right_clusters)

    left_lane, left_slope = fit_lane_line(best_left_cluster, image.shape)
    right_lane, right_slope = fit_lane_line(best_right_cluster, image.shape)

    if check_lines_cross(left_lane, right_lane):
        left_lane = None
        right_lane = None

    left_lane = smooth_lane_line(left_lane, prev_left_line)
    right_lane = smooth_lane_line(right_lane, prev_right_line)
    
    prev_left_line = left_lane
    prev_right_line = right_lane

    result = draw_lane_lines(image, left_lane, right_lane)

    return result, left_slope, right_slope


def process_image(image_path, output_path=None, auto_save=True):
    logger.info("=" * 60)
    logger.info(f"开始处理图片: {image_path}")
    
    try:
        if not os.path.exists(image_path):
            logger.error(f"✗ 图片文件不存在: {image_path}")
            return False
        
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"✗ 无法读取图片: {image_path}")
            return False
        
        logger.info(f"✓ 图片加载成功，尺寸: {image.shape[1]}x{image.shape[0]}")
        
        result, left_slope, right_slope = detect_lane_lines(image)
        
        if auto_save and output_path is None:
            if not ensure_result_directory():
                logger.error("✗ 无法创建结果目录，保存失败")
                return False
            
            original_filename = os.path.basename(image_path)
            output_filename = generate_output_filename(original_filename)
            output_path = os.path.join(RESULT_DIR, output_filename)
        
        if output_path:
            success = save_result_image(result, output_path, f"车道线检测结果 - {os.path.basename(image_path)}")
            if not success:
                return False
        
        logger.info(f"✓ 检测完成 - 左车道斜率: {left_slope:.4f if left_slope else 'N/A'}, 右车道斜率: {right_slope:.4f if right_slope else 'N/A'}")
        
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.title('车道线检测结果')
        plt.axis('off')
        plt.show()
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 处理图片时发生错误: {image_path}")
        logger.error(f"  错误详情: {str(e)}")
        logger.error(f"  堆栈跟踪:\n{traceback.format_exc()}")
        return False


def show_detection_steps(image_path):
    image = cv2.imread(image_path)

    if image is None:
        print(f"无法读取图片: {image_path}")
        return

    height, width = image.shape[:2]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('车道线检测步骤展示', fontsize=16)

    axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('原图')
    axes[0, 0].axis('off')

    enhanced = enhance_image(image)
    axes[0, 1].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title('CLAHE增强')
    axes[0, 1].axis('off')

    hsl_result = select_white_yellow_colors_hsl(enhanced)
    axes[0, 2].imshow(cv2.cvtColor(hsl_result, cv2.COLOR_BGR2RGB))
    axes[0, 2].set_title('HSL颜色过滤')
    axes[0, 2].axis('off')

    edges = canny_edge_detection(hsl_result)
    axes[1, 0].imshow(edges, cmap='gray')
    axes[1, 0].set_title('Canny边缘检测')
    axes[1, 0].axis('off')

    left_mask, right_mask = create_left_right_roi(image.shape)
    combined_mask = cv2.bitwise_or(left_mask, right_mask)
    roi_visual = cv2.cvtColor(combined_mask, cv2.COLOR_GRAY2BGR)
    axes[1, 1].imshow(cv2.cvtColor(roi_visual, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title('左右分离ROI')
    axes[1, 1].axis('off')

    result, _, _ = detect_lane_lines(image)
    axes[1, 2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('最终车道线检测结果')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()


def save_result_image(image, output_path, filename_description=""):
    try:
        ext = os.path.splitext(output_path)[1]
        success = cv2.imencode(ext, image)[1].tofile(output_path)
        
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            logger.info(f"✓ 保存成功: {output_path}")
            logger.info(f"  文件大小: {file_size / 1024:.2f} KB")
            if filename_description:
                logger.info(f"  描述: {filename_description}")
            return True
        else:
            logger.error(f"✗ 保存失败: 文件未创建 - {output_path}")
            return False
            
    except PermissionError as e:
        logger.error(f"✗ 权限不足，无法保存文件: {output_path}")
        logger.error(f"  错误详情: {str(e)}")
        return False
    except cv2.error as e:
        logger.error(f"✗ OpenCV编码错误，无法保存文件: {output_path}")
        logger.error(f"  错误详情: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"✗ 保存文件时发生未知错误: {output_path}")
        logger.error(f"  错误详情: {str(e)}")
        logger.error(f"  堆栈跟踪:\n{traceback.format_exc()}")
        return False

def cv_imread(file_path):
    try:
        cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
        if cv_img is None:
            logger.error(f"✗ 无法解码图片文件: {file_path}")
        return cv_img
    except Exception as e:
        logger.error(f"✗ 读取图片文件失败: {file_path}")
        logger.error(f"  错误详情: {str(e)}")
        return None

def cv_imwrite(file_path, img):
    ext = os.path.splitext(file_path)[1]
    cv2.imencode(ext, img)[1].tofile(file_path)


def process_all_test_images():
    global prev_left_line, prev_right_line, detection_counter
    
    logger.info("=" * 60)
    logger.info("开始批量处理测试图片")
    logger.info(f"脚本目录: {SCRIPT_DIR}")
    logger.info(f"测试图片目录: {TEST_IMAGES_DIR}")
    logger.info(f"结果保存目录: {RESULT_DIR}")
    
    if not os.path.exists(TEST_IMAGES_DIR):
        logger.error(f"✗ 测试图片目录不存在: {TEST_IMAGES_DIR}")
        logger.info("  请创建test_images目录并放入测试图片")
        return False
    
    if not ensure_result_directory():
        logger.error("✗ 无法创建结果目录，批量处理终止")
        return False
    
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    try:
        all_files = os.listdir(TEST_IMAGES_DIR)
        image_files = [f for f in all_files if f.lower().endswith(image_extensions)]
    except PermissionError as e:
        logger.error(f"✗ 权限不足，无法访问测试图片目录: {TEST_IMAGES_DIR}")
        logger.error(f"  错误详情: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"✗ 读取测试图片目录失败: {TEST_IMAGES_DIR}")
        logger.error(f"  错误详情: {str(e)}")
        return False
    
    if not image_files:
        logger.warning(f"⚠ 测试图片目录中没有找到图片文件: {TEST_IMAGES_DIR}")
        logger.info(f"  支持的图片格式: {', '.join(image_extensions)}")
        return False
    
    logger.info(f"✓ 找到 {len(image_files)} 个图片文件")
    logger.info("-" * 60)
    
    dedup = BatchDeduplicator(strategy=DedupStrategy.FILENAME)
    
    detection_counter = 0
    success_count = 0
    fail_count = 0
    skipped_count = 0
    
    for idx, filename in enumerate(image_files, 1):
        prev_left_line = None
        prev_right_line = None
        
        logger.info(f"\n[{idx}/{len(image_files)}] 处理文件: {filename}")
        
        should_process, _ = dedup.should_process_file(filename)
        if not should_process:
            skipped_count += 1
            logger.info(f"  ⚠ 跳过重复文件: {filename}")
            continue
        
        image_path = os.path.join(TEST_IMAGES_DIR, filename)
        
        try:
            image = cv_imread(image_path)
            if image is None:
                logger.error(f"  ✗ 跳过: 无法读取图片")
                fail_count += 1
                continue
            
            logger.info(f"  ✓ 图片尺寸: {image.shape[1]}x{image.shape[0]}")
            
            result, left_slope, right_slope = detect_lane_lines(image)
            
            output_filename = generate_output_filename(filename)
            output_path = os.path.join(RESULT_DIR, output_filename)
            
            if dedup.is_output_duplicate(output_path):
                skipped_count += 1
                continue
            
            success = save_result_image(result, output_path, f"批量处理 - {filename}")
            
            if success:
                success_count += 1
                logger.info(f"  ✓ 进度: 成功 {success_count}/{len(image_files)}")
            else:
                fail_count += 1
                logger.error(f"  ✗ 进度: 失败 {fail_count}/{len(image_files)}")
                
        except Exception as e:
            fail_count += 1
            logger.error(f"  ✗ 处理失败: {str(e)}")
            logger.error(f"  堆栈跟踪:\n{traceback.format_exc()}")
    
    dedup.log_stats()
    
    logger.info("\n" + "=" * 60)
    logger.info("批量处理完成!")
    logger.info(f"  总计: {len(image_files)} 个文件")
    logger.info(f"  成功: {success_count} 个")
    logger.info(f"  失败: {fail_count} 个")
    logger.info(f"  跳过(重复): {skipped_count} 个")
    logger.info(f"  结果保存位置: {RESULT_DIR}")
    logger.info("=" * 60)
    
    return True


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("智能车道检测系统")
    logger.info("=" * 60)
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--batch":
            process_all_test_images()
        else:
            image_path = sys.argv[1]
            output_path = sys.argv[2] if len(sys.argv) > 2 else None

            if "--steps" in sys.argv:
                show_detection_steps(image_path)
            else:
                auto_save = "--no-save" not in sys.argv
                process_image(image_path, output_path, auto_save=auto_save)
    else:
        logger.info("\n使用方法:")
        logger.info("  python lane_detection.py <图片路径> [输出路径]")
        logger.info("  python lane_detection.py <图片路径> --steps")
        logger.info("  python lane_detection.py <图片路径> --no-save")
        logger.info("  python lane_detection.py --batch")
        logger.info("\n参数说明:")
        logger.info("  <图片路径>    输入图片的路径")
        logger.info("  [输出路径]    可选，指定输出路径（默认自动保存到结果目录）")
        logger.info("  --steps      显示检测步骤的可视化")
        logger.info("  --no-save    不自动保存结果")
        logger.info("  --batch      批量处理test_images目录下的所有图片")
        logger.info("\n示例:")
        logger.info("  python lane_detection.py test.jpg")
        logger.info("  python lane_detection.py test.jpg output.jpg")
        logger.info("  python lane_detection.py test.jpg --steps")
        logger.info("  python lane_detection.py --batch")
        logger.info("\n目录结构:")
        logger.info(f"  脚本目录:     {SCRIPT_DIR}")
        logger.info(f"  测试图片目录: {TEST_IMAGES_DIR}")
        logger.info(f"  结果保存目录: {RESULT_DIR}")
        logger.info("\n输出文件命名规则:")
        logger.info("  result_<时间戳>_ID<序号>_<原文件名>.<扩展名>")
        logger.info("  例如: result_20260530_143025_ID0001_test.jpg")
        logger.info("=" * 60)
