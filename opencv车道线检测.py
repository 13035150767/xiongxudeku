import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

# -------------------------- 全局平滑（防闪烁）
# --------------------------
prev_left = None
prev_right = None
alpha = 0.7


def smooth_line(current, prev, alpha=0.7):
    if current is None:
        return prev
    if prev is None:
        return current
    return [int(round(alpha * c + (1 - alpha) * p)) for c, p in zip(current, prev)]


# -------------------------- 颜色过滤
# --------------------------
def color_filter(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 200), (180, 30, 255))
    yellow = cv2.inRange(hsv, (15, 50, 100), (35, 255, 255))
    mask = cv2.bitwise_or(white, yellow)
    return cv2.bitwise_and(img, img, mask=mask)


# -------------------------- 灰度
# --------------------------
def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# -------------------------- 高斯模糊
# --------------------------
def blur(img):
    return cv2.GaussianBlur(img, (7, 7), 0)


# -------------------------- Canny
# --------------------------
def canny(img):
    med = np.median(img)
    low = int(max(0, med * 0.7))
    high = int(min(255, med * 1.3))
    return cv2.Canny(img, low, high)


# -------------------------- ROI
# --------------------------
def roi(img):
    h, w = img.shape[:2]
    pts = np.array([[
        (int(w * 0.05), h),
        (int(w * 0.95), h),
        (int(w * 0.6), int(h * 0.6)),
        (int(w * 0.4), int(h * 0.6))
    ]], np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, pts, 255)
    return cv2.bitwise_and(img, mask)


# -------------------------- 车道线拟合
# --------------------------
def fit_lines(img, lines):
    left, right = [], []
    if lines is None:
        return None, None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        k, b = np.polyfit((x1, x2), (y1, y2), 1)
        if 0.4 < abs(k) < 2.0:
            if k < 0:
                left.append((k, b))
            else:
                right.append((k, b))

    def make(lst):
        if not lst: return None
        k, b = np.average(lst, axis=0)
        y1 = img.shape[0]
        y2 = int(y1 * 0.6)
        x1 = int((y1 - b) / k)
        x2 = int((y2 - b) / k)
        return [x1, y1, x2, y2]

    return make(left), make(right)


# -------------------------- 画线
# --------------------------
def draw_lines(img, lines):
    out = np.zeros_like(img)
    for line in lines:
        if line is None: continue
        x1, y1, x2, y2 = line
        cv2.line(out, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return out


# -------------------------- 帧处理
# --------------------------
def process_frame(frame):
    global prev_left, prev_right
    img = frame.copy()
    cf = color_filter(img)
    g = gray(cf)
    b = blur(g)
    c = canny(b)
    c = cv2.dilate(c, np.ones((3, 3)), iterations=1)
    r = roi(c)
    lines = cv2.HoughLinesP(r, 2, np.pi / 180, 50, minLineLength=30, maxLineGap=20)
    left, right = fit_lines(img, lines)
    left = smooth_line(left, prev_left, alpha)
    right = smooth_line(right, prev_right, alpha)
    prev_left, prev_right = left, right
    line_img = draw_lines(img, [left, right])
    final = cv2.addWeighted(img, 0.8, line_img, 1, 0)
    return final, [cf, g, b, c, r, line_img]


# -------------------------- 图片检测：第一张直接输出车道检测图
# --------------------------
def detect_image(path):
    img = cv2.imread(path)
    final, steps = process_frame(img)

    # 第一张：最终车道线检测结果
    plt.figure()
    plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
    plt.title("Lane Detection Result")
    plt.axis("off")
    plt.show()

    # 后面依次输出步骤图（不含原图）
    names = ["ColorFilter", "Gray", "Blur", "Canny", "ROI", "Lane Lines"]
    for step, name in zip(steps, names):
        plt.figure()
        if len(step.shape) == 2:
            plt.imshow(step, cmap="gray")
        else:
            plt.imshow(cv2.cvtColor(step, cv2.COLOR_BGR2RGB))
        plt.title(name)
        plt.axis("off")
        plt.show()


# -------------------------- 视频检测（稳定不报错）
# --------------------------
def detect_video(path, out_path="lane_output.mp4"):
    cap = cv2.VideoCapture(path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    while True:
        ret, frame = cap.read()
        if not ret: break
        res, _ = process_frame(frame)
        writer.write(res)
        cv2.imshow("Lane Detection", res)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    writer.release()
    cv2.destroyAllWindows()


# ====================== 运行 ======================
if __name__ == "__main__":
    # 图片：第一张直接是车道检测结果！
    # detect_image("data/test_images/solidWhiteCurve.jpg")

    # 视频：稳定不闪烁
    # detect_video("data/test_videos/solidWhiteRight.mp4")
    detect_video("data/test_videos/testVideo2.mp4")