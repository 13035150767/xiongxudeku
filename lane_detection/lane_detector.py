import cv2
import numpy as np


class LaneDetector:
    def __init__(self):
        self.canny_threshold1 = 50
        self.canny_threshold2 = 150
        self.gaussian_kernel = 5
        self.hough_rho = 1
        self.hough_theta = np.pi / 180
        self.hough_threshold = 15
        self.hough_min_line_length = 40
        self.hough_max_line_gap = 20
        self.roi_vertices_ratio = [
            (0.05, 1.0),
            (0.40, 0.55),
            (0.60, 0.55),
            (0.95, 1.0),
        ]
        self.prev_left_slope = None
        self.prev_right_slope = None
        self.prev_left_intercept = None
        self.prev_right_intercept = None
        self.smooth_alpha = 0.3
        self.use_hsl = True

    def reset_state(self):
        self.prev_left_slope = None
        self.prev_right_slope = None
        self.prev_left_intercept = None
        self.prev_right_intercept = None

    def color_filter_hsl(self, image):
        hsl = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        lower_white = np.array([0, 190, 0], dtype=np.uint8)
        upper_white = np.array([180, 255, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hsl, lower_white, upper_white)

        lower_yellow = np.array([8, 0, 80], dtype=np.uint8)
        upper_yellow = np.array([45, 255, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hsl, lower_yellow, upper_yellow)

        mask = cv2.bitwise_or(white_mask, yellow_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cv2.bitwise_and(image, image, mask=mask)

    def color_filter_rgb(self, image):
        lower_white = np.array([190, 190, 190], dtype=np.uint8)
        upper_white = np.array([255, 255, 255], dtype=np.uint8)
        white_mask = cv2.inRange(image, lower_white, upper_white)

        lower_yellow = np.array([170, 170, 0], dtype=np.uint8)
        upper_yellow = np.array([255, 255, 150], dtype=np.uint8)
        yellow_mask = cv2.inRange(image, lower_yellow, upper_yellow)

        mask = cv2.bitwise_or(white_mask, yellow_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return cv2.bitwise_and(image, image, mask=mask)

    def grayscale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def gaussian_blur(self, image):
        ksize = self.gaussian_kernel
        if ksize % 2 == 0:
            ksize += 1
        return cv2.GaussianBlur(image, (ksize, ksize), 0)

    def canny_edge(self, image):
        return cv2.Canny(image, self.canny_threshold1, self.canny_threshold2)

    def region_of_interest(self, image):
        h, w = image.shape[:2]
        vertices = np.array([[
            (int(w * r[0]), int(h * r[1])) for r in self.roi_vertices_ratio
        ]], dtype=np.int32)

        mask = np.zeros_like(image)
        if len(image.shape) > 2:
            channel_count = image.shape[2]
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        cv2.fillPoly(mask, vertices, ignore_mask_color)
        return cv2.bitwise_and(image, mask)

    def hough_lines(self, image):
        return cv2.HoughLinesP(
            image,
            self.hough_rho,
            self.hough_theta,
            self.hough_threshold,
            minLineLength=self.hough_min_line_length,
            maxLineGap=self.hough_max_line_gap,
        )

    def calculate_lane_lines(self, lines, image_shape):
        if lines is None:
            return None, None

        left_lines = []
        right_lines = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                if x2 == x1:
                    continue
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1
                length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

                if slope < -0.3:
                    left_lines.append((slope, intercept, length))
                elif slope > 0.3:
                    right_lines.append((slope, intercept, length))

        left_lane = self._fit_lane(left_lines, "left", image_shape)
        right_lane = self._fit_lane(right_lines, "right", image_shape)

        return left_lane, right_lane

    def _fit_lane(self, lines, side, image_shape):
        if not lines:
            if side == "left" and self.prev_left_slope is not None:
                return self.prev_left_slope, self.prev_left_intercept
            elif side == "right" and self.prev_right_slope is not None:
                return self.prev_right_slope, self.prev_right_intercept
            return None

        slopes = [l[0] for l in lines]
        intercepts = [l[1] for l in lines]
        lengths = [l[2] for l in lines]

        median_slope = np.median(slopes)
        tolerance = 0.4

        filtered = [
            (s, i, ln) for s, i, ln in lines
            if abs(s - median_slope) < tolerance
        ]

        if not filtered:
            filtered = lines

        total_length = sum(f[2] for f in filtered)
        if total_length == 0:
            return None

        avg_slope = sum(f[0] * f[2] for f in filtered) / total_length
        avg_intercept = sum(f[1] * f[2] for f in filtered) / total_length

        if side == "left":
            if self.prev_left_slope is not None:
                avg_slope = self.smooth_alpha * avg_slope + (1 - self.smooth_alpha) * self.prev_left_slope
                avg_intercept = self.smooth_alpha * avg_intercept + (1 - self.smooth_alpha) * self.prev_left_intercept
            self.prev_left_slope = avg_slope
            self.prev_left_intercept = avg_intercept
        else:
            if self.prev_right_slope is not None:
                avg_slope = self.smooth_alpha * avg_slope + (1 - self.smooth_alpha) * self.prev_right_slope
                avg_intercept = self.smooth_alpha * avg_intercept + (1 - self.smooth_alpha) * self.prev_right_intercept
            self.prev_right_slope = avg_slope
            self.prev_right_intercept = avg_intercept

        return avg_slope, avg_intercept

    def draw_lane_lines(self, image, left_lane, right_lane):
        line_image = np.zeros_like(image)
        h = image.shape[0]
        w = image.shape[1]

        left_pts = None
        right_pts = None

        if left_lane is not None:
            slope, intercept = left_lane
            y1 = h
            y2 = int(h * 0.55)
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            left_pts = ((x1, y1), (x2, y2))
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 80, 80), 6, cv2.LINE_AA)

        if right_lane is not None:
            slope, intercept = right_lane
            y1 = h
            y2 = int(h * 0.55)
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            right_pts = ((x1, y1), (x2, y2))
            cv2.line(line_image, (x1, y1), (x2, y2), (80, 80, 255), 6, cv2.LINE_AA)

        if left_pts is not None and right_pts is not None:
            poly = np.array([[
                left_pts[0], left_pts[1], right_pts[1], right_pts[0]
            ]], dtype=np.int32)
            overlay = line_image.copy()
            cv2.fillPoly(overlay, poly, (0, 255, 0))
            line_image = cv2.addWeighted(line_image, 1.0, overlay, 0.15, 0.0)

        result = cv2.addWeighted(image, 0.8, line_image, 1.0, 0.0)

        if left_pts is not None:
            cv2.circle(result, left_pts[0], 5, (255, 80, 80), -1)
            cv2.circle(result, left_pts[1], 5, (255, 80, 80), -1)
        if right_pts is not None:
            cv2.circle(result, right_pts[0], 5, (80, 80, 255), -1)
            cv2.circle(result, right_pts[1], 5, (80, 80, 255), -1)

        return result

    def process_image(self, image, return_steps=False):
        if return_steps:
            steps = {}

            steps["original"] = image.copy()

            if self.use_hsl:
                color_filtered = self.color_filter_hsl(image)
            else:
                color_filtered = self.color_filter_rgb(image)
            steps["color_filter"] = color_filtered

            gray = self.grayscale(color_filtered)
            steps["grayscale"] = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

            blurred = self.gaussian_blur(gray)
            steps["gaussian_blur"] = cv2.cvtColor(blurred, cv2.COLOR_GRAY2RGB)

            edges = self.canny_edge(blurred)
            steps["canny_edge"] = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

            roi = self.region_of_interest(edges)
            steps["roi"] = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

            lines = self.hough_lines(roi)
            line_img = np.zeros_like(image)
            if lines is not None:
                for line in lines:
                    for x1, y1, x2, y2 in line:
                        cv2.line(line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            steps["hough_lines"] = line_img

            left_lane, right_lane = self.calculate_lane_lines(lines, image.shape)
            result = self.draw_lane_lines(image, left_lane, right_lane)
            steps["result"] = result

            return steps
        else:
            if self.use_hsl:
                color_filtered = self.color_filter_hsl(image)
            else:
                color_filtered = self.color_filter_rgb(image)
            gray = self.grayscale(color_filtered)
            blurred = self.gaussian_blur(gray)
            edges = self.canny_edge(blurred)
            roi = self.region_of_interest(edges)
            lines = self.hough_lines(roi)
            left_lane, right_lane = self.calculate_lane_lines(lines, image.shape)
            return self.draw_lane_lines(image, left_lane, right_lane)

    def process_video_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result_rgb = self.process_image(frame_rgb)
        return cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
