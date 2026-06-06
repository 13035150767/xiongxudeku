import cv2
import numpy as np
from ultralytics import YOLO

DETECT_MODEL_PATH = r"C:\Users\孙书文\Desktop\yolov8n.pt"
SEG_MODEL_PATH = r"C:\Users\孙书文\Desktop\yolov8n-seg.pt"

VEHICLE_CLASSES = [2, 5, 7]

CLASS_NAMES = {
    2: "car",
    5: "bus",
    7: "truck",
}

CLASS_NAMES_CN = {
    2: "汽车",
    5: "公交车",
    7: "卡车",
}

COLORS = {
    2: (0, 255, 0),
    5: (0, 165, 255),
    7: (0, 0, 255),
}

_detect_model = None
_seg_model = None


def get_detect_model():
    global _detect_model
    if _detect_model is None:
        _detect_model = YOLO(DETECT_MODEL_PATH)
    return _detect_model


def get_seg_model():
    global _seg_model
    if _seg_model is None:
        _seg_model = YOLO(SEG_MODEL_PATH)
    return _seg_model


def detect_vehicles(image_bgr, conf=0.25):
    model = get_detect_model()
    results = model(image_bgr, conf=conf, verbose=False)
    vehicles = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            if cls_id in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf_val = float(box.conf[0])
                vehicles.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": cls_id,
                    "class_name": CLASS_NAMES.get(cls_id, "unknown"),
                    "class_name_cn": CLASS_NAMES_CN.get(cls_id, "未知"),
                    "confidence": round(conf_val, 3),
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                })
    return vehicles


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


def hungarian_match(cost_matrix):
    n_row = len(cost_matrix)
    n_col = len(cost_matrix[0]) if n_row > 0 else 0
    if n_row == 0 or n_col == 0:
        return []

    row_matched = [False] * n_row
    col_matched = [False] * n_col
    matches = []

    for _ in range(min(n_row, n_col)):
        best_val = float("inf")
        best_r, best_c = -1, -1
        for r in range(n_row):
            if row_matched[r]:
                continue
            for c in range(n_col):
                if col_matched[c]:
                    continue
                if cost_matrix[r][c] < best_val:
                    best_val = cost_matrix[r][c]
                    best_r, best_c = r, c
        if best_r >= 0 and best_val < 1e9:
            matches.append((best_r, best_c, best_val))
            row_matched[best_r] = True
            col_matched[best_c] = True
        else:
            break

    return matches


class Track:
    def __init__(self, track_id, bbox, class_id, confidence):
        self.track_id = track_id
        self.bbox = list(bbox)
        self.class_id = class_id
        self.confidence = confidence
        self.age = 0
        self.missed = 0
        self.center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        self.prev_center = self.center
        self.counted = False

    def update(self, bbox, class_id, confidence):
        self.prev_center = self.center
        self.bbox = list(bbox)
        self.class_id = class_id
        self.confidence = confidence
        self.center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        self.age += 1
        self.missed = 0

    def mark_missed(self):
        self.missed += 1
        dx = self.center[0] - self.prev_center[0]
        dy = self.center[1] - self.prev_center[1]
        self.prev_center = self.center
        self.center = (int(self.center[0] + dx), int(self.center[1] + dy))
        self.bbox[0] += dx
        self.bbox[1] += dy
        self.bbox[2] += dx
        self.bbox[3] += dy


class MultiTracker:
    def __init__(self, iou_threshold=0.3, max_missed=15):
        self.tracks = {}
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed

    def reset(self):
        self.tracks = {}
        self.next_id = 1

    def update(self, vehicles):
        active_ids = [tid for tid, t in self.tracks.items() if t.missed <= self.max_missed]

        if len(active_ids) == 0 and len(vehicles) == 0:
            return {}

        if len(active_ids) == 0:
            assignments = {}
            for i, v in enumerate(vehicles):
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = Track(tid, v["bbox"], v["class_id"], v["confidence"])
                assignments[i] = tid
            return assignments

        if len(vehicles) == 0:
            for tid in active_ids:
                self.tracks[tid].mark_missed()
            return {}

        n_tracks = len(active_ids)
        n_dets = len(vehicles)

        cost_matrix = []
        for i, tid in enumerate(active_ids):
            row = []
            track = self.tracks[tid]
            for j, v in enumerate(vehicles):
                iou = compute_iou(track.bbox, v["bbox"])
                cx_diff = abs(track.center[0] - v["center"][0])
                cy_diff = abs(track.center[1] - v["center"][1])
                dist = np.sqrt(cx_diff ** 2 + cy_diff ** 2)

                if iou > 0.1 or dist < 120:
                    cost = 1.0 - iou + dist / 500.0
                else:
                    cost = 1e9
                row.append(cost)
            cost_matrix.append(row)

        matches = hungarian_match(cost_matrix)

        matched_tracks = set()
        matched_dets = set()
        assignments = {}

        for r, c, cost in matches:
            if cost < 1e8:
                tid = active_ids[r]
                v = vehicles[c]
                self.tracks[tid].update(v["bbox"], v["class_id"], v["confidence"])
                assignments[c] = tid
                matched_tracks.add(tid)
                matched_dets.add(c)

        for i, tid in enumerate(active_ids):
            if tid not in matched_tracks:
                self.tracks[tid].mark_missed()

        for j, v in enumerate(vehicles):
            if j not in matched_dets:
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = Track(tid, v["bbox"], v["class_id"], v["confidence"])
                assignments[j] = tid

        dead = [tid for tid, t in self.tracks.items() if t.missed > self.max_missed]
        for tid in dead:
            del self.tracks[tid]

        return assignments


class TrafficCounter:
    def __init__(self):
        self.count_line_y_ratio = 0.6
        self.total_count = 0
        self.count_up = 0
        self.count_down = 0
        self.tracker = MultiTracker(iou_threshold=0.25, max_missed=15)

    def reset(self):
        self.total_count = 0
        self.count_up = 0
        self.count_down = 0
        self.tracker.reset()

    def process_frame(self, image_bgr, vehicles):
        h, w = image_bgr.shape[:2]
        line_y = int(h * self.count_line_y_ratio)

        assignments = self.tracker.update(vehicles)

        for det_idx, tid in assignments.items():
            track = self.tracker.tracks.get(tid)
            if track is None or track.counted:
                continue
            if track.missed > 0:
                continue

            prev_cy = track.prev_center[1]
            curr_cy = track.center[1]

            if prev_cy < line_y and curr_cy >= line_y:
                self.count_down += 1
                self.total_count += 1
                track.counted = True
            elif prev_cy >= line_y and curr_cy < line_y:
                self.count_up += 1
                self.total_count += 1
                track.counted = True

        result = image_bgr.copy()

        cv2.line(result, (0, line_y), (w, line_y), (0, 255, 255), 2)
        cv2.putText(result, "COUNT LINE", (w - 160, line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        for tid, track in self.tracker.tracks.items():
            if track.missed > 0:
                continue
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            color = COLORS.get(track.class_id, (0, 255, 0))
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

            cn = CLASS_NAMES_CN.get(track.class_id, "?")
            label = f"ID:{tid} {cn} {track.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(result, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

            cv2.circle(result, (int(track.center[0]), int(track.center[1])), 4, color, -1)

        cv2.rectangle(result, (10, 10), (340, 110), (0, 0, 0), -1)
        cv2.putText(result, f"Total: {self.total_count}",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(result, f"Down: {self.count_down}  Up: {self.count_up}",
                    (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        return result

    def process_image(self, image_bgr, conf=0.25):
        self.reset()
        vehicles = detect_vehicles(image_bgr, conf=conf)
        annotated = self.process_frame(image_bgr, vehicles)
        stats = {
            "total_count": len(vehicles),
            "count_up": self.count_up,
            "count_down": self.count_down,
            "vehicles": vehicles,
            "by_type": {},
        }
        for v in vehicles:
            cn = v["class_name_cn"]
            stats["by_type"][cn] = stats["by_type"].get(cn, 0) + 1
        return annotated, stats


class SegTrafficCounter:
    def __init__(self):
        self.seg_colors = {}
        self.tracker = MultiTracker(iou_threshold=0.25, max_missed=15)
        self.count_line_y_ratio = 0.6
        self.total_count = 0
        self.count_up = 0
        self.count_down = 0

    def _get_color(self, cls_id):
        if cls_id not in self.seg_colors:
            np.random.seed(cls_id * 37 + 7)
            self.seg_colors[cls_id] = (
                int(np.random.randint(80, 255)),
                int(np.random.randint(80, 255)),
                int(np.random.randint(80, 255)),
            )
        return self.seg_colors[cls_id]

    def reset(self):
        self.tracker.reset()
        self.total_count = 0
        self.count_up = 0
        self.count_down = 0

    def process_frame(self, image_bgr, conf=0.25):
        model = get_seg_model()
        results = model(image_bgr, conf=conf, verbose=False)

        h, w = image_bgr.shape[:2]
        line_y = int(h * self.count_line_y_ratio)

        overlay = image_bgr.copy()
        result = image_bgr.copy()
        vehicle_count = 0
        class_counts = {}
        vehicles = []

        for r in results:
            if r.masks is None:
                continue
            for i, box in enumerate(r.boxes):
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue
                vehicle_count += 1
                cn = CLASS_NAMES_CN.get(cls_id, "未知")
                class_counts[cn] = class_counts.get(cn, 0) + 1

                color = self._get_color(cls_id)
                mask = r.masks.data[i].cpu().numpy()
                mask_resized = cv2.resize(mask, (w, h))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

                colored_mask = np.zeros_like(image_bgr)
                colored_mask[mask_binary == 1] = color
                overlay = cv2.addWeighted(overlay, 1.0, colored_mask, 0.5, 0)

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                conf_val = float(box.conf[0])
                vehicles.append({
                    "bbox": [x1, y1, x2, y2],
                    "class_id": cls_id,
                    "class_name_cn": cn,
                    "confidence": round(conf_val, 3),
                    "center": (cx, cy),
                })

        assignments = self.tracker.update(vehicles)

        for det_idx, tid in assignments.items():
            track = self.tracker.tracks.get(tid)
            if track is None or track.missed > 0 or track.counted:
                continue
            prev_cy = track.prev_center[1]
            curr_cy = track.center[1]
            if prev_cy < line_y and curr_cy >= line_y:
                self.count_down += 1
                self.total_count += 1
                track.counted = True
            elif prev_cy >= line_y and curr_cy < line_y:
                self.count_up += 1
                self.total_count += 1
                track.counted = True

        result = cv2.addWeighted(result, 0.6, overlay, 0.4, 0)

        cv2.line(result, (0, line_y), (w, line_y), (0, 255, 255), 2)
        cv2.putText(result, "COUNT LINE", (w - 160, line_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        for tid, track in self.tracker.tracks.items():
            if track.missed > 0:
                continue
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            color = COLORS.get(track.class_id, (0, 255, 0))
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            cn = CLASS_NAMES_CN.get(track.class_id, "?")
            label = f"ID:{tid} {cn}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(result, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(result, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)

        cv2.rectangle(result, (10, 10), (380, 110), (0, 0, 0), -1)
        cv2.putText(result, f"Vehicles: {vehicle_count}  Count: {self.total_count}",
                    (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(result, f"Down: {self.count_down}  Up: {self.count_up}",
                    (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        stats = {
            "vehicle_count": vehicle_count,
            "total_count": self.total_count,
            "count_down": self.count_down,
            "count_up": self.count_up,
            "by_type": class_counts,
        }
        return result, stats

    def process_image(self, image_bgr, conf=0.25):
        self.reset()
        annotated, stats = self.process_frame(image_bgr, conf=conf)
        stats["total_count"] = stats["vehicle_count"]
        return annotated, stats

    def process_video_frame(self, image_bgr, conf=0.25):
        annotated, _ = self.process_frame(image_bgr, conf=conf)
        return annotated


class LaneOccupancyAnalyzer:
    def __init__(self, num_lanes=3):
        self.num_lanes = num_lanes
        self.tracker = MultiTracker(iou_threshold=0.25, max_missed=15)
        self.lane_polys = []
        self.lane_density = []
        self.frame_count = 0
        self.smooth_alpha = 0.3

    def _init_lanes(self, image_bgr):
        h, w = image_bgr.shape[:2]
        cx = w // 2
        top_y = int(h * 0.3)
        bot_y = int(h * 0.98)

        top_half_w = int(w * 0.08)
        bot_half_w = int(w * 0.48)

        self.lane_polys = []
        self.lane_density = [0.0] * self.num_lanes

        for i in range(self.num_lanes):
            left_t = cx - top_half_w + i * (2 * top_half_w // self.num_lanes)
            right_t = cx - top_half_w + (i + 1) * (2 * top_half_w // self.num_lanes)
            left_b = cx - bot_half_w + i * (2 * bot_half_w // self.num_lanes)
            right_b = cx - bot_half_w + (i + 1) * (2 * bot_half_w // self.num_lanes)

            self.lane_polys.append(np.array([[
                (left_b, bot_y),
                (left_t, top_y),
                (right_t, top_y),
                (right_b, bot_y),
            ]], dtype=np.int32))

    def _point_in_poly(self, px, py, poly):
        poly_flat = poly.reshape(-1, 2)
        n = len(poly_flat)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly_flat[i]
            xj, yj = poly_flat[j]
            if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def _compute_density(self, lane_idx, vehicles, image_bgr):
        poly = self.lane_polys[lane_idx]
        poly_flat = poly.reshape(-1, 2)
        area = cv2.contourArea(poly_flat)
        if area <= 0:
            return 0.0

        vehicle_area = 0
        for v in vehicles:
            vx1, vy1, vx2, vy2 = v["bbox"]
            cx, cy = v["center"]
            if self._point_in_poly(cx, cy, poly):
                v_area = (vx2 - vx1) * (vy2 - vy1)
                vehicle_area += v_area

        density = min(vehicle_area / area, 1.0)
        return density

    def process_image(self, image_bgr, conf=0.25):
        self._init_lanes(image_bgr)
        vehicles = detect_vehicles(image_bgr, conf=conf)
        assignments = self.tracker.update(vehicles)
        self.frame_count += 1

        lane_vehicles = [[] for _ in range(self.num_lanes)]
        outside_vehicles = []

        for det_idx, tid in assignments.items():
            track = self.tracker.tracks.get(tid)
            if track is None or track.missed > 0:
                continue
            cx, cy = track.center
            assigned = False
            for i in range(self.num_lanes):
                if self._point_in_poly(cx, cy, self.lane_polys[i]):
                    lane_vehicles[i].append(track)
                    assigned = True
                    break
            if not assigned:
                outside_vehicles.append(track)

        raw_density = []
        for i in range(self.num_lanes):
            d = self._compute_density(i, vehicles, image_bgr)
            raw_density.append(d)

        if self.frame_count == 1:
            self.lane_density = raw_density[:]
        else:
            for i in range(self.num_lanes):
                self.lane_density[i] = self.smooth_alpha * raw_density[i] + (1 - self.smooth_alpha) * self.lane_density[i]

        result = image_bgr.copy()

        status_colors = {
            "FREE": (0, 200, 0),
            "NORMAL": (0, 200, 200),
            "SLOW": (0, 165, 255),
            "JAM": (0, 0, 255),
        }

        for i in range(self.num_lanes):
            n = len(lane_vehicles[i])
            density = self.lane_density[i]
            rate = round(density * 100, 1)

            if density < 0.05:
                status = "FREE"
            elif density < 0.2:
                status = "NORMAL"
            elif density < 0.4:
                status = "SLOW"
            else:
                status = "JAM"

            color = status_colors[status]

            overlay_lane = result.copy()
            cv2.fillPoly(overlay_lane, self.lane_polys[i], color)
            result = cv2.addWeighted(result, 0.8, overlay_lane, 0.2, 0)
            cv2.polylines(result, self.lane_polys[i], True, color, 2)

            poly_flat = self.lane_polys[i].reshape(-1, 2)
            mid_x = int(np.mean(poly_flat[:, 0]))
            mid_y = int(np.mean(poly_flat[:, 1]))

            cv2.rectangle(result, (mid_x - 55, mid_y - 35), (mid_x + 55, mid_y + 45), (0, 0, 0), -1)
            cv2.putText(result, f"Lane {i + 1}", (mid_x - 45, mid_y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(result, f"{n} cars {rate}%", (mid_x - 50, mid_y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            cv2.putText(result, status, (mid_x - 25, mid_y + 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        for tid, track in self.tracker.tracks.items():
            if track.missed > 0:
                continue
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            color = COLORS.get(track.class_id, (0, 255, 0))
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            cn = CLASS_NAMES_CN.get(track.class_id, "?")
            label = f"ID:{tid} {cn}"
            cv2.putText(result, label, (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        total_in_lanes = sum(len(lv) for lv in lane_vehicles)
        total_tracked = total_in_lanes + len(outside_vehicles)

        cv2.rectangle(result, (10, 10), (420, 70), (0, 0, 0), -1)
        cv2.putText(result, f"Total Vehicles: {total_tracked}",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        lane_stats = []
        for i in range(self.num_lanes):
            n = len(lane_vehicles[i])
            density = self.lane_density[i]
            rate = round(density * 100, 1)
            if density < 0.05:
                status = "FREE"
            elif density < 0.2:
                status = "NORMAL"
            elif density < 0.4:
                status = "SLOW"
            else:
                status = "JAM"
            lane_stats.append({
                "lane": i + 1,
                "vehicle_count": n,
                "occupancy_rate": rate,
                "status": status,
            })

        stats = {
            "total_vehicles": total_tracked,
            "lanes": lane_stats,
        }
        return result, stats

    def process_video_frame(self, image_bgr, conf=0.25):
        annotated, _ = self.process_image(image_bgr, conf=conf)
        return annotated


class SpeedEstimator:
    def __init__(self):
        self.ppm = 8.0
        self.fps = 25.0
        self.tracker = MultiTracker(iou_threshold=0.25, max_missed=15)
        self.speeds = {}

    def reset(self):
        self.tracker.reset()
        self.speeds = {}

    def process_frame(self, image_bgr, vehicles):
        assignments = self.tracker.update(vehicles)

        for det_idx, tid in assignments.items():
            track = self.tracker.tracks.get(tid)
            if track is None or track.missed > 0:
                continue
            if track.age > 0:
                dx = track.center[0] - track.prev_center[0]
                dy = track.center[1] - track.prev_center[1]
                pixel_dist = np.sqrt(dx ** 2 + dy ** 2)
                real_dist_m = pixel_dist / self.ppm
                speed_kmh = (real_dist_m * self.fps) * 3.6
                if tid in self.speeds:
                    self.speeds[tid] = 0.7 * self.speeds[tid] + 0.3 * speed_kmh
                else:
                    self.speeds[tid] = speed_kmh

        result = image_bgr.copy()
        for tid, track in self.tracker.tracks.items():
            if track.missed > 0:
                continue
            x1, y1, x2, y2 = [int(v) for v in track.bbox]
            speed = self.speeds.get(tid, 0)
            color = (0, 255, 0) if speed < 40 else (0, 165, 255) if speed < 60 else (0, 0, 255)
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            label = f"ID:{tid} {speed:.1f}km/h"
            cv2.putText(result, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return result

    def process_image(self, image_bgr, conf=0.25):
        self.reset()
        vehicles = detect_vehicles(image_bgr, conf=conf)
        annotated = self.process_frame(image_bgr, vehicles)
        stats = {
            "vehicle_count": len(vehicles),
            "speeds": {},
            "avg_speed": 0,
        }
        for tid, spd in self.speeds.items():
            stats["speeds"][str(tid)] = round(spd, 1)
        if self.speeds:
            stats["avg_speed"] = round(sum(self.speeds.values()) / len(self.speeds), 1)
        return annotated, stats


class HeatmapGenerator:
    def __init__(self):
        self.heatmap = None
        self.decay = 0.95
        self.frame_count = 0

    def reset(self, shape=None):
        if shape is not None:
            self.heatmap = np.zeros((shape[0], shape[1]), dtype=np.float32)
        else:
            self.heatmap = None
        self.frame_count = 0

    def process_frame(self, image_bgr, vehicles):
        h, w = image_bgr.shape[:2]
        if self.heatmap is None:
            self.heatmap = np.zeros((h, w), dtype=np.float32)

        self.heatmap *= self.decay

        for v in vehicles:
            x1, y1, x2, y2 = v["bbox"]
            cx, cy = v["center"]
            bw = max(x2 - x1, 10)
            bh = max(y2 - y1, 10)
            sigma_x = bw / 4
            sigma_y = bh / 4

            y_min = max(0, y1 - bh // 2)
            y_max = min(h, y2 + bh // 2)
            x_min = max(0, x1 - bw // 2)
            x_max = min(w, x2 + bw // 2)

            yy, xx = np.mgrid[y_min:y_max, x_min:x_max]
            g = np.exp(-((xx - cx) ** 2 / (2 * sigma_x ** 2) + (yy - cy) ** 2 / (2 * sigma_y ** 2)))
            self.heatmap[y_min:y_max, x_min:x_max] += g * 50

        self.frame_count += 1

        heatmap_norm = np.clip(self.heatmap, 0, 255).astype(np.uint8)
        heatmap_colored = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image_bgr, 0.6, heatmap_colored, 0.4, 0)

        for v in vehicles:
            x1, y1, x2, y2 = v["bbox"]
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 1)

        return overlay

    def process_image(self, image_bgr, conf=0.25):
        self.reset(shape=image_bgr.shape[:2])
        vehicles = detect_vehicles(image_bgr, conf=conf)
        for _ in range(5):
            self.process_frame(image_bgr, vehicles)
        annotated = self.process_frame(image_bgr, vehicles)
        stats = {
            "vehicle_count": len(vehicles),
            "max_intensity": round(float(np.max(self.heatmap)), 1),
            "avg_intensity": round(float(np.mean(self.heatmap[self.heatmap > 0])), 1) if np.any(self.heatmap > 0) else 0,
        }
        return annotated, stats
