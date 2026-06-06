import os
import uuid
import base64
import subprocess
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from lane_detector import LaneDetector
from vehicle_analyzer import TrafficCounter, SpeedEstimator, HeatmapGenerator, SegTrafficCounter, LaneOccupancyAnalyzer

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join("static", "uploads")
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

detector = LaneDetector()
traffic_counter = TrafficCounter()
speed_estimator = SpeedEstimator()
heatmap_gen = HeatmapGenerator()
seg_traffic_counter = SegTrafficCounter()
lane_occupancy = LaneOccupancyAnalyzer()


def image_to_base64(image):
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buffer).decode("utf-8")


def load_image_bgr(file_storage):
    file_bytes = np.frombuffer(file_storage.read(), np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image_bgr


def load_image_rgb(file_storage):
    image_bgr = load_image_bgr(file_storage)
    if image_bgr is None:
        return None
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _get_conf(form):
    try:
        return float(form.get("conf", "0.25"))
    except (ValueError, TypeError):
        return 0.25


def _process_video_with_fn(input_path, process_fn, fps_override=None):
    raw_filename = uuid.uuid4().hex + "_raw.mp4"
    raw_path = os.path.join(app.config["UPLOAD_FOLDER"], raw_filename)
    output_filename = uuid.uuid4().hex + ".mp4"
    output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None, 0, 0

    fps = fps_override or cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(raw_path, fourcc, fps, (w, h))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        result_frame = process_fn(frame)
        out.write(result_frame)
        frame_count += 1

    cap.release()
    out.release()

    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", raw_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        os.remove(raw_path)
    except subprocess.CalledProcessError:
        if os.path.exists(output_path):
            os.remove(output_path)
        os.rename(raw_path, output_path)

    return output_filename, total_frames, frame_count


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/detect_image", methods=["POST"])
def detect_image():
    if "file" not in request.files:
        return jsonify({"error": "请上传图片文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400

    image = load_image_rgb(file)
    if image is None:
        return jsonify({"error": "无法读取图片"}), 400

    _apply_params(request.form)
    detector.reset_state()

    steps = detector.process_image(image, return_steps=True)

    result = {}
    for key, img in steps.items():
        result[key] = image_to_base64(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    return jsonify({"steps": result})


@app.route("/api/detect_video", methods=["POST"])
def detect_video():
    if "file" not in request.files:
        return jsonify({"error": "请上传视频文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400

    ext = os.path.splitext(file.filename)[1] or ".mp4"
    input_filename = uuid.uuid4().hex + ext
    input_path = os.path.join(app.config["UPLOAD_FOLDER"], input_filename)
    file.save(input_path)

    _apply_params(request.form)
    detector.reset_state()

    output_filename, total_frames, processed = _process_video_with_fn(
        input_path, detector.process_video_frame
    )
    os.remove(input_path)

    if output_filename is None:
        return jsonify({"error": "无法打开视频文件"}), 400

    return jsonify({
        "video_url": f"/static/uploads/{output_filename}",
        "total_frames": total_frames,
        "processed_frames": processed,
    })


@app.route("/api/traffic_count", methods=["POST"])
def traffic_count():
    if "file" not in request.files:
        return jsonify({"error": "请上传文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400

    conf = _get_conf(request.form)
    is_video = file.content_type and file.content_type.startswith("video/")

    if is_video:
        ext = os.path.splitext(file.filename)[1] or ".mp4"
        input_filename = uuid.uuid4().hex + ext
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], input_filename)
        file.save(input_path)

        traffic_counter.reset()
        from vehicle_analyzer import detect_vehicles

        def process_fn(frame):
            vehicles = detect_vehicles(frame, conf=conf)
            return traffic_counter.process_frame(frame, vehicles)

        output_filename, total_frames, processed = _process_video_with_fn(
            input_path, process_fn
        )
        os.remove(input_path)

        if output_filename is None:
            return jsonify({"error": "无法打开视频文件"}), 400

        return jsonify({
            "video_url": f"/static/uploads/{output_filename}",
            "total_frames": total_frames,
            "processed_frames": processed,
            "stats": {
                "total_count": traffic_counter.total_count,
                "by_type": {},
            },
        })
    else:
        image_bgr = load_image_bgr(file)
        if image_bgr is None:
            return jsonify({"error": "无法读取图片"}), 400

        traffic_counter.reset()
        annotated, stats = traffic_counter.process_image(image_bgr, conf=conf)

        return jsonify({
            "image": image_to_base64(annotated),
            "stats": stats,
        })


@app.route("/api/segment", methods=["POST"])
def segment():
    if "file" not in request.files:
        return jsonify({"error": "请上传文件"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400
    conf = _get_conf(request.form)
    is_video = file.content_type and file.content_type.startswith("video/")

    if is_video:
        ext = os.path.splitext(file.filename)[1] or ".mp4"
        input_filename = uuid.uuid4().hex + ext
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], input_filename)
        file.save(input_path)
        seg_traffic_counter.reset()

        def process_fn(frame):
            return seg_traffic_counter.process_video_frame(frame, conf=conf)

        output_filename, total_frames, processed = _process_video_with_fn(input_path, process_fn)
        os.remove(input_path)
        if output_filename is None:
            return jsonify({"error": "无法打开视频文件"}), 400
        return jsonify({
            "video_url": f"/static/uploads/{output_filename}",
            "total_frames": total_frames,
            "processed_frames": processed,
            "stats": {
                "total_count": seg_traffic_counter.total_count,
                "count_down": seg_traffic_counter.count_down,
                "count_up": seg_traffic_counter.count_up,
            },
        })
    else:
        image_bgr = load_image_bgr(file)
        if image_bgr is None:
            return jsonify({"error": "无法读取图片"}), 400
        seg_traffic_counter.reset()
        annotated, stats = seg_traffic_counter.process_image(image_bgr, conf=conf)
        return jsonify({"image": image_to_base64(annotated), "stats": stats})


@app.route("/api/lane_occupancy", methods=["POST"])
def lane_occupancy_api():
    if "file" not in request.files:
        return jsonify({"error": "请上传文件"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400
    conf = _get_conf(request.form)
    is_video = file.content_type and file.content_type.startswith("video/")

    if is_video:
        ext = os.path.splitext(file.filename)[1] or ".mp4"
        input_filename = uuid.uuid4().hex + ext
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], input_filename)
        file.save(input_path)
        lane_occupancy.tracker.reset()

        def process_fn(frame):
            return lane_occupancy.process_video_frame(frame, conf=conf)

        output_filename, total_frames, processed = _process_video_with_fn(input_path, process_fn)
        os.remove(input_path)
        if output_filename is None:
            return jsonify({"error": "无法打开视频文件"}), 400
        return jsonify({
            "video_url": f"/static/uploads/{output_filename}",
            "total_frames": total_frames,
            "processed_frames": processed,
        })
    else:
        image_bgr = load_image_bgr(file)
        if image_bgr is None:
            return jsonify({"error": "无法读取图片"}), 400
        annotated, stats = lane_occupancy.process_image(image_bgr, conf=conf)
        return jsonify({"image": image_to_base64(annotated), "stats": stats})


@app.route("/api/speed_detect", methods=["POST"])
def speed_detect():
    if "file" not in request.files:
        return jsonify({"error": "请上传文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400

    conf = _get_conf(request.form)
    is_video = file.content_type and file.content_type.startswith("video/")

    if is_video:
        ext = os.path.splitext(file.filename)[1] or ".mp4"
        input_filename = uuid.uuid4().hex + ext
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], input_filename)
        file.save(input_path)

        speed_estimator.reset()
        from vehicle_analyzer import detect_vehicles

        def process_fn(frame):
            vehicles = detect_vehicles(frame, conf=conf)
            return speed_estimator.process_frame(frame, vehicles)

        output_filename, total_frames, processed = _process_video_with_fn(
            input_path, process_fn
        )
        os.remove(input_path)

        if output_filename is None:
            return jsonify({"error": "无法打开视频文件"}), 400

        return jsonify({
            "video_url": f"/static/uploads/{output_filename}",
            "total_frames": total_frames,
            "processed_frames": processed,
            "stats": {
                "speeds": {k: round(v, 1) for k, v in speed_estimator.speeds.items()},
                "avg_speed": round(sum(speed_estimator.speeds.values()) / len(speed_estimator.speeds), 1) if speed_estimator.speeds else 0,
            },
        })
    else:
        image_bgr = load_image_bgr(file)
        if image_bgr is None:
            return jsonify({"error": "无法读取图片"}), 400

        speed_estimator.reset()
        annotated, stats = speed_estimator.process_image(image_bgr, conf=conf)

        return jsonify({
            "image": image_to_base64(annotated),
            "stats": stats,
        })


@app.route("/api/heatmap", methods=["POST"])
def heatmap():
    if "file" not in request.files:
        return jsonify({"error": "请上传文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400

    conf = _get_conf(request.form)
    is_video = file.content_type and file.content_type.startswith("video/")

    if is_video:
        ext = os.path.splitext(file.filename)[1] or ".mp4"
        input_filename = uuid.uuid4().hex + ext
        input_path = os.path.join(app.config["UPLOAD_FOLDER"], input_filename)
        file.save(input_path)

        heatmap_gen.reset()
        from vehicle_analyzer import detect_vehicles

        def process_fn(frame):
            vehicles = detect_vehicles(frame, conf=conf)
            return heatmap_gen.process_frame(frame, vehicles)

        output_filename, total_frames, processed = _process_video_with_fn(
            input_path, process_fn
        )
        os.remove(input_path)

        if output_filename is None:
            return jsonify({"error": "无法打开视频文件"}), 400

        return jsonify({
            "video_url": f"/static/uploads/{output_filename}",
            "total_frames": total_frames,
            "processed_frames": processed,
        })
    else:
        image_bgr = load_image_bgr(file)
        if image_bgr is None:
            return jsonify({"error": "无法读取图片"}), 400

        heatmap_gen.reset(shape=image_bgr.shape[:2])
        annotated, stats = heatmap_gen.process_image(image_bgr, conf=conf)

        return jsonify({
            "image": image_to_base64(annotated),
            "stats": stats,
        })


@app.route("/api/params", methods=["GET"])
def get_params():
    return jsonify({
        "canny_threshold1": detector.canny_threshold1,
        "canny_threshold2": detector.canny_threshold2,
        "gaussian_kernel": detector.gaussian_kernel,
        "hough_threshold": detector.hough_threshold,
        "hough_min_line_length": detector.hough_min_line_length,
        "hough_max_line_gap": detector.hough_max_line_gap,
        "use_hsl": detector.use_hsl,
    })


def _apply_params(form):
    if "canny_threshold1" in form:
        detector.canny_threshold1 = int(form["canny_threshold1"])
    if "canny_threshold2" in form:
        detector.canny_threshold2 = int(form["canny_threshold2"])
    if "gaussian_kernel" in form:
        detector.gaussian_kernel = int(form["gaussian_kernel"])
    if "hough_threshold" in form:
        detector.hough_threshold = int(form["hough_threshold"])
    if "hough_min_line_length" in form:
        detector.hough_min_line_length = int(form["hough_min_line_length"])
    if "hough_max_line_gap" in form:
        detector.hough_max_line_gap = int(form["hough_max_line_gap"])
    if "use_hsl" in form:
        detector.use_hsl = form["use_hsl"].lower() in ("true", "1", "yes")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
