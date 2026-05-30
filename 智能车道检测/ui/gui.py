"""
图形用户界面模块

提供基于Tkinter的图形界面，支持左右双屏显示。
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional
import threading
import cv2
import numpy as np
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import Config, get_config
from utils.logger import setup_logger, get_logger
from core.preprocessor import Preprocessor
from core.detector import LaneDetector
from utils.visualization import Visualizer


class LaneDetectionGUI:
    """车道线检测图形界面"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.logger = get_logger()
        
        self.preprocessor = Preprocessor()
        self.detector = LaneDetector()
        self.visualizer = Visualizer()
        
        self.root = tk.Tk()
        self.root.title("车道线检测系统")
        self.root.geometry("1400x800")
        self.root.minsize(1200, 600)
        
        self.original_image: Optional[np.ndarray] = None
        self.processed_image: Optional[np.ndarray] = None
        self.current_video_path: Optional[str] = None
        self.video_running: bool = False
        self.video_cap: Optional[cv2.VideoCapture] = None
        self.video_writer: Optional[cv2.VideoWriter] = None
        self.video_output_path: Optional[str] = None
        self.video_frame_count: int = 0
        
        self.tk_original_image: Optional[ImageTk.PhotoImage] = None
        self.tk_processed_image: Optional[ImageTk.PhotoImage] = None
        
        self._setup_ui()
        self._setup_style()
        
        self.root.after(100, self._init_display_areas)
    
    def _setup_style(self):
        """设置界面样式"""
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Microsoft YaHei', 12, 'bold'))
        style.configure('Status.TLabel', font=('Microsoft YaHei', 10))
    
    def _setup_ui(self):
        """设置界面"""
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        control_frame = ttk.LabelFrame(main_frame, text="控制面板", padding="10")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        
        file_frame = ttk.LabelFrame(control_frame, text="文件操作", padding="5")
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(file_frame, text="打开图像", command=self._open_image, width=15).pack(fill=tk.X, pady=3)
        ttk.Button(file_frame, text="打开视频", command=self._open_video, width=15).pack(fill=tk.X, pady=3)
        ttk.Button(file_frame, text="保存结果", command=self._save_result, width=15).pack(fill=tk.X, pady=3)
        
        process_frame = ttk.LabelFrame(control_frame, text="处理操作", padding="5")
        process_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(process_frame, text="处理图像", command=self._process_image, width=15).pack(fill=tk.X, pady=3)
        ttk.Button(process_frame, text="开始视频", command=self._start_video, width=15).pack(fill=tk.X, pady=3)
        ttk.Button(process_frame, text="停止视频", command=self._stop_video, width=15).pack(fill=tk.X, pady=3)
        ttk.Button(process_frame, text="重置检测器", command=self._reset_detector, width=15).pack(fill=tk.X, pady=3)
        
        param_frame = ttk.LabelFrame(control_frame, text="参数设置", padding="5")
        param_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(param_frame, text="白色亮度下限:").pack(anchor=tk.W, pady=(5, 0))
        self.white_l_var = tk.IntVar(value=self.config.color_threshold.white_lower[1])
        white_scale = ttk.Scale(param_frame, from_=150, to=255, variable=self.white_l_var,
                                orient=tk.HORIZONTAL, command=self._update_params)
        white_scale.pack(fill=tk.X, pady=(0, 5))
        self.white_l_label = ttk.Label(param_frame, text=f"{self.white_l_var.get()}")
        self.white_l_label.pack(anchor=tk.E)
        
        ttk.Label(param_frame, text="黄色H下限:").pack(anchor=tk.W, pady=(5, 0))
        self.yellow_h_var = tk.IntVar(value=self.config.color_threshold.yellow_lower[0])
        yellow_scale = ttk.Scale(param_frame, from_=0, to=30, variable=self.yellow_h_var,
                                 orient=tk.HORIZONTAL, command=self._update_params)
        yellow_scale.pack(fill=tk.X, pady=(0, 5))
        self.yellow_h_label = ttk.Label(param_frame, text=f"{self.yellow_h_var.get()}")
        self.yellow_h_label.pack(anchor=tk.E)
        
        ttk.Label(param_frame, text="平滑系数:").pack(anchor=tk.W, pady=(5, 0))
        self.smooth_var = tk.DoubleVar(value=self.config.smooth.alpha)
        smooth_scale = ttk.Scale(param_frame, from_=0.1, to=0.9, variable=self.smooth_var,
                                 orient=tk.HORIZONTAL, command=self._update_params)
        smooth_scale.pack(fill=tk.X, pady=(0, 5))
        self.smooth_label = ttk.Label(param_frame, text=f"{self.smooth_var.get():.2f}")
        self.smooth_label.pack(anchor=tk.E)
        
        info_frame = ttk.LabelFrame(control_frame, text="检测信息", padding="5")
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.left_status_var = tk.StringVar(value="左车道线: 未检测")
        self.right_status_var = tk.StringVar(value="右车道线: 未检测")
        self.fps_status_var = tk.StringVar(value="FPS: --")
        
        ttk.Label(info_frame, textvariable=self.left_status_var).pack(anchor=tk.W)
        ttk.Label(info_frame, textvariable=self.right_status_var).pack(anchor=tk.W)
        ttk.Label(info_frame, textvariable=self.fps_status_var).pack(anchor=tk.W)
        
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        original_frame = ttk.LabelFrame(display_frame, text="原始图像", padding="5")
        original_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        self.original_canvas = tk.Canvas(original_frame, bg='#2b2b2b', highlightthickness=0)
        self.original_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.original_title = ttk.Label(original_frame, text="等待加载...", style='Status.TLabel')
        self.original_title.pack(pady=5)
        
        processed_frame = ttk.LabelFrame(display_frame, text="处理结果", padding="5")
        processed_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.processed_canvas = tk.Canvas(processed_frame, bg='#2b2b2b', highlightthickness=0)
        self.processed_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.processed_title = ttk.Label(processed_frame, text="等待处理...", style='Status.TLabel')
        self.processed_title.pack(pady=5)
        
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, padding=(10, 5))
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        self.original_canvas.bind('<Configure>', self._on_original_canvas_resize)
        self.processed_canvas.bind('<Configure>', self._on_processed_canvas_resize)
    
    def _init_display_areas(self):
        """初始化显示区域"""
        self._show_placeholder(self.original_canvas, "请打开图像或视频文件")
        self._show_placeholder(self.processed_canvas, "等待处理...")
    
    def _show_placeholder(self, canvas: tk.Canvas, text: str):
        """显示占位符"""
        canvas.delete("all")
        width = canvas.winfo_width()
        height = canvas.winfo_height()
        if width > 1 and height > 1:
            canvas.create_text(
                width // 2, height // 2,
                text=text,
                fill='#888888',
                font=('Microsoft YaHei', 14)
            )
    
    def _on_original_canvas_resize(self, event):
        """原始图像画布大小改变"""
        if self.original_image is not None:
            self._display_original_image()
    
    def _on_processed_canvas_resize(self, event):
        """处理结果画布大小改变"""
        if self.processed_image is not None:
            self._display_processed_image()
    
    def _update_params(self, *args):
        """更新参数"""
        self.config.color_threshold.white_lower[1] = self.white_l_var.get()
        self.config.color_threshold.yellow_lower[0] = self.yellow_h_var.get()
        self.config.smooth.alpha = self.smooth_var.get()
        
        self.white_l_label.config(text=f"{self.white_l_var.get()}")
        self.yellow_h_label.config(text=f"{self.yellow_h_var.get()}")
        self.smooth_label.config(text=f"{self.smooth_var.get():.2f}")
        
        self.preprocessor = Preprocessor()
        self.detector = LaneDetector()
    
    def _open_image(self):
        """打开图像文件"""
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff"), ("所有文件", "*.*")]
        )
        
        if file_path:
            try:
                self.original_image = cv2.imdecode(
                    np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if self.original_image is None:
                    raise ValueError("无法解码图像文件")
                
                self.processed_image = None
                self._display_original_image()
                self._show_placeholder(self.processed_canvas, "点击\"处理图像\"开始处理")
                
                h, w = self.original_image.shape[:2]
                self.original_title.config(text=f"{os.path.basename(file_path)} ({w}x{h})")
                self.processed_title.config(text="等待处理...")
                self.status_var.set(f"已加载: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("错误", f"无法加载图像: {e}")
                self.logger.error(f"加载图像失败: {e}")
    
    def _open_video(self):
        """打开视频文件"""
        file_path = filedialog.askopenfilename(
            title="选择视频文件",
            filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv"), ("所有文件", "*.*")]
        )
        
        if file_path:
            self.current_video_path = file_path
            
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                ret, first_frame = cap.read()
                cap.release()
                
                if ret and first_frame is not None:
                    self.original_image = first_frame.copy()
                    self.processed_image = None
                    self._display_original_image()
                    self._show_placeholder(self.processed_canvas, "点击\"开始视频\"开始播放")
                else:
                    self.original_image = None
                    self.processed_image = None
                    self._show_placeholder(self.original_canvas, "无法读取视频帧")
                    self._show_placeholder(self.processed_canvas, "等待处理...")
                
                self.original_title.config(text=f"{os.path.basename(file_path)} ({width}x{height}, {fps:.1f}fps)")
                self.processed_title.config(text=f"共 {frame_count} 帧 - 已暂停")
                self.status_var.set(f"已选择视频: {os.path.basename(file_path)} (首帧已显示)")
            else:
                messagebox.showerror("错误", "无法打开视频文件")
    
    def _process_image(self):
        """处理当前图像"""
        if self.original_image is None:
            messagebox.showwarning("警告", "请先打开图像")
            return
        
        try:
            self.status_var.set("正在处理...")
            self.root.update()
            
            self.detector.reset()
            
            enhanced, color_filtered, edges = self.preprocessor.preprocess(self.original_image)
            
            if edges is not None:
                result = self.detector.detect(edges, self.original_image.shape)
                output = self.visualizer.draw_lane_lines(
                    self.original_image, result['left_line'], result['right_line']
                )
                output = self.visualizer.draw_detection_status(
                    output, result['left_detected'], result['right_detected']
                )
                
                self.left_status_var.set(f"左车道线: {'检测成功' if result['left_detected'] else '未检测'}")
                self.right_status_var.set(f"右车道线: {'检测成功' if result['right_detected'] else '未检测'}")
                
                if result.get('left_slope'):
                    self.left_status_var.set(f"左车道线: 斜率={result['left_slope']:.3f}")
                if result.get('right_slope'):
                    self.right_status_var.set(f"右车道线: 斜率={result['right_slope']:.3f}")
            else:
                output = self.original_image.copy()
                cv2.putText(output, "Edge detection failed", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            self.processed_image = output
            self._display_processed_image()
            self.processed_title.config(text="处理完成")
            self.status_var.set("处理完成")
            
        except Exception as e:
            messagebox.showerror("错误", f"处理失败: {e}")
            self.logger.error(f"处理图像失败: {e}")
    
    def _start_video(self):
        """开始视频处理"""
        if self.current_video_path is None:
            messagebox.showwarning("警告", "请先打开视频")
            return
        
        if self.video_running:
            return
        
        self.video_running = True
        self.video_cap = cv2.VideoCapture(self.current_video_path)
        self.detector.reset()
        
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result_dir = os.path.join(script_dir, "结果")
        try:
            if not os.path.exists(result_dir):
                os.makedirs(result_dir, exist_ok=True)
        except Exception:
            pass
        
        width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(self.current_video_path))[0]
        video_filename = f"result_{timestamp}_{base_name}.avi"
        self.video_output_path = os.path.join(result_dir, video_filename)
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(self.video_output_path, fourcc, fps, (width, height))
        
        if not self.video_writer.isOpened():
            self.logger.error("无法创建视频写入器")
            self.video_writer = None
            self.video_output_path = None
        
        self.video_frame_count = 0
        
        self.status_var.set("视频处理中...")
        self.processed_title.config(text="实时处理中...")
        
        self.frame_count = 0
        self.start_time = None
        
        threading.Thread(target=self._process_video_thread, daemon=True).start()
    
    def _process_video_thread(self):
        """视频处理线程"""
        import time
        self.start_time = time.time()
        
        while self.video_running and self.video_cap is not None:
            ret, frame = self.video_cap.read()
            if not ret:
                self.root.after(0, self._stop_video)
                break
            
            self.original_image = frame.copy()
            
            enhanced, color_filtered, edges = self.preprocessor.preprocess(frame)
            
            if edges is not None:
                result = self.detector.detect(edges, frame.shape)
                output = self.visualizer.draw_lane_lines(
                    frame, result['left_line'], result['right_line']
                )
                output = self.visualizer.draw_detection_status(
                    output, result['left_detected'], result['right_detected']
                )
                
                self.root.after(0, self._update_detection_status, result)
            else:
                output = frame
            
            self.processed_image = output
            
            if self.video_writer is not None:
                try:
                    self.video_writer.write(output)
                    self.video_frame_count += 1
                except Exception as e:
                    self.logger.error(f"写入视频帧失败: {e}")
            
            self.frame_count += 1
            if self.frame_count % 5 == 0:
                elapsed = time.time() - self.start_time
                if elapsed > 0:
                    fps = self.frame_count / elapsed
                    self.root.after(0, self.fps_status_var.set, f"FPS: {fps:.1f}")
            
            self.root.after(0, self._display_original_image)
            self.root.after(0, self._display_processed_image)
        
        self.root.after(0, self.status_var.set, "视频处理完成")
    
    def _update_detection_status(self, result: dict):
        """更新检测状态"""
        left_status = "检测成功" if result.get('left_detected') else "未检测"
        right_status = "检测成功" if result.get('right_detected') else "未检测"
        
        if result.get('left_slope'):
            left_status = f"斜率={result['left_slope']:.3f}"
        if result.get('right_slope'):
            right_status = f"斜率={result['right_slope']:.3f}"
        
        self.left_status_var.set(f"左车道线: {left_status}")
        self.right_status_var.set(f"右车道线: {right_status}")
    
    def _stop_video(self):
        """停止视频处理"""
        self.video_running = False
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        
        saved_info = ""
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
            
            if self.video_output_path and os.path.exists(self.video_output_path):
                file_size_mb = os.path.getsize(self.video_output_path) / (1024 * 1024)
                saved_info = f" | 视频已保存: {os.path.basename(self.video_output_path)} ({file_size_mb:.2f} MB, {self.video_frame_count} 帧)"
                self.logger.info(f"视频已保存: {self.video_output_path} ({file_size_mb:.2f} MB, {self.video_frame_count} 帧)")
            self.video_output_path = None
        
        self.status_var.set(f"已停止{saved_info}")
        self.processed_title.config(text=f"已停止 | 已录制 {self.video_frame_count} 帧")
    
    def _reset_detector(self):
        """重置检测器及全部状态"""
        if self.video_running:
            self._stop_video()
        
        self.detector.reset()
        self.preprocessor = Preprocessor()
        self.detector = LaneDetector()
        
        self.original_image = None
        self.processed_image = None
        self.current_video_path = None
        self.video_output_path = None
        self.video_frame_count = 0
        
        self.tk_original_image = None
        self.tk_processed_image = None
        
        self._show_placeholder(self.original_canvas, "请打开图像或视频文件")
        self._show_placeholder(self.processed_canvas, "等待处理...")
        
        self.original_title.config(text="等待加载...")
        self.processed_title.config(text="等待处理...")
        self.left_status_var.set("左车道线: 未检测")
        self.right_status_var.set("右车道线: 未检测")
        self.fps_status_var.set("FPS: --")
        self.status_var.set("已重置")
    
    def _save_result(self):
        """保存结果到结果文件夹"""
        if self.current_video_path and self.video_output_path and os.path.exists(self.video_output_path):
            messagebox.showinfo("提示", f"视频已自动保存到:\n{self.video_output_path}\n\n共 {self.video_frame_count} 帧")
            return
        
        if self.processed_image is None:
            messagebox.showwarning("警告", "没有可保存的结果")
            return
        
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        result_dir = os.path.join(script_dir, "结果")
        
        try:
            if not os.path.exists(result_dir):
                os.makedirs(result_dir, exist_ok=True)
                self.logger.info(f"创建结果目录: {result_dir}")
        except PermissionError:
            messagebox.showerror("错误", f"权限不足，无法创建结果目录: {result_dir}")
            return
        except OSError as e:
            messagebox.showerror("错误", f"创建结果目录失败: {e}")
            return
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result_{timestamp}.jpg"
        
        file_path = os.path.join(result_dir, filename)
        
        try:
            cv2.imencode('.jpg', self.processed_image)[1].tofile(file_path)
            file_size_kb = os.path.getsize(file_path) / 1024
            self.status_var.set(f"已保存: {filename} ({file_size_kb:.1f} KB) -> {result_dir}")
            self.logger.info(f"结果已保存: {file_path} ({file_size_kb:.1f} KB)")
        except Exception as e:
            messagebox.showerror("错误", f"保存失败: {e}")
            self.logger.error(f"保存结果失败: {e}")
    
    def _display_original_image(self):
        """显示原始图像"""
        if self.original_image is None:
            return
        
        try:
            rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            
            canvas_width = self.original_canvas.winfo_width()
            canvas_height = self.original_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                return
            
            h, w = rgb_image.shape[:2]
            scale = min(canvas_width / w, canvas_height / h) * 0.95
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            if new_w > 0 and new_h > 0:
                resized = cv2.resize(rgb_image, (new_w, new_h))
                pil_image = Image.fromarray(resized)
                self.tk_original_image = ImageTk.PhotoImage(pil_image)
                
                self.original_canvas.delete("all")
                x = (canvas_width - new_w) // 2
                y = (canvas_height - new_h) // 2
                self.original_canvas.create_image(x, y, anchor=tk.NW, image=self.tk_original_image)
        except Exception as e:
            self.logger.error(f"显示原始图像失败: {e}")
    
    def _display_processed_image(self):
        """显示处理后的图像"""
        if self.processed_image is None:
            return
        
        try:
            rgb_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB)
            
            canvas_width = self.processed_canvas.winfo_width()
            canvas_height = self.processed_canvas.winfo_height()
            
            if canvas_width <= 1 or canvas_height <= 1:
                return
            
            h, w = rgb_image.shape[:2]
            scale = min(canvas_width / w, canvas_height / h) * 0.95
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            if new_w > 0 and new_h > 0:
                resized = cv2.resize(rgb_image, (new_w, new_h))
                pil_image = Image.fromarray(resized)
                self.tk_processed_image = ImageTk.PhotoImage(pil_image)
                
                self.processed_canvas.delete("all")
                x = (canvas_width - new_w) // 2
                y = (canvas_height - new_h) // 2
                self.processed_canvas.create_image(x, y, anchor=tk.NW, image=self.tk_processed_image)
        except Exception as e:
            self.logger.error(f"显示处理图像失败: {e}")
    
    def run(self):
        """运行界面"""
        self.root.mainloop()
    
    def close(self):
        """关闭界面"""
        self._stop_video()
        self.root.destroy()


def main():
    """启动GUI"""
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(script_dir, "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "gui.log")
    setup_logger(
        name="lane_detection",
        level="INFO",
        log_file=log_file,
        console_enabled=False
    )
    
    app = LaneDetectionGUI()
    app.run()


if __name__ == "__main__":
    main()
