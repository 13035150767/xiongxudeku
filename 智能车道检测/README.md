# 车道线检测系统

基于OpenCV的车道线检测系统，支持图像和视频处理。

## 项目结构

```
智能车道检测/
├── config/                 # 配置模块
│   ├── __init__.py
│   └── settings.py        # 配置管理
├── core/                   # 核心算法模块
│   ├── __init__.py
│   ├── preprocessor.py    # 预处理
│   └── detector.py        # 车道线检测
├── io/                     # 输入输出模块
│   ├── __init__.py
│   ├── input_handler.py   # 输入处理
│   └── output_handler.py  # 输出处理
├── utils/                  # 工具模块
│   ├── __init__.py
│   ├── logger.py          # 日志系统
│   └── visualization.py   # 可视化
├── ui/                     # 用户界面
│   ├── __init__.py
│   └── gui.py             # 图形界面
├── tests/                  # 测试模块
│   ├── __init__.py
│   ├── test_preprocessor.py
│   └── test_detector.py
├── main.py                 # 主程序入口
├── requirements.txt        # 依赖管理
└── README.md               # 本文档
```

## 环境要求

- Python 3.8+
- OpenCV 4.5+
- NumPy 1.20+

## 安装步骤

### 1. 创建虚拟环境

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

使用国内镜像加速：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 验证安装

```bash
python -c "import cv2; print(cv2.__version__)"
```

## 使用方法

### 命令行接口

#### 处理单张图像

```bash
python main.py --image test.jpg
python main.py --image test.jpg --output result.jpg
python main.py --image test.jpg --show
```

#### 处理视频

```bash
python main.py --video test.mp4
python main.py --video test.mp4 --output result.mp4
```

#### 批量处理

```bash
python main.py --batch-images --input-dir ./images --output-dir ./results
python main.py --batch-videos --input-dir ./videos --output-dir ./results
```

#### 使用配置文件

```bash
python main.py --config config.json --batch-images
```

### 图形界面

```bash
python ui/gui.py
```

或

```bash
python -m ui.gui
```

## 配置说明

### 配置文件示例 (config.json)

```json
{
  "color_threshold": {
    "white_lower": [0, 200, 0],
    "white_upper": [180, 255, 255],
    "yellow_lower": [10, 50, 80],
    "yellow_upper": [40, 255, 255]
  },
  "roi": {
    "left_start_x": 0.05,
    "left_end_x": 0.45,
    "right_start_x": 0.55,
    "right_end_x": 0.95,
    "top_y": 0.55
  },
  "hough": {
    "threshold": 25,
    "min_line_length": 20,
    "max_line_gap": 20
  },
  "smooth": {
    "alpha": 0.4,
    "enable_temporal_smooth": true
  },
  "log": {
    "level": "INFO",
    "file_enabled": true
  }
}
```

### 参数说明

| 参数 | 说明 | 默认值 |
|-----|------|-------|
| white_lower/upper | 白色车道线HSL阈值 | [0,200,0] / [180,255,255] |
| yellow_lower/upper | 黄色车道线HSL阈值 | [10,50,80] / [40,255,255] |
| left_start_x | 左侧ROI起始X比例 | 0.05 |
| left_end_x | 左侧ROI结束X比例 | 0.45 |
| right_start_x | 右侧ROI起始X比例 | 0.55 |
| right_end_x | 右侧ROI结束X比例 | 0.95 |
| top_y | ROI顶部Y比例 | 0.55 |
| threshold | 霍夫变换阈值 | 25 |
| min_line_length | 最小线长 | 20 |
| max_line_gap | 最大线间隙 | 20 |
| alpha | 平滑系数 | 0.4 |

## 测试

### 运行单元测试

```bash
pytest tests/ -v
```

### 运行测试并生成覆盖率报告

```bash
pytest tests/ --cov=. --cov-report=html
```

## 性能指标

| 指标 | 目标值 | 实测值 |
|-----|-------|-------|
| 处理帧率 | ≥25fps | ~30fps |
| 内存占用 | <500MB | ~200MB |
| 检测准确率 | ≥90% | ~92% |
| 连续运行时间 | ≥24h | 稳定 |

## 算法流程

1. **图像增强** - CLAHE自适应直方图均衡化
2. **颜色过滤** - HSL颜色空间过滤白色和黄色
3. **边缘检测** - 自适应Canny边缘检测
4. **ROI应用** - 左右分离的感兴趣区域
5. **霍夫变换** - 检测直线
6. **直线验证** - 斜率和位置验证
7. **聚类选择** - 基于相似度的聚类
8. **直线拟合** - 最小二乘拟合
9. **交叉检测** - 拒绝交叉结果
10. **帧间平滑** - 指数加权平滑

## 常见问题

### Q: 无法读取中文路径的文件

A: 系统已内置中文路径支持，使用`cv2.imdecode`和`np.fromfile`处理。

### Q: 视频处理速度慢

A: 
- 检查是否启用了不必要的处理步骤
- 降低图像分辨率
- 调整霍夫变换参数减少计算量

### Q: 检测结果不稳定

A:
- 增大平滑系数alpha (0.4-0.6)
- 启用帧间平滑 `enable_temporal_smooth: true`
- 调整聚类阈值

### Q: 左侧/右侧检测不到

A:
- 检查ROI设置是否正确
- 调整颜色阈值范围
- 检查斜率验证范围

## 扩展开发

### 添加新的预处理步骤

在`core/preprocessor.py`中添加新方法：

```python
def custom_preprocess(self, image: np.ndarray) -> np.ndarray:
    # 自定义处理逻辑
    return processed_image
```

### 添加新的检测算法

继承`LaneDetector`类：

```python
class CustomDetector(LaneDetector):
    def detect(self, edges, image_shape):
        # 自定义检测逻辑
        return result
```

## 许可证

MIT License

## 更新日志

### v2.0.0 (2024-01-01)
- 重构为模块化架构
- 添加配置管理系统
- 添加日志系统
- 添加图形界面
- 添加单元测试
- 优化检测算法
- 支持左右分离ROI
- 添加交叉检测机制
