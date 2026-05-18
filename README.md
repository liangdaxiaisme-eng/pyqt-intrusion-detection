# 基于改进YOLOv12的非法入侵检测报警系统

> PyQt5 桌面版 | YOLOv12 + ECA 注意力机制 | 实时检测 + 电子围栏 + 报警记录

---

## 📋 项目简介

本系统采用改进的 YOLOv12 目标检测模型，融合 ECA（Efficient Channel Attention）注意力机制，实现对监控画面中非法入侵行为的实时检测与报警。支持摄像头实时监控、视频文件分析、照片检测三种模式，具备电子围栏（领地划分）功能，闯入即报警。

**作者**：黄丽佳  
**学号**：22460525  
**指导教师**：曾德真  
**学院**：大数据与人工智能学院

---

## ✨ 核心功能

| 功能 | 说明 |
|------|------|
| 📹 实时检测 | 支持摄像头/视频文件实时检测，显示检测框、目标轨迹、FPS |
| 🏗 电子围栏 | 手动绘制多边形防护区域，闯入即报警，支持多级警报（高/中/低） |
| 📷 照片检测 | 支持单张照片检测，多尺度扫描（640+1280+1920）提升小目标检出率 |
| 📋 报警记录 | 自动记录入侵事件（5秒防重复），支持清空和导出 CSV |
| 🌙 夜间增强 | CLAHE 直方图均衡化，提升低光环境下的检测效果 |
| 🎯 多目标跟踪 | 基于 IoU 的多目标跟踪算法，支持目标轨迹可视化 |

---

## 📊 模型性能

| 指标 | 数值 |
|------|------|
| mAP@0.5 | 88.3% |
| Precision | 82.7% |
| Recall | 79.0% |
| FPR | 3.2% |
| 推理帧率 | ~57.8 FPS (GPU) |

---

## 🏗 技术架构

```
输入源 (摄像头/视频/图片)
    ↓
YOLOv12 + ECA 注意力机制 (目标检测)
    ↓
SimpleTracker (IoU 多目标跟踪)
    ↓
ROI 多边形判断 (电子围栏入侵检测)
    ↓
PyQt5 界面显示 + 报警记录
```

### ECA 注意力机制

ECA（Efficient Channel Attention）模块通过一维卷积捕获通道间依赖关系，在几乎不增加计算量的情况下提升模型对关键特征的敏感度。

```python
class ECA(nn.Module):
    def __init__(self, c1, c2, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size-1)//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv1(y.squeeze(-1).transpose(-1,-2)).transpose(-1,-2).unsqueeze(-1)
        return x * self.sigmoid(y)
```

### 多尺度检测策略

照片检测采用多尺度自适应检测策略：
- 以 640、1280、1920 三种输入分辨率进行推理
- 低置信度阈值 (0.15) 确保微小目标不被遗漏
- 通过 IoU 去重融合机制合并检测结果（IoU > 0.5 的重叠框只保留置信度最高的）

### 报警分级机制

| 级别 | 置信度 | 颜色 | 说明 |
|------|--------|------|------|
| 🔴 高级 | ≥ 0.6 | 红色 | 高置信度入侵，需立即响应 |
| 🟠 中级 | 0.3 ~ 0.6 | 橙色 | 中等置信度，需关注 |
| 🟡 低级 | < 0.3 | 黄色 | 低置信度，可能误检 |

---

## 📁 项目结构

```
pyqt_intrusion/
├── main.py              # 主程序（运行此文件启动系统）
├── models/
│   └── best.pt          # YOLOv12+ECA 训练好的模型权重（22MB）
├── uploads/             # 上传文件存储目录
├── upload_server.py     # 视频上传服务（端口 9999）
├── upload_photo.py      # 照片上传服务（端口 9999）
└── README.md            # 本文档
```

---

## 🚀 快速开始

### 方式一：PyCharm 运行（推荐）

#### 第一步：安装 PyCharm

1. 下载 PyCharm Community Edition（免费版）：https://www.jetbrains.com/pycharm/download/
2. 安装完成后打开 PyCharm

#### 第二步：导入项目

1. 点击 **File → Open**
2. 选择下载的 `pyqt_intrusion` 文件夹
3. 点击 **OK**
4. 等待 PyCharm 索引完成

#### 第三步：配置 Python 解释器

1. 点击 **File → Settings**（macOS 是 **PyCharm → Preferences**）
2. 进入 **Project: pyqt_intrusion → Python Interpreter**
3. 点击右上角齿轮图标 → **Add**
4. 选择 **Virtualenv Environment → New environment**
   - Location：默认即可
   - Base interpreter：选择系统安装的 Python 3.8~3.11（推荐 3.10）
5. 点击 **OK** → **Apply**

#### 第四步：安装依赖

在 PyCharm 底部打开 **Terminal**，执行：

```bash
pip install PyQt5 opencv-python ultralytics torch torchvision numpy
```

如果使用 GPU 加速（推荐）：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> **注意**：`opencv-python` 可能与 PyQt5 冲突，如遇到 `Could not load the Qt platform plugin` 错误，改用：
> ```bash
> pip uninstall opencv-python
> pip install opencv-python-headless
> ```

#### 第五步：运行程序

1. 在左侧项目栏找到 `main.py`
2. 右键 → **Run 'main'**
3. 等待模型加载（首次约 3-5 秒）
4. 程序窗口出现后即可使用

### 方式二：命令行运行

```bash
# 进入项目目录
cd pyqt_intrusion

# 安装依赖
pip install PyQt5 opencv-python ultralytics torch torchvision numpy

# 运行
python main.py
```

---

## 📖 使用说明

### 📹 实时检测

1. 点击 **📹 实时检测** 标签页
2. 选择输入源：
   - **摄像头0** — 默认摄像头
   - **摄像头1** — 第二个摄像头
   - **选择视频文件...** — 浏览选择本地视频文件
3. 点击 **▶ 启动检测**
4. 查看结果：
   - 绿色框：普通目标
   - 红色/橙色/黄色框：入侵目标（按置信度分级）
   - 左上角显示：模型信息、FPS、目标数、入侵数
   - 目标中心显示运动轨迹
5. 点击 **⏹ 停止** 结束检测

### 🏗 电子围栏（领地划分）

**⚠️ 必须先设置围栏，否则所有目标都会被检测但不会触发入侵报警！**

操作步骤：

1. 点击 **🏗 领地划分** 标签页
2. 加载背景：
   - 点击 **📂 加载背景图片** 选择一张场景图片
   - 或点击 **🎬 从视频提取首帧** 从视频中提取
3. 绘制围栏区域：
   - **鼠标左键**：点击添加围栏顶点（至少 3 个点）
   - **鼠标右键**：完成绘制
4. 点击 **✅ 应用到检测**
5. 返回 **📹 实时检测** 页，状态栏显示 `✅ 领地已设置`

**应用场景示例**：
- 在围墙区域绘制围栏 → 有人翻墙进入 → 触发高级警报
- 在禁区入口绘制围栏 → 有人闯入 → 自动记录并报警

### 📷 照片检测

1. 点击 **📷 照片检测** 标签页
2. 点击 **📂 选择照片**
3. 系统自动使用多尺度检测（640+1280+1920），提升小目标检出率
4. 检测结果直接显示在界面上，包括目标数量和置信度

**适用场景**：翻墙背影、远距离目标、低分辨率图片

### 📋 报警记录

- 每次检测到入侵自动记录（5 秒防重复）
- 显示：时间、类型（高/中/低级）、数量、详情
- 支持 **🗑 清空** 和 **💾 导出 CSV**

### 🌙 夜间增强

点击 **🌙 夜间增强** 按钮开启 CLAHE 直方图均衡化，提升低光环境下的检测效果。

---

## ⚙️ 配置说明

在 `main.py` 顶部可调整以下参数：

```python
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.pt')  # 模型路径
CONF_THRESHOLD = 0.2    # 默认置信度
CONF_PHOTO = 0.15       # 照片检测置信度（低=更灵敏）
CONF_VIDEO = 0.25       # 视频检测置信度（高=更准确）
CLASS_NAMES = {0: 'Intruder'}  # 类别名称
```

---

## 🖥 系统要求

### 硬件要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| CPU | Intel i5 / AMD Ryzen 5 | Intel i7 / AMD Ryzen 7 |
| 内存 | 8GB | 16GB |
| GPU | 无（CPU 可运行） | NVIDIA GTX 1060 6GB+ |
| 硬盘 | 2GB 可用空间 | 5GB+ |

### 软件要求

| 软件 | 版本 |
|------|------|
| 操作系统 | Windows 10/11 (64位)、macOS、Ubuntu 20.04+ |
| Python | 3.8 ~ 3.11（推荐 3.10） |
| CUDA（可选） | 11.7+ (GPU 加速) |

---

## ❓ 常见问题

### Q: 启动报错 "Could not load the Qt platform plugin"

**原因**：OpenCV 和 PyQt5 的 Qt 插件冲突

**解决**：
```bash
pip uninstall opencv-python
pip install opencv-python-headless
```

### Q: 启动报错 "模型加载失败"

**原因**：缺少 `models/best.pt` 文件或依赖未安装完整

**解决**：
1. 确认 `models/best.pt` 文件存在
2. 执行 `pip install ultralytics torch`
3. 确认 PyTorch 版本 ≥ 2.0

### Q: 视频检测很卡

**原因**：每帧都需要模型推理，FPS 取决于硬件

**解决**：
- CPU：约 5-15 FPS（基本可用）
- GPU（CUDA）：约 30-60 FPS（流畅）
- 可降低视频分辨率提升流畅度

### Q: 没有 GPU 能用吗

**可以**。CPU 模式下帧率较低，但功能完全正常。安装 CPU 版 PyTorch：
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Q: 检测不到人

**原因**：目标太小或背影特征不明显

**解决**：
- 照片模式已自动采用多尺度检测（640+1280+1920）
- 可尝试降低置信度阈值（修改 `CONF_PHOTO`）
- 确保光线充足，目标清晰可见

### Q: 围栏设置后不报警

**原因**：可能未点击 "✅ 应用到检测"

**解决**：
1. 确认在领地划分页点击了 "✅ 应用到检测"
2. 确认状态栏显示 "✅ 领地已设置"
3. 检查围栏区域是否覆盖了入侵路径

### Q: 如何更换模型

将新的 `.pt` 模型文件放入 `models/` 目录，修改 `main.py` 中的 `MODEL_PATH`：

```python
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'your_new_model.pt')
```

### Q: 启动报错 "Can't get attribute 'ECA'"

**原因**：这是正常现象，程序启动时会自动注册 ECA 模块

**解决**：不影响运行，可忽略此警告

---

## 📦 依赖列表

```
PyQt5>=5.15
opencv-python>=4.5
ultralytics>=8.0
torch>=2.0
torchvision>=0.15
numpy>=1.21
```

一键安装：
```bash
pip install PyQt5 opencv-python ultralytics torch torchvision numpy
```

---

## 📄 许可证

本项目仅供学术研究使用。

---

**最后更新**：2026-05-18
