"""
基于改进YOLOv12的非法入侵检测报警系统 - PyQt5桌面版
功能：领地划分 + 入侵检测 + 照片检测 + 报警
"""
import sys, os, cv2, time, torch, torch.nn as nn, numpy as np
from collections import defaultdict, deque
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QComboBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QMessageBox, QGroupBox, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint
from PyQt5.QtGui import QImage, QPixmap, QFont, QColor, QPen, QPainter, QBrush, QPolygon

# ============ ECA ============
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

import ultralytics.nn.modules as modules
import ultralytics.nn.tasks as tasks
modules.ECA = ECA; tasks.ECA = ECA

# ============ 配置 ============
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best.pt')
CONF_THRESHOLD = 0.2
CONF_PHOTO = 0.15    # 照片检测置信度（低=更灵敏）
CONF_VIDEO = 0.25    # 视频检测置信度（高=更准确）
CLASS_NAMES = {0: 'Intruder'}

# ============ 跟踪器 ============
class SimpleTracker:
    def __init__(self, max_disappeared=30, iou_threshold=0.3):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold
        self.trajectories = defaultdict(lambda: deque(maxlen=50))

    def _iou(self, b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1])
        a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter / (a1+a2-inter+1e-6)

    def update(self, dets):
        if len(dets) == 0:
            for k in list(self.disappeared):
                self.disappeared[k] += 1
                if self.disappeared[k] > self.max_disappeared:
                    del self.objects[k]; del self.disappeared[k]
            return self.objects
        new_objs = {}
        for d in dets:
            x1,y1,x2,y2,conf,cls = d
            best_id, best_iou = None, 0
            for oid, obox in self.objects.items():
                iou = self._iou(d[:4], obox[:4])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou, best_id = iou, oid
            if best_id is not None:
                new_objs[best_id] = (x1,y1,x2,y2,cls,conf)
                self.disappeared[best_id] = 0
            else:
                new_objs[self.next_id] = (x1,y1,x2,y2,cls,conf)
                self.disappeared[self.next_id] = 0
                self.next_id += 1
            tid = best_id if best_id is not None else self.next_id-1
            self.trajectories[tid].append(((x1+x2)/2, (y1+y2)/2))
        self.objects = new_objs
        return self.objects

# ============ 检测线程 ============
class DetectThread(QThread):
    result_ready = pyqtSignal(object, dict, list, int)
    fps_update = pyqtSignal(float)

    def __init__(self, model, tracker, roi_polygon=None):
        super().__init__()
        self.model = model
        self.tracker = tracker
        self.conf = CONF_VIDEO
        self.running = False
        self.source = 0
        self.roi_polygon = roi_polygon
        self.night_mode = False  # 夜间增强模式
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    def run(self):
        self.running = True
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened(): return
        while self.running and cap.isOpened():
            t0 = time.time()
            ret, frame = cap.read()
            if not ret:
                if isinstance(self.source, str): break
                continue
            annotated, tracked, intrusions, count = self._process(frame)
            self.result_ready.emit(annotated, tracked, intrusions, count)
            dt = time.time() - t0
            if dt > 0: self.fps_update.emit(1.0/dt)
        cap.release()

    def stop(self): self.running = False

    def _in_roi(self, box):
        if self.roi_polygon is None: return True
        cx, cy = (box[0]+box[2])//2, (box[1]+box[3])//2
        return cv2.pointPolygonTest(self.roi_polygon, (cx, cy), False) >= 0

    def _enhance_night(self, frame):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self.clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    def _process(self, frame):
        if self.night_mode:
            frame = self._enhance_night(frame)
        results = self.model.predict(source=frame, conf=self.conf, verbose=False)
        dets = []
        for r in results:
            if r.boxes is None: continue
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                dets.append((x1,y1,x2,y2,conf,cls))
        tracked = self.tracker.update(dets)
        intrusions = [tid for tid, (x1,y1,x2,y2,_,_) in tracked.items() if self._in_roi((x1,y1,x2,y2))]
        annotated = self._draw(frame.copy(), tracked, set(intrusions))
        return annotated, tracked, intrusions, len(intrusions)

    def _draw(self, frame, tracked, intrusions):
        h, w = frame.shape[:2]
        if self.roi_polygon is not None:
            overlay = frame.copy()
            cv2.fillPoly(overlay, [self.roi_polygon], (0, 0, 150))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            cv2.polylines(frame, [self.roi_polygon], True, (0, 0, 255), 3)
            cx = int(np.mean(self.roi_polygon[:, 0]))
            cy = int(np.mean(self.roi_polygon[:, 1]))
            cv2.putText(frame, 'TERRITORY', (cx-60, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        for tid, (x1,y1,x2,y2,cls,conf) in tracked.items():
            is_intrusion = tid in intrusions
            if is_intrusion:
                if conf >= 0.6:
                    color = (0,0,255); level = 'HIGH'; thickness = 3
                elif conf >= 0.3:
                    color = (0,140,255); level = 'MID'; thickness = 2
                else:
                    color = (0,255,255); level = 'LOW'; thickness = 2
                label = f'[{level}] #{tid} {conf:.2f}'
            else:
                color = (0,200,0); label = f'Target #{tid} {conf:.2f}'; thickness = 1
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1-th-6), (x1+tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            pts = list(self.tracker.trajectories.get(tid, []))
            for i in range(1, len(pts)):
                cv2.line(frame, (int(pts[i-1][0]),int(pts[i-1][1])), (int(pts[i][0]),int(pts[i][1])), color, 2)
        cv2.putText(frame, f'YOLOv12+ECA | Conf:{self.conf}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f'Targets: {len(tracked)} | Intrusions: {len(intrusions)}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, time.strftime('%H:%M:%S'), (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        if intrusions:
            cv2.circle(frame, (w-30, 30), 15, (0,0,255), -1)
            cv2.putText(frame, 'INTRUSION!', (w-180, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        return frame


# ============ ROI绘制控件 ============
class ROIDrawLabel(QLabel):
    roi_changed = pyqtSignal(list)
    def __init__(self):
        super().__init__()
        self.roi_points = []
        self._bg_pixmap = None
    def set_background(self, pixmap):
        self._bg_pixmap = pixmap; self.roi_points = []; self.update()
    def _w2i(self, wx, wy):
        if self._bg_pixmap is None or self.pixmap() is None: return wx, wy
        pw, ph = self.pixmap().width(), self.pixmap().height()
        if pw == 0 or ph == 0: return wx, wy
        offx = (self.width() - pw) // 2; offy = (self.height() - ph) // 2
        ix = max(0, min(int((wx - offx) * self._bg_pixmap.width() / pw), self._bg_pixmap.width()-1))
        iy = max(0, min(int((wy - offy) * self._bg_pixmap.height() / ph), self._bg_pixmap.height()-1))
        return ix, iy
    def _i2w(self, ix, iy):
        if self._bg_pixmap is None or self.pixmap() is None: return ix, iy
        pw, ph = self.pixmap().width(), self.pixmap().height()
        if pw == 0 or ph == 0: return ix, iy
        offx = (self.width() - pw) // 2; offy = (self.height() - ph) // 2
        return int(ix * pw / self._bg_pixmap.width()) + offx, int(iy * ph / self._bg_pixmap.height()) + offy
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.roi_points.append(self._w2i(event.x(), event.y())); self.update()
        elif event.button() == Qt.RightButton and len(self.roi_points) >= 3:
            self.roi_changed.emit(self.roi_points)
    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.roi_points: return
        painter = QPainter(self); painter.setPen(QPen(QColor(255,0,0), 2))
        wpts = [self._i2w(p[0], p[1]) for p in self.roi_points]
        for i, (wx, wy) in enumerate(wpts):
            painter.drawEllipse(wx-4, wy-4, 8, 8)
            if i > 0: painter.drawLine(wpts[i-1][0], wpts[i-1][1], wx, wy)
        if len(wpts) >= 3:
            painter.drawLine(wpts[-1][0], wpts[-1][1], wpts[0][0], wpts[0][1])
            painter.setBrush(QBrush(QColor(255,0,0,40)))
            painter.drawPolygon(QPolygon([QPoint(p[0], p[1]) for p in wpts]))
        painter.end()
    def clear_roi(self): self.roi_points = []; self.update()


# ============ 主窗口 ============
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('非法入侵检测报警系统 - YOLOv12+ECA')
        self.setMinimumSize(1200, 750)
        self.model = None; self.tracker = SimpleTracker()
        self.detect_thread = None; self.alert_history = []
        self.roi_polygon = None; self._source_changing = False; self._last_alert_time = 0
        self._load_model(); self._init_ui()

    def _load_model(self):
        from ultralytics import YOLO
        try:
            self.model = YOLO(MODEL_PATH)
            print(f"✅ 模型加载成功: {self.model.names}")
        except Exception as e: print(f"❌ 模型加载失败: {e}")

    def _init_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        tabs = QTabWidget()
        tabs.addTab(self._tab_camera(), '📹 实时检测')
        tabs.addTab(self._tab_roi(), '🏗 领地划分')
        tabs.addTab(self._tab_photo(), '📷 照片检测')
        tabs.addTab(self._tab_history(), '📋 报警记录')
        layout.addWidget(tabs)
        self.statusBar().showMessage('就绪 | mAP@0.5: 88.3% | Precision: 82.7% | Recall: 79.0% | 输入尺寸: 1280 | 置信度: 0.2')

    # ---------- 实时检测页 ----------
    def _tab_camera(self):
        tab = QWidget(); vbox = QVBoxLayout(tab)
        ctrl = QHBoxLayout()
        self.btn_start = QPushButton('▶ 启动检测'); self.btn_start.clicked.connect(self._start_camera)
        self.btn_stop = QPushButton('⏹ 停止'); self.btn_stop.clicked.connect(self._stop_detect); self.btn_stop.setEnabled(False)
        self.btn_night = QPushButton('🌙 夜间增强'); self.btn_night.setCheckable(True); self.btn_night.clicked.connect(self._toggle_night)
        self.cmb_source = QComboBox(); self.cmb_source.addItems(['摄像头0', '摄像头1', '选择视频文件...'])
        self.cmb_source.currentIndexChanged.connect(self._source_changed)
        self.lbl_fps = QLabel('FPS: --'); self.lbl_targets = QLabel('目标: 0'); self.lbl_intrusions = QLabel('0')
        ctrl.addWidget(QLabel('输入源:')); ctrl.addWidget(self.cmb_source)
        ctrl.addWidget(self.btn_start); ctrl.addWidget(self.btn_stop); ctrl.addWidget(self.btn_night); ctrl.addStretch()
        ctrl.addWidget(self.lbl_fps); ctrl.addWidget(self.lbl_targets); ctrl.addWidget(self.lbl_intrusions)
        vbox.addLayout(ctrl)
        self.lbl_roi_status = QLabel('⚠ 未设置领地 | 请先在"领地划分"页设置防护区域')
        self.lbl_roi_status.setStyleSheet('color: orange; font-weight: bold; padding: 5px;')
        vbox.addWidget(self.lbl_roi_status)
        grp = QGroupBox('模型性能'); mlay = QHBoxLayout()
        for name, val, c in [('mAP@0.5','88.3%','#2196F3'),('Precision','82.7%','#4CAF50'),('Recall','79.0%','#FF9800'),('FPS','57.8','#9C27B0')]:
            mlay.addWidget(self._card(name, val, c))
        grp.setLayout(mlay); vbox.addWidget(grp)
        self.video_label = QLabel('点击"启动检测"开始'); self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(800, 500); self.video_label.setStyleSheet('border: 2px dashed #aaa; background: #000;')
        vbox.addWidget(self.video_label)
        return tab

    def _card(self, name, value, color):
        w = QWidget(); lay = QVBoxLayout(w); lay.setContentsMargins(8,4,8,4)
        v = QLabel(value); v.setAlignment(Qt.AlignCenter); v.setStyleSheet(f'font-size:20px;font-weight:bold;color:{color};')
        n = QLabel(name); n.setAlignment(Qt.AlignCenter); n.setStyleSheet('font-size:11px;color:#666;')
        lay.addWidget(v); lay.addWidget(n)
        w.setStyleSheet(f'border:1px solid {color};border-radius:6px;background:white;')
        return w

    # ---------- 照片检测页 ----------
    def _tab_photo(self):
        tab = QWidget(); vbox = QVBoxLayout(tab)
        ctrl = QHBoxLayout()
        btn_open = QPushButton('📂 选择照片'); btn_open.clicked.connect(self._photo_detect)
        self.lbl_photo_result = QLabel('选择照片进行检测')
        self.lbl_photo_result.setStyleSheet('font-size: 14px; color: #666;')
        ctrl.addWidget(btn_open); ctrl.addStretch(); ctrl.addWidget(self.lbl_photo_result)
        vbox.addLayout(ctrl)
        self.photo_label = QLabel('点击"选择照片"开始检测'); self.photo_label.setAlignment(Qt.AlignCenter)
        self.photo_label.setMinimumSize(800, 500); self.photo_label.setStyleSheet('border: 2px dashed #aaa; background: #f5f5f5;')
        vbox.addWidget(self.photo_label)
        return tab

    def _photo_detect(self):
        if not self.model:
            QMessageBox.warning(self, '错误', '模型未加载'); return
        path, _ = QFileDialog.getOpenFileName(self, '选择照片', '', 'Images (*.jpg *.jpeg *.png)')
        if not path: return
        frame = cv2.imread(path)
        if frame is None:
            QMessageBox.warning(self, '错误', '图片读取失败'); return
        # 多尺度检测
        all_dets = []
        for size in [640, 1280, 1920]:
            results = self.model.predict(source=frame, conf=CONF_PHOTO, imgsz=size, verbose=False)
            for r in results:
                if r.boxes is None: continue
                for box in r.boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    all_dets.append((x1,y1,x2,y2,conf,cls))
        # 去重
        dets = []
        for d in sorted(all_dets, key=lambda x: -x[4]):
            dup = False
            for k in dets:
                x1,y1 = max(d[0],k[0]), max(d[1],k[1])
                x2,y2 = min(d[2],k[2]), min(d[3],k[3])
                inter = max(0,x2-x1)*max(0,y2-y1)
                a1 = (d[2]-d[0])*(d[3]-d[1])
                a2 = (k[2]-k[0])*(k[3]-k[1])
                iou = inter/(a1+a2-inter+1e-6)
                if iou > 0.5: dup = True; break
            if not dup: dets.append(d)
        count = 0
        for x1,y1,x2,y2,conf,cls in dets:
            w_box = x2 - x1; h_box = y2 - y1
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
            label = f'{self.model.names[cls]} {conf:.2f} ({w_box}x{h_box})'
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1-th-6), (x1+tw, y1), (0,0,255), -1)
            cv2.putText(frame, label, (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
            count += 1
        self._show_frame(frame, self.photo_label)
        self.lbl_photo_result.setText(f'检测完成: {os.path.basename(path)} | 目标: {count}')
        self.lbl_photo_result.setStyleSheet(f'font-size:14px;font-weight:bold;color:{"red" if count > 0 else "green"};')

    # ---------- 领地划分页 ----------
    def _tab_roi(self):
        tab = QWidget(); hbox = QHBoxLayout(tab)
        left = QVBoxLayout()
        left.addWidget(QLabel('在图片上点击画领地边界（左键画点，右键完成）:'))
        self.roi_label = ROIDrawLabel(); self.roi_label.setAlignment(Qt.AlignCenter)
        self.roi_label.setMinimumSize(640, 480); self.roi_label.setStyleSheet('border: 2px dashed #aaa; background: #f5f5f5;')
        self.roi_label.roi_changed.connect(self._on_roi_done)
        left.addWidget(self.roi_label); hbox.addLayout(left, 3)
        right = QVBoxLayout(); grp = QGroupBox('领地设置'); glay = QVBoxLayout()
        for text, slot in [('📂 加载背景图片', self._load_roi_bg), ('🎬 从视频提取首帧', self._load_roi_video), ('🗑 清除领地', self._clear_roi), ('✅ 应用到检测', self._apply_roi)]:
            btn = QPushButton(text); btn.clicked.connect(slot); glay.addWidget(btn)
        glay.addWidget(QLabel('')); glay.addWidget(QLabel('操作说明:'))
        for t in ['1. 加载背景图或视频首帧', '2. 左键点击画领地顶点', '3. 右键点击完成绘制', '4. 点击"应用到检测"']:
            glay.addWidget(QLabel(t))
        self.lbl_roi_info = QLabel('当前: 未设置'); self.lbl_roi_info.setStyleSheet('color: #666;')
        glay.addWidget(self.lbl_roi_info)
        grp.setLayout(glay); right.addWidget(grp); right.addStretch(); hbox.addLayout(right, 1)
        return tab

    def _load_roi_bg(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择背景图片', '', 'Images (*.jpg *.jpeg *.png)')
        if not path: return
        frame = cv2.imread(path)
        if frame is not None: self._show_roi_frame(frame)

    def _load_roi_video(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择视频', '', 'Videos (*.mp4 *.avi *.mkv *.mov)')
        if not path: return
        cap = cv2.VideoCapture(path); ret, frame = cap.read(); cap.release()
        if not ret: QMessageBox.warning(self, '错误', '无法读取视频首帧'); return
        self._show_roi_frame(frame)

    def _show_roi_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        self.roi_label.set_background(pix)
        self.roi_label.setPixmap(pix.scaled(self.roi_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.roi_label.clear_roi()
        self.statusBar().showMessage(f'已加载背景 ({w}x{h})')

    def _on_roi_done(self, points):
        self.roi_polygon = np.array(points, dtype=np.int32)
        self.lbl_roi_info.setText(f'当前: {len(points)} 个顶点的多边形')
        self.lbl_roi_info.setStyleSheet('color: green; font-weight: bold;')

    def _apply_roi(self):
        if self.roi_polygon is None:
            QMessageBox.warning(self, '提示', '请先画领地区域'); return
        self.lbl_roi_status.setText('✅ 领地已设置 | 检测到入侵将自动报警')
        self.lbl_roi_status.setStyleSheet('color: green; font-weight: bold; padding: 5px;')
        QMessageBox.information(self, '成功', '领地已应用到检测！')

    def _clear_roi(self):
        self.roi_polygon = None; self.roi_label.clear_roi()
        self.lbl_roi_info.setText('当前: 未设置'); self.lbl_roi_info.setStyleSheet('color: #666;')
        self.lbl_roi_status.setText('⚠ 未设置领地 | 请先在"领地划分"页设置防护区域')
        self.lbl_roi_status.setStyleSheet('color: orange; font-weight: bold; padding: 5px;')

    # ---------- 报警记录页 ----------
    def _tab_history(self):
        tab = QWidget(); vbox = QVBoxLayout(tab)
        ctrl = QHBoxLayout()
        btn_clear = QPushButton('🗑 清空'); btn_clear.clicked.connect(lambda: (self.alert_history.clear(), self.tbl_history.setRowCount(0)))
        btn_export = QPushButton('💾 导出'); btn_export.clicked.connect(self._export)
        ctrl.addWidget(btn_clear); ctrl.addWidget(btn_export); ctrl.addStretch()
        vbox.addLayout(ctrl)
        self.tbl_history = QTableWidget(0, 4)
        self.tbl_history.setHorizontalHeaderLabels(['时间', '类型', '数量', '详情'])
        self.tbl_history.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_history.setAlternatingRowColors(True)
        vbox.addWidget(self.tbl_history)
        return tab

    # ---------- 控制 ----------
    def _source_changed(self, idx):
        if self._source_changing: return
        if idx == 2:
            self._source_changing = True
            path, _ = QFileDialog.getOpenFileName(self, '选择视频', '', 'Videos (*.mp4 *.avi *.mkv *.mov)')
            if path: self._video_path = path; self.cmb_source.setItemText(2, f'视频: {os.path.basename(path)}')
            else: self.cmb_source.setCurrentIndex(0)
            self._source_changing = False

    def _start_camera(self):
        if not self.model: QMessageBox.warning(self, '错误', '模型未加载'); return
        idx = self.cmb_source.currentIndex()
        source = idx if idx < 2 else getattr(self, '_video_path', None)
        if source is None: QMessageBox.warning(self, '错误', '请先选择视频文件'); return
        self._stop_detect(); self.tracker = SimpleTracker()
        self.detect_thread = DetectThread(self.model, self.tracker, self.roi_polygon)
        self.detect_thread.night_mode = self.btn_night.isChecked()
        self.detect_thread.source = source
        self.detect_thread.result_ready.connect(self._on_result)
        self.detect_thread.fps_update.connect(self._on_fps)
        self.detect_thread.finished.connect(self._on_finished)
        self.detect_thread.start()
        self.btn_start.setEnabled(False); self.btn_stop.setEnabled(True)

    def _stop_detect(self):
        if self.detect_thread and self.detect_thread.isRunning():
            self.detect_thread.stop(); self.detect_thread.wait(2000)
        self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)

    def _toggle_night(self):
        if self.detect_thread and self.detect_thread.isRunning():
            self.detect_thread.night_mode = self.btn_night.isChecked()
        self.btn_night.setText('☀ 关闭增强' if self.btn_night.isChecked() else '🌙 夜间增强')

    def _on_result(self, frame, tracked, intrusions, count):
        self._show_frame(frame, self.video_label)
        self.lbl_targets.setText(f'目标: {len(tracked)}'); self.lbl_intrusions.setText(str(count))
        if count > 0:
            # 判断最高报警级别
            max_conf = 0
            for tid in intrusions:
                if tid in tracked:
                    c = tracked[tid][5]
                    if c > max_conf: max_conf = c
            if max_conf >= 0.6:
                level = '高级'; level_color = 'red'
            elif max_conf >= 0.3:
                level = '中级'; level_color = 'orange'
            else:
                level = '低级'; level_color = '#FFD700'
            self.lbl_intrusions.setText(f'{count} [{level}]')
            self.lbl_intrusions.setStyleSheet(f'color: {level_color}; font-weight: bold;')
            now = time.time()
            if now - self._last_alert_time >= 5.0:
                self._last_alert_time = now
                self.alert_history.append({'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'type': f'{level}警报', 'count': count, 'detail': f'{count}个目标闯入领地 (置信度:{max_conf:.2f})'})
                row = self.tbl_history.rowCount(); self.tbl_history.insertRow(row)
                self.tbl_history.setItem(row, 0, QTableWidgetItem(self.alert_history[-1]['time']))
                self.tbl_history.setItem(row, 1, QTableWidgetItem(f'{level}警报'))
                self.tbl_history.setItem(row, 2, QTableWidgetItem(str(count)))
                self.tbl_history.setItem(row, 3, QTableWidgetItem(self.alert_history[-1]['detail']))
                self.tbl_history.scrollToBottom()
        else: self.lbl_intrusions.setStyleSheet('color: green;')

    def _on_fps(self, fps): self.lbl_fps.setText(f'FPS: {fps:.1f}')
    def _on_finished(self): self.btn_start.setEnabled(True); self.btn_stop.setEnabled(False)

    def _show_frame(self, frame, label):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch*w, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _export(self):
        if not self.alert_history: QMessageBox.information(self, '提示', '暂无记录'); return
        path, _ = QFileDialog.getSaveFileName(self, '导出', '报警记录.csv', 'CSV (*.csv)')
        if not path: return
        with open(path, 'w', encoding='utf-8-sig') as f:
            f.write('时间,类型,数量,详情\n')
            for a in self.alert_history: f.write(f"{a['time']},{a['type']},{a['count']},\"{a['detail']}\"\n")
        QMessageBox.information(self, '导出成功', f'已保存到: {path}')

    def closeEvent(self, event): self._stop_detect(); event.accept()


if __name__ == '__main__':
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = ''
    app = QApplication(sys.argv); app.setStyle('Fusion'); app.setFont(QFont('Microsoft YaHei', 10))
    win = MainWindow(); win.show()
    sys.exit(app.exec_())
