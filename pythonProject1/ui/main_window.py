from PyQt5.QtWidgets import QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QFileDialog, QCheckBox, QMessageBox
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
from detector.face_detector import FaceDetector
from utils.image_enhancement import ImageEnhancer

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸检测系统")
        self.setGeometry(100, 100, 1200, 800)
        
        try:
            # 初始化检测器和增强器
            self.face_detector = FaceDetector()
            self.image_enhancer = ImageEnhancer()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"初始化人脸检测器失败: {str(e)}")
            self.face_detector = None
        
        # 创建主窗口部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 创建布局
        self.main_layout = QHBoxLayout(self.central_widget)
        self.video_layout = QVBoxLayout()
        self.control_layout = QVBoxLayout()
        
        # 视频显示区域
        self.video_label = QLabel()
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_layout.addWidget(self.video_label)
        
        # 控制按钮
        self.start_camera_btn = QPushButton("启动摄像头")
        self.stop_camera_btn = QPushButton("停止摄像头")
        self.load_image_btn = QPushButton("加载图片")
        self.save_image_btn = QPushButton("保存图片")
        
        # 复选框
        self.detect_faces_cb = QCheckBox("人脸检测")
        self.enhance_image_cb = QCheckBox("图像增强")
        
        # 添加控件到控制布局
        self.control_layout.addWidget(self.start_camera_btn)
        self.control_layout.addWidget(self.stop_camera_btn)
        self.control_layout.addWidget(self.detect_faces_cb)
        self.control_layout.addWidget(self.enhance_image_cb)
        self.control_layout.addWidget(self.load_image_btn)
        self.control_layout.addWidget(self.save_image_btn)
        self.control_layout.addStretch()
        
        # 将布局添加到主布局
        self.main_layout.addLayout(self.video_layout, stretch=7)
        self.main_layout.addLayout(self.control_layout, stretch=3)
        
        # 初始化摄像头
        self.camera = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        
        # 连接信号和槽
        self.start_camera_btn.clicked.connect(self.start_camera)
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.load_image_btn.clicked.connect(self.load_image)
        self.save_image_btn.clicked.connect(self.save_image)
        
        # 状态变量
        self.is_camera_active = False
        self.current_frame = None
        
        # 初始化按钮状态
        self.stop_camera_btn.setEnabled(False)
        self.save_image_btn.setEnabled(False)
        
        # 如果检测器初始化失败，禁用相关功能
        if self.face_detector is None:
            self.detect_faces_cb.setEnabled(False)
    
    def start_camera(self):
        if not self.is_camera_active:
            self.camera = cv2.VideoCapture(0)
            if self.camera.isOpened():
                self.is_camera_active = True
                self.timer.start(30)  # 30ms 刷新率
                self.start_camera_btn.setEnabled(False)
                self.stop_camera_btn.setEnabled(True)
                self.save_image_btn.setEnabled(True)
            else:
                QMessageBox.warning(self, "警告", "无法打开摄像头")
    
    def stop_camera(self):
        if self.is_camera_active:
            self.timer.stop()
            self.camera.release()
            self.is_camera_active = False
            self.start_camera_btn.setEnabled(True)
            self.stop_camera_btn.setEnabled(False)
            self.save_image_btn.setEnabled(False)
            self.video_label.clear()
    
    def update_frame(self):
        ret, frame = self.camera.read()
        if ret:
            try:
                self.current_frame = frame.copy()
                
                # 图像增强
                if self.enhance_image_cb.isChecked():
                    frame = self.image_enhancer.enhance_for_detection(frame)
                
                # 人脸检测
                if self.detect_faces_cb.isChecked() and self.face_detector is not None:
                    detections = self.face_detector.detect(frame)
                    frame = self.face_detector.draw_detections(frame, detections)
                
                # 转换图像格式用于显示
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                    self.video_label.size(), Qt.KeepAspectRatio))
            except Exception as e:
                QMessageBox.warning(self, "警告", f"处理图像时出错: {str(e)}")
    
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)")
        if file_name:
            try:
                image = cv2.imread(file_name)
                if image is not None:
                    self.current_frame = image.copy()
                    
                    # 图像增强
                    if self.enhance_image_cb.isChecked():
                        image = self.image_enhancer.enhance_for_detection(image)
                    
                    # 人脸检测
                    if self.detect_faces_cb.isChecked() and self.face_detector is not None:
                        detections = self.face_detector.detect(image)
                        image = self.face_detector.draw_detections(image, detections)
                    
                    # 显示图像
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    h, w, ch = image.shape
                    bytes_per_line = ch * w
                    qt_image = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    self.video_label.setPixmap(QPixmap.fromImage(qt_image).scaled(
                        self.video_label.size(), Qt.KeepAspectRatio))
                    self.save_image_btn.setEnabled(True)
                else:
                    QMessageBox.warning(self, "警告", "无法加载图片")
            except Exception as e:
                QMessageBox.warning(self, "警告", f"处理图片时出错: {str(e)}")
    
    def save_image(self):
        if self.current_frame is not None:
            try:
                file_name, _ = QFileDialog.getSaveFileName(
                    self, "保存图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp)")
                if file_name:
                    # 保存当前显示的图像
                    image = self.current_frame.copy()
                    
                    # 图像增强
                    if self.enhance_image_cb.isChecked():
                        image = self.image_enhancer.enhance_for_detection(image)
                    
                    # 人脸检测
                    if self.detect_faces_cb.isChecked() and self.face_detector is not None:
                        detections = self.face_detector.detect(image)
                        image = self.face_detector.draw_detections(image, detections)
                    
                    cv2.imwrite(file_name, image)
                    QMessageBox.information(self, "成功", "图片保存成功")
            except Exception as e:
                QMessageBox.warning(self, "警告", f"保存图片时出错: {str(e)}") 