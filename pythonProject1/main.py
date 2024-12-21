import sys
from PyQt5.QtWidgets import QApplication
from ui.main_window import MainWindow
from detector.face_detector import FaceDetector
import utils.image_enhancement

def main():
    # 创建应用程序
    app = QApplication(sys.argv)
    
    # 创建主窗口
    window = MainWindow()
    
    # 初始化人脸检测器
    detector = FaceDetector()
    
    # 初始化图像增强器
    enhancer = ImageEnhancer()
    
    # 显示窗口
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())

if __name__ == '__main__':
    main() 