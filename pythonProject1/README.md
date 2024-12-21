# 基于YOLOv5的人脸检测系统

本项目是一个基于YOLOv5和PyQt5的实时人脸检测系统，支持实时视频流处理和图像增强功能。

## 功能特点

- 实时人脸检测
- 摄像头视频流处理
- 图像增强处理
- 友好的图形用户界面
- 实时人脸标注显示

## 环境要求

- Python 3.8+
- PyQt5
- YOLOv5
- OpenCV
- CUDA（可选，用于GPU加速）

## 安装说明

1. 克隆项目到本地
2. 安装依赖：
```bash
pip install -r requirements.txt
```
3. 下载YOLOv5预训练模型

## 使用说明

运行主程序：
```bash
python main.py
```

## 项目结构

```
├── main.py              # 主程序入口
├── ui/                  # 用户界面相关代码
├── detector/            # 检测器相关代码
├── utils/              # 工具函数
└── requirements.txt    # 项目依赖
``` 