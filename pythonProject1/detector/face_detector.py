import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import os

class FaceDetector:
    def __init__(self, model_path="D:/yolov5-v7.0/yolov5s.pt"):
        """
        初始化人脸检测器
        :param model_path: YOLOv5模型路径，默认使用本地模型
        """
        try:
            # 直接加载模型
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = torch.jit.load(model_path) if model_path.endswith('.torchscript') else torch.load(model_path, map_location=self.device)
            if isinstance(self.model, dict):
                self.model = self.model['model']  # 提取模型
            self.model.to(self.device).eval()  # 设置为评估模式
            
            # 设置推理参数
            self.conf_threshold = 0.5
            self.iou_threshold = 0.45
            self.img_size = 640
            
        except Exception as e:
            print(f"模型加载错误: {str(e)}")
            raise
    
    def detect(self, image):
        """
        检测图像中的人脸
        :param image: OpenCV格式的图像(BGR)
        :return: 检测到的人脸框列表 [(x1,y1,x2,y2,conf),...]
        """
        try:
            # 预处理图像
            img = cv2.resize(image, (self.img_size, self.img_size))
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # 执行检测
            with torch.no_grad():
                pred = self.model(img)
                if isinstance(pred, tuple):
                    pred = pred[0]
                
                # 应用非极大值抑制
                pred = self._non_max_suppression(pred)
            
            # 处理检测结果
            detections = []
            if len(pred) > 0 and pred[0] is not None:
                # 调整检测框到原始图像大小
                pred = pred[0].cpu().numpy()
                h, w = image.shape[:2]
                scale_w = w / self.img_size
                scale_h = h / self.img_size
                
                for det in pred:
                    x1, y1, x2, y2, conf, cls = det
                    x1 *= scale_w
                    x2 *= scale_w
                    y1 *= scale_h
                    y2 *= scale_h
                    detections.append([x1, y1, x2, y2, conf, cls])
            
            return np.array(detections)
            
        except Exception as e:
            print(f"检测错误: {str(e)}")
            return np.array([])
    
    def _non_max_suppression(self, prediction, conf_thres=0.5, iou_thres=0.45, classes=None):
        """
        执行非极大值抑制
        """
        # 获取置信度大于阈值的预测结果
        mask = prediction[..., 4] > conf_thres
        output = [None] * prediction.shape[0]
        
        for i, pred in enumerate(prediction):
            pred = pred[mask[i]]
            if not pred.shape[0]:
                continue
                
            # 计算置信度分数
            pred[:, 5:] *= pred[:, 4:5]
            
            # 获取边界框坐标
            box = self._xywh2xyxy(pred[:, :4])
            
            # 应用NMS
            conf, j = pred[:, 5:].max(1, keepdim=True)
            pred = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            
            if not pred.shape[0]:
                continue
                
            # 执行NMS
            boxes, scores = pred[:, :4], pred[:, 4]
            i = self._nms(boxes, scores, iou_thres)
            output[i] = pred[i]
            
        return output
    
    @staticmethod
    def _xywh2xyxy(x):
        """
        将中心点坐标和宽高转换为左上右下坐标
        """
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y
    
    @staticmethod
    def _nms(boxes, scores, iou_threshold):
        """
        执行非极大值抑制
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        _, order = scores.sort(0, descending=True)
        keep = []
        
        while order.numel() > 0:
            if order.numel() == 1:
                keep.append(order.item())
                break
            i = order[0]
            keep.append(i)
            
            xx1 = x1[order[1:]].clamp(min=x1[i])
            yy1 = y1[order[1:]].clamp(min=y1[i])
            xx2 = x2[order[1:]].clamp(max=x2[i])
            yy2 = y2[order[1:]].clamp(max=y2[i])
            
            w = (xx2 - xx1).clamp(min=0)
            h = (yy2 - yy1).clamp(min=0)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            ids = (ovr <= iou_threshold).nonzero().squeeze()
            if ids.numel() == 0:
                break
            order = order[ids + 1]
        
        return torch.tensor(keep)
    
    def draw_detections(self, image, detections):
        """
        在图像上绘制检测结果
        :param image: OpenCV格式的图像
        :param detections: 检测结果列表
        :return: 绘制了检测框的图像
        """
        try:
            img = image.copy()
            for det in detections:
                x1, y1, x2, y2, conf, cls = det
                # 转换为整数坐标
                box = np.array([x1, y1, x2, y2]).astype(int)
                # 绘制边界框
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                # 添加置信度标签
                label = f'Face {conf:.2f}'
                cv2.putText(img, label, (box[0], box[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return img
        except Exception as e:
            print(f"绘制检测框错误: {str(e)}")
            return image
    
    @staticmethod
    def enhance_image(image):
        """
        图像增强处理
        :param image: OpenCV格式的图像
        :return: 增强后的图像
        """
        try:
            # 直方图均衡化
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # 锐化
            kernel = np.array([[-1,-1,-1],
                             [-1, 9,-1],
                             [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            return enhanced
        except Exception as e:
            print(f"图像增强错误: {str(e)}")
            return image