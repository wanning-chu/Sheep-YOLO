from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('D:/deeplearning/compare_model_weights/improved_yolov8/best.pt') # 自己训练结束后的模型权重
    model.val(data='ultralytics/cfg/datasets/sheep.yaml',
              split='test',
              imgsz=640,
              batch=32,
              save_json=True, # if you need to cal coco metrice
              project='runs/test',
              name='exp',
              )
