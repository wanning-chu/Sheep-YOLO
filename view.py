from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(r'D:\ultralytics-main\ultralytics\cfg\models\v8\yolov8n.yaml')  # build a new model from YAML
    model.info()

