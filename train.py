from ultralytics import YOLO
from ultralytics.models import RTDETR

if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/Sheep-YOLO/Sheep-YOLO.yaml')  # build a new model from YAML
    model.train(
        data='ultralytics/cfg/datasets/sheep.yaml',
        epochs=10,
        imgsz=640,
        batch=1,
        save_period=5,
        resume=True,
        patience=200,
        project='runs/5-fold_compare/train/',
    )

