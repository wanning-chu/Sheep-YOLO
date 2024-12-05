## Sheep-YOLO: Improved and Lightweight YOLOv8n for Precise and Intelligent Recognition of Fattening Lambs' Behaviors and Vitality Statuses

## 目录
1. [Network structure diagram of Sheep-YOLO](#Network structure)
2. [Main changed files](#Main changed files)
3. [Environment](#Environment)
4. [Download](#Download)
5. [Train](#Train)
6. [Test](#Test)
7. [Predict](#Predict)


### Network structure
<img width="1024" src="Sheep-YOLO.jpg" alt="Sheep-YOLO的网络结构图">

### Main changed files
Main changed files compared to YOLOv8n basic model
```bash
# (1)这个文件直观地展示了Sheep-YOLO与YOLOv8n模型的不同之处。该文件展示了Sheep-YOLO的网络架构。我们在该文件中对Sheep-YOLO的每一层网络都做了详细的说明，并且在注释中详细分析了经过每一层网络后的特征图尺寸的变化。
'ultralytics/cfg/models/v8/Sheep-YOLO/Sheep-YOLO.yaml' 

#(2)文章中涉及到的FasterNet轻量级网络代码可以在以下文件找到 
'ultralytics/nn/modules/block.py'  # Lines 3657-3725
'ultralytics/nn/tasks.py' # Line 733, Line 745

# (3)文章中涉及到的Mixed Local Channel Attention(MLCA)module 可以在以下文件找到
'ultralytics/nn/modules/MLCA.py'

# (4)文章中涉及到的Content-Aware ReAssembly of FEatures(CARAFE) 可以在以下文件找到
'ultralytics/nn/modules/block.py'# Lines 3525-3565
```


### Environment
```bash
# 硬件配置
Intel Xeon E5-2686 v4 (2.3GHz) CPU with 64 GB of RAM
NVIDIA GeForce RTX 4090 56 GB GPU
# 软件环境
python == 3.10.9
torch == 2.0.1
torchvision == 0.15.2
CUDA==11.8
cuDNN==8.7.0
OpenCV-Python==4.9.0
```

### Download
(1)权重文件：使用Sheep-YOLO训练的权重可在谷歌网盘中下载。    
链接: https://pan.baidu.com/s/1A22fC5cPRb74gqrpq7O9-A [上传到谷歌云盘]

(2)数据集
本文所制作的育肥羔羊行为与活力状态的数据集可在谷歌网盘中下载。  
链接: https://pan.baidu.com/s/1vkk3lMheUm6IjTXznlg7Ng [上传到谷歌云盘]

### Train
(1) 配置环境 
```bash
pip install ultralytics
pip install requirements.txt
```
(2) 修改'ultralytics/cfg/datasets/sheep.yaml'中的数据集路径为你自己的路径


(3) 修改train.py中yaml文件的路径与data的路径，在这里可以修改epoch与batch等超参数的值
```bash
if __name__ == '__main__':
    model = YOLO('ultralytics/cfg/models/v8/Sheep-YOLO/Sheep-YOLO.yaml')  # build a new model from YAML
    model.train(
        data='ultralytics/cfg/datasets/sheep.yaml',
        epochs=32,
        imgsz=640,
        batch=1,
        save_period=5,
        resume=True,
        patience=200,
        project='runs/train/',
    )
```


### Test
运行test.py，得到所训练模型的性能指标表现
```bash
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
```

### Predict
运行 predict.py，能够得到图片/视频的检测结果


