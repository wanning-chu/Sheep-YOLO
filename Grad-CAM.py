# 博客地址: https://blog.csdn.net/weixin_43694096/article/details/134517606
# pip install grad-cam -i https://pypi.tuna.tsinghua.edu.cn/simple
import os
import shutil
import warnings
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam import GradCAM
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.torch_utils import intersect_dicts
from ultralytics.utils.ops import xywh2xyxy


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    """
    将图像调整为指定大小并进行填充。

    Args:
        im (ndarray): 输入图像。
        new_shape (tuple): 目标大小。
        color (tuple): 填充颜色。
        auto (bool): 是否自动调整为最小矩形。
        scaleFill (bool): 是否拉伸填充。
        scaleup (bool): 是否允许放大。
        stride (int): 步长。

    Returns:
        tuple: 处理后的图像，缩放比例，填充大小。
    """
    shape = im.shape[:2]  # 当前图像的高和宽
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # 计算缩放比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # 计算填充
    ratio = r, r
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return im, ratio, (dw, dh)


def nms(boxes, scores, iou_threshold=0.5):
    # boxes (array): 边界框列表，形状为[num_boxes, 4]
    # scores (array): 每个边界框的置信度或分数
    # iou_threshold (float): IOU阈值，用于决定是否抑制

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]

    return keep


class YOLOv8GradCAM:
    """
    使用 Grad-CAM 可视化 YOLOv8 模型的类。

    Args:
        weight (str): 模型权重文件的路径。
        cfg (str): 模型配置文件的路径。
        device (str): 使用的设备（如 'cpu' 或 'cuda:0'）。
        method (str): Grad-CAM 方法名。
        layer (str): 目标层的名称，用于提取特征。
        backward_type (str): 反向传播的类型，可选值为 'class', 'box', 'all'。
        conf_threshold (float): 置信度阈值。
    """

    def __init__(
        self, weight, cfg, device, method, layer, backward_type, conf_threshold
    ):
        self.device = torch.device(device)
        self.conf_threshold = conf_threshold
        self.backward_type = backward_type

        # 加载模型
        try:
            ckpt = torch.load(weight, map_location=self.device)
            self.class_names = ckpt["model"].names
            csd = ckpt["model"].float().state_dict()
            self.model = DetectionModel(cfg, ch=3, nc=len(self.class_names)).to(
                self.device
            )
            csd = intersect_dicts(csd, self.model.state_dict(), exclude=["anchor"])
            self.model.load_state_dict(csd, strict=False)
            self.model.eval()
        except Exception as e:
            raise FileNotFoundError(f"模型加载失败：{e}")

        # 设置目标层
        try:
            self.target_layers = [eval(layer)]
        except Exception as e:
            raise ValueError(f"目标层设置错误：{e}")

        self.method = eval(method)

    def post_process(self, result):
        """
        对模型输出进行后处理。

        Args:
            result (Tensor): 模型的输出结果。

        Returns:
            tuple: 分类结果、边界框坐标（归一化前和归一化后）。
        """
        logits_ = result[:, 4:]
        boxes_ = result[:, :4]
        sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
        return (
            torch.transpose(logits_[0], 0, 1)[indices[0]],
            torch.transpose(boxes_[0], 0, 1)[indices[0]],
            xywh2xyxy(torch.transpose(boxes_[0], 0, 1)[indices[0]])
            .cpu()
            .detach()
            .numpy(),
        )

    def draw_detections(self, box, img):
        """
        在图像上绘制检测结果。

        Args:
            box (array): 边界框坐标。
            img (ndarray): 图像数组。

        Returns:
            ndarray: 绘制后的图像。
        """
        xmin, ymin, xmax, ymax = list(map(int, box))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        return img

    def __call__(self, img_path, save_path):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        unique_save_path = os.path.join(save_path, f"result_{timestamp}")

        if os.path.exists(unique_save_path):
            shutil.rmtree(unique_save_path)

        os.makedirs(unique_save_path, exist_ok=True)

        # 读取和预处理图像
        try:
            img = cv2.imread(img_path)
            assert img is not None, "无法读取输入图像，请检查路径。"
            img = letterbox(img)[0]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_rgb = np.float32(img_rgb) / 255.0
            tensor = (
                torch.from_numpy(np.transpose(img_rgb, (2, 0, 1)))
                .unsqueeze(0)
                .to(self.device)
            )
        except Exception as e:
            raise ValueError(f"图像预处理失败：{e}")

        grads = ActivationsAndGradients(
            self.model, self.target_layers, reshape_transform=None
        )

        result = grads(tensor)
        activations = grads.activations[0]

        post_result, pre_post_boxes, post_boxes = self.post_process(result[0])

        # 初始化累加的 saliency_map
        accumulated_saliency_map = None

        # 遍历所有检测到的目标
        for i in range(post_result.size(0)):
            # 检查置信度是否满足阈值
            if float(post_result[i].max()) < self.conf_threshold:
                continue

            self.model.zero_grad()

            # 根据 backward_type 进行反向传播
            if self.backward_type in ("class", "all"):
                score = post_result[i].max()
                score.backward(retain_graph=True)

            if self.backward_type in ("box", "all"):
                for j in range(4):
                    score = pre_post_boxes[i, j]
                    score.backward(retain_graph=True)

            # 获取梯度
            if self.backward_type == "class":
                gradients = grads.gradients[0]
            elif self.backward_type == "box":
                gradients = sum(grads.gradients[:4])
            else:
                gradients = sum(grads.gradients[:5])

            # 计算权重和 saliency_map
            print(gradients)
            gradients = gradients.cpu().detach().numpy()
            activations_np = activations.cpu().detach().numpy()
            weights = self.method.get_cam_weights(
                self.method, None, None, None, activations_np, gradients
            )
            weights = weights.reshape((weights.shape[0], weights.shape[1], 1, 1))
            saliency_map = np.sum(weights * activations_np, axis=1)
            saliency_map = np.maximum(saliency_map, 0)
            saliency_map = cv2.resize(saliency_map[0], (tensor.size(3), tensor.size(2)))
            if saliency_map.max() == saliency_map.min():
                continue
            saliency_map = (saliency_map - saliency_map.min()) / (
                saliency_map.max() - saliency_map.min()
            )

            # 累加 saliency_map
            if accumulated_saliency_map is None:
                accumulated_saliency_map = saliency_map
            else:
                accumulated_saliency_map += saliency_map

        if accumulated_saliency_map is not None:
            # 归一化累计的 saliency_map
            accumulated_saliency_map /= accumulated_saliency_map.max()

            # 生成最终的 CAM 图像
            cam_image = show_cam_on_image(
                img_rgb, accumulated_saliency_map, use_rgb=True
            )

            # 应用NMS
            scores = (
                post_result.max(1)[0].detach().cpu()
            )  # Detach and move scores to CPU
            if isinstance(post_boxes, torch.Tensor):
                post_boxes = (
                    post_boxes.detach().cpu()
                )  # Detach and move the boxes tensor to CPU
                keep = nms(
                    post_boxes.numpy(), scores.numpy(), iou_threshold=0.5
                )  # Convert tensors to numpy arrays here for NMS
            else:
                keep = nms(
                    post_boxes, scores.numpy(), iou_threshold=0.5
                )  # If already a numpy array, pass directly
            # 绘制所有的检测框
            for i in keep:
                if scores[i] < self.conf_threshold:
                    continue
                cam_image = self.draw_detections(post_boxes[i], cam_image)

            # 保存最终的图像
            cam_image = Image.fromarray(cam_image)
            cam_image.save(os.path.join(unique_save_path, "combined.png"))
            print(f"结果已保存到 {unique_save_path}")
        else:
            print("没有检测到高置信度的目标。")

def get_params():
    """
    获取默认参数。

    Returns:
        dict: 参数字典。
    """
    params = {
        # "weight": "D:/deeplearning/compare_model_weights/improved_yolov8/best.pt",
        # "cfg": "D:/deeplearning/PycharmProjects/copy_yolov8/ultralytics/cfg/models/v8/compare_test/Sheep-YOLO.yaml",
        "weight": "D:/deeplearning/compare_model_weights/yolov8n/best.pt",
        "cfg": "D:/deeplearning/PycharmProjects/copy_yolov8/ultralytics/cfg/models/v8/yolov8.yaml",
        "device": "cuda:0",
        "method": "GradCAM",
        "layer": "self.model.model[15]", #sheep-yolo 14/ yolov8:15
        "backward_type": "all",  # 值：'class', 'box', 'all'
        "conf_threshold": 0.4,
    }
    return params


if __name__ == "__main__":
    # 获取参数
    params = get_params()

    # 实例化 YOLOv8GradCAM
    model = YOLOv8GradCAM(
        weight=params["weight"],
        cfg=params["cfg"],
        device=params["device"],
        method=params["method"],
        layer=params["layer"],
        backward_type=params["backward_type"],
        conf_threshold=params["conf_threshold"],
    )

    # 处理图像
    model(
        img_path="D:/deeplearning/PycharmProjects/ultralytics-main/sheep_dataset_second/test/images/D02_20240418143729_009.jpg",  # 输入图像路径
        save_path="D:/deeplearning/compare_model_weights/对比效果/挑选出来的对比效果展示图/heatmap/yolov8",  # 结果保存路径
        # save_path="D:/deeplearning/compare_model_weights/对比效果/挑选出来的对比效果展示图/heatmap/sheep-yolo",  # 结果保存路径
    )


# import warnings
#
# warnings.filterwarnings('ignore')
# warnings.simplefilter('ignore')
# import torch, yaml, cv2, os, shutil
# import numpy as np
#
# np.random.seed(0)
# import matplotlib.pyplot as plt
# from tqdm import trange
# from PIL import Image
# from ultralytics.nn.tasks import DetectionModel as Model
# from ultralytics.utils.torch_utils import intersect_dicts
# from ultralytics.utils.ops import xywh2xyxy
# from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
#
#
# def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
#     # Resize and pad image while meeting stride-multiple constraints
#     shape = im.shape[:2]  # current shape [height, width]
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)
#
#     # Scale ratio (new / old)
#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     if not scaleup:  # only scale down, do not scale up (for better val mAP)
#         r = min(r, 1.0)
#
#     # Compute padding
#     ratio = r, r  # width, height ratios
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
#     if auto:  # minimum rectangle
#         dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
#     elif scaleFill:  # stretch
#         dw, dh = 0.0, 0.0
#         new_unpad = (new_shape[1], new_shape[0])
#         ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
#
#     dw /= 2  # divide padding into 2 sides
#     dh /= 2
#
#     if shape[::-1] != new_unpad:  # resize
#         im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
#     return im, ratio, (dw, dh)
#
#
# class yolov8_heatmap:
#     def __init__(self, weight, cfg, device, method, layer, backward_type, conf_threshold, ratio):
#         device = torch.device(device)
#         ckpt = torch.load(weight)
#         model_names = ckpt['model'].names
#         csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
#         model = Model(cfg, ch=3, nc=len(model_names)).to(device)
#         csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])  # intersect
#         model.load_state_dict(csd, strict=False)  # load
#         model.eval()
#         print(f'Transferred {len(csd)}/{len(model.state_dict())} items')
#
#         target_layers = [eval(layer)]
#         method = eval(method)
#
#         colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)
#         self.__dict__.update(locals())
#
#     def post_process(self, result):
#         logits_ = result[:, 4:]
#         boxes_ = result[:, :4]
#         sorted, indices = torch.sort(logits_.max(1)[0], descending=True)
#         return torch.transpose(logits_[0], dim0=0, dim1=1)[indices[0]], torch.transpose(boxes_[0], dim0=0, dim1=1)[
#             indices[0]], xywh2xyxy(torch.transpose(boxes_[0], dim0=0, dim1=1)[indices[0]]).cpu().detach().numpy()
#
#     def draw_detections(self, box, color, name, img):
#         xmin, ymin, xmax, ymax = list(map(int, list(box)))
#         cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
#         cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2,
#                     lineType=cv2.LINE_AA)
#         return img
#
#     def __call__(self, img_path, save_path):
#         # remove dir if exist
#         if os.path.exists(save_path):
#             shutil.rmtree(save_path)
#         # make dir if not exist
#         os.makedirs(save_path, exist_ok=True)
#
#         # img process
#         img = cv2.imread(img_path)
#         img = letterbox(img)[0]
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         img = np.float32(img) / 255.0
#         tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
#
#         # init ActivationsAndGradients
#         grads = ActivationsAndGradients(self.model, self.target_layers, reshape_transform=None)
#
#         # get ActivationsAndResult
#         result = grads(tensor)
#         activations = grads.activations[0].cpu().detach().numpy()
#
#         # postprocess to yolo output
#         post_result, pre_post_boxes, post_boxes = self.post_process(result[0])
#         for i in trange(int(post_result.size(0) * self.ratio)):
#             if float(post_result[i].max()) < self.conf_threshold:
#                 break
#
#             self.model.zero_grad()
#             # get max probability for this prediction
#             if self.backward_type == 'class' or self.backward_type == 'all':
#                 score = post_result[i].max()
#                 score.backward(retain_graph=True)
#
#             if self.backward_type == 'box' or self.backward_type == 'all':
#                 for j in range(4):
#                     score = pre_post_boxes[i, j]
#                     score.backward(retain_graph=True)
#
#             # process heatmap
#             if self.backward_type == 'class':
#                 gradients = grads.gradients[0]
#             elif self.backward_type == 'box':
#                 gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3]
#             else:
#                 gradients = grads.gradients[0] + grads.gradients[1] + grads.gradients[2] + grads.gradients[3] + grads.gradients[4]
#
#             b, k, u, v = gradients.size()
#             weights = self.method.get_cam_weights(self.method, None, None, None, activations,
#                                                   gradients.detach().numpy())
#             weights = weights.reshape((b, k, 1, 1))
#             saliency_map = np.sum(weights * activations, axis=1)
#             saliency_map = np.squeeze(np.maximum(saliency_map, 0))
#             saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)))
#             saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
#             if (saliency_map_max - saliency_map_min) == 0:
#                 continue
#             saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)
#
#             # add heatmap and box to image
#             cam_image = show_cam_on_image(img.copy(), saliency_map, use_rgb=True)
#             "不想在图片中绘画出边界框和置信度，注释下面的一行代码即可"
#             cam_image = self.draw_detections(post_boxes[i], self.colors[int(post_result[i, :].argmax())],
#                                              f'{self.model_names[int(post_result[i, :].argmax())]} {float(post_result[i].max()):.2f}',
#                                              cam_image)
#             cam_image = Image.fromarray(cam_image)
#             cam_image.save(f'{save_path}/{i}.png')
#
#
# def get_params():
#     params = {
#         'weight': r"D:\deeplearning\compare_model_weights\improved_yolov8\best.pt",  # 训练出来的权重文件
#         'cfg': r"D:\deeplearning\PycharmProjects\copy_yolov8\ultralytics\cfg\models\v8\compare_test\Sheep-YOLO.yaml",  # 训练权重对应的yaml配置文件
#         "device": "cpu",
#         'method': 'GradCAM',  # GradCAMPlusPlus, GradCAM, XGradCAM , 使用的热力图库文件不同的效果不一样可以多尝试
#         'layer': 'model.model[21]',  # 想要检测的对应层
#         'backward_type': 'all',  # class, box, all
#         'conf_threshold': 0.6,  # 0.6  # 置信度阈值，有的时候你的进度条到一半就停止了就是因为没有高于此值的了
#         'ratio': 0.02  # 0.02-0.1
#     }
#     return params
#
#
# if __name__ == '__main__':
#     model = yolov8_heatmap(**get_params())
#     model(r'D:/deeplearning/PycharmProjects/ultralytics-main/sheep_dataset_second/test/images/D02_20240419102221_007.jpg', 'D:/deeplearning/compare_model_weights/对比效果/挑选出来的对比效果展示图/heatmap/sheep-yolo')  # 第一个是检测的文件, 第二个是保存的路径