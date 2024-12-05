import os
import shutil
import random
from itertools import zip_longest


def split_data(src_images_dir, src_labels_dir, dest_base_dir, folds=5):
    # 获取所有图片和label的文件名
    images = os.listdir(src_images_dir)
    labels = os.listdir(src_labels_dir)

    # 确保图片和label数量一致
    assert len(images) == len(labels), "图片和标签数量不一致"

    # 打乱文件顺序以实现随机分配
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images[:], labels[:] = zip(*combined)

    # 分割数据到folds个部分
    fold_size = len(images) // folds
    for i in range(folds):
        dest_fold_dir = os.path.join(dest_base_dir, f"trainval{i + 1}")
        os.makedirs(os.path.join(dest_fold_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(dest_fold_dir, "labels"), exist_ok=True)

        start = i * fold_size
        end = (i + 1) * fold_size if i < folds - 1 else len(images)

        # 拷贝图片和label到对应fold的子文件夹
        for img, lbl in combined[start:end]:
            shutil.copy(os.path.join(src_images_dir, img),
                        os.path.join(dest_fold_dir, "images", img))
            shutil.copy(os.path.join(src_labels_dir, lbl),
                        os.path.join(dest_fold_dir, "labels", lbl))


# 示例使用
src_images_dir = 'D:/ultralytics-main/sheep_dataset_second/traninval_all_txt/images'
src_labels_dir = 'D:/ultralytics-main/sheep_dataset_second/traninval_all_txt/labels'
dest_base_dir = 'D:/ultralytics-main/sheep_dataset_second'

split_data(src_images_dir, src_labels_dir, dest_base_dir)