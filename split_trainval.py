import os
import shutil

def merge_and_split_for_cross_validation(src_base_dir, dest_base_dir):
    # 确保目标目录不存在，避免覆盖现有数据
    if os.path.exists(dest_base_dir):
        print("Destination directory already exists. Please remove or rename it before proceeding.")
        return

    os.makedirs(dest_base_dir, exist_ok=True)

    # 初始化五个目标文件夹
    for i in range(1, 6):
        target_folder = os.path.join(dest_base_dir, f"data_{i}")
        os.makedirs(os.path.join(target_folder, "train/images"), exist_ok=True)
        os.makedirs(os.path.join(target_folder, "train/labels"), exist_ok=True)
        os.makedirs(os.path.join(target_folder, "val/images"), exist_ok=True)
        os.makedirs(os.path.join(target_folder, "val/labels"), exist_ok=True)

    # 遍历每个验证集（trainval1至trainval5）
    for val_set in range(1, 6):
        val_folder = os.path.join(src_base_dir, f"trainval{val_set}")
        val_images = os.path.join(val_folder, "images")
        val_labels = os.path.join(val_folder, "labels")

        # 确定当前验证集应该放入的目标文件夹
        target_val_folder = os.path.join(dest_base_dir, f"yolov8n_{val_set}")

        # 复制验证集的图片和标签到对应的目标文件夹的val子文件夹
        for item in os.listdir(val_images):
            shutil.copy(os.path.join(val_images, item), os.path.join(target_val_folder, "val/images", item))
        for item in os.listdir(val_labels):
            shutil.copy(os.path.join(val_labels, item), os.path.join(target_val_folder, "val/labels", item))

        # 对于每个训练集，除了当前的验证集，将图片和标签复制到所有目标文件夹的train子文件夹
        for train_set in range(1, 6):
            if train_set != val_set:
                src_train_folder = os.path.join(src_base_dir, f"trainval{train_set}")
                src_images = os.path.join(src_train_folder, "images")
                src_labels = os.path.join(src_train_folder, "labels")

                # 复制到当前目标文件夹的train子文件夹
                for item in os.listdir(src_images):
                    shutil.copy(os.path.join(src_images, item), os.path.join(target_val_folder, "train/images", item))
                for item in os.listdir(src_labels):
                    shutil.copy(os.path.join(src_labels, item), os.path.join(target_val_folder, "train/labels", item))

    print("Cross-validation datasets prepared.")

# 示例使用
src_base_dir = 'D:/ultralytics-main/sheep_dataset_second/'  # 存放trainval1-5的目录
dest_base_dir = 'D:/ultralytics-main/sheep_dataset_second/data/'  # 输出五折交叉验证数据集的目录

merge_and_split_for_cross_validation(src_base_dir, dest_base_dir)