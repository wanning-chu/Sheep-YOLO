import os
import random
import shutil

# 指定源文件夹
images_source_directory = r'D:\deeplearning\Sheep_YOLO_Dataset\test\images'
labels_source_directory = r'D:\deeplearning\Sheep_YOLO_Dataset\test\labels'

# 指定目标文件夹
images_target_directory = r'D:\deeplearning\Sheep_YOLO_Dataset\testnew\images'
labels_target_directory = r'D:\deeplearning\Sheep_YOLO_Dataset\testnew\labels'

# 确保目标文件夹存在
if not os.path.exists(images_target_directory):
    os.makedirs(images_target_directory)
if not os.path.exists(labels_target_directory):
    os.makedirs(labels_target_directory)

# 获取源文件夹中的文件名
image_files = [f for f in os.listdir(images_source_directory) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
label_files = [f for f in os.listdir(labels_source_directory) if f.endswith('.txt')]

# 检查文件名是否一一对应（这里简单假设文件名（不包括扩展名）相同即为对应）
file_pairs = []
for img_file in image_files:
    img_name, img_ext = os.path.splitext(img_file)
    for lbl_file in label_files:
        lbl_name, _ = os.path.splitext(lbl_file)
        if img_name == lbl_name:
            file_pairs.append((img_file, lbl_file))
            break  # 找到对应文件后跳出内层循环

# 如果file_pairs的长度不等于image_files或label_files的长度，说明有文件没有对应上
if len(file_pairs) != len(image_files) or len(file_pairs) != len(label_files):
    raise ValueError("Some image and label files do not have corresponding names.")

# 打乱文件名对列表
random.shuffle(file_pairs)

# 重命名文件并复制到目标文件夹
for i, (img_file, lbl_file) in enumerate(file_pairs):
    # 获取文件扩展名
    img_extension = os.path.splitext(img_file)[1]
    lbl_extension = os.path.splitext(lbl_file)[1]

    # 构建新文件名
    new_img_filename = f"{i + 1}{img_extension}"
    new_lbl_filename = f"{i + 1}{lbl_extension}"

    # 构建源文件路径和新文件路径
    img_source_path = os.path.join(images_source_directory, img_file)
    lbl_source_path = os.path.join(labels_source_directory, lbl_file)
    img_new_path = os.path.join(images_target_directory, new_img_filename)
    lbl_new_path = os.path.join(labels_target_directory, new_lbl_filename)

    # 复制并重命名文件
    shutil.copy2(img_source_path, img_new_path)
    shutil.copy2(lbl_source_path, lbl_new_path)

print("图片和标签文件已打乱顺序并重命名完成，并复制到了新目录中。原始文件保持不变。")