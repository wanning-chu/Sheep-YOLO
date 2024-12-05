import cv2
import os
import glob


class VideoFrameExtractor:
    def __init__(self):
        self.start_name = '1.jpg'
        self.timeF = 100

    def extract_frames(self, video_path, save_path):
        video_files = glob.glob(os.path.join(video_path, '*.mp4'))  # 获取目标文件夹下所有的MP4文件路径

        for video_file in video_files:
            cv = cv2.VideoCapture(video_file)  # 读入视频文件，命名为 cv

            if cv.isOpened():  # 判断是否正常打开
                rval, frame = cv.read()
                i = 1
                n = 1  # 重置帧计数器
            else:
                rval = False
                print(f'无法打开视频文件: {video_file}')
                continue

            video_name = os.path.splitext(os.path.basename(video_file))[0]  # 获取视频文件名（不带后缀）

            while rval:  # 正常打开 开始处理
                rval, frame = cv.read()
                jpg_name = os.path.join(save_path, f"{video_name}_{str(n).zfill(3)}.jpg")  # 命名保存的图片

                if (i % self.timeF == 0):  # 每隔timeF帧进行存储操作
                    try:
                        cv2.imwrite(jpg_name, frame)  # 存储为图像
                    except Exception as e:
                        print(f'保存图片出错: {e}')
                    n += 1
                i += 1
            cv2.waitKey(1)
            cv.release()
            print(f'{video_file} 处理完成')
        print('抽帧完成')


if __name__ == "__main__":
    video_folder_path = 'D:/deeplearning/sheep_status_video/20240420-三只/萎靡'  # 设置为视频文件存储的目标文件夹路径
    img_output_path = 'D:/deeplearning/sheep_status_dataset_forlabel/20240420-multi/dropping' # 文件夹不能含中文

    if not os.path.exists(img_output_path):  # 如果存储图片的文件夹不存在，自动创建保存图片文件夹
        os.makedirs(img_output_path)

    frame_extractor = VideoFrameExtractor()
    frame_extractor.extract_frames(video_folder_path, img_output_path)  # 执行提取图片程序