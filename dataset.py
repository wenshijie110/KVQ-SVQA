import os
import random
import cv2
import numpy as np
import pandas as pd
import csv
from decord import VideoReader, gpu, cpu
from tqdm import tqdm
import torch
random.seed(42)


def get_image_feature(video_path, coordinate_file, clip_type, clip_length=8, cropped_size=(448, 768), sal_cropped_size=(224,224), phase='train'):
    vr = VideoReader(video_path)
    num_frames = len(vr)
    width, height = vr[0].shape[1], vr[0].shape[0]
    # 获取帧数
    selected_frames_indices = []
    if clip_type == 'uniform':
        step = num_frames // clip_length
        start_idx = random.randint(0, step-1) # 0-7之间的随机数
        # 选取的帧的索引
        selected_frames_indices = [int(z * step)+start_idx for z in range(clip_length)]
    # if num_frames < selected_frames_indices[-1]:
    #     print(video_path, num_frames, selected_frames_indices)
    # 获取相应的sal坐标
    coordinate_data = np.load(coordinate_file, allow_pickle=True).item()
    video_name = os.path.basename(video_path)
    if len(coordinate_data[video_name]) < selected_frames_indices[-1]:
        print(video_name, len(coordinate_data[video_name]), selected_frames_indices)
    sal_coordinates = [coordinate_data[video_name][i] for i in selected_frames_indices]
    # 获取对应的帧
    frames = vr.get_batch(selected_frames_indices).asnumpy()
    #获取裁剪的普通图像块
    processed_frames = []
    sal_cropped_frames = []

    flip_decision = random.choice([-1, 0, 1, 2])

    for frame in frames:
        resized_size = (480, 840)
        resized_frame = cv2.resize(frame, resized_size)
        # 根据随机结果翻转图像
        if flip_decision == 2:
            flipped_image = resized_frame
        else:
            flipped_image = cv2.flip(resized_frame, flip_decision)
        resized_frame = flipped_image
        # plt.imshow(resized_frame)
        # plt.show()
        # 裁剪帧（例如，裁剪中心区域100x100）,计算开始点
        start_x = random.randrange(resized_size[0] - cropped_size[0])
        start_y = random.randrange(resized_size[1] - cropped_size[1])
        cropped_frame = resized_frame[start_y:start_y + cropped_size[1], start_x:start_x + cropped_size[0]]
        processed_frames.append(cropped_frame)
        # plt.imshow(cropped_frame)
        # plt.show()
    for frame, coord in zip(frames, sal_coordinates):
        y1, x1 = coord
        y2, x2 = y1 + sal_cropped_size[0], x1 + sal_cropped_size[1]
        if y1 > height - sal_cropped_size[0]:
            y2 = height
            y1 = height - sal_cropped_size[0]
        if x1 > width - sal_cropped_size[1]:
            x2 = width
            x1 = width - sal_cropped_size[1]
        sal_cropped_frame = frame[y1:y2, x1:x2, :]

        # 根据随机结果翻转图像
        if flip_decision == 2:
            flipped_image = sal_cropped_frame
        else:
            flipped_image = cv2.flip(sal_cropped_frame, flip_decision)

        flipped_image = cv2.resize(flipped_image, (256, 256))
        sal_cropped_frame = flipped_image
        sal_cropped_frames.append(sal_cropped_frame)

    processed_frames = torch.tensor(np.array(processed_frames)).float()
    sal_cropped_frames = torch.tensor(np.array(sal_cropped_frames)).float()

    return selected_frames_indices, processed_frames, sal_cropped_frames


def get_image_feature_test(video_path, coordinate_file, clip_type, clip_length=8, cropped_size=(448, 768),
                      sal_cropped_size=(224, 224), phase='test', num_clips=5):

    vr = VideoReader(video_path)
    num_frames = len(vr)  # * 0.75
    width, height = vr[0].shape[1], vr[0].shape[0]
    coordinate_data = np.load(coordinate_file, allow_pickle=True).item()
    video_name = os.path.basename(video_path)
    # 获取帧数
    video_clips = []
    sal_clips = []
    for ii in range(num_clips):

        selected_frames_indices = []
        if clip_type == 'uniform':
            step = num_frames // clip_length
            start_idx = random.randint(0, step - 1)  # 0-7之间的随机数
            # 选取的帧的索引
            selected_frames_indices = [int(z * step) + start_idx for z in range(clip_length)]
        # 获取相应的sal坐标
        sal_coordinates = [coordinate_data[video_name][i] for i in selected_frames_indices]
        # 获取对应的帧
        frames = vr.get_batch(selected_frames_indices).asnumpy()
        # 获取裁剪的普通图像块
        processed_frames = []
        sal_cropped_frames = []

        for frame in frames:
            resized_size = (480, 840)
            resized_frame = cv2.resize(frame, resized_size)
            # 裁剪帧（例如，裁剪中心区域100x100）,计算开始点
            start_x = random.randrange(resized_size[0] - cropped_size[0])
            start_y = random.randrange(resized_size[1] - cropped_size[1])
            cropped_frame = resized_frame[start_y:start_y + cropped_size[1], start_x:start_x + cropped_size[0]]
            processed_frames.append(cropped_frame)

        for frame, coord in zip(frames, sal_coordinates):
            y1, x1 = coord
            y2, x2 = y1 + sal_cropped_size[0], x1 + sal_cropped_size[1]
            if y1 > height - sal_cropped_size[0]:
                y2 = height
                y1 = height - sal_cropped_size[0]
            if x1 > width - sal_cropped_size[1]:
                x2 = width
                x1 = width - sal_cropped_size[1]
            sal_cropped_frame = frame[y1:y2, x1:x2, :]

            sal_cropped_frame = cv2.resize(sal_cropped_frame, (256, 256))

            sal_cropped_frames.append(sal_cropped_frame)

        processed_frames = torch.tensor(np.array(processed_frames)).float()
        sal_cropped_frames = torch.tensor(np.array(sal_cropped_frames)).float()
        video_clips.append(processed_frames)
        sal_clips.append(sal_cropped_frames)

    return video_clips, sal_clips


class SUGCsal(torch.utils.data.Dataset):
    def __init__(self, opt):
        super().__init__()

        self.video_infos = {}
        self.ann_file = opt["anno_file"]
        self.data_prefix = opt["data_prefix"]
        self.data_prefix_3D = opt["data_prefix_3D"]
        self.phase = opt["phase"]
        self.mean = torch.FloatTensor([0.485*255, 0.456*255, 0.406*255])
        self.std = torch.FloatTensor([0.229*255, 0.224*255, 0.225*255])

        self.clip_len = opt["clip_len"]
        self.clip_type = 'uniform'

        with open(self.ann_file, newline='') as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)  # 跳过标题行
            for row in csvreader:
                filename = row[0].split('/')[-1]
                label = float(row[1])
                video_name = filename
                filename = os.path.join(self.data_prefix, filename)
                self.video_infos[video_name]=[filename, label]

        if self.phase != 'test':
            del_index = ['0150.mp4', '0182.mp4', '0658.mp4', '0672.mp4', '0700.mp4', '0707.mp4', '0714.mp4', '0735.mp4',
                         '0736.mp4', '0737.mp4', '0738.mp4', '0739.mp4', '0740.mp4', '0741.mp4', '0742.mp4', '0749.mp4',
                         '0791.mp4', '0938.mp4', '1218.mp4', '1239.mp4', '1533.mp4', '1554.mp4', '1589.mp4', '1694.mp4',
                         '2912.mp4']
            copy_dict = self.video_infos.copy()
            for video in list(copy_dict):
                if video.split('/')[-1] in del_index:
                    del self.video_infos[video]

            valid_videos = pd.read_csv('select_valid_videos_v2.csv')
            valid_videos = valid_videos['videos'].tolist()
            valid_videos = [str('%04d.mp4' % int(i)) for i in valid_videos if str('%04d.mp4' % int(i)) not in del_index]
            valid_infos = {key: self.video_infos[key] for key in valid_videos}
            train_infos = {key: value for key, value in self.video_infos.items() if key.split('/')[-1] not in valid_infos}

            if self.phase == 'valid':
                self.video_infos = valid_infos
            elif self.phase == 'train':
                self.video_infos = self.video_infos  # train_infos

    def __len__(self):
        return len(self.video_infos)

    def __getitem__(self, index):
        video_list = list(self.video_infos.keys())
        info_list = list(self.video_infos.values())
        video_name = video_list[index].split('/')[-1]
        info = info_list[index]
        filename = info[0]
        label = info[1]

        data = {}
        video_length_read = self.clip_len  # 8
        feature_folder_name = os.path.join(self.data_prefix_3D, video_name)
        slowfast_feature = torch.zeros([video_length_read, 2048 + 256])
        for i in range(video_length_read):
            i_index = i
            feature_3d_slow = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_slow_feature.npy'))
            feature_3d_slow = torch.from_numpy(feature_3d_slow)
            feature_3d_slow = feature_3d_slow.squeeze()
            feature_3d_fast = np.load(os.path.join(feature_folder_name, 'feature_' + str(i_index) + '_fast_feature.npy'))
            feature_3d_fast = torch.from_numpy(feature_3d_fast)
            feature_3d_fast = feature_3d_fast.squeeze()
            feature_3d = torch.cat([feature_3d_slow, feature_3d_fast])
            slowfast_feature[i] = feature_3d

        if self.phase != 'test':
            coordinate_file = 'data_train_valid.npy'
        else:
            coordinate_file = 'data_test.npy'

        if self.phase == 'train':
            frame_ids, frame_feature, sal_feature = get_image_feature(filename, coordinate_file, self.clip_type, clip_length=self.clip_len, phase=self.phase)
            frame_feature = ((frame_feature - self.mean) / self.std).permute(3, 0, 1, 2)
            sal_feature = ((sal_feature - self.mean) / self.std).permute(3, 0, 1, 2)
        else:
            frame_feature, sal_feature = get_image_feature_test(filename, coordinate_file, self.clip_type, clip_length=self.clip_len, phase=self.phase, num_clips=5)

            for idx, (x, y) in enumerate(zip(frame_feature, sal_feature)):
                x = ((x - self.mean) / self.std).permute(3, 0, 1, 2)
                y = ((y - self.mean) / self.std).permute(3, 0, 1, 2)
                frame_feature[idx] = x
                sal_feature[idx] = y

        data["name"] = video_name
        data["feat"] = slowfast_feature
        data["label"] = label
        data['frame_feature'] = frame_feature
        data['sal_feature'] = sal_feature

        return data


if __name__ == '__main__':
    train_settings = {'phase': 'train', 'anno_file': '/mnt/hdd1/wsj/SUGC_VQA/S-UGC/train_data.csv',
                      'data_prefix': '/mnt/hdd1/wsj/SUGC_VQA/S-UGC/Train/',
                      'data_prefix_3D': '/mnt/hdd1/wsj/code/KVQ-challenge/feature/SUGC/train/',
                      'feature_type': 'SlowFast',
                      'clip_len': 8,
                      'clip_type': 'uniform',
                      }
    train_dataset = SUGCsal(train_settings)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True,
                                               num_workers=16, drop_last=True)

    valid_settings = {'phase': 'valid', 'anno_file': '/mnt/hdd1/wsj/SUGC_VQA/S-UGC/train_data.csv',
                      'data_prefix': '/mnt/hdd1/wsj/SUGC_VQA/S-UGC/Train/',
                      'data_prefix_3D': '/mnt/hdd1/wsj/code/KVQ-challenge/feature/SUGC/train/',
                      'feature_type': 'SlowFast',
                      'clip_len': 8,
                      'clip_type': 'uniform'}
    valid_dataset = SUGCsal(valid_settings)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False,
                                               num_workers=16)

    test_settings = {'phase': 'test', 'anno_file': '/mnt/hdd1/wsj/code/S-UGC-New/test.csv',
                     'data_prefix': '/mnt/hdd1/wsj/SUGC_VQA/S-UGC/Test/',
                     'data_prefix_3D': '/mnt/hdd1/wsj/code/KVQ-challenge/feature/SUGC/test/',
                     'feature_type': 'SlowFast',
                     'clip_len': 8,
                     'clip_type': 'uniform'}
    test_dataset = SUGCsal(test_settings)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=16)
    for z in range(50):
        for i, data in enumerate(tqdm(train_loader, desc="Training")):
            pass

        for i, data in enumerate(tqdm(valid_loader, desc="Validating")):
            pass

        for i, data in enumerate(tqdm(test_loader, desc="Testing")):
            pass


