from tqdm import tqdm
import pandas as pd
from dataset import SUGCsal
from model import VQA_Network
import argparse
import torch
import numpy as np


def test(config):
    device = torch.device("cuda:" + config.gpu_id.split(',')[0])
    model = VQA_Network().to(device)
    pretrained_dict = torch.load(config.pretrained_pth, map_location='cpu')['state_dict']
    new_dict = {}
    for k, v in pretrained_dict.items():
        k = k[7:]
        new_dict[k] = v
    model.load_state_dict(new_dict)

    test_settings = {'phase': 'test', 'anno_file': config.anno_file,
                     'data_prefix': config.video_dir,
                     'data_prefix_3D': config.feature_dir,
                     'feature_type': 'SlowFast',
                     'clip_len': 5,
                     'clip_type': 'uniform'}
    test_dataset = SUGCsal(test_settings)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=config.num_workers*2)

    model.eval()
    predictions = []
    videos = []

    for i, data in enumerate(tqdm(test_loader, desc="Testing")):
        avg_result = []
        with torch.no_grad():

            for zz in range(len(data['frame_feature'])):
                new_data = {}
                new_data['feat'] = data['feat'].to(device)
                new_data['frame_feature'] = data['frame_feature'][zz].to(device)
                new_data['sal_feature'] = data['sal_feature'][zz].to(device)
                result = model(new_data).cpu().numpy()
                avg_result.append(result[0][0])

        videos.append(data["name"][0])
        predictions.append(np.mean(avg_result).item())

    videos = ['test/'+ i for i in videos]
    data = {'filename': videos, 'score': predictions}  #
    df = pd.DataFrame(data)

    df.to_csv('prediction.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parameters
    parser.add_argument('--pretrained_pth', type=str, default='SUGC_6.pth')
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--anno_file', type=str, default='test.csv')
    parser.add_argument('--video_dir', type=str, default='./test_videos/')
    parser.add_argument('--feature_dir', type=str, default='./feature/test/')

    args = parser.parse_args()
    test(args)

