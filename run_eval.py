import argparse
import torch
import torch.nn as nn
from models.model_unet import HackathonModel
from dataset import HackathonDataset

import pandas as pd

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib import image

from collections import defaultdict

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', '-p')

    args = parser.parse_args()

    data = pd.read_csv('test_to_send/x-ai_test_data_without_labels.csv')

    l = []

    df = data['filename']
    for filename in df:
        img = image.imread(f'test_to_send/images/{filename}')

        img = torch.tensor(img)
        print(len(l))

        input = defaultdict()
        input['img'] = img[None, :]

        from models.model_unet import HackathonModel
        model = HackathonModel.load_from_checkpoint('model_weights/unet.ckpt')
        model.eval()

        segmented = model(input)
        plt.imsave(f'test_to_send/segmented/{filename}', segmented.cpu().detach().numpy(), cmap='Greys')

        from models.model_efficient import HackathonModel
        model = HackathonModel.load_from_checkpoint('model_weights/efficientnet.ckpt')
        model.eval()

        has_silo = model(input)
        has_silo = nn.Sigmoid()(has_silo)
        has_silo = (has_silo > 0.5).int()
        l.append(has_silo.cpu().detach().numpy())

    data.insert(2, 'class', l, allow_duplicates=True)
    data.to_csv('./test_to_send/answer.csv')