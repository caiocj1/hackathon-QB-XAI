import argparse
import torch
from models.model_unet import HackathonModel
from dataset import HackathonDataset

from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
from matplotlib import image

from collections import defaultdict

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', '-p')

    args = parser.parse_args()

    img = image.imread(args.image_path)
    img = torch.tensor(img)

    input = defaultdict()
    input['img'] = img[None, :]

    from models.model_unet import HackathonModel
    model = HackathonModel.load_from_checkpoint('model_weights/unet.ckpt')
    model.eval()

    segmented = model(input)

    plt.imshow(segmented.cpu().detach().numpy(), cmap='Greys')
    plt.savefig('eval_outs/seg')

    from models.model_efficient import HackathonModel
    model = HackathonModel.load_from_checkpoint('model_weights/efficientnet.ckpt')
    model.eval()

    has_silo = model(input)
    f = open("eval_outs/out.txt", "w")
    f.write(f'{has_silo}')
    f.close()
