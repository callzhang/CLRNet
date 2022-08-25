
import os, glob, random
import torch, cv2
from clrnet.models.registry import build_net
from clrnet.utils.net_utils import save_model, load_network, resume_network
from clrnet.utils.config import Config
from clrnet.utils.visualization import imshow_lanes
from PIL import Image
from torchvision import transforms
from mmcv.parallel import MMDataParallel
from tqdm import tqdm
from clrnet.datasets.process import Process

DEVICE = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE)
config = 'configs/clrnet/clr_dla34_culane.py'
weights = 'weights/culane_dla34.pth'
cfg = Config.fromfile(config)
# cfg.gpus = len(gpus)
cfg.load_from = weights
cfg.view = True
cfg.workers = 2
cfg.batch_size = 1

# load model
model = build_net(cfg)
model = MMDataParallel(model, device_ids=[DEVICE]).to(DEVICE)
load_network(model, cfg.load_from, finetune_from=None, logger=None)
# model.to(DEVICE)
model.eval()
print(f'Model loaded: {weights}')

# load data
images = glob.glob('data/CULane/**/*.jpg', recursive=True)
images.sort()
processor = Process(cfg.val_process, cfg)

# data processor
def process(img_path):
    img = cv2.imread(img_path)
    img = img[cfg.cut_height:, :, :]
    sample = {'img': img, 'lanes': []}
    sample = processor(sample)
    sample['img'] = sample['img'].unsqueeze(0).to(DEVICE)
    return sample


for img_path in tqdm(images):
    data = process(img_path)

    # inference
    with torch.no_grad():
        output = model(data)
        results = model.module.heads.get_lanes(output)[0]
    if results:
        # render output
        lanes = [lane.to_array(cfg) for lane in results]
        img = cv2.imread(img_path)
        out_file = f"output/{img_path.split('/')[-1]}"
        imshow_lanes(img, lanes, out_file=out_file)
