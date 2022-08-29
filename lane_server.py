
import os, glob, logging
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
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
import uvicorn
from rich import print
from rich.traceback import install
from rich.logging import RichHandler
install(show_locals=True, suppress=[torch], width=200)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, markup=True)]
)

os.environ['no_proxy'] = '*'

app = FastAPI()

DEVICE = 0
os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE)
# config = 'configs/clrnet/clr_dla34_culane.py'
# weights = 'weights/culane_dla34.pth'
# config = 'configs/clrnet/clr_resnet18_tusimple.py'
# weights = 'weights/tusimple_r18.pth'
config = 'configs/clrnet/clr_dla34_generic.py'
weights = 'weights/llamas_dla34.pth'
cfg = Config.fromfile(config)
# cfg.gpus = len(gpus)
cfg.load_from = weights
cfg.view = True
cfg.workers = 2
cfg.batch_size = 1

global model, processor

@app.on_event("startup")
def load_model():
    # load model
    global model, processor
    model = build_net(cfg)#.half()
    model = MMDataParallel(model, device_ids=[DEVICE]).to(DEVICE)
    load_network(model, cfg.load_from, finetune_from=None, logger=None)
    # model.to(DEVICE)
    model.eval()
    print(f'Model loaded: {weights}')
    # image processor
    processor = Process(cfg.val_process, cfg)

# data processor
def process(img_path, cut:int=0):
    img = cv2.imread(img_path)
    if cut == 0:
        # calculate cut according to the height of the image
        # assuming the vanishing point is at the center of the image
        # and cut size is at 40% of the height of the image
        cut = int(img.shape[0] * 0.4)
    cfg.cut_height = cut
    cfg.ori_img_h, cfg.ori_img_w = img.shape[:2]
    cfg.sample_y = range(cfg.ori_img_h, cut, -10)
    img = img[cut:, :, :]
    
    sample = {'img': img, 'lanes': []}
    sample = processor(sample)
    sample['img'] = sample['img'].unsqueeze(0).to(DEVICE)
    return sample


def test():
    # load data
    images = glob.glob('data/CULane/**/*.jpg', recursive=True)
    images.sort()
    for img_path in tqdm(images):
        inference(img_path, render=True)


@app.post('/inference')
@torch.no_grad()
@torch.autocast('cuda')
def inference(image=File(default=None), cut:int = 0, render:bool=False, threshold:float = 0.2):
    """
    Inference lane detection on a single image:

    - **image**: image file using **multi-part-form-data**
    - **cut**: cut size of the sky part, default=0 (calculated automatically 40% of the image height)
    - **render**: wether to render the result and returning an image, default=False
    - **threshold**: score threshold to filter out the lanes, default=0.2
    """
    img_path = f'cache/{image.filename}'
    with open(img_path, 'wb') as f:
        f.write(image.file.read())
    # process image
    data = process(img_path, cut=cut)
    logging.info(f'Inference: {image.filename}, cut: {cfg.cut_height}, render: {render}, threshold: {threshold}')
    # inference
    output = model(data)
    results = model.module.heads.get_lanes(output, threshold=threshold)[0]
    lanes = [lane.to_array(cfg) for lane in results]
    scores = [l.metadata['conf'] for l in results]
    logging.info(f"Result: {len(lanes)} lanes with conf: {scores}")
        
    if render:
        # render output
        img = cv2.imread(img_path)
        out_file = f"output/{img_path.split('/')[-1]}"
        imshow_lanes(img, lanes, scores=scores, out_file=out_file)
        return FileResponse(out_file)
    else:
        result = {'lanes': lanes, 'scores': scores}
        return result


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9020)
