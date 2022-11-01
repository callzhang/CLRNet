
import json,requests,uuid
from fastapi.middleware.cors import CORSMiddleware
import os, glob, logging, time, random
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
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, Request
from fastapi.responses import FileResponse, JSONResponse
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

#跨域设置
origins = [
    "https://torbjorn.startask.net"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

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


@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    '''return error code for debugging'''
    import traceback
    content="".join(
        traceback.format_exception(etype=type(exc), value=exc, tb=exc.__traceback__)
    )
    print(content)
    return JSONResponse(status_code=500, content=content, media_type='text/plain')


# data processor
def process(img_path, cut_ratio:float=0.0):
    img = cv2.imread(img_path)
    if cut_ratio == 0:
        # calculate cut according to the height of the image
        # assuming the vanishing point is at the center of the image
        # and cut size is at 40% of the height of the image
        cut_height = int(img.shape[0] * 0.4)
    else:
        cut_height = int(img.shape[0] * cut_ratio)
    cfg.cut_height = cut_height
    cfg.ori_img_h, cfg.ori_img_w = img.shape[:2]
    cfg.sample_y = range(cfg.ori_img_h, cut_height, -10)
    img = img[cut_height:, :, :]
    
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
def inference(image=File(default=None), cut_ratio:float = 0.0, render:bool=False, threshold:float = 0.2, background_tasks: BackgroundTasks = BackgroundTasks()):
    """
    Inference lane detection on a single image:

    - **image**: image file using **multi-part-form-data**
    - **cut**: cut size of the sky part, default=0 (calculated automatically 40% of the image height)
    - **render**: wether to render the result and returning an image, default=False
    - **threshold**: score threshold to filter out the lanes, default=0.2
    """
    fname = image.filename
    fname = 'image.jpg' if fname == 'image' else fname
    img_path = f'cache/{fname}'
    with open(img_path, 'wb') as f:
        f.write(image.file.read())
    # process image
    data = process(img_path, cut_ratio)
    logging.info(f'Inference: {fname}, cut_ratio: {cfg.cut_height}, render: {render}, threshold: {threshold}')
    # inference
    output = model(data)
    results = model.module.heads.get_lanes(output, threshold=threshold)[0]
    lanes = [lane.to_array(cfg) for lane in results]
    scores = [l.metadata['conf'] for l in results]
    logging.info(f"Result: {len(lanes)} lanes with conf: {scores}")

    background_tasks.add_task(prune_cache, 100)
        
    if render:
        # render output
        img = cv2.imread(img_path)
        out_file = f"output/{img_path.split('/')[-1]}"
        imshow_lanes(img, lanes, scores=scores,
                     out_file=out_file, cut=cut_ratio)
        return FileResponse(out_file)
    else:
        result = {'lanes': lanes, 'scores': scores}
        return result

def download(url):
    # # print(url)
    r = requests.get(
        url,
        headers={
            "sec-ch-ua":
            'Not A;Brand";v="99", "Chromium";v="96", "Google Chrome";v="96"',
            "Referer": "https://work.startask.net/",
            "sec-ch-ua-mobile": "?0",
            "User-Agent":
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.93 Safari/537.36",
            "sec-ch-ua-platform": "macOS"
        })
    if r.status_code != 200:
        # print('==========\n')
        raise Exception(f"download failed {url}")
    # # print(f"_download {os.path.basename(url)} ok")
    return r.content


@app.post('/stardust_inference')
@torch.no_grad()
@torch.autocast('cuda')
def stardust_inference(sourceRecord:dict = {},type:str = None, cut_ratio:float = 0.42, render:bool=False, threshold:float = 0.22):

    """
    Inference lane detection on a single image:

    - **image**: image file using **multi-part-form-data**
    - **cut**: cut size of the sky part, default=0 (calculated automatically 40% of the image height)
    - **render**: wether to render the result and returning an image, default=False
    - **threshold**: score threshold to filter out the lanes, default=0.2
    """
    image_data = download(sourceRecord['sourceRecord']['attachment'])
    path = sourceRecord['sourceRecord']['attachment'].split('/')[-1]

    with open(f'cache/{path}', 'wb') as fb:
        fb.write(image_data)
    img_path = f'cache/{path}'

    # img_path = f'cache/{image.filename}'
    # with open(img_path, 'wb') as f:
    #     f.write(image.file.read())

    # process image
    data = process(img_path, cut_ratio)
    logging.info(f'Inference: {path}, cut_ratio: {cfg.cut_height}, render: {render}, threshold: {threshold}')
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
        slots = {"result":{ "type": "slotChildren","slotChildren": []}}
        for i in lanes:
            li = []
            for k in i:
                li.append({'x':float(k[0]),'y':k[1]})
            slots["result"]['slotChildren'].append({
               "slot":{ "id": str(uuid.uuid4()),
                "vertices":li,
                "label": "",
                "type": "line"},
                "children":{}
                })
        # result = {'lanes': lanes, 'scores': scores}
       # slots = {"result":{ "type": "slot","slot": []}}
       # for i in lanes:
        #    li = []
         #   for k in i:
          #      li.append({'x':float(k[0]),'y':k[1]})
           # slots["result"]['slot'].append({
            #    "id": str(uuid.uuid4()),
             #   "vertices":li,
              #  "label": "",
               # "type": "line"
               # })

        json.dumps(slots)
        # response.headers.add("Access-Control-Allow-Origin", "*")
        return slots

def prune_cache(n=100):
    if random.random() < 0.99:
        return
    time.sleep(1)
    images = glob.glob('cache/*', recursive=True)
    if len(images) > n:
        print(f'Pruning cache: {len(images)-n}')
        images.sort(key=os.path.getmtime)
        for img_path in images[:len(images)-n]:
            os.remove(img_path)
            print(f'Pruned: {img_path}')

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9020)
    # prune_cache(n=100)
