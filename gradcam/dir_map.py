
import os
import os.path as osp
import sys

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
import torch, yaml, cv2, os, shutil
import numpy as np
np.random.seed(0)
import matplotlib.pyplot as plt
from tqdm import trange
from PIL import Image

# from models.yolo import Model

# from utils.augmentations import letterbox
# from utils.general import xywh2xyxy
import torch.nn as nn
import logging

from pytorch_grad_cam import GradCAMPlusPlus, GradCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients

def set_logging(name=None):
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)

LOGGER = set_logging(__name__)


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    '''Resize and pad image while meeting stride-multiple constraints.'''
    shape = im.shape[:2]  # 当前图片的形状

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    elif isinstance(new_shape, list) and len(new_shape) == 1:
       new_shape = (new_shape[0], new_shape[0])

    # 计算比例
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # 计算填充
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return im, r, (left, top)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results.
    This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
    Args:
        prediction: (tensor), with shape [N, 5 + num_classes], N is the number of bboxes.
        conf_thres: (float) confidence threshold.
        iou_thres: (float) iou threshold.
        classes: (None or list[int]), if a list is provided, nms only keep the classes you provide.
        agnostic: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
        multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.
        max_det:(int), max number of output bboxes.

    Returns:
         list of detections, echo item is one tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
    """

    num_classes = prediction.shape[2] - 5  # number of classes
    pred_candidates = torch.logical_and(prediction[..., 4] > conf_thres, torch.max(prediction[..., 5:], axis=-1)[0] > conf_thres)  # candidates
    # Check the parameters.
    assert 0 <= conf_thres <= 1, f'conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided.'
    assert 0 <= iou_thres <= 1, f'iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided.'

    # Function settings.
    max_wh = 4096  # maximum box width and height
    max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
    time_limit = 10.0  # quit the function when nms cost time exceed the limit time.
    multi_label &= num_classes > 1  # multiple labels per box

    tik = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for img_idx, x in enumerate(prediction):  # image index, image inference
        x = x[pred_candidates[img_idx]]  # confidence

        # If no box remains, skip the next process.
        if not x.shape[0]:
            continue

        # confidence multiply the objectness
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
        if multi_label:
            box_idx, class_idx = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[box_idx], x[box_idx, class_idx + 5, None], class_idx[:, None].float()), 1)
        else:  # Only keep the class with highest scores.
            conf, class_idx = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, class_idx.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class, only keep boxes whose category is in classes.
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        num_box = x.shape[0]  # number of boxes
        if not num_box:  # no boxes kept.
            continue
        elif num_box > max_nms:  # excess max boxes' number.
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        class_offset = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + class_offset, x[:, 4]  # boxes (offset by class), scores
        keep_box_idx = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if keep_box_idx.shape[0] > max_det:  # limit detections
            keep_box_idx = keep_box_idx[:max_det]

        output[img_idx] = x[keep_box_idx]
        if (time.time() - tik) > time_limit:
            print(f'WARNING: NMS cost time exceed the limited {time_limit}s.')
            break  # time limit exceeded

    return output

def xywh2xyxy(bboxes):
    '''Transform bbox(xywh) to box(xyxy).'''
    bboxes[..., 0] = bboxes[..., 0] - bboxes[..., 2] * 0.5
    bboxes[..., 1] = bboxes[..., 1] - bboxes[..., 3] * 0.5
    bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2]
    bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3]
    return bboxes

def load_checkpoint(weights, map_location=None, inplace=True):
    """Load model from checkpoint file."""
    LOGGER.info("Loading checkpoint from {}".format(weights))
    ckpt = torch.load(weights, map_location=map_location)  # load
    model = ckpt['ema' if ckpt.get('ema') else 'model'].float()
    
    model = model.eval()
    return model


def intersect_dicts(da, db, exclude=()):
    # Dictionary intersection of matching keys and shapes, omitting 'exclude' keys, using da values
    return {k: v for k, v in da.items() if k in db and all(x not in k for x in exclude) and v.shape == db[k].shape}


class yolov5_heatmap:
    def __init__(self, weight, device, method, layer, backward_type, conf_threshold, ratio):
        device = torch.device(device)
        ckpt = torch.load(weight)
        model_names = ckpt['model'].names
        csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32

        # model = Model(cfg, ch=3, nc=len(model_names)).to(device)
        model =load_checkpoint(weight, device)

        csd = intersect_dicts(csd, model.state_dict(), exclude=['anchor'])  # intersect
        #model.load_state_dict(csd, strict=False)  # load
        
        print(f'Transferred {len(csd)}/{len(model.state_dict())} items')
        print(model.detect.cls_convs[0].block.conv)
        target_layers = [eval(layer)]
        method = eval(method)

        colors = np.random.uniform(0, 255, size=(len(model_names), 3)).astype(np.int32)
        self.__dict__.update(locals())
    
    def post_process(self, result):
        logits_ = result[..., 4:]
        boxes_ = result[..., :4]
        sorted, indices = torch.sort(logits_[..., 0], descending=True)
        return logits_[0][indices[0]], xywh2xyxy(boxes_[0][indices[0]]).cpu().detach().numpy()

    def draw_detections(self, box, color, name, img):
        xmin, ymin, xmax, ymax = list(map(int, list(box)))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), tuple(int(x) for x in color), 2)
        cv2.putText(img, str(name), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, tuple(int(x) for x in color), 2, lineType=cv2.LINE_AA)
        return img

    def __call__(self, img_path, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        # img process
        img = cv2.imread(img_path)
        img = letterbox(img)[0]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.float32(img) / 255.0
        tensor = torch.from_numpy(np.transpose(img, axes=[2, 0, 1])).unsqueeze(0).to(self.device)
        tensor.requires_grad = True  # 确保输入张量需要梯度
        
        # init ActivationsAndGradients
        grads = ActivationsAndGradients(self.model, self.target_layers, reshape_transform=None)

        # get ActivationsAndResult
        result = grads(tensor)
        activations = grads.activations[0].cpu().detach().numpy()

        # postprocess to yolo output
        post_result, post_boxes = self.post_process(result[0])
        for i in trange(int(post_result.size(0) * self.ratio)):
            if post_result[i][0] < self.conf_threshold:
                break

            self.model.zero_grad()
            if self.backward_type == 'conf':
                post_result[i, 0].backward(retain_graph=True)
            else:
                # get max probability for this prediction
                score = post_result[i, 1:].max()
                score.backward(retain_graph=True)
                

            # process heatmap
            gradients = grads.gradients[0]
            b, k, u, v = gradients.size()
            weights = self.method.get_cam_weights(self.method, None, None, None, activations, gradients.detach().numpy())
            weights = weights.reshape((b, k, 1, 1))
            saliency_map = np.sum(weights * activations, axis=1)
            saliency_map = np.squeeze(np.maximum(saliency_map, 0))
            saliency_map = cv2.resize(saliency_map, (tensor.size(3), tensor.size(2)))
            saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
            if (saliency_map_max - saliency_map_min) == 0:
                continue
            saliency_map = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)

            # add heatmap and box to image
            cam_image = show_cam_on_image(img.copy(), saliency_map, use_rgb=True)
            #cam_image = self.draw_detections(post_boxes[i], self.colors[int(post_result[i, 1:].argmax())], f'{self.model_names[int(post_result[i, 1:].argmax())]} {post_result[i][0]:.2f}', cam_image)
            cam_image = Image.fromarray(cam_image)
            #cam_image.save(f'{save_path}/{i}.png')
            cam_image.save(f'{save_path}/{base_name}_{i}.png')
        del tensor, result, activations, post_result, post_boxes, img, grads, gradients, weights, saliency_map, cam_image
        torch.cuda.empty_cache()


def process_folder(input_folder, output_folder):
    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # 创建模型实例
    model = yolov5_heatmap(**get_params())

    # 处理每个图像文件
    for image_file in image_files:

        # 判断文件名是否以 "inclusion" 开头
        if not image_file.startswith("inclusion"):
            continue

        input_path = os.path.join(input_folder, image_file)
        model(input_path, output_folder)

def get_params():
    params = {
        #'weight': r'D:\experiment\result\analysis_neck_compare\cm_fpn\base\cm_fpn\weights\best_ckpt.pt',
        'weight': r'D:\experiment\result\neu\mywork\best_distill\weights\best_ckpt.pt',
        
        'device': 'cuda:0',
        'method': 'GradCAM', # GradCAMPlusPlus, GradCAM, XGradCAM
        'layer': "model.detect.cls_convs[0]",
        'backward_type': 'class', # class or conf
        'conf_threshold': 0.0001, # 0.6
        'ratio': 0.00015 # 0.02-0.1
    }
    return params


if __name__ == '__main__':
    input_folder = r'C:\Users\jiahao\Desktop\paper\vis\NEU\ori'  # 输入文件夹路径
    output_folder = r'C:\Users\jiahao\Desktop\paper\vis\NEU\gas_cam'  # 输出文件夹路径

    process_folder(input_folder, output_folder)