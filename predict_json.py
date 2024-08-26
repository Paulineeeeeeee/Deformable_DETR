import cv2
from PIL import Image
import numpy as np
import os
import time
import json

import torch
from torch import nn
import torchvision.transforms as T
from main import get_args_parser as get_main_args_parser
from models import build_model

torch.set_grad_enabled(False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] 当前使用{}做推断".format(device))

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu().numpy()
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b

def load_model(model_path, args):
    model, _, _ = build_model(args)
    model.cuda()
    model.eval()
    state_dict = torch.load(model_path)
    model.load_state_dict(state_dict["model"])
    model.to(device)
    print("load model sucess")
    return model

def detect(im, model, transform, prob_threshold=0.7):
    img = transform(im).unsqueeze(0)
    img = img.to(device)
    start = time.time()
    outputs = model(img)
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > prob_threshold
    probas = probas.cpu().detach().numpy()
    keep = keep.cpu().detach().numpy()
    ids = probas[keep].argmax(-1).tolist()
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    end = time.time()
    return probas[keep], bboxes_scaled, ids, end - start

if __name__ == "__main__":
    main_args = get_main_args_parser().parse_args()
    dfdetr = load_model('checkpoint0049.pth', main_args)
    list_path = "hw1_dataset/test2017/"
    files = os.listdir(list_path)

    results = {}
    cn = 0
    waste = 0

    for file in files:
        img_path = os.path.join(list_path, file)
        im = Image.open(img_path)
        scores, boxes, ids, waste_time = detect(im, dfdetr, transform)
        
        results[file] = {
                "boxes": [],
                "labels": [],
                "scores": []
        }

        for score, box, idx in zip(scores, boxes, ids):
            results[file]["boxes"].append(box.tolist())
            results[file]["labels"].append(idx)  # 你在圖片中稱它為"label"，在你的代碼中是"id"
            results[file]["scores"].append(score.max().item())
        
        print("{} [INFO] {} time: {} done!!!".format(cn, file, waste_time))
        cn += 1
        waste += waste_time
        waste_avg = waste / cn
        print(waste_avg)

    with open("predict_test.json", "w") as f:
        json.dump(results, f)
