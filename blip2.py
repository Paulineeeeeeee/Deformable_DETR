import json
import pandas as pd
import os
import numpy as np 

from transformers import (
    Blip2VisionConfig,
    Blip2QFormerConfig,
    OPTConfig,
    Blip2Config,
    Blip2ForConditionalGeneration,
)
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from PIL import Image

with open('hw1_dataset/annotations/instances_train2017.json') as json_data:
    data = json.load(json_data)
    categories = pd.DataFrame(data['categories'])
    images = pd.DataFrame(data['images'])
    annotations = pd.DataFrame(data['annotations'])

train_index = os.listdir('hw1_dataset/train2017')
train_index.remove('.DS_Store')
train_index.sort()

check_points = ["Salesforce/blip2-opt-2.7b", "Salesforce/blip2-flan-t5-xl"]


images_info = []
for check_point in check_points:
    print("start blip2")
    torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(check_point)
    model = Blip2ForConditionalGeneration.from_pretrained(check_point, torch_dtype = torch.float16, device_map="auto")

    for index in train_index:
        
        url = f"hw1_dataset/train2017/{index}"
        picture = Image.open(url)
        model.to(device).to(torch.float16)
        
        image = images[images['file_name'] == index]
        id = image['id'].item()
        height = image['height'].item()
        width = image['width'].item()
        annotation = annotations[annotations['image_id'] == id]
        creatures = annotation['category_id'].drop_duplicates()

        creature_names = categories[categories['id'].isin(creatures)]['name'].tolist()
        bboxes = annotation['bbox'].apply(lambda x: [x[0] / width, x[1] / height, (x[0] + x[2]) / width, (x[1] + x[3]) / height]).tolist()

        prompt = " and ".join([f"a photo of {name}" for name in creature_names])
        inputs = processor(picture, text=prompt, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        image_info = {
            "image": index,
            "labels": creature_names,
            "height": height,
            "width": width,
            "bboxes": bboxes,
            "generated_text": generated_text,
            "prompt_w_label": f'{generated_text}, {" and ".join(creature_names)}, height: {height}, width: {width}',
            "prompt_w_suffix": f'{generated_text}, {" and ".join(creature_names)}, height: {height}, width: {width}, HQ quality, highly detailed'
        }
        images_info.append(image_info)
    check_point = check_point.split('/')[-1]
    with open(f'{check_point}_image_data.json', 'w') as json_file:
        json.dump(images_info, json_file, indent=4)

        '''for creature in creatures:    
            creature_name = categories[categories['id'] == creature]['name'].item()
            # prompt = f"Question:Provide a detailed description of the {creature_name} shown. Answer:"
            prompt = f"The {creature_name}"
            inputs = processor(picture, text=prompt, return_tensors="pt").to(device, torch.float16)
            generated_ids = model.generate(**inputs, max_new_tokens=20)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            
            bbox = annotation[annotation['category_id'] == creature]['bbox']
            normalized_bboxes = bbox.apply(lambda x: [x[0] / width, x[1] / height, (x[0] + x[2]) / width, (x[1] + x[3]) / height])
            
            image_info = {
                "image": index,
                "label": creature_name,
                "height": height,
                "width": width,
                "bboxes":normalized_bboxes.tolist(),
                "generated_text": generated_text,
                "prompt_w_label": f'{generated_text},{creature_name} height: {height}, width: {width}',
                "prompt_w_suffix": f'{generated_text},{creature_name} height: {height}, width: {width}, HQ quality, highly detailed'
            }
            images_info.append(image_info)
    check_point = check_point.split('/')[-1]
    with open(f'{check_point}_image_data.json', 'w') as json_file:
        json.dump(images_info, json_file, indent=4) '''
