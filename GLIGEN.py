import json
import torch
from diffusers import StableDiffusionGLIGENPipeline
from PIL import Image

import os
import os

# 加载预训练的 GLIGEN 模型
pipe = StableDiffusionGLIGENPipeline.from_pretrained(
    "masterful/gligen-1-4-generation-text-box", variant="fp16", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

# JSON 文件路径
json_file_paths = ['blip2-opt-2.7b_image_data.json' , 'blip2-flan-t5-xl_image_data.json']
ps = ['prompt_w_label','prompt_w_suffix']

for json_file_path in json_file_paths:
    
    # 获取 JSON 文件所在的文件夹路径
    check_point_name = json_file_path.split("_")[0]  # 这将得到 'blip2-opt-2.7b'

    # 读取 BLIP-2 生成的 JSON 文件
    with open(json_file_path, 'r') as json_file:
        blip2_data = json.load(json_file)

    # 对于 JSON 文件中的每个数据点生成图像
    for data_point in blip2_data:
        url = f'hw1_dataset/train2017/{data_point["image"]}'
        gligen_image = Image.open(url)
        # 过滤条件：确保每张图像只包含一个类别并且边界框数量不超过 6 个
        if len(data_point["labels"]) == 1 and len(data_point["bboxes"]) <= 6:
            labels_str = str(data_point["labels"][0])  # 将标签列表转换为字符串
            for p in ps:
                prompt = f'a photo of {labels_str} {data_point[p]}' 
                # phrases = [data_point["labels"]]
                phrases = [labels_str]  # 如果 data_point["labels"] 只有一个元素

                boxes = data_point["bboxes"]

                # 生成图像
                images = pipe(
                    prompt=prompt,
                    gligen_phrases=phrases,
                    gligen_boxes=boxes,
                    gligen_scheduled_sampling_beta=1,
                    # output_type="pil",
                    num_inference_steps=50,
                ).images

                # 图像文件名
                json_folder_path = check_point_name.split("-")[1] + '_' + p  # 这将得到 'opt'
                if not os.path.exists(json_folder_path):
                    os.makedirs(json_folder_path)

                image_file_name = f'{json_folder_path}/{data_point["image"]}'

                # 保存图像到 JSON 文件所在的文件夹
                images[0].save(image_file_name)
        else:
            for p in ps: 
                # 图像文件名
                json_folder_path = check_point_name.split("-")[1] + '_' + p  # 这将得到 'opt'
                if not os.path.exists(json_folder_path):
                    os.makedirs(json_folder_path)

                image_file_name = f'{json_folder_path}/{data_point["image"]}'

                    # 保存图像到 JSON 文件所在的文件夹
                gligen_image.save(image_file_name)



'''
# text and image
pipe = StableDiffusionGLIGENTextImagePipeline.from_pretrained(
    "anhnct/Gligen_Text_Image", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

prompt = "a flower sitting on the beach"
boxes = [[0.0, 0.09, 0.53, 0.76]]
phrases = ["flower"]
gligen_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gligen/pexels-pixabay-60597.jpg"
)

images = pipe(
    prompt=prompt,
    gligen_phrases=phrases,
    gligen_images=[gligen_image],
    gligen_boxes=boxes,
    gligen_scheduled_sampling_beta=1,
    output_type="pil",
    num_inference_steps=50,
).images

images[0].save("./gligen-generation-text-image-box.jpg")'''