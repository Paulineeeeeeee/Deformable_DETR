from PIL import Image
import os

# 图像所在的文件夹路径
folders = ['image/FID','image/FID_image','label/FID_label']

# 遍历文件夹中的所有图像文件
for folder in folders:
    for img_file in os.listdir(folder):
        if img_file.endswith('.jpg') or img_file.endswith('.png'):  # 检查文件扩展名
            # 打开图像
            print(f"{folder}")
            img_path = os.path.join(folder, img_file)
            with Image.open(img_path) as img:
                # 调整图像大小
                img = img.resize((512, 512), Image.ANTIALIAS)
                
                # 保存调整后的图像
                img.save(img_path)
    print(f"{folder}")
print("All images have been resized to 512x512.")
