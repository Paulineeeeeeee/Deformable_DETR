import torch
pretrained_weights  = torch.load('r50_deformable_detr-checkpoint.pth')

num_class = 8 

for i in range(6):
    pretrained_weights["model"][f"class_embed.{i}.weight"].resize_(num_class+1, 256)
    pretrained_weights["model"][f"class_embed.{i}.bias"].resize_(num_class+1)

pretrained_weights["model"]["query_embed.weight"].resize_(100,512)

'''pretrained_weights["model"]["class_embed.1.weight"].resize_(num_class+1, 256)
pretrained_weights["model"]["class_embed.1.bias"].resize_(num_class+1)'''
torch.save(pretrained_weights, "detr-r50_%d.pth"%num_class)
