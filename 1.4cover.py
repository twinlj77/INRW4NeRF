import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
from PIL import Image
from lib.nerf.model import SmallModel, BigModel

# 初始化所有参数为0
def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)


# 保存数组列表到文件中
with open('key.pkl', 'rb') as f:
    key = pickle.load(f)

# 创建模型实例
model = BigModel(hidden_dim=256).to('cuda')
model.apply(init_weights)

big_model = BigModel(hidden_dim=256).to('cuda')
big_model.load_state_dict(torch.load('./models/big.pth'))

# 传导参数
count_key = 0
for new_name, new_param in model.named_parameters():
    for name, param in big_model.named_parameters():
        if new_name == name:
            if 'weight' in name:
                for i in range(new_param.shape[0]):
                    if key[count_key][i] == 1:
                            new_param[i].data.copy_(param[i])

            elif 'bias' in name:
                for i in range(new_param.shape[0]):
                    if key[count_key][i] == 1:
                        new_param[i].data.copy_(param[i])

                count_key = count_key + 1

# 打印 big_model 的所有参数
for name, param in model.named_parameters():
  print(f"Parameter Name: {name}, Shape: {param.shape}, Value: {param.data}")
# 保存模型
torch.save(model.state_dict(), './models/recover.pth')
