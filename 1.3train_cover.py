import os
import cv2
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from lib.utils import isfloat, seed_everything, get_lr
from lib.nerf.model import SmallModel, BigModel
from lib.nerf.utils import get_rays, render_rays, pose_spherical, compute_accumulated_transmittance
import pickle

seed_everything(seed=42)

folder = "./data/nerf_example_data/nerf_synthetic/lego"
model_path = "./models/big.pth"

with open(os.path.join(folder, "transforms_train.json"), 'r') as f:
    data_train = json.load(f)
camera_angle_x_train, frames_train = data_train["camera_angle_x"], data_train["frames"]
camera_rotation_train = frames_train[0]["rotation"]
print("camera_angle_x_train", camera_angle_x_train)
print("camera_rotation_train", camera_rotation_train)

H, W = 400, 400
training_sets = []
focal = 0.5 *W / np.tan(0.5 * camera_angle_x_train)
for i_frame in trange(len(frames_train)):
    frame = frames_train[i_frame]
    f_path = os.path.join(folder, frame["file_path"][2:])+".png"
    t_mat = np.array(frame["transform_matrix"])
    img = cv2.imread(f_path)
    pixel_colors = cv2.resize(img, (H, W)) / 255.
    rays_o, rays_d = get_rays(H, W, focal, t_mat[:3, :4])
    rays_o, rays_d, pixel_colors = [x.reshape([-1, 3]) for x in [rays_o, rays_d, pixel_colors]]
    training_set = np.concatenate([rays_o, rays_d, pixel_colors], axis = 1)
    training_sets.append(training_set)
training_sets = np.concatenate(training_sets, axis = 0)
print(training_sets.shape)

lr = 2e-4
hn = 2
hf = 6
nb_bins = 192
batch_size = 256

# 初始化所有参数为0
def init_weights(m):
    if isinstance(m, nn.Linear):
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)

device = torch.device("cuda")
big_model = BigModel(hidden_dim=256).to(device)

my_model = BigModel(hidden_dim=256).to(device)
my_model.load_state_dict(torch.load('./models/secret.pth'))


# 打印 my_model 的所有参数
for name, param in my_model.named_parameters():
    print(f"Parameter Name: {name}, Shape: {param.shape}, Value: {param.data}")

# 读取key
# 加载保存的数组列表
with open('key.pkl', 'rb') as f:
    key = pickle.load(f)
print(key)

optimizer = optim.Adam(big_model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2,4,8], gamma=0.5)
dataloader = DataLoader(training_sets.astype(np.float32), batch_size=batch_size, shuffle=True)

n_epochs = 1
# n_steps = len(dataloader)
n_steps = 30000 # Due to limited computation power

training_loss = []
start_time = time.time()
for i_epoch in range(n_epochs):
    print("Learning rate =", get_lr(optimizer))
    for i_step, xs in enumerate(dataloader):
        if i_step >= n_steps: break  # Due to limited computation power
        ray_origins = xs[:, :3].to(device)
        ray_directions = xs[:, 3:6].to(device)
        ground_truth_pixels = xs[:, 6:].to(device)

        regenerated_pixels = render_rays(
            big_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins
        )
        loss = ((ground_truth_pixels - regenerated_pixels) ** 2).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 传导参数
        with torch.no_grad():
            count_key = 0
            for oral_name, oral_param in my_model.named_parameters():
                for name, param in big_model.named_parameters():
                    if oral_name == name:
                        if 'weight' in name:
                            for i in range(param.shape[0]):
                                if key[count_key][i] == 1:
                                    param[i].data.copy_(oral_param[i])


                        elif 'bias' in name:
                            for i in range(oral_param.shape[0]):
                                if key[count_key][i] == 1:
                                    param[i].data.copy_(oral_param[i])


                            count_key = count_key + 1


        training_loss.append(loss.item())

        if i_step == 0 or (i_step + 1) % 50 == 0 or i_step + 1 == len(dataloader):
            # print logs
            time_cost = (time.time() - start_time) / 3600
            print(
                "[Epoch %d/%d][%d/%d]\tLoss: %.4f\tTime: %.4f hrs" % (
                    i_epoch + 1, n_epochs, i_step + 1, n_steps, loss.item(), time_cost
                )
            )
            torch.save(big_model.state_dict(), model_path)

    scheduler.step()


 # 打印 big_model 的所有参数
for name, param in big_model.named_parameters():
        print(f"Parameter Name: {name}, Shape: {param.shape}, Value: {param.data}")

print("Finished")
