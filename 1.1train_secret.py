import os
import cv2
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
from lib.utils import isfloat, seed_everything, get_lr
from lib.nerf.model import SmallModel,BigModel
from lib.nerf.utils import get_rays, render_rays, pose_spherical, compute_accumulated_transmittance

seed_everything(seed=42)
#加载数据集
folder = "./data/nerf_example_data/nerf_synthetic/hotdog"
model_path = "./models/secret.pth"

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
#设置参数
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
my_model = BigModel(hidden_dim=256).to(device)

new_model = BigModel(hidden_dim=256).to(device)
new_model.apply(init_weights)


count = 256

random_1 = np.concatenate((np.zeros(count//2), np.ones(count//2)))
random_2 = np.concatenate((np.zeros(count//2), np.ones(count//2)))
random_3 = np.concatenate((np.zeros(count//2), np.ones(count//2)))
random_4 = np.concatenate((np.zeros(count//2), np.ones(count//2)))
random_5 = np.concatenate((np.zeros(count//2), np.ones(count//2)))
random_6 = np.concatenate((np.zeros(count//2), np.ones(count//2)))
random_7 = np.concatenate((np.zeros(count//2), np.ones(count//2)))
random_8 = np.concatenate((np.zeros(count//2), np.ones(count//2)))
random_9 = np.concatenate((np.zeros(count//2), np.ones(count//2)))
random_10 = np.concatenate((np.zeros(count//2), np.ones(count//2)))
random_11 = np.concatenate((np.zeros(count//2), np.ones(count//2)))
random_12 = np.concatenate((np.zeros(count//2), np.ones(count//2)))
random_13 = np.concatenate((np.zeros(count//2), np.ones(count//2)))
random_14 = np.concatenate((np.zeros(count//2), np.ones(count//2)))
random_15 = np.concatenate((np.zeros(count//2), np.ones(count//2)))
random_16 = np.concatenate((np.zeros(count//2), np.ones(count//2+1)))
random_17 = np.concatenate((np.zeros(141), np.ones(142)))
random_18 = np.concatenate((np.zeros(141), np.ones(142)))
random_19 = np.concatenate((np.zeros(count//4), np.ones(count//4)))
random_20 = np.concatenate((np.zeros(count//4), np.ones(count//4)))
random_21 = np.concatenate((np.zeros(count//4), np.ones(count//4)))
random_22 = np.ones(3)


np.random.shuffle(random_1)
np.random.shuffle(random_2)
np.random.shuffle(random_3)
np.random.shuffle(random_4)
np.random.shuffle(random_5)
np.random.shuffle(random_6)
np.random.shuffle(random_7)
np.random.shuffle(random_8)
np.random.shuffle(random_9)
np.random.shuffle(random_10)
np.random.shuffle(random_11)
np.random.shuffle(random_12)
np.random.shuffle(random_13)
np.random.shuffle(random_14)
np.random.shuffle(random_15)
np.random.shuffle(random_16)
np.random.shuffle(random_17)
np.random.shuffle(random_18)
np.random.shuffle(random_19)
np.random.shuffle(random_20)
np.random.shuffle(random_21)
np.random.shuffle(random_22)

random = [random_1, random_2, random_3, random_4, random_5, random_6, random_7, random_8, random_9,random_10,random_11, random_12, random_13, random_14, random_15, random_16, random_17, random_18, random_19,random_20, random_21,random_22]
print(random)


optimizer = optim.Adam(my_model.parameters(), lr=lr)
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
            my_model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins
        )
        loss = ((ground_truth_pixels - regenerated_pixels) ** 2).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 传导参数
        count_key = 0
        for zero_name, zero_param in new_model.named_parameters():
            for name, param in my_model.named_parameters():
                if zero_name == name:
                    if 'weight' in name:
                        for i in range(param.shape[0]):
                            if random[count_key][i] == 0:
                                param[i].data.copy_(zero_param[i])

                    elif 'bias' in name:
                        for i in range(zero_param.shape[0]):
                            if random[count_key][i] == 0:
                                param[i].data.copy_(zero_param[i])

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
            torch.save(my_model.state_dict(), model_path)

    scheduler.step()

    # 将不重要的参数和重要的参数编码成key
    key_1 = np.zeros(256)
    key_2 = np.zeros(256)
    key_3 = np.zeros(256)
    key_4 = np.zeros(256)
    key_5 = np.zeros(256)
    key_6 = np.zeros(256)
    key_7 = np.zeros(256)
    key_8 = np.zeros(256)
    key_9 = np.zeros(256)
    key_10 = np.zeros(256)
    key_11 = np.zeros(256)
    key_12 = np.zeros(256)
    key_13 = np.zeros(256)
    key_14 = np.zeros(256)
    key_15 = np.zeros(256)
    key_16 = np.zeros(257)
    key_17 = np.zeros(283)
    key_18 = np.zeros(283)
    key_19 = np.zeros(128)
    key_20 = np.zeros(128)
    key_21 = np.zeros(128)
    key_22 = np.ones(3)

    key = [key_1, key_2, key_3, key_4, key_5, key_6, key_7, key_8, key_9, key_10, key_11, key_12, key_13, key_14, key_15, key_16, key_17, key_18, key_19, key_20, key_21, key_22]

    # 进行模型剪枝，删掉不重要的参数，保留重要的参数
    count_key = 0
    for name, param in my_model.named_parameters():

        if 'weight' in name:
            for i in range(param.shape[0]):
                if torch.sum(torch.abs(param[i])) >= 0.001:
                    key[count_key][i] = 1

        elif 'bias' in name:
            for i in range(param.shape[0]):
                if torch.sum(torch.abs(param[i])) >= 0.001:
                    key[count_key][i] = 1

            count_key = count_key + 1

    # 保存数组列表到文件中
    with open('key.pkl', 'wb') as f:
        pickle.dump(key, f)
        print(key)
print("Finished")

