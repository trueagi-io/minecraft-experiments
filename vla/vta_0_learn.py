import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from aeblock import StackedAE
import random
from utils import *
from vis_transf import *
import os

def load_data(folder, area=320*240, max_frames=10000):
    import cv2
    import json
    count = 0
    frames = []
    acts = []
    for i in range(1, max_frames+1):
        fname = folder + "/" + str(i) + ".jpg"
        frame = cv2.imread(fname)
        if frame is None:
            break
        coef = math.sqrt(frame.shape[0]*frame.shape[1]/area)
        resize = (int(frame.shape[1]/coef), int(frame.shape[0]/coef))
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fname = folder + "/" + str(i) + ".json"
        with open(fname) as f:
            d = json.load(f)
            acts.append(d)
        frames.append(frame)
        count += 1
    frames = np.array(frames).astype(np.float32) / 255.0
    print("Loaded: ", frames.shape)
    frames = torch.tensor(frames).permute(0, 3, 1, 2)  # (N, C, H, W)
    return frames, acts

@torch.no_grad
def process_frames(ae_model, frames):
    dataset = VideoFramesDataset(frames)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    frames_tensor_list = []
    with torch.no_grad():
        for frame in dataloader:
            frame = frame.to('cuda')
            z = ae_model.encode_straight(frame)[-1][0]
            frames_tensor_list.append(z.to('cpu'))
    return frames_tensor_list

N_ADD_FEAT = 8
def process_acts(acts):
    acts_tensor_list = []
    act_str = ['pitch', 'turn', 'strafe', 'move', 'attack', 'jump']
    atn = torch.zeros(N_ADD_FEAT)
    for aa in acts:
        # NOTE: If some actions are not presented for current frame,
        # we don't set them to zero, but keep their previous values
        atn = atn.clone()
        for a in aa['actions']:
            if len(a) == 0:
                continue
            if a[0] in act_str:
                atn[act_str.index(a[0])] = float(a[1])
        atn[6] = 1 if "inRange" in aa["lineOfSight"] and aa["lineOfSight"]["inRange"] else 0
        atn[7] = aa["agentPos"][3] / 180.
        acts_tensor_list.append(atn)
    return acts_tensor_list

def act_loss_1pt(pred, target):
    H = pred.shape[1]
    W = pred.shape[2]
    return nn.MSELoss()(pred[:,H//2,W//2,-N_ADD_FEAT:],
                        target[:,H//2,W//2,-N_ADD_FEAT:])

def act_loss_mean(pred, target):
    return nn.MSELoss()(pred[:,:,:,-N_ADD_FEAT:],
                        target[:,:,:,-N_ADD_FEAT:])

if __name__ == "__main__":
    # In this experiment, we train the same FactorizedVideoTransformer on top
    # of the same autoencoder.
    params = {
        'dense' : {'ae' : 'result/ae_0_stack.pth',
                   'vta': 'result/vta_0_ae0dense.pth'},
        'sparse': {'ae' : 'result/ae_1_sparse.pth',
                   'vta': 'result/vta_0_ae1sparse.pth'},
    }
    ps = params['sparse']
    T = 8
    video_tensor_list = []
    ae_model = StackedAE.from_dict(torch.load(ps['ae'], weights_only=True)).to('cuda')
    for i in range(29):
        frames, acts = load_data(f"data/dataset_wood/dataset_log{i}")#, max_frames = 100)
        act_tensor_list=process_acts(acts)
        frames_tensor_list=process_frames(ae_model, frames)
        fract_tensor_list = []
        for a, f in zip(act_tensor_list, frames_tensor_list):
            # Here, we just append addtional action/observation features to image features
            a = a.view(N_ADD_FEAT, 1, 1).expand(N_ADD_FEAT, f.size(1), f.size(2))
            fract_tensor_list.append(torch.cat([f, a], dim=0).permute(1, 2, 0))
        for i in range(len(fract_tensor_list)-T-1):
            video_tensor_list.append(torch.stack(fract_tensor_list[i:i+T+1]))
    print("Samples:", len(video_tensor_list))
    (H, W, C) = video_tensor_list[0][0].shape
    random.shuffle(video_tensor_list)
    N = int(0.8 * len(video_tensor_list))
    train_data = VideoPredictDataset(video_tensor_list[:N])
    val_data = VideoPredictDataset(video_tensor_list[N:])

    fvt = FactorizedVideoTransformer(
        latent_dim=C, H=H, W=W,
        num_frames=T,
        pos_dim=(C-N_ADD_FEAT)//2 if C < 100 else (C-N_ADD_FEAT)//4,
        spatial_depth=3,
        temporal_depth=4,
        n_heads=8,
    )
    if os.path.exists(ps['vta']):
        fvt.load_state_dict(torch.load(ps['vta'], weights_only=True))
    train_model(fvt, train_data, val_data, epochs=20, batch_size=1, lr=1e-5, device='cuda',
                loss_fn=nn.MSELoss()) # act_loss_mean) #
    torch.save(fvt.state_dict(), ps['vta'])
