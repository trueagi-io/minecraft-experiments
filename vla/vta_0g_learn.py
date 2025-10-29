import torch
import torch.nn as nn
from aeblock import StackedAE
import random
from utils import *
from vis_transf import *
import os
from vta_0_learn import load_data, process_frames

skip_goal = ['Obtain', 'FindAndMine']
goals = ['NeuralSearch', 'ApproachPos', 'LookAt', 'AttackBlockTool', 'PickNear']
def traverse(goal):
    if goal['goal'] in ['SAnd', 'SOr']:
        c = goal['current']
        g = goal['subgoals']
        return g[c if c < len(g) else c-1]
    if goal['goal'] not in skip_goal:
        return goal
    if goal['delegate'] == 'None':
        return goal
    return traverse(goal['delegate'])

N_ADD_FEAT = 8
def process_acts_goals(acts):
    acts_tensor_list = []
    act_str = ['pitch', 'turn', 'strafe', 'move', 'attack', 'jump']
    atn = torch.zeros(N_ADD_FEAT)
    for aa in acts:
        # Add goals first
        gtn = torch.zeros(N_ADD_FEAT)
        g = traverse(aa)['goal']
        if g in goals:
            gtn[goals.index(g)] = 1
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
        gtn = torch.cat([gtn, atn])
        acts_tensor_list.append(gtn)
    return acts_tensor_list


def act_goal_loss_mean(pred, target):
    return nn.MSELoss()(pred[:,:,:,-N_ADD_FEAT*2:],
                        target[:,:,:,-N_ADD_FEAT*2:])

if __name__ == "__main__":
    # In this experiment, we train the same FactorizedVideoTransformer on top
    # of the same autoencoder.
    params = {
        'dense' : {'ae' : 'result/ae_0_stack.pth',
                   'vta': 'result/vta_0g_ae0dense.pth'},
        'sparse': {'ae' : 'result/ae_1_sparse.pth',
                   'vta': 'result/vta_0g_ae1sparse.pth'},
    }
    ps = params['dense']
    T = 8
    video_tensor_list = []
    ae_model = StackedAE.from_dict(torch.load(ps['ae'], weights_only=True)).to('cuda')
    for i in range(29):
        frames, acts = load_data(f"data/dataset_wood/dataset_log{i}")#, max_frames = 100)
        act_tensor_list=process_acts_goals(acts)
        frames_tensor_list=process_frames(ae_model, frames)
        fract_tensor_list = []
        for a, f in zip(act_tensor_list, frames_tensor_list):
            # Here, we just append addtional action/observation features to image features
            a = a.view(N_ADD_FEAT*2, 1, 1).expand(N_ADD_FEAT*2, f.size(1), f.size(2))
            fract_tensor_list.append(torch.cat([f, a], dim=0).permute(1, 2, 0))
        for i in range(len(fract_tensor_list)-T-1):
            video_tensor_list.append(torch.stack(fract_tensor_list[i:i+T+1]))
    print("Samples:", len(video_tensor_list))
    (H, W, C) = video_tensor_list[0][0].shape
    print("Shape: ", H, W, C)
    random.shuffle(video_tensor_list)
    N = int(0.8 * len(video_tensor_list))
    train_data = VideoPredictDataset(video_tensor_list[:N])
    val_data = VideoPredictDataset(video_tensor_list[N:])

    fvt = FactorizedVideoTransformer(
        latent_dim=C, H=H, W=W,
        num_frames=T,
        pos_dim=(C-N_ADD_FEAT*2)//2 if C < 100 else (C-N_ADD_FEAT*2)//4,
        spatial_depth=3,
        temporal_depth=4,
        n_heads=8,
    )
    if os.path.exists(ps['vta']):
        fvt.load_state_dict(torch.load(ps['vta'], weights_only=True))
    train_model(fvt, train_data, val_data, epochs=20, batch_size=1, lr=1e-6, device='cuda',
                loss_fn=nn.MSELoss()) # act_goal_loss_mean) #
    torch.save(fvt.state_dict(), ps['vta'])
#0.00180
