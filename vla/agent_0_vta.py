import torch
import cv2
import numpy as np
import time

#from utils import *
from aeblock import StackedAE
from vis_transf import FactorizedVideoTransformer

from tagilmo.utils.vereya_wrapper import MCConnector, RobustObserverWithCallbacks
from mcdemoaux.vision.neural import process_pixel_data
#from tagilmo.utils.mathutils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_ADD_FEAT = 8
act_str = ['pitch', 'turn', 'strafe', 'move', 'attack', 'jump']
def process_act(act):
    atn = torch.zeros(N_ADD_FEAT)
    for a in act['actions']:
        if len(a) > 0 and a[0] in act_str:
            atn[act_str.index(a[0])] = float(a[1])
    atn[6] = 1 if "inRange" in act["lineOfSight"] and act["lineOfSight"]["inRange"] else 0
    atn[7] = act["agentPos"][3] / 180.
    return atn

def proc_frame(rob, acts):
    rob.mc.observeProc()
    frame = rob.getCachedObserve('getImageFrame')
    frame = process_pixel_data(frame.pixels, keep_aspect_ratio=True, maximum_size=(320,240))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    frame = torch.tensor(frame).permute(2, 0, 1)  # (C, H, W)
    frame = frame.to(device)
    with torch.no_grad():
        f = ae_model.encode_straight(frame.unsqueeze(0))[-1][0]
        j = {'actions': acts}
        j['lineOfSight'] = rob.cached['getLineOfSights'][0]
        j['agentPos'] = rob.cached['getAgentPos'][0]
        a = process_act(j).to(device)
        a = a.view(N_ADD_FEAT, 1, 1).expand(N_ADD_FEAT, f.size(1), f.size(2))
        return torch.cat([f, a], dim=0).permute(1, 2, 0)


if __name__ == "__main__":
    map_location = torch.device('cpu') if device == 'cpu' else None
    ae_model = StackedAE.from_dict(torch.load("result/ae_1_sparse.pth", weights_only=True, map_location=map_location))
    C = ae_model.aes[-1].decoder[0].in_channels
    ae_model = ae_model.to(device)
    T = 8
    fvt = FactorizedVideoTransformer(
        latent_dim=C+N_ADD_FEAT,
        H=240//(2 ** len(ae_model.aes)),
        W=320//(2 ** len(ae_model.aes)),
        num_frames=T,
        pos_dim=C//2 if C < 100 else C//4,
        spatial_depth=3,
        temporal_depth=4,
        n_heads=8,
    )
    fvt.load_state_dict(torch.load("result/vta_0a_ae1sparse.pth", weights_only=True, map_location=torch.device('cpu')))
    fvt = fvt.to(device)

    mc = MCConnector.connect(name='Robbo', video=True) #, seed='8823213')
    if not mc.is_mission_running():
        mc.safeStart()
    rob = RobustObserverWithCallbacks(mc)
    time.sleep(0.2)
    frames_tensor_list = []
    # seeding the agent with initial move
    for n in range(T):
        rob.sendCommand('move 1')
        va_tensor = proc_frame(rob, [['move', 1]])
        frames_tensor_list.append(va_tensor)
        time.sleep(0.05)

    #rob.sendCommand('move 0')
    frame = rob.getCachedObserve('getImageFrame')
    w = frame.iWidth
    h = frame.iHeight
    out = cv2.VideoWriter("agent.mp4", cv2.VideoWriter_fourcc(*'XVID'), 6, (w, h))
    for t in range(600):
        with torch.no_grad():
            frames_tensor = torch.stack(frames_tensor_list[-T:]).to(device)
            predict = fvt(frames_tensor.unsqueeze(0))[0][-1]
            predict = predict[:,:,-8:].mean(dim=(0, 1)) #amax
            #predict = predict[predict.shape[0]//2,predict.shape[1]//2,-8:]
            acts = []
            for ia in range(4):
                command = act_str[ia] + ' ' + str(predict[ia].item())
                rob.sendCommand(command)
                acts.append([act_str[ia], predict[ia].item()])
            #print(predict)
            if predict[4] > 0.3:
                rob.sendCommand('attack 1')
                acts.append(['attack', 1])
            else:
                rob.sendCommand('attack 0')
                acts.append(['attack', 0])
            if predict[5] > 0.2:
                rob.sendCommand('jump 1')
                acts.append(['jump', 1])
            else:
                rob.sendCommand('jump 0')
                acts.append(['jump', 0])
            va_tensor = proc_frame(rob, acts)
            frames_tensor_list.append(va_tensor)
            #print(predict)
            out.write(rob.getCachedObserve('getImageFrame').pixels)
            time.sleep(0.07)

    out.release()
