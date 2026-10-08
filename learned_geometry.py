"""Model-scored complete cuts under hard grid, scene and clearance constraints.

No exposure, axis, travel, shoulder or chain-frequency preferences. The learned
heads supply the joint score; sampling is conditioned on a complete legal cut.
"""
import torch
from torch.nn import functional as F

from parity import DIR_VEC
from swing_clearance import blocked_approaches


def sample_geometry(flow, hidden, hand, step, anchor, allowed, other, wall_cols,
                    generator, temperature, followers=None, *, relative=True):
    directions = sorted(allowed)
    if not directions:
        raise ValueError('no allowed cut direction')
    device = hidden.device
    dvec = F.one_hot(torch.tensor(directions, device=device), 9).float()
    hs = hidden.expand(len(directions), -1)
    dp = F.log_softmax(flow.dir_head(hidden), -1)
    cp = F.log_softmax(flow.col_head(torch.cat([hs, dvec], -1)), -1)
    kp = F.log_softmax(flow.chain_head(torch.cat([hs, dvec], -1)), -1)
    ncol = cp.shape[-1]
    layers = torch.cat([hs.repeat_interleave(ncol, 0), dvec.repeat_interleave(ncol, 0),
                       torch.eye(ncol, device=device).repeat(len(directions), 1)], -1)
    lp = F.log_softmax(flow.lay_head(layers), -1).reshape(len(directions), ncol, -1)
    occupied = {(n[2], n[3]) for n in other}
    choices = []; indices = []
    ac, al = anchor
    for i, direction in enumerate(directions):
        vx, vy = DIR_VEC[direction]
        for col in range(4):
            for layer in range(3):
                for count in (range(1, kp.shape[-1]+1) if followers is None else (followers+1,)):
                    group = [(step, hand, col+j*vx, layer+j*vy, direction if j==0 else 8)
                             for j in range(count)]
                    if not all(0<=x<=3 and 0<=y<=2 and x not in wall_cols
                               and (x,y) not in occupied
                               for _,_,x,y,_ in group): continue
                    if blocked_approaches(other + group): continue
                    dc, dl = (col-ac+3, layer-al+2) if relative else (col, layer)
                    choices.append((direction, col, layer, count))
                    indices.append((i, direction, dc, dl, count-1))
    if not choices:
        raise ValueError('no complete clear cut under current hand state')
    ids = torch.tensor(indices, device=device)
    scores = dp[ids[:,1]] + cp[ids[:,0],ids[:,2]] + lp[ids[:,0],ids[:,2],ids[:,3]] + kp[ids[:,0],ids[:,4]]
    probabilities = torch.softmax(scores/temperature, dim=0)
    index = int(torch.multinomial(probabilities, 1, generator=generator))
    return choices[index]
