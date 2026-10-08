"""Conditional geometry with explicit histories for both hands, never target poses."""
import math
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from eval.joint_model import EMPTY,CONTEXT_DIM,context,literal_slots,source_rate
from eval.joint_decode import JointState
from eval.joint_phrase import encode_events
from eval.phrase_native_phase import signature,direction_counters

PAD_PHASE=5
PHASE=(0,1,2,3,0,0,1,1,4)
PHASE_DIRECTIONS=((0,4,5),(1,6,7),(2,),(3,),(8,),())
KEYS=('context','history','phase','age','count','slots')


def features(source,audio,event,state,rate,njs=18):
    history=[EMPTY]*6;ages=[0.]*4;phase=[PAD_PHASE]*6
    for hand,notes in enumerate(event['hands']):
        for slot,(_,_,d) in enumerate(notes):phase[hand*3+slot]=PHASE[d]
        prior=state.last_by_hand[hand]
        if prior is not None:
            seconds=(event['beat']-prior['beat'])*60/source['bpm']
            if seconds<0:raise ValueError('future hand state')
            ages[hand]=math.log1p(seconds)/4;ages[hand+2]=1.
            for slot,(c,l,d) in enumerate(prior['notes']):history[hand*3+slot]=(c*3+l)*9+d
    return {'context':context(audio,event['beat'],source['bpm'],rate,njs,source['walls'],source['bombs']),
            'history':history,'phase':phase,'age':ages,
            'count':len(event['hands'][0])*4+len(event['hands'][1])}


def legal_mask(phase,slots):
    """Identical teacher/serving masks; each slot reads only its earlier prefix."""
    if phase.shape!=slots.shape or phase.shape[-1]!=6:raise ValueError('invalid slot shape')
    cells=torch.arange(12,device=slots.device)
    direction_table=torch.zeros((6,9),dtype=torch.bool,device=slots.device)
    for p,ds in enumerate(PHASE_DIRECTIONS):direction_table[p,list(ds)]=True
    counts=(phase!=PAD_PHASE).reshape(*phase.shape[:-1],2,3).sum(-1)
    masks=[]
    for slot in range(6):
        active=phase[...,slot]!=PAD_PHASE
        prior_cell=slots[...,slot-1]//9 if slot%3 else torch.full_like(slots[...,slot],-1)
        available=cells>prior_cell[...,None]
        if slot:available=available&~(slots[...,:slot,None]//9==cells).any(-2)
        remaining=counts[...,slot//3]-slot%3-1
        capacity=available.sum(-1)-remaining
        available=available&(available.cumsum(-1)<=capacity[...,None])
        mask=(available[...,None]&direction_table[phase[...,slot]][...,None,:]).flatten(-2)
        # Inactive losses are omitted; finite logits here avoid 0*NaN in reduction.
        masks.append(torch.where(active[...,None],mask,torch.ones_like(mask)))
    return torch.stack(masks,-2)


class GeometryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token=nn.Embedding(109,16);self.phase=nn.Embedding(6,8)
        self.body=nn.Sequential(nn.Linear(CONTEXT_DIM+96+48+4,128),nn.GELU(),nn.Linear(128,128),nn.GELU())
        self.count=nn.Embedding(16,16);self.position=nn.Embedding(6,16)
        self.slot=nn.Sequential(nn.Linear(176,128),nn.GELU(),nn.Linear(128,108))

    def encode(self,batch):
        return self.body(torch.cat([batch['context'],self.token(batch['history']).flatten(-2),
            self.phase(batch['phase']).flatten(-2),batch['age']],-1))

    def slot_logits(self,h,count,targets):
        emb=self.token(targets)
        prefix=torch.cat([torch.zeros_like(emb[...,:1,:]),emb[...,:-1,:]],-2).cumsum(-2)
        shape=(*h.shape[:-1],6,-1)
        return self.slot(torch.cat([h.unsqueeze(-2).expand(shape),self.count(count).unsqueeze(-2).expand(shape),
            self.position(torch.arange(6,device=h.device)).expand(*h.shape[:-1],6,16),prefix],-1))

    def forward(self,batch):
        return self.slot_logits(self.encode(batch),batch['count'],batch['slots'])


def loss(logits,batch):
    active=batch['slots']!=EMPTY;mask=legal_mask(batch['phase'],batch['slots'])
    targets=batch['slots'].clamp_max(107)
    if not mask.gather(-1,targets[...,None]).squeeze(-1)[active].all():raise ValueError('unreachable teacher target')
    values=F.cross_entropy(logits.masked_fill(~mask,-torch.inf).reshape(-1,108),targets.reshape(-1),reduction='none').reshape_as(targets)
    return ((values*active).sum(-1)/active.sum(-1).clamp_min(1)).mean()


def teacher(source,audio,njs=None):
    source={**source,'events':encode_events(source['notes'])}
    state=JointState(source['duration_beats']);rate=source_rate(source);rows=[]
    if njs is None:njs=source.get('authored',{}).get('njs',18)
    for event in encode_events(source['notes']):
        rows.append({**features(source,audio,event,state,rate,njs),'slots':literal_slots(event)})
        state.append(event)
    if not rows:raise ValueError('empty teacher')
    tensors={k:torch.as_tensor(np.asarray([r[k] for r in rows]),dtype=torch.float32 if k in ('context','age') else torch.long) for k in KEYS}
    mask=legal_mask(tensors['phase'],tensors['slots']);active=tensors['slots']!=EMPTY
    if not mask.gather(-1,tensors['slots'].clamp_max(107)[...,None]).squeeze(-1)[active].all():raise ValueError('teacher outside decoder support')
    assert state.notes()==list(map(tuple,source['notes']))
    return tensors


def refine(model,template,audio,seed=0,temperature=1.):
    if not np.isfinite(temperature) or temperature<=0:raise ValueError('invalid temperature')
    model.eval();device=next(model.parameters()).device;gen=torch.Generator().manual_seed(seed)
    template={**template,'events':encode_events(template['notes'])}
    state=JointState(template['duration_beats']);rate=source_rate(template);started=time.monotonic()
    events=encode_events(template['notes'])
    def result(ok,reason):
        return {'ok':ok,'reason':reason,'source':{**template,'notes':state.notes(),'events':state.events},
            'attempt_seed':seed,'actions':len(state.events),'elapsed_s':time.monotonic()-started}
    with torch.no_grad():
        for event in events:
            if len(state.events)>=20000:return result(False,'action_budget_exhausted')
            raw=features(template,audio,event,state,rate)
            batch={k:torch.as_tensor(v,dtype=torch.float32 if k in ('context','age') else torch.long,device=device) for k,v in raw.items()}
            h=model.encode(batch);tokens=torch.full((6,),EMPTY,dtype=torch.long,device=device);hands=[[],[]]
            for slot,p in enumerate(raw['phase']):
                if p==PAD_PHASE:continue
                mask=legal_mask(batch['phase'],tokens)[slot]
                indices=mask.nonzero().flatten().cpu()
                logits=model.slot_logits(h,batch['count'],tokens)[slot].detach().float().cpu()
                if not len(indices) or not torch.isfinite(logits).all():return result(False,'invalid_or_infeasible_geometry')
                token=int(indices[torch.multinomial(torch.softmax(logits[indices]/temperature,0),1,generator=gen)])
                cell,d=divmod(token,9);c,l=divmod(cell,3);hands[slot//3].append((c,l,d));tokens[slot]=token
            state.append({'beat':event['beat'],'hands':hands})
    got=result(bool(state.events),None if state.events else 'empty_chart')
    if got['ok']:
        assert signature(got['source'])==signature(template)
        assert direction_counters(got['source'])==direction_counters(template)
    return got
