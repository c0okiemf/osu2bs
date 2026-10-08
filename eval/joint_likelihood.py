"""Reference-free geometry likelihood selection from a frozen candidate pool."""
import json
import math
import time
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

from eval.joint_model import JointModel,EMPTY,actions,context,source_rate
from eval.joint_phrase import ROOT,_atomic,_sha,encode_events
from eval.joint_export import read_chart
from eval import joint_refine as typed,joint_pilot as old

OUT=ROOT/'experiments/joint-phrase-v6/likelihood-selection'
SPEC=ROOT/'docs/specs/2026-09-30-geometry-likelihood-selection-design.md'


def geometry_nll(model,source,audio,chunk_size=256):
    if chunk_size<1:raise ValueError('invalid chunk size')
    source={**source,'events':encode_events(source['notes'])};rows=actions(source)
    if not source['notes']:raise ValueError('empty geometry')
    device=next(model.parameters()).device;model.eval();hidden=None;total=0.;count=0
    rate=source_rate(source)
    with torch.no_grad():
        for start in range(0,len(rows),chunk_size):
            part=rows[start:start+chunk_size]
            batch={name:torch.tensor([r[key] for r in part],device=device)[None]
                   for name,key in (('prev','prev_slots'),('kind','prev_kind'),('prev_gap','prev_gap'),
                                    ('gap','gap'),('count','count'),('slots','slots'))}
            batch['prev_gap']=batch['prev_gap'].float()
            batch['context']=torch.tensor(np.stack([context(audio,r['cursor'],source['bpm'],rate,18,
                                                source['walls'],source['bombs']) for r in part]),device=device)[None]
            out,hidden=model(batch,hidden)
            valid=(batch['gap']>0)[...,None]&(batch['slots']!=EMPTY)
            if valid.any():
                logits=out['slots'][valid].double()
                if not torch.isfinite(logits).all():raise ValueError('nonfinite geometry score')
                total+=float(F.cross_entropy(logits,batch['slots'][valid],reduction='sum'))
                count+=int(valid.sum())
    if count!=len(source['notes']):raise ValueError('scored geometry count mismatch')
    return {'nll_sum':total,'tokens':count,'mean_nll':total/count}


def select(records,scores,retrieval):
    if len(records)!=len(scores):raise ValueError('score inventory mismatch')
    eligible=[(s['mean_nll'],i) for i,(r,s) in enumerate(zip(records,scores))
              if r['ok'] and r.get('machine',{}).get('admitted') and s is not None]
    if any(not math.isfinite(v) for v,i in eligible):raise ValueError('nonfinite selection score')
    if not eligible:return None,retrieval
    _,seed=min(eligible)
    return seed,records[seed]


def freeze_experiment():
    from eval.expressive_manifest import freeze_run
    parent=typed.freeze_experiment();bound={}
    for path in sorted((typed.OUT/'development').glob('*/refined/*/*')):
        if path.name in ('record.json','ExpertPlus.dat','Info.dat'):bound[str(path.relative_to(ROOT))]=_sha(path)
    if len(bound)!=144:raise ValueError('incomplete fixed candidate pool')
    return freeze_run(OUT,{'parent_identity':parent['identity'],'spec_sha256':_sha(SPEC),
                'code_sha256':_sha(__file__),'v4_report_sha256':_sha(typed.OUT/'report.json'),
                'candidate_artifacts':bound,'common_masks':parent['config']['common_masks'],
                'opportunities':parent['config']['opportunities'],'checkpoint':parent['config']['checkpoint'],
                'checkpoint_sha256':parent['config']['checkpoint_sha256']})


def run(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700);torch.set_num_threads(4)
    f=freeze_experiment();scales=old.evaluation_freeze()['config']['scales']
    model=JointModel().eval();model.load_state_dict(torch.load(ROOT/f['config']['checkpoint'],map_location='cpu',weights_only=True))
    songs={};scores_by_family={}
    for family,item in typed.inputs().items():
        root=typed.OUT/'development'/family.replace(':','_');song=json.loads((root/'song.json').read_text())
        scores=[]
        for seed,record in enumerate(song['attempts']):
            score_path=OUT/'scores'/family.replace(':','_')/f'{seed}.json'
            score=old.cached_record(score_path,f['identity'])
            if score is None:
                if time.monotonic()>=deadline:return {'status':'INCOMPLETE','family':family,'next_seed':seed}
                source=read_chart(root/'refined'/str(seed)/'ExpertPlus.dat',root/'refined'/str(seed)/'Info.dat')
                source['duration_beats']=item['source']['duration_beats']
                if typed.signature(source)!=typed.signature(item['source']):raise ValueError('typed signature changed')
                score={**geometry_nll(model,source,item['data']['audio']),'identity':f['identity']}
                _atomic(score_path,score)
            scores.append(score)
        seed,chosen=select(song['attempts'],scores,item['retrieval'])
        songs[family]={**song,'selected_seed':seed,'refined':chosen}
        scores_by_family[family]={'scores':scores,'first_admitted_seed':song['selected_seed'],'selected_seed':seed}
        _atomic(OUT/'development'/family.replace(':','_')/'song.json',songs[family])
        print(f'likelihood {family}: seed {seed}',flush=True)
    # The frozen diagnostic helper has a module-level output path. Override only
    # that destination during this synchronous call; never touch earlier reports.
    with patch.object(typed,'OUT',OUT):
        report=typed.decide(songs,f,scales)
    report.update(selection='minimum raw geometry NLL among admitted; earliest seed tie',
                  likelihood=scores_by_family,inherited_candidate_pool=str(typed.OUT.relative_to(ROOT)),
                  new_generation_attempts=0,new_optimizer_updates=0)
    _atomic(OUT/'report.json',report)
    freeze_experiment() # includes the unchanged v4 report, every chart and weights
    return report


if __name__=='__main__':
    print(json.dumps(run(),indent=1))
