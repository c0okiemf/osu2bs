"""Observational replay of frozen validation rollouts to locate tiny timing proposals."""
from collections import Counter
import json
from unittest.mock import patch

import numpy as np
import torch

from eval import joint_approved_fit as fit,joint_timing as timing,joint_pilot as old
from eval.expressive_manifest import freeze_run
from eval.joint_model import JointModel,actions,GAPS
from eval.joint_phrase import ROOT,_atomic,_sha
from eval.joint_export import read_chart

OUT=ROOT/'experiments/joint-timing-tail-audit-v1'
PLAN=ROOT/'docs/specs/2026-09-30-joint-timing-tail-audit.md'


def replay(model,source,audio,rate,seed):
    trace=[];logits=[];support=timing.timing_support;counts=model.count_logits
    def gap_hook(module,args,result):logits.append(result.detach().cpu().numpy().copy())
    def observed_support(cursor,end,kind,residuals):
        targets,allowed=support(cursor,end,kind,residuals)
        weights=torch.softmax(torch.tensor(logits[-1]).masked_fill(~torch.tensor(allowed),-float('inf')),0).numpy()
        trace.append({'cursor':cursor,'end':end,'kind':kind,'targets':targets,'probabilities':weights.tolist(),'category':0})
        return targets,allowed
    def observed_counts(hidden,category):
        trace[-1]['category']=int(category);return counts(hidden,category)
    hook=model.gap.register_forward_hook(gap_hook)
    try:
        with patch.object(timing,'timing_support',observed_support),patch.object(model,'count_logits',observed_counts):
            result=timing.rollout(model,source,audio,rate,seed=seed)
    finally:hook.remove()
    events=iter(result['source']['events']);last=[None,None]
    for row in trace:
        cat=row['category'];p=np.array(row.pop('probabilities'));order=np.argsort(-p,kind='stable')
        row.update(probability=float(p[cat]),rank=int(np.where(order==cat)[0][0])+1,modal_category=int(order[0]),smallest_category_probability=float(p[2]))
        targets=row.pop('targets');row['target']=targets[cat];row['tiny_same_hand']=[]
        if not cat:continue
        event=next(events);assert event['beat']==row['target']
        for hand,notes in enumerate(event['hands']):
            if not notes:continue
            if last[hand] is not None:
                ms=(event['beat']-last[hand])*60000/source['bpm']
                if ms<1:row['tiny_same_hand'].append({'hand':hand,'gap_ms':ms})
            last[hand]=event['beat']
    assert next(events,None) is None
    return result,trace


def run():
    torch.set_num_threads(2);parent,prepared=fit.prepare();selection=json.loads((fit.OUT/'selection.json').read_text())
    snapshot=fit.OUT/f"snapshot-{selection['selected']}.pt";valroot=fit.OUT/'validation'/str(selection['selected'])
    if selection['snapshot_sha256']!=_sha(snapshot):raise ValueError('selected checkpoint changed')
    frozen=freeze_run(OUT,{'parent_identity':parent['identity'],'selection_sha256':_sha(fit.OUT/'selection.json'),
        'prepared_sha256':_sha(fit.OUT/'prepared.json'),'snapshot_sha256':_sha(snapshot),'code_sha256':_sha(__file__),'plan_sha256':_sha(PLAN),
        'validation_artifacts':{str(p.relative_to(ROOT)):_sha(p) for p in valroot.rglob('*') if p.is_file()}})
    model=JointModel().eval();model.load_state_dict(torch.load(snapshot,map_location='cpu',weights_only=True))
    histogram=Counter();small=[];teacher_probability=[];teacher_modal=Counter();native_gap=[]
    for family,item in sorted(parent['config']['sources'].items()):
        if item['role']!='train':continue
        data=torch.load(ROOT/item['payload'],weights_only=False);rows=actions(data['source']);native_gap.append(fit.gap_diagnostic(data['source']))
        tensors=torch.load(fit.OUT/'data'/(family.replace(':','_')+'.pt'),weights_only=False)
        # Continuous teacher conditioning is descriptive, not the cropped training loss.
        with torch.no_grad():
            h,_=model.hidden(tensors['context'][None],tensors['prev'][None],tensors['kind'][None],tensors['prev_gap'][None])
            p=model.gap(h).softmax(-1)[0].numpy()
        teacher_probability.extend(p[:,2].tolist());teacher_modal.update(p.argmax(-1).tolist())
        for row in rows:
            histogram[row['gap']]+=1
            if row['gap']==2:small.append({'family':family,'beat':row['target_beat'],'delta_beats':row['target_beat']-row['cursor'],
                'residual':row['residual'],'count':row['count'],'previous':row['prev_slots'],'slots':row['slots']})
    summaries=[];all_trace=[]
    for family in sorted(fit.VALIDATION):
        data=torch.load(ROOT/parent['config']['sources'][family]['payload'],weights_only=False);audio=fit.load_view(family,parent)
        for seed in range(2):
            root=valroot/family.replace(':','_')/str(seed);record=json.loads((root/'record.json').read_text())
            result,trace=replay(model,old.generation_source(data),audio,prepared['validation_rate'],seed)
            assert result['ok']==record['ok'] and result['reason']==record['reason']
            actual=read_chart(root/'ExpertPlus.dat',root/'Info.dat');assert result['source']['notes']==actual['notes']
            assert result['source']['bombs']==actual['bombs'] and result['source']['walls']==actual['walls']
            assert len(trace)==record['timing_support']['decisions']
            tiny=[r for r in trace if r['tiny_same_hand']];all_trace.extend(trace)
            summary={'family':family,'seed':seed,'decisions':len(trace),'gap_diagnostic':fit.gap_diagnostic(actual),
                'category_counts':dict(Counter(r['category'] for r in trace)),'tiny_category_counts':dict(Counter(r['category'] for r in tiny)),
                'tiny_modal_choices':sum(r['rank']==1 for r in tiny),'tiny_nonmodal_choices':sum(r['rank']>1 for r in tiny),'tiny_rows':tiny}
            _atomic(OUT/'traces'/family.replace(':','_')/f'{seed}.json',trace);summaries.append(summary)
    tiny=[r for r in all_trace if r['tiny_same_hand']]
    result={'identity':frozen['identity'],'replayed_exports':len(summaries),'exact_notes_verified':True,
        'train':{'action_categories':dict(histogram),'smallest_category_targets':small,'same_hand_intervals':sum(r['same_hand_intervals'] for r in native_gap),
            'under_1ms':sum(r['under_1ms'] for r in native_gap),'teacher_smallest_probability_mean':float(np.mean(teacher_probability)),
            'teacher_modal_categories':dict(teacher_modal)},
        'rollout':{'decisions':len(all_trace),'category_counts':dict(Counter(r['category'] for r in all_trace)),
            'smallest_probability_mean':float(np.mean([r['smallest_category_probability'] for r in all_trace])),
            'tiny_decisions':len(tiny),'tiny_same_hand_intervals':sum(len(r['tiny_same_hand']) for r in tiny),
            'tiny_category_counts':dict(Counter(r['category'] for r in tiny)),
            'tiny_modal_choices':sum(r['rank']==1 for r in tiny),'tiny_nonmodal_choices':sum(r['rank']>1 for r in tiny)},
        'families':summaries,'claim':'exact observational replay; no repair, candidate selection, new outputs or universal tiny-gap rule'}
    _atomic(OUT/'report.json',result);return {k:v for k,v in result.items() if k not in ('train','families')}


if __name__=='__main__':print(json.dumps(run(),indent=1))
