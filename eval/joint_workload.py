"""Correct donor workload coordinates in the simpler original retrieval baseline."""
import json
import time
from unittest.mock import patch

import numpy as np
import torch

from eval import joint_ordered as ordered,joint_pilot as old
from eval.joint_model import _torch_save
from eval.joint_phrase import ROOT,_atomic,_sha
from eval.joint_export import read_chart
from eval.joint_continuity import describe

OUT=ROOT/'experiments/joint-phrase-v8/workload-units'
SPEC=ROOT/'docs/specs/2026-09-30-retrieval-workload-units-design.md'


def corrected_bank(base):
    density=np.array([len({(b,h) for b,h,c,l,d in e['notes']})/8 for e in base['entries']],dtype=np.float32)
    mean=base['mean'].copy();sd=base['sd'].copy();x=base['x'].copy();entries=[]
    mean[-1]=density.mean();sd[-1]=max(float(density.std()),1e-6)
    x[:,-1]=(density-mean[-1])/sd[-1]
    for entry,rate in zip(base['entries'],density):
        descriptor=entry['descriptor'].copy();descriptor[-1]=rate
        entries.append({**entry,'descriptor':descriptor})
    result={**base,'entries':entries,'mean':mean,'sd':sd,'x':x}
    if not np.array_equal(result['x'][:,:54],base['x'][:,:54]):raise ValueError('audio descriptors changed')
    return result


def retrieve(source,audio,bank,rate,seed=0):
    if not np.isfinite(rate) or rate<0 or not np.isfinite(source['bpm']) or source['bpm']<=0:
        raise ValueError('invalid workload or tempo')
    return old.retrieve_song(source,audio,bank,rate*60/source['bpm'],seed)


def freeze_experiment():
    from eval.expressive_manifest import freeze_run
    parent,bank=ordered.prepare();report=json.loads((ordered.OUT/'report.json').read_text())
    if report['identity']!=parent['identity'] or not report['gates']['complete_panel']:
        raise ValueError('ordered control must finish before workload experiment')
    controls={str(p.relative_to(ROOT)):_sha(p) for p in (ordered.OUT/'development').rglob('*')
              if p.is_file() and p.name in ('song.json','record.json','ExpertPlus.dat','Info.dat')}
    return freeze_run(OUT,{'parent_identity':parent['identity'],'spec_sha256':_sha(SPEC),'code_sha256':_sha(__file__),
             'control_report_sha256':_sha(ordered.OUT/'report.json'),'control_bank_receipt':bank,
             'control_artifacts':controls,'continuity_code_sha256':_sha(ROOT/'eval/joint_continuity.py')})


def prepare():
    f=freeze_experiment();path=OUT/'retrieval.pt';receipt_path=OUT/'retrieval-receipt.json'
    if not receipt_path.exists():
        base=torch.load(ordered.OUT/'retrieval.pt',weights_only=False)
        bank=corrected_bank(base);_torch_save(path,bank)
        _atomic(receipt_path,{'identity':f['identity'],'sha256':_sha(path),'entries':len(bank['entries']),
                              'changed_feature':'hand-events per beat; audio/temporal arrays unchanged'})
    receipt=json.loads(receipt_path.read_text())
    if receipt['identity']!=f['identity'] or receipt['sha256']!=_sha(path):raise ValueError('workload bank changed')
    return f,receipt


def decide(songs,frozen,controls,continuity):
    # Reuse the exact v7 numeric decision with an isolated output destination.
    with patch.object(ordered,'OUT',OUT):report=ordered.decide(songs,frozen)
    ablation={};rng=np.random.default_rng(20260930);ix=rng.integers(0,len(songs),(2000,len(songs)))
    for axis in ('rhythm','geometry'):
        before=[controls[s['family']]['arms']['retrieval']['selected']['errors'][axis]['mean'] for s in songs]
        after=[s['arms']['ordered']['selected']['errors'][axis]['mean'] for s in songs]
        delta=np.array(after)-np.array(before)
        ablation[axis]={'uncorrected_original_mean':float(np.mean(before)),'corrected_mean':float(np.mean(after)),
                        'mean_difference':float(delta.mean()),'bootstrap_95_difference':np.quantile(delta[ix].mean(1),[.025,.975]).tolist(),
                        'paired_differences':{s['family']:float(v) for s,v in zip(songs,delta)}}
    report.update(arm_semantics={'ordered':'original retrieval with beat-density correction (legacy candidate field name)',
                                'retrieval':'frozen uncorrected original bag-of-frames deployment retrieval','b0':'original production baseline'},
                  workload_ablation=ablation,continuity=continuity,
                  retired_ordered_control=json.loads((ordered.OUT/'report.json').read_text())['aggregate']['ordered'],
                  budget='six new corrected candidates per family; original/ordered six-candidate v7 controls reused; no fit',
                  continuity_scope='native output beat-window boundaries; descriptive single-head physical-time transitions, not an admission gate')
    _atomic(OUT/'report.json',report);return report


def run(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700);torch.set_num_threads(4)
    frozen,receipt=prepare();bank=torch.load(OUT/'retrieval.pt',weights_only=False)
    scales=old.evaluation_freeze()['config']['scales'];kit=old.machine_tools();songs=[];controls={};continuity={}
    for family,item in ordered.inputs().items():
        key=family.replace(':','_');dest=OUT/'development'/key;control_root=ordered.OUT/'development'/key
        control=json.loads((control_root/'song.json').read_text());controls[family]=control;records=[]
        for seed in range(6):
            target=dest/'ordered'/str(seed);rec=old.cached_record(target/'record.json',frozen['identity'])
            if rec is None:
                if time.monotonic()>=deadline:return {'status':'INCOMPLETE','families':len(songs),'family':family,'next_seed':seed}
                started=time.monotonic();attempt=retrieve(item['source'],item['data']['audio'],bank,item['rate'],seed)
                attempt['elapsed_s']=time.monotonic()-started
                rec=ordered.measure(attempt,item['data'],target,frozen['identity'],scales,kit,
                                    control_root/'retrieval'/str(seed)/'record.json')
                print(f'workload {family} s{seed}: {rec["ok"]} {rec.get("machine",{}).get("verdict")}',flush=True)
            records.append(rec)
        chosen=ordered.select(records,item['b0'])
        song={**control,'identity':frozen['identity'],'arms':{'retrieval':control['arms']['retrieval'],'ordered':chosen}}
        _atomic(dest/'song.json',song);songs.append(song)
        continuity[family]={}
        for arm,selected,root,folder in (('corrected',chosen,dest,'ordered'),('uncorrected',control['arms']['retrieval'],control_root,'retrieval')):
            path=old.PILOT/'development'/key/'b0' if selected['fallback'] else root/folder/str(selected['selected_seed'])
            continuity[family][arm]=describe(read_chart(path/'ExpertPlus.dat',path/'Info.dat'))
    prepare();return decide(songs,frozen,controls,continuity)


if __name__=='__main__':print(json.dumps(run(),indent=1))
