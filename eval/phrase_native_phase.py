"""Refine coupled placement while preserving native directional phase."""
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

import numpy as np
import torch

from eval import phrase_native_position as fixed,phrase_native_refine as base,joint_ordered as ordered,joint_pilot as old,joint_uniform_audio as uniform
from eval.joint_model import JointModel,EMPTY,actions,context,literal_slots,source_rate
from eval.joint_decode import JointState
from eval.joint_export import read_chart
from eval.joint_phrase import ROOT,_atomic,_sha,encode_events
from eval.expressive_manifest import freeze_run

OUT=ROOT/'experiments/phrase-native-phase-v1'
SPEC=ROOT/'docs/specs/2026-09-30-budget-native-phase-design.md'
TEMPERATURES=base.TEMPERATURES
inputs=base.inputs
select=base.select


def directions(d):
    if d in (0,4,5):return (0,4,5)
    if d in (1,6,7):return (1,6,7)
    return (d,)


def signature(source):
    return [(e['beat'],tuple(tuple(directions(d) for c,l,d in ns) for ns in e['hands'])) for e in encode_events(source['notes'])]


def refine(model,template,audio,seed=0,temperature=1.):
    if not np.isfinite(temperature) or temperature<=0:raise ValueError('invalid temperature')
    model.eval();device=next(model.parameters()).device
    gen=torch.Generator().manual_seed(seed)
    source={**template,'events':encode_events(template['notes'])}
    state=JointState(source['duration_beats'])
    previous,kind,last_gap,hidden=[EMPTY]*6,2,0.,None
    rate=source_rate(source);started=time.monotonic()

    def result(ok,reason,steps):
        return {'ok':ok,'reason':reason,'source':{**source,'notes':state.notes(),'events':state.events},
                'attempt_seed':seed,'actions':steps,'elapsed_s':time.monotonic()-started}

    rows=actions(source)
    with torch.no_grad():
        for step,row in enumerate(rows):
            if step>=20000:return result(False,'action_budget_exhausted',step)
            ctx=context(audio,state.cursor,source['bpm'],rate,18,source['walls'],source['bombs'])
            h,hidden=model.hidden(torch.tensor(ctx,device=device)[None,None],
                     torch.tensor(previous,device=device)[None,None],torch.tensor([[kind]],device=device),
                     torch.tensor([[last_gap]],device=device,dtype=torch.float32),hidden)
            h=h[0,0];delta=row['target_beat']-state.cursor
            if row['gap']==0:
                state.rest(row['target_beat']);kind,last_gap=0,delta
                continue
            category=torch.tensor(row['gap'],device=device)
            count=torch.tensor(row['count'],device=device)
            counts=[row['count']//4,row['count']%4]
            # Only directional roles are read from source slots; source poses never feed history.
            roles=[[row['slots'][hand*3+i]%9 for i in range(n)] for hand,n in enumerate(counts)]
            tokens=[EMPTY]*6;hands=[[],[]];occupied=set()
            try:
                for hand,total in enumerate(counts):
                    last_cell=-1
                    for slot in range(total):
                        cells=[c for c in range(last_cell+1,12) if c not in occupied]
                        remaining=total-slot-1
                        cells=cells[:len(cells)-remaining] if len(cells)>remaining else []
                        allowed_directions=directions(roles[hand][slot])
                        allowed=[c*9+d for c in cells for d in allowed_directions]
                        logits=model.slot_logits(h,category,count,torch.tensor(tokens,device=device))[hand*3+slot]
                        logits=logits.detach().float().cpu()
                        if logits.shape!=(108,) or not torch.isfinite(logits).all() or not allowed:
                            raise ValueError('invalid_or_infeasible_typed_geometry')
                        indices=torch.tensor(allowed)
                        token=int(indices[torch.multinomial(torch.softmax(logits[indices]/temperature,0),1,generator=gen)])
                        cell,direction=divmod(token,9);col,layer=divmod(cell,3)
                        hands[hand].append((col,layer,direction));occupied.add(cell);last_cell=cell
                        tokens[hand*3+slot]=token
                event={'beat':row['target_beat'],'hands':hands}
                state.append(event)
            except (ValueError,RuntimeError) as exc:return result(False,str(exc),step)
            previous=literal_slots(event);kind,last_gap=1,delta
    candidate=result(bool(state.events),None if state.events else 'empty_chart',len(rows))
    if candidate['ok'] and signature(candidate['source'])!=signature(source):
        raise ValueError('phase signature changed')
    return candidate


direction_counters=fixed.direction_counters

def verify_export(template,actual,record,baseline):
    base.verify_export(template,actual,record,baseline)
    assert record['phase_signature_ok'] and signature(template)==signature(actual)
    assert record['direction_counters_ok'] and direction_counters(template)==direction_counters(actual)


def freeze():
    parent=fixed.freeze();audit=json.loads((fixed.OUT/'artifact-audit.json').read_text())
    if audit['repeat_report_sha256']!=_sha(fixed.OUT/'report.json'):raise ValueError('exact-cut refinement not audited')
    return freeze_run(OUT,{'parent_identity':parent['identity'],'checkpoint':parent['config']['checkpoint'],
        'checkpoint_sha256':parent['config']['checkpoint_sha256'],'templates':parent['config']['templates'],
        'code_sha256':_sha(__file__),'spec_sha256':_sha(SPEC),'temperatures':TEMPERATURES,
        'parent_audit_sha256':_sha(fixed.OUT/'artifact-audit.json'),'parent_report_sha256':_sha(fixed.OUT/'report.json'),
        'protected':parent['config']['protected'],'query_views':parent['config']['query_views'],
        'dependencies':{p:_sha(ROOT/p) for p in ('eval/phrase_native_position.py','eval/phrase_native_refine.py','motion.py','eval/joint_model.py','eval/joint_decode.py',
            'eval/joint_ordered.py','eval/joint_pilot.py','eval/joint_continuity.py','eval/joint_export.py','eval/quality_metrics.py','parity.py')}})


def evidence(item,records,selected,root):
    result=base.evidence(item,records,selected,root)
    for seed,r in enumerate(records):
        if r['ok']:verify_export(item['template'],read_chart(root/'ordered'/str(seed)/'ExpertPlus.dat',root/'ordered'/str(seed)/'Info.dat'),r,item['template_record'])
    from motion import lr_doubles
    from eval.quality_metrics import opposite_horizontal
    def patterns(source):
        notes=[(b*60000/source['bpm'],h,c,l,d) for b,h,c,l,d in source['notes']]
        return {'opposite_horizontal':opposite_horizontal(notes),'horizontal_pairs':lr_doubles(notes)}
    chosen=item['template_path'] if selected['refinement_fallback'] else root/'ordered'/str(selected['selected_seed'])
    result['patterns_template']=patterns(item['template'])
    result['patterns_selected']=patterns(read_chart(chosen/'ExpertPlus.dat',chosen/'Info.dat'))
    return result


def decide(songs,frozen,proofs):
    with tempfile.TemporaryDirectory(prefix='osu2bs-native-phase-report-') as tmp:
        with patch.object(base,'OUT',Path(tmp)):report=base.decide(songs,frozen,proofs)
    records=[r for s in songs for r in s['arms']['ordered']['attempts']]
    report['gates']['exact_phase_signatures']=all(r['phase_signature_ok'] for r in records)
    report['gates']['directional_categories_preserved']=all(r['direction_counters_ok'] for r in records)
    report['status']='DEVELOPMENT_POSITIVE' if all(report['gates'].values()) else 'DEVELOPMENT_NEGATIVE'
    report['refinement_contract']='exact native timestamps, hands, directional phase, lateral cuts, dots and geometry support; coupled position/direction sampling'
    report['fixed_cut_paired_changes']={axis:{s['family']:s['arms']['ordered']['selected']['errors'][axis]['mean']-
        json.loads((fixed.OUT/'development'/s['family'].replace(':','_')/'song.json').read_text())['arms']['ordered']['selected']['errors'][axis]['mean']
        for s in songs} for axis in ('rhythm','geometry')}
    report['budget']='six cached native proposals plus six new phase-preserving geometry proposals per song; no fit'
    path=OUT/'report.json'
    if path.exists():assert json.loads(path.read_text())==report
    _atomic(path,report);return report


def run():
    torch.set_num_threads(2);deadline=time.monotonic()+2700;frozen=freeze();model=JointModel().eval()
    model.load_state_dict(torch.load(ROOT/frozen['config']['checkpoint'],map_location='cpu',weights_only=True))
    kit=old.machine_tools();scales=old.evaluation_freeze()['config']['scales'];songs=[];proofs={}
    for family,item in inputs().items():
        key=family.replace(':','_');root=OUT/'development'/key;records=[]
        audio=torch.load(uniform.OUT/'audio'/key/'features.pt',weights_only=False)
        for seed,temperature in enumerate(TEMPERATURES):
            target=root/'ordered'/str(seed);record=old.cached_record(target/'record.json',frozen['identity'])
            if record is None:
                if time.monotonic()>=deadline:return {'status':'INCOMPLETE','family':family,'seed':seed}
                attempt=refine(model,item['template'],audio,seed,temperature)
                attempt['phase_signature_ok']=signature(attempt['source'])==signature(item['template'])
                attempt['direction_counters_ok']=direction_counters(attempt['source'])==direction_counters(item['template'])
                attempt['signature_ok']=base.typed.signature(attempt['source'])==base.typed.signature(item['template'])
                if attempt['ok'] and not (attempt['phase_signature_ok'] and attempt['direction_counters_ok']):raise ValueError('directional plan changed')
                attempt['template_artifacts']=item['template_record']['artifacts']
                record=ordered.measure(attempt,item['data'],target,frozen['identity'],scales,kit,item['template_path']/'record.json')
                print('native phase',family,seed,record['ok'],record.get('machine',{}).get('verdict',record['reason']),flush=True)
            if record['ok']:verify_export(item['template'],read_chart(target/'ExpertPlus.dat',target/'Info.dat'),record,item['template_record'])
            records.append(record)
        chosen=select(records,item['template_record'],item['template_seed'])
        song={**item['prior'],'identity':frozen['identity'],'arms':{'retrieval':item['prior']['arms']['retrieval'],'ordered':chosen}}
        _atomic(root/'song.json',song);songs.append(song);proofs[family]=evidence(item,records,chosen,root)
    freeze();return decide(songs,frozen,proofs)


if __name__=='__main__':print(json.dumps(run(),indent=1))
