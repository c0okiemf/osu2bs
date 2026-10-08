"""No-fit geometry refinement with exact temporal and arrow/dot invariants."""
import json
import time

import numpy as np
import torch

from eval.joint_model import JointModel,EMPTY,actions,context,literal_slots,source_rate
from eval.joint_decode import JointState
from eval.joint_export import read_chart
from eval.joint_phrase import ROOT,_atomic,_sha,encode_events
from eval import joint_pilot as old
from eval import joint_start as start

OUT=ROOT/'experiments/joint-phrase-v4/typed-refinement'
SPEC=ROOT/'docs/specs/2026-09-30-typed-phrase-refinement-design.md'


def signature(source):
    return [(e['beat'],tuple(tuple(d==8 for c,l,d in ns) for ns in e['hands']))
            for e in encode_events(source['notes'])]


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
            # Only arrow/dot type is read from source slots; source poses never feed history.
            roles=[[row['slots'][hand*3+i]%9==8 for i in range(n)] for hand,n in enumerate(counts)]
            tokens=[EMPTY]*6;hands=[[],[]];occupied=set()
            try:
                for hand,total in enumerate(counts):
                    last_cell=-1
                    for slot in range(total):
                        cells=[c for c in range(last_cell+1,12) if c not in occupied]
                        remaining=total-slot-1
                        cells=cells[:len(cells)-remaining] if len(cells)>remaining else []
                        directions=[8] if roles[hand][slot] else range(8)
                        allowed=[c*9+d for c in cells for d in directions]
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
        raise ValueError('typed signature changed')
    return candidate


def common_error(reference,candidate,scales,mask):
    ref=np.array(reference['geometry'],float);cand=np.array(candidate['geometry'],float)
    mask=np.array(mask,bool)
    if ref.shape!=cand.shape or mask.shape!=ref.shape:raise ValueError('geometry window mismatch')
    counts=mask.sum(0)
    if np.any(counts==0) or np.any(mask&(~np.isfinite(ref)|~np.isfinite(cand))):
        raise ValueError('missing frozen geometry opportunity')
    values=np.where(mask,np.abs(ref-cand)/np.array(scales['geometry']),0).sum(0)/counts
    return {'mean':float(values.mean()),'components':values.tolist(),'supported_windows':counts.tolist()}


def inputs():
    result={}
    for d in old.load_data('development'):
        key=d['family'].replace(':','_');root=old.PILOT/'development'/key
        song=json.loads((root/'song.json').read_text());arm=song['arms']['retrieval']
        if arm['fallback'] or not arm['selected']['machine']['admitted']:
            raise ValueError('requires an admitted retrieved input')
        path=root/'retrieval'/str(arm['selected_seed'])
        source=read_chart(path/'ExpertPlus.dat',path/'Info.dat')
        source['duration_beats']=d['source']['duration_beats']
        source['events']=encode_events(source['notes'])
        result[d['family']]={'data':d,'source':source,'retrieval':arm['selected'],
                            'b0':song['b0'],'path':path,'song_path':root/'song.json'}
    if len(result)!=8:raise ValueError('development panel changed')
    return result


def freeze_experiment():
    from eval.expressive_manifest import freeze_run
    parent=start.freeze_experiment();selection=json.loads((start.OUT/'selection.json').read_text())
    model=start.OUT/f"snapshot-{selection['selected']}.pt"
    comparisons={};masks={};opportunities={}
    for family,item in inputs().items():
        for path in (item['song_path'],*(item['path']/n for n in ('record.json','ExpertPlus.dat','Info.dat'))):
            comparisons[str(path.relative_to(ROOT))]=_sha(path)
        r=item['retrieval'];ref=np.array(r['reference_profile']['geometry'],float)
        b=np.array(item['b0']['profile']['geometry'],float);retr=np.array(r['profile']['geometry'],float)
        mask=np.isfinite(ref)&np.isfinite(b)&np.isfinite(retr)
        if np.any(mask.sum(0)==0):raise ValueError('no common geometry support')
        masks[family]=mask.tolist()
        opportunities[family]={'common':mask.sum(0).tolist(),
                               'human':np.isfinite(ref).sum(0).tolist(),
                               'b0':np.isfinite(b).sum(0).tolist(),
                               'retrieval':np.isfinite(retr).sum(0).tolist()}
    return freeze_run(OUT,{'parent_identity':parent['identity'],'spec_sha256':_sha(SPEC),
            'code_sha256':_sha(__file__),'checkpoint':str(model.relative_to(ROOT)),
            'checkpoint_sha256':_sha(model),'comparison_artifacts':comparisons,
            'common_masks':masks,'opportunities':opportunities,'no_fit':True})


def select(records,retrieval):
    seed=next((i for i,r in enumerate(records) if r['ok'] and r.get('machine',{}).get('admitted')),None)
    return seed,retrieval if seed is None else records[seed]


def decide(songs,frozen,scales):
    aggregate={};per_family={}
    for family,song in songs.items():
        ref=song['retrieval']['reference_profile'];mask=frozen['config']['common_masks'][family]
        per_family[family]={arm:common_error(ref,song[arm]['profile'],scales,mask)
                            for arm in ('b0','retrieval','refined')}
    for arm in ('b0','retrieval','refined'):
        entries=[x[arm] for x in per_family.values()]
        aggregate[arm]={'mean':float(np.mean([e['mean'] for e in entries])),
                        'components':np.mean([e['components'] for e in entries],axis=0).tolist(),
                        'supported_windows':np.sum([e['supported_windows'] for e in entries],axis=0).tolist()}
    gates={'complete_panel':len(songs)==8,'six_nonfallback':sum(s['selected_seed'] is not None for s in songs.values())>=6,
           'exact_signatures':all(r.get('signature_ok',False) for s in songs.values() for r in s['attempts']),
           'rhythm_unchanged':all(s['refined']['profile']['rhythm']==s['retrieval']['profile']['rhythm'] for s in songs.values()),
           'opportunities_unchanged':all(np.array_equal(np.isfinite(np.array(s['refined']['profile']['geometry'],float)),
                         np.isfinite(np.array(s['retrieval']['profile']['geometry'],float))) for s in songs.values()),
           'no_new_contradictions':all(s['refined']['machine']['contradictions']['status'] not in
                         ('STRUCTURAL_CONTRADICTION','MODEL_CONTRADICTION') for s in songs.values())}
    intervals={};rng=np.random.default_rng(20260930);ix=rng.integers(0,len(songs),(2000,len(songs)))
    for other in ('b0','retrieval'):
        a,b=aggregate['refined'],aggregate[other]
        gates['geometry_10percent_vs_'+other]=a['mean']<=.9*b['mean'] and a['mean']<b['mean']
        gates['components_vs_'+other]=all(x<=1.1*y+.01 for x,y in zip(a['components'],b['components']))
        gates['support_vs_'+other]=all(s['refined']['machine']['support']['share_supported'] is not None and
                    s[other]['machine']['support']['share_supported'] is not None and
                    s['refined']['machine']['support']['share_supported']>=s[other]['machine']['support']['share_supported']-.05
                    for s in songs.values())
        delta=np.array([v['refined']['mean']-v[other]['mean'] for v in per_family.values()])
        intervals['refined_minus_'+other]=np.quantile(delta[ix].mean(1),[.025,.975]).tolist()
    panel=json.loads((start.OUT/'report.json').read_text())['panel']
    report={'identity':frozen['identity'],'status':'GEOMETRY_REFINEMENT_POSITIVE' if all(gates.values()) else 'GEOMETRY_REFINEMENT_NEGATIVE',
            'release_eligible':False,'gates':gates,'aggregate_common_geometry':aggregate,
            'per_family_common_geometry':per_family,'common_opportunities':frozen['config']['opportunities'],
            'bootstrap_95_difference':intervals,'panel':panel,
            'families':[{'family':family,'selected_seed':s['selected_seed'],'fallback':s['selected_seed'] is None,
                         'original_full_errors':{a:s[a]['errors'] for a in ('b0','retrieval','refined')},
                         'attempts':[{'ok':r['ok'],'reason':r['reason'],'verdict':r.get('machine',{}).get('verdict'),
                                      'signature_ok':r.get('signature_ok',False)} for r in s['attempts']]}
                        for family,s in songs.items()],
            'claim':'controlled geometry headroom with fixed rhythm/types; not a whole-generator pass or promotion',
            'budget':'six earlier retrieval candidates plus six new geometry attempts per family'}
    _atomic(OUT/'report.json',report)
    return report


def run(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700)
    torch.set_num_threads(4)
    frozen=freeze_experiment();scales=old.evaluation_freeze()['config']['scales']
    model=JointModel().eval();model.load_state_dict(torch.load(ROOT/frozen['config']['checkpoint'],map_location='cpu',weights_only=True))
    kit=old.machine_tools();songs={}
    for family,item in inputs().items():
        records=[];template=item['source'];d=item['data'];dest=OUT/'development'/family.replace(':','_')
        for seed,temp in enumerate((.85,1.,1.15,.85,1.,1.15)):
            target=dest/'refined'/str(seed);rec=old.cached_record(target/'record.json',frozen['identity'])
            if rec is None:
                if time.monotonic()>=deadline:return {'status':'INCOMPLETE','families':len(songs),'family':family,'next_seed':seed}
                attempt=refine(model,template,d['audio'],seed,temp)
                attempt['signature_ok']=signature(attempt['source'])==signature(template)
                if not attempt['signature_ok']:attempt.update(ok=False,reason='template_signature_changed')
                rec=old.measure_attempt(attempt,d,target,frozen['identity'],scales,kit)
                print(f"refinement {family} s{seed}: {rec['ok']} {rec.get('machine',{}).get('verdict')}",flush=True)
            # Recheck the typed contract for resumed artifacts as well as new ones.
            if rec['ok']:
                exported=read_chart(target/'ExpertPlus.dat',target/'Info.dat')
                if not rec.get('signature_ok') or signature(exported)!=signature(template):
                    raise ValueError('export changed template signature')
                if rec['profile']['rhythm']!=item['retrieval']['profile']['rhythm']:
                    raise ValueError('rhythm changed despite template signature')
                if not np.array_equal(np.isfinite(np.array(rec['profile']['geometry'],float)),
                                      np.isfinite(np.array(item['retrieval']['profile']['geometry'],float))):
                    raise ValueError('geometry opportunity changed')
                common_error(item['retrieval']['reference_profile'],rec['profile'],scales,frozen['config']['common_masks'][family])
            records.append(rec)
        selected,refined=select(records,item['retrieval'])
        song={'identity':frozen['identity'],'family':family,'selected_seed':selected,'refined':refined,
              'b0':item['b0'],'retrieval':item['retrieval'],'attempts':records}
        _atomic(dest/'song.json',song);songs[family]=song
    freeze_experiment()
    return decide(songs,frozen,scales)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['freeze','run'])
    parser.add_argument('--deadline-seconds',type=float,default=2700)
    args=parser.parse_args()
    print(json.dumps(freeze_experiment() if args.command=='freeze' else run(args.deadline_seconds),indent=1))
