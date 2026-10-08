"""Joint phrase pilot: train-only retrieval, paired output metrics and rollouts."""
from collections import Counter
import inspect
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from eval.joint_decode import JointState, sample_event
from eval.joint_export import export_chart, qa_scene
from eval.joint_model import (OUT, EMPTY, GAPS, N_GAP, JointModel, context, decode_gap,
                              freeze, literal_slots, source_rate)
from eval.joint_phrase import ROOT, _atomic, _sha, encode_events, phrase_windows


def profile(source, audio):
    """Window descriptors; closeness to one authored reference is not quality."""
    from eval.expression_profile import profile as expression
    windows = phrase_windows(source["notes"], source.get("bombs", []),
                              source.get("walls", []), source["duration_beats"])
    rhythm, geometry, intensities = [], [], []
    for w in windows:
        evs = w["events"]
        duration_s = (w["end"] - w["start"]) * 60 / source["bpm"]
        left = sum(bool(e["hands"][0]) for e in evs)
        right = sum(bool(e["hands"][1]) for e in evs)
        doubles = sum(all(e["hands"]) for e in evs)
        rhythm.append([len(evs)/duration_s, left/duration_s, right/duration_s,
                       doubles/len(evs) if evs else 0., float(not evs)])
        notes = [(e["beat"],h,c,l,d) for e in evs
                 for h, ns in enumerate(e["hands"]) for c,l,d in ns]
        if notes:
            p = expression([(h,c,l,d) for b,h,c,l,d in notes],
                           [b*60/source["bpm"] for b,h,c,l,d in notes])
            features = p.get("features") or {}
            spatial, arcs = features.get("spatial", {}), features.get("arcs", {})
            patterns = features.get("patterns", {})
            last, dist = {}, []
            for b,h,c,l,d in notes:
                if d == 8:
                    continue
                if h in last:
                    dist.append(math.dist(last[h], (c,l)))
                last[h]=(c,l)
            geometry.append([spatial.get("low_vertical_share"), arcs.get("runs_per_100_heads"),
                             (p.get("double_vocabulary") or {}).get("entropy"),
                             patterns.get("max_4gram_share"), np.mean(dist) if dist else None])
        else:
            geometry.append([None]*5)
        lo, hi = np.searchsorted(audio["times"], np.array([w["start"],w["end"]])*60/source["bpm"])
        intensities.append(float(np.mean(audio["rms"][lo:hi])) if hi>lo else 0.)
    times = np.array([e["beat"]*60/source["bpm"] for e in encode_events(source["notes"])])
    onset = float(np.mean(np.interp(times,audio["times"],audio["onset"]))) if len(times) else None
    act = np.array(rhythm)[:,0]
    corr = float(np.corrcoef(act,intensities)[0,1]) \
        if np.std(act)>1e-8 and np.std(intensities)>1e-8 else None
    return {"rhythm": rhythm, "geometry": geometry,
            "audio": {"mean_onset_strength": onset, "activity_rms_correlation": corr},
            "n_notes": len(source["notes"]), "n_events": len(times)}


def fit_scales(profiles):
    result={}
    for axis in ("rhythm","geometry"):
        x=np.concatenate([np.array(p[axis],dtype=float) for p in profiles])
        scale=np.nanquantile(x,.75,axis=0)-np.nanquantile(x,.25,axis=0)
        result[axis]=np.maximum(np.nan_to_num(scale,nan=0.),.05).tolist()
    return result


def errors(reference, candidate, scales):
    result={}
    for axis in ("rhythm","geometry"):
        ref, cand = (np.array(p[axis],dtype=float) for p in (reference,candidate))
        if ref.shape!=cand.shape:
            raise ValueError("window identity mismatch")
        valid=np.isfinite(ref)&np.isfinite(cand)
        delta=np.abs(ref-cand)/np.array(scales[axis])
        counts=valid.sum(0)
        values=np.divide(np.where(valid,delta,0).sum(0),counts,
                         out=np.full(5,np.nan),where=counts>0)
        # Lost opportunities are disclosed; a missing entire axis cannot win.
        result[axis]={"mean":float(np.nanmean(values)) if np.any(counts) else None,
                      "components":[float(v) if np.isfinite(v) else None for v in values],
                      "supported_windows":counts.tolist(),
                      "missing_candidate_windows":(np.isfinite(ref)&~np.isfinite(cand)).sum(0).tolist()}
    return result


def load_data(role):
    run=freeze()
    result=[]
    for r in run["config"]["records"]:
        if r["fit_role"]!=role:
            continue
        path=OUT/'data'/(r['family'].replace(':','_')+'.pt')
        receipt=json.loads(path.with_suffix('.json').read_text())
        if receipt['identity']!=run['identity'] or receipt['sha256']!=_sha(path):
            raise ValueError('prepared data identity mismatch')
        result.append(torch.load(path,weights_only=False))
    return result


def phrase_descriptor(audio, start, end, bpm, rate):
    lo, hi=np.searchsorted(audio['times'],np.array([start,end])*60/bpm)
    x=audio['x'][lo:hi]
    if len(x)==0:
        return np.zeros(55,dtype=np.float32)
    return np.concatenate([x.mean(0),x.std(0),[rate]]).astype(np.float32)


def retrieval_bank(train):
    bank=[]
    for d in train:
        if d["fit_role"] != "train":
            raise ValueError("retrieval accepts fit-train families only")
        src=d['source']
        for w in src['windows']:
            if w['end']-w['start']!=8:
                continue
            notes=[(e['beat']-w['start'],h,c,l,di) for e in w['events']
                   for h,ns in enumerate(e['hands']) for c,l,di in ns]
            rate=sum(bool(h) for e in w['events'] for h in e['hands'])/(8*60/src['bpm'])
            bank.append({'family':d['family'],'start':w['start'],'notes':notes,
                         'descriptor':phrase_descriptor(d['audio'],w['start'],w['end'],src['bpm'],rate)})
    x=np.stack([r['descriptor'] for r in bank])
    mean,sd=x.mean(0),np.maximum(x.std(0),1e-6)
    return {'entries':bank,'mean':mean,'sd':sd,'x':(x-mean)/sd}


def _join_ok(previous, upcoming, bpm):
    from parity import family
    for hand in (0,1):
        prev=[n for n in previous if n[1]==hand and n[4]!=8]
        nxt=[n for n in upcoming if n[1]==hand and n[4]!=8]
        if not prev or not nxt:
            continue
        a,b=prev[-1],nxt[0]
        # Only unambiguous directional boundary heads participate in this heuristic.
        if sum(n[0]==a[0] for n in prev)>1 or sum(n[0]==b[0] for n in nxt)>1:
            continue
        if (b[0]-a[0])*60/bpm<1 and family(a[4])==family(b[4]) \
                and family(a[4])!='lateral':
            return False
    return True


def retrieve_song(source, audio, bank, rate, seed=0):
    state=JointState(source['duration_beats'])
    used=[]
    for start in np.arange(0.,source['duration_beats'],8.):
        end=min(start+8,source['duration_beats'])
        q=(phrase_descriptor(audio,start,end,source['bpm'],rate)-bank['mean'])/bank['sd']
        distances=np.square(bank['x']-q).mean(1)
        order=np.argsort(distances,kind='stable')[:32]
        compatible=[]
        old=state.notes()
        for ix in order:
            entry=bank['entries'][ix]
            key=(entry['family'],entry['start'])
            if used and key==used[-1]:
                continue
            notes=[(b+float(start),h,c,l,d) for b,h,c,l,d in entry['notes'] if b<end-start]
            if _join_ok(old,notes,source['bpm']):
                compatible.append((key,notes))
                if len(compatible)==6:
                    break
        if not compatible:
            return {'ok':False,'reason':'no_compatible_retrieval_join','beat':float(start),'donors':used}
        key, notes=compatible[seed%len(compatible)]
        for event in encode_events(notes):
            state.append(event)
        state.rest(float(end))
        used.append(key)
    return {'ok':bool(state.events),'reason':None if state.events else 'empty_chart',
            'source':{**source,'notes':state.notes(),'events':state.events},
            'donors':used,'unique_donors':len(set(used)),'attempt_seed':seed}


def rollout(model, source, audio, rate, njs=18, seed=0, temperature=1., max_actions=20000):
    """Actual free-running joint sequence. Fail invalid proposals, never repair."""
    model.eval()
    device=next(model.parameters()).device
    gen=torch.Generator().manual_seed(seed)
    state=JointState(source['duration_beats'])
    hidden=None
    previous=[EMPTY]*6
    kind, last_gap=2,0.
    end=min(8.,source['duration_beats'])
    start_time=time.monotonic()
    def failed(reason, **details):
        return {'ok':False,'reason':reason,'source':{**source,'notes':state.notes(),
                'events':state.events},'attempt_seed':seed,**details}
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('invalid temperature')
    with torch.no_grad():
        for step in range(max_actions):
            if state.cursor>=source['duration_beats']:
                return {'ok':bool(state.events),'reason':None if state.events else 'empty_chart',
                        'source':{**source,'notes':state.notes(),'events':state.events},
                        'actions':step,'elapsed_s':time.monotonic()-start_time,'attempt_seed':seed}
            ctx=context(audio,state.cursor,source['bpm'],rate,njs,source.get('walls',[]),source.get('bombs',[]))
            h,hidden=model.hidden(torch.tensor(ctx,device=device)[None,None],
                 torch.tensor(previous,device=device)[None,None],torch.tensor([[kind]],device=device),
                 torch.tensor([[last_gap]],device=device,dtype=torch.float32),hidden)
            h=h[0,0]
            logits=model.gap(h).float().cpu()
            # Zero gap legal only at BOS or immediately after REST.
            if kind==1:
                logits[1]=-float('inf')
            if torch.isnan(logits).any() or torch.isposinf(logits).any() or not torch.isfinite(logits).any():
                return failed('invalid_gap_logits')
            category=int(torch.multinomial(torch.softmax(logits/temperature,0),1,generator=gen))
            if category==0:
                last_gap=end-state.cursor
                state.rest(end)
                kind=0
                end=min(end+8,source['duration_beats'])
                continue
            residual=float(model.residual(h)[category].clamp(-16,8))
            delta=decode_gap(category,residual)
            target=state.cursor+delta
            if not math.isfinite(target) or target>=end or (kind==1 and target<=state.cursor):
                return failed('event_outside_window',beat=state.cursor,actions=step)
            cat=torch.tensor(category,device=device)
            counts=model.count_logits(h,cat)
            # The slot head's joint-count condition must match the sampled count.
            if not torch.isfinite(counts).all():
                return failed('invalid_count_logits')
            clone=torch.Generator();clone.set_state(gen.get_state())
            sampled_count=int(torch.multinomial(torch.softmax(counts.detach().float().cpu()[1:]/temperature,0),
                                               1,generator=clone))+1
            def slots(hand,slot,prefix):
                tokens=[EMPTY]*6
                for hh,ns in enumerate(prefix):
                    for ss,(c,l,d) in enumerate(ns):
                        tokens[hh*3+ss]=(c*3+l)*9+d
                pred=model.slot_logits(h,cat,torch.tensor(sampled_count,device=device),
                                       torch.tensor(tokens,device=device))
                return pred[hand*3+slot]
            try:
                event=sample_event(target,counts,slots,gen,temperature)
                state.append(event)
            except (ValueError,RuntimeError) as exc:
                return failed(str(exc),beat=state.cursor,actions=step)
            previous=literal_slots(event)
            kind,last_gap=1,delta
    return failed('action_budget_exhausted',actions=max_actions)


PILOT = OUT.parent / 'pilot'
ADDENDUM = ROOT / 'docs/specs/2026-09-30-joint-pilot-evaluation-addendum.md'


def baseline(data):
    """Literal cached B0, mapped through its TimeGrid to the audio origin."""
    from convert import parse_osu, grid_steps
    key=data['family'].replace(':','_')
    root=ROOT/'experiments/expressive-v1/fresh'
    rawpath,osupath=root/'partial'/key/'b0.json',root/'mi'/key/'gen.osu'
    raw=json.loads(rawpath.read_text())
    meta,objects,bpm,offset=parse_osu(osupath)
    *_,grid=grid_steps(objects,bpm,offset,meta.get('_timing'),thin=True)
    src=data['source']
    factor=src['bpm']/60000
    notes=sorted((grid.time(s)*factor,h,c,l,d) for s,h,c,l,d in raw['notes'])
    walls=sorted((grid.time(s)*factor,(grid.time(s+length)-grid.time(s))*factor,c,1,0,5)
                 for s,length,c in raw['walls'])
    phrase_windows(notes,[],walls,src['duration_beats'])
    return {**src,'notes':notes,'events':encode_events(notes),'bombs':[],'walls':walls}, \
           {'raw_sha256':_sha(rawpath),'osu_sha256':_sha(osupath),
            'transform':'TimeGrid milliseconds * authored BPM / 60000; no lead shift'}


def controls():
    times=np.arange(0,16,.01)
    notes=[(b,h,c,0,di) for b,c,di in [(0,0,1),(1,0,0),(2,1,1),(3,1,0),
           (4,0,1),(5,0,0),(6,1,1),(7,1,0)] for h,c in [(0,c),(1,3-c)]]
    src={'notes':notes,'bombs':[],'walls':[],'bpm':60,'duration_beats':16}
    audio={'times':times,'x':np.zeros((len(times),27)),
           'onset':sum(np.exp(-((times-b)/.04)**2) for b in range(8)),
           'rms':np.where(times<8,1.,.1)}
    ref=profile(src,audio)
    scales={'rhythm':[1.]*5,'geometry':[1.]*5}
    self_error=errors(ref,ref,scales)
    silent=profile({**src,'notes':[]},audio)
    shifted=profile({**src,'notes':[(b+.5,h,c,l,d) for b,h,c,l,d in notes]},audio)
    rng=np.random.default_rng(13)
    # Shuffle whole simultaneous events, preserving literal occupancy and times.
    events=encode_events(notes)
    order=rng.permutation(len(events))
    shuffled_notes=[(event['beat'],h,c,l,d) for event,ix in zip(events,order)
                    for h,ns in enumerate(events[ix]['hands']) for c,l,d in ns]
    shuffled=profile({**src,'notes':shuffled_notes},audio)
    silence_error=errors(ref,silent,scales)
    shuffle_error=errors(ref,shuffled,scales)
    assert self_error['rhythm']['mean']==self_error['geometry']['mean']==0
    assert silence_error['rhythm']['mean']>0
    assert sum(silence_error['geometry']['missing_candidate_windows'])>0
    assert shifted['audio']['mean_onset_strength']<ref['audio']['mean_onset_strength']*.1
    assert shuffle_error['rhythm']['mean']==0 and shuffle_error['geometry']['mean']>0
    return {'status':'CONTROLS_PASS','self_error':self_error,'silence_error':silence_error,
            'shuffle_error':shuffle_error,'onset_reference':ref['audio']['mean_onset_strength'],
            'onset_shifted':shifted['audio']['mean_onset_strength']}


def generation_recipe():
    funcs=(profile,fit_scales,errors,phrase_descriptor,retrieval_bank,_join_ok,
           retrieve_song,rollout,baseline,controls)
    return hashlib.sha256('\n'.join(inspect.getsource(f) for f in funcs).encode()).hexdigest()


def prepare_evaluation():
    from eval.expressive_manifest import freeze_run
    from eval.joint_model import _torch_save
    torch.set_num_threads(4)
    run=freeze()
    checks=controls()
    train=load_data('train')
    bank=retrieval_bank(train)
    scales=fit_scales([profile(d['source'],d['audio']) for d in train])
    rate=float(np.median([d['rate'] for d in train]))
    config={'model_identity':run['identity'],'generation_recipe_sha256':generation_recipe(),
            'addendum_sha256':_sha(ADDENDUM),'scales':scales,'validation_rate':rate,
            'bank_families':sorted({e['family'] for e in bank['entries']}),
            'bank_entries':len(bank['entries']),'controls':checks,
            'dependencies':{p:_sha(ROOT/p) for p in ('eval/joint_decode.py',
                'eval/joint_export.py','eval/expression_profile.py','eval/joint_model.py',
                'qa/comparator_v2.py','qa/comparator_support.py','qa/certificates.py',
                'experiments/qa-v4/comparator-v2/evaluator_freeze.json')}}
    frozen=freeze_run(PILOT,config)
    bp=PILOT/'retrieval.pt'
    if not bp.exists():
        _torch_save(bp,bank)
        _atomic(PILOT/'retrieval-receipt.json',{'sha256':_sha(bp),'identity':frozen['identity']})
    receipt=json.loads((PILOT/'retrieval-receipt.json').read_text())
    if receipt['sha256']!=_sha(bp) or receipt['identity']!=frozen['identity']:
        raise ValueError('retrieval artifact changed')
    _atomic(PILOT/'controls.json',checks)
    return {'status':'EVALUATION_FROZEN','identity':frozen['identity'],
            'train':len(train),'bank_entries':len(bank['entries']),
            'min_actions':min(len(d['gap']) for d in train)}


def evaluation_freeze():
    run=freeze()
    f=json.loads((PILOT/'run.json').read_text())
    c=f['config']
    if c['model_identity']!=run['identity'] or c['generation_recipe_sha256']!=generation_recipe() \
            or c['addendum_sha256']!=_sha(ADDENDUM):
        raise ValueError('evaluation recipe changed')
    if any(_sha(ROOT/p)!=sha for p,sha in c['dependencies'].items()):
        raise ValueError('evaluation dependency changed')
    return f


def generation_source(data, environment=None):
    src=data['source']
    environment=environment or {'walls':[],'bombs':[]}
    return {'bpm':src['bpm'],'duration_beats':src['duration_beats'],
            'notes':[],'events':[],'walls':environment['walls'],'bombs':environment['bombs']}


def machine_tools():
    from qa.neighbours import bank_from_role
    from qa.certificates import load_speed_warning
    bank,_records=bank_from_role(identity='qa-train-v2')
    return (bank,
            {'T_support':json.loads((ROOT/'experiments/qa-v4/comparator-v2/evaluator_freeze.json')
                                   .read_text())['threshold']},load_speed_warning())


def machine_measure(readback, kit):
    from qa.certificates import contradictions
    from qa.comparator_support import measure_support
    from qa.comparator_v2 import chart_verdict
    bank,threshold,warn=kit
    scene=qa_scene(readback)
    contr=contradictions(scene,{'speed_warning':warn})
    support=measure_support(scene,bank,threshold,None)
    verdict=chart_verdict(contr,support)
    return {'verdict':verdict,'admitted':verdict=='MACHINE_PASS_QUALITY_NOT_EVALUATED',
            'contradictions':contr,'support':support}


def cached_record(path, identity):
    if not path.exists():
        return None
    rec=json.loads(path.read_text())
    if rec['identity']!=identity:
        raise ValueError('candidate identity mismatch')
    for name,sha in rec.get('artifacts',{}).items():
        if _sha(path.parent/name)!=sha:
            raise ValueError('candidate artifact changed')
    return rec


def measure_attempt(attempt, data, dest, identity, scales, kit=None):
    from eval.joint_export import assert_same
    source=attempt.pop('source',None)
    record={**attempt,'identity':identity,'family':data['family'],'artifacts':{}}
    if source is not None and source['notes']:
        readback=export_chart(source,dest,njs=18)
        assert_same(source,readback,18,0)
        record['artifacts']={name:_sha(dest/name) for name in ('ExpertPlus.dat','Info.dat')}
        emitted={**source,**{k:readback[k] for k in ('notes','bombs','walls','bpm')}}
        record['emitted_notes']=len(emitted['notes'])
        ev=encode_events(emitted['notes'])
        gaps=np.diff([0.]+[e['beat']*60/source['bpm'] for e in ev]+[source['duration_beats']*60/source['bpm']])
        record['cadence']={'max_gap_seconds':float(max(gaps)),'min_gap_seconds':float(min(gaps)),
                           'events_per_second':len(ev)/(source['duration_beats']*60/source['bpm'])}
        if attempt['ok']:
            p=profile(emitted,data['audio'])
            ref=profile(data['source'],data['audio'])
            record['profile']=p
            record['errors']=errors(ref,p,scales)
            record['reference_profile']=ref
            record['worst_windows']={}
            for axis in ('rhythm','geometry'):
                a,b=np.array(ref[axis],float),np.array(p[axis],float)
                valid=np.isfinite(a)&np.isfinite(b)
                count=valid.sum(1)
                error=np.divide(np.where(valid,np.abs(a-b)/np.array(scales[axis]),0).sum(1),
                                count,out=np.zeros(len(count)),where=count>0)
                order=sorted(range(len(count)),key=lambda i:(-error[i],i))[:3]
                record['worst_windows'][axis]=[{'start_beat':int(i*8),'mean_error':float(error[i]),
                                                 'supported_components':int(count[i])} for i in order]
            if kit is not None:
                record['machine']=machine_measure(readback,kit)
    elif attempt['ok']:
        record.update(ok=False,reason='empty_chart')
    _atomic(dest/'record.json',record)
    return record


def validation(deadline_seconds=2700):
    from eval.joint_model import SNAPSHOTS
    torch.set_num_threads(2)
    frozen=evaluation_freeze()
    config=frozen['config']
    data=load_data('validation')
    deadline=time.monotonic()+min(deadline_seconds,2700)
    summaries=[]
    for update in SNAPSHOTS:
        snapshot=OUT/f'snapshot-{update}.pt'
        if not snapshot.exists():
            return {'status':'WAITING_FOR_SNAPSHOT','update':update,'summaries':summaries}
        model=JointModel()
        model.load_state_dict(torch.load(snapshot,map_location='cpu',weights_only=True))
        identity=frozen['identity']+':'+_sha(snapshot)
        records=[]
        for d in data:
            for seed in (0,1):
                dest=PILOT/'validation'/str(update)/d['family'].replace(':','_')/str(seed)
                rec=cached_record(dest/'record.json',identity)
                if rec is None:
                    if time.monotonic()>=deadline:
                        return {'status':'INCOMPLETE','update':update,'completed':len(records)}
                    result=rollout(model,generation_source(d),d['audio'],config['validation_rate'],seed=seed)
                    rec=measure_attempt(result,d,dest,identity,config['scales'])
                    print(f"validation {update} {d['family']} s{seed}: {rec['ok']} {rec.get('reason')}",flush=True)
                records.append(rec)
        good=[r for r in records if r['ok']]
        errors_sum=[sum(r['errors'][a]['mean'] for a in ('rhythm','geometry')) for r in good
                    if all(r['errors'][a]['mean'] is not None for a in ('rhythm','geometry'))]
        summary={'update':update,'snapshot_sha256':_sha(snapshot),'attempts':len(records),
                 'complete':len(good),'failures':len(records)-len(good),
                 'failure_reasons':dict(Counter(r.get('reason') for r in records if not r['ok'])),
                 'error':float(np.mean(errors_sum)) if len(errors_sum)==len(good) and good else None}
        summaries.append(summary)
        _atomic(PILOT/'validation'/f'summary-{update}.json',summary)
    selected=min(summaries,key=lambda r:(r['failures'],float('inf') if r['error'] is None else r['error'],r['update']))
    report={'status':'VALIDATION_COMPLETE','identity':frozen['identity'],'snapshots':summaries,
            'selected':selected['update'],'learned_candidate':selected['update']>0,
            'selection':'failure count; rhythm+geometry error; earliest update'}
    _atomic(PILOT/'selection.json',report)
    return report


def development(deadline_seconds=2700):
    torch.set_num_threads(4)
    frozen=evaluation_freeze()
    config=frozen['config']
    selection=json.loads((PILOT/'selection.json').read_text())
    if selection['identity']!=frozen['identity']:
        raise ValueError('selection identity mismatch')
    snapshot=OUT/f"snapshot-{selection['selected']}.pt"
    model=JointModel()
    model.load_state_dict(torch.load(snapshot,map_location='cpu',weights_only=True))
    bankpath=PILOT/'retrieval.pt'
    receipt=json.loads((PILOT/'retrieval-receipt.json').read_text())
    if receipt['sha256']!=_sha(bankpath) or receipt['identity']!=frozen['identity']:
        raise ValueError('retrieval bank identity changed')
    bank=torch.load(bankpath,weights_only=False)
    kit=machine_tools()
    deadline=time.monotonic()+min(deadline_seconds,2700)
    songs=[]
    for d in load_data('development'):
        src,transform=baseline(d)
        identity=hashlib.sha256(json.dumps({'evaluation':frozen['identity'],
                    'snapshot':_sha(snapshot),'baseline':transform},sort_keys=True).encode()).hexdigest()
        dest=PILOT/'development'/d['family'].replace(':','_')
        b0=cached_record(dest/'b0'/'record.json',identity)
        if b0 is None:
            if time.monotonic()>=deadline:
                return {'status':'INCOMPLETE','families':len(songs)}
            b0=measure_attempt({'ok':True,'source':src,'transform':transform},d,dest/'b0',
                               identity,config['scales'],kit)
        arms={}
        for arm in ('model','retrieval'):
            records=[]
            for seed,temp in enumerate((.85,1.,1.15,.85,1.,1.15)):
                target=dest/arm/str(seed)
                rec=cached_record(target/'record.json',identity)
                if rec is None:
                    if time.monotonic()>=deadline:
                        return {'status':'INCOMPLETE','families':len(songs),'family':d['family'],
                                'arm':arm,'next_seed':seed}
                    gen_source=generation_source(d,src)
                    if arm=='model':
                        result=rollout(model,gen_source,d['audio'],source_rate(src),seed=seed,temperature=temp)
                    else:
                        result=retrieve_song(gen_source,d['audio'],bank,source_rate(src),seed=seed)
                    rec=measure_attempt(result,d,target,identity,config['scales'],kit)
                    print(f"development {d['family']} {arm} s{seed}: {rec['ok']} "
                          f"{rec.get('reason')} {rec.get('machine',{}).get('verdict')}",flush=True)
                records.append(rec)
            selected=next((i for i,r in enumerate(records) if r['ok'] and
                           r.get('machine',{}).get('admitted')),None)
            arms[arm]={'selected_seed':selected,'fallback':selected is None,
                       'selected':b0 if selected is None else records[selected],
                       'attempts':[{'ok':r['ok'],'reason':r.get('reason'),
                                    'verdict':r.get('machine',{}).get('verdict'),
                                    'emitted_notes':r.get('emitted_notes',0)} for r in records]}
        song={'identity':identity,'family':d['family'],'b0':b0,'arms':arms,
              'requested_rate':source_rate(src),'reference_rate':d['rate']}
        _atomic(dest/'song.json',song)
        songs.append(song)
    return decision(songs,selection,frozen)


def decision(songs,selection,frozen,output_root=None):
    # Fixed family-balanced means, no post hoc serving selection by human error.
    aggregate={}
    for arm in ('b0','model','retrieval'):
        records=[s['b0'] if arm=='b0' else s['arms'][arm]['selected'] for s in songs]
        aggregate[arm]={}
        for axis in ('rhythm','geometry'):
            es=[r['errors'][axis] for r in records]
            components=[[e['components'][i] for e in es] for i in range(5)]
            aggregate[arm][axis]={
                'mean':float(np.mean([e['mean'] for e in es])) if all(e['mean'] is not None for e in es) else None,
                'components':[float(np.mean(v)) if all(x is not None for x in v) else None for v in components],
                'supported_windows':np.sum([e['supported_windows'] for e in es],axis=0).tolist()}
    gates={'complete_panel':len(songs)==8,'learned_checkpoint':selection['learned_candidate'],
           'six_nonfallback':sum(not s['arms']['model']['fallback'] for s in songs)>=6}
    intervals={}
    rng=np.random.default_rng(20260930)
    samples=rng.integers(0,len(songs),(2000,len(songs)))
    for other in ('b0','retrieval'):
        for axis in ('rhythm','geometry'):
            a,b=aggregate['model'][axis],aggregate[other][axis]
            gates[f'{axis}_10percent_vs_{other}']=a['mean'] is not None and b['mean'] is not None and a['mean']<=.9*b['mean'] and a['mean']<b['mean']
            gates[f'{axis}_components_vs_{other}']=all(x is not None and y is not None and x<=1.1*y+.01
                                                     for x,y in zip(a['components'],b['components']))
            gates[f'{axis}_coverage_vs_{other}']=all(x>=y for x,y in zip(a['supported_windows'],b['supported_windows']))
            delta=[]
            for song in songs:
                ma= song['arms']['model']['selected']['errors'][axis]['mean']
                ob=(song['b0'] if other=='b0' else song['arms'][other]['selected'])['errors'][axis]['mean']
                delta.append(None if ma is None or ob is None else ma-ob)
            intervals[f'{axis}_model_minus_{other}']=np.quantile(np.array(delta)[samples].mean(1),[.025,.975]).tolist() \
                if all(v is not None for v in delta) else None
    gates['no_new_contradictions']=all(s['arms']['model']['selected']['machine']['contradictions']['status']
        not in ('STRUCTURAL_CONTRADICTION','MODEL_CONTRADICTION') or
        s['arms']['model']['selected']['machine']['contradictions']==s['b0']['machine']['contradictions'] for s in songs)
    gates['support_no_drop_over_005']=all(
        s['arms']['model']['selected']['machine']['support']['share_supported'] is not None and
        s['b0']['machine']['support']['share_supported'] is not None and
        s['arms']['model']['selected']['machine']['support']['share_supported']>=s['b0']['machine']['support']['share_supported']-.05
        for s in songs)
    from eval.joint_phrase import OUT as readiness
    rr=json.loads((readiness/'report.json').read_text())
    inventory=json.loads((readiness/'run.json').read_text())
    dev={r['family'] for r in inventory['config']['records'] if r['role']=='development'}
    panel=[{'family':r['family'],'readiness':r['status'],'included':r['family'] in {s['family'] for s in songs},
            'reason':json.loads((readiness/'partial'/(r['family'].replace(':','_')+'.json')).read_text()).get('reason')}
           for r in rr['families'] if r['family'] in dev]
    report={'status':'PILOT_POSITIVE' if all(gates.values()) else 'PILOT_NEGATIVE',
            'identity':frozen['identity'],'gates':gates,'aggregate':aggregate,
            'bootstrap_95_difference':intervals,'panel':panel,'selection':selection,
            'families':[{'family':s['family'],'requested_rate':s['requested_rate'],
                         'reference_rate':s['reference_rate'],'arms':{a:{'fallback':v['fallback'],
                         'selected_seed':v['selected_seed'],'attempts':v['attempts']} for a,v in s['arms'].items()}}
                        for s in songs],
            'claim':'paired development reference distances and scoped machine evidence; not enjoyment or perfection'}
    evaluation_freeze()  # recheck production/data/evaluator identities at completion
    _atomic((PILOT if output_root is None else Path(output_root))/'report.json',report)
    return report


def diagnostics():
    """Teacher losses and historical descriptors; never checkpoint/serving gates."""
    from eval.joint_model import KEYS,SNAPSHOTS
    import torch.nn.functional as F
    from eval.expression_profile import profile as expression
    from eval.e1_machine_ab import PROPS,GUARD,_get
    from eval.joint_export import read_chart
    torch.set_num_threads(2)
    frozen=evaluation_freeze()
    result={'identity':frozen['identity'],'teacher_validation':[],'development_expression':[]}
    data=load_data('validation')
    for update in SNAPSHOTS:
        snapshot=OUT/f'snapshot-{update}.pt'
        path=PILOT/'diagnostics'/f'teacher-{update}.json'
        ident=frozen['identity']+':'+_sha(snapshot)
        saved=cached_record(path,ident)
        if saved is None:
            model=JointModel().eval()
            model.load_state_dict(torch.load(snapshot,weights_only=True,map_location='cpu'))
            sums={k:0. for k in ('gap_ce','count_ce','slot_ce','residual_smooth_l1')}
            counts={k:0 for k in sums}
            with torch.no_grad():
                for d in data:
                    state=None
                    for start in range(0,len(d['gap']),256):
                        batch={k:d[k][start:start+256][None] for k in KEYS}
                        out,state=model(batch,state)
                        valid=torch.ones_like(batch['gap'],dtype=torch.bool)
                        if start==0:
                            valid[:,:32]=False
                        event=valid&(batch['gap']>0)
                        positive=valid&(batch['gap']>1)
                        slots=event[...,None]&(batch['slots']!=EMPTY)
                        for key,mask,pred,target in (
                            ('gap_ce',valid,out['gap'],batch['gap']),
                            ('count_ce',event,out['count'],batch['count']),
                            ('slot_ce',slots,out['slots'],batch['slots'])):
                            if mask.any():
                                sums[key]+=float(F.cross_entropy(pred[mask],target[mask],reduction='sum'))
                                counts[key]+=int(mask.sum())
                        pred=out['residual'].gather(-1,batch['gap'][...,None]).squeeze(-1)
                        if positive.any():
                            sums['residual_smooth_l1']+=float(F.smooth_l1_loss(pred[positive],batch['residual'][positive],reduction='sum'))
                            counts['residual_smooth_l1']+=int(positive.sum())
            saved={'identity':ident,'update':update,'families':len(data),
                   'counts':counts,'loss':{k:sums[k]/counts[k] if counts[k] else None for k in sums}}
            _atomic(path,saved)
        result['teacher_validation'].append(saved)
    for path in sorted((PILOT/'development').glob('*/song.json')):
        song=json.loads(path.read_text())
        arms={'b0':path.parent/'b0'}
        for arm in ('model','retrieval'):
            seed=song['arms'][arm]['selected_seed']
            arms[arm]=path.parent/'b0' if seed is None else path.parent/arm/str(seed)
        row={'family':song['family'],'arms':{}}
        for arm,dest in arms.items():
            r=read_chart(dest/'ExpertPlus.dat',dest/'Info.dat')
            p=expression([(h,c,l,di) for b,h,c,l,di in r['notes']],
                         [b*60/r['bpm'] for b,h,c,l,di in r['notes']])
            row['arms'][arm]={'props':{k:_get(p,v[0]) for k,v in PROPS.items()},
                              'repetition_guard':_get(p,GUARD[0])}
        result['development_expression'].append(row)
    _atomic(PILOT/'diagnostics.json',result)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','validate','development','diagnostics'])
    parser.add_argument('--deadline-seconds',type=float,default=2700)
    args=parser.parse_args()
    result=diagnostics() if args.command=='diagnostics' else prepare_evaluation() if args.command=='prepare' else \
           {'validate':validation,'development':development}[args.command](args.deadline_seconds)
    print(json.dumps(result,indent=1))
