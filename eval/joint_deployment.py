"""Deployment-only timing inputs and physical-time aligned measurement views."""
import json
import math

from eval.joint_phrase import _sha,encode_events,phrase_windows


def positive(value,name):
    if not math.isfinite(value) or value<=0:
        raise ValueError(f'invalid {name}')
    return value


def in_bpm(source,bpm):
    """Change beat units, not physical timing; never export this evaluation view."""
    positive(bpm,'bpm');positive(source['bpm'],'source bpm')
    positive(source['duration_beats'],'duration')
    ratio=bpm/source['bpm']
    result={k:v for k,v in source.items() if k not in ('windows','authored','events')}
    if ratio==1:
        result.update(notes=list(source['notes']),bombs=list(source['bombs']),walls=list(source['walls']))
    else:
        result.update(notes=[(b*ratio,h,c,l,d) for b,h,c,l,d in source['notes']],
                      bombs=[(b*ratio,c,l) for b,c,l in source['bombs']],
                      walls=[(b*ratio,dur*ratio,c,w,y,h) for b,dur,c,w,y,h in source['walls']])
    result.update(bpm=bpm,duration_beats=source['duration_beats']*ratio,
                  events=encode_events(result['notes']))
    phrase_windows(result['notes'],result['bombs'],result['walls'],result['duration_beats'])
    return result


def reference_view(source,reference):
    """Reference BPM is used only after generation, to align scoring windows."""
    result=in_bpm(source,reference['bpm'])
    positive(reference['duration_beats'],'reference duration')
    if not math.isclose(result['duration_beats'],reference['duration_beats'],rel_tol=0,abs_tol=1e-8):
        raise ValueError('audio duration mismatch')
    result['duration_beats']=reference['duration_beats']
    phrase_windows(result['notes'],result['bombs'],result['walls'],result['duration_beats'])
    return result


def deployment_input(audio_path,osu_path,raw_path):
    """No human chart argument: audio/MI/B0 provide all generation conditions."""
    from convert import parse_osu,grid_steps
    import soundfile as sf
    duration=positive(sf.info(str(audio_path)).duration,'audio duration')
    meta,objects,bpm,offset=parse_osu(osu_path);positive(bpm,'MI bpm')
    *_,grid=grid_steps(objects,bpm,offset,meta.get('_timing'),thin=True)
    raw=json.loads(raw_path.read_text());factor=bpm/60000
    source={'bpm':bpm,'duration_beats':duration*bpm/60,'notes':[],'events':[],
            'bombs':[],'walls':sorted((grid.time(s)*factor,(grid.time(s+n)-grid.time(s))*factor,c,1,0,5)
                                     for s,n,c in raw['walls'])}
    phrase_windows([],[],source['walls'],source['duration_beats'])
    rate=len({(n[0],n[1]) for n in raw['notes']})/duration
    return source,rate,{'audio_sha256':_sha(audio_path),'osu_sha256':_sha(osu_path),'raw_sha256':_sha(raw_path),
                        'bpm':bpm,'duration_seconds':duration,'requested_rate':rate,
                        'grid':grid.decision,'audio_origin_seconds':0,
                        'serving_inputs':'audio duration; MI first BPM; B0 TimeGrid walls and hand-event rate'}


def audit():
    """Opened-panel input audit; no new generation, fit or QA-cache writes."""
    from pathlib import Path
    import numpy as np
    from eval import joint_pilot as old
    from eval.joint_model import source_rate
    from eval.joint_export import export_chart,read_chart,qa_scene
    from eval.joint_phrase import ROOT,_atomic
    from eval.expressive_manifest import freeze_run
    root=ROOT/'experiments/joint-phrase-v5/deployment-input-audit-native-duration'
    data=old.load_data('development');parent=old.evaluation_freeze();inputs={}
    model_run=json.loads((ROOT/'experiments/joint-phrase-v1/model/run.json').read_text())
    audio_paths={r['family']:Path(r['sources']['audio']['path']) for r in model_run['config']['records']}
    prepared=[]
    for d in data:
        key=d['family'].replace(':','_');base=ROOT/'experiments/expressive-v1/fresh'
        osu,raw=base/'mi'/key/'gen.osu',base/'partial'/key/'b0.json'
        served,rate,proof=deployment_input(audio_paths[d['family']],osu,raw)
        inputs[d['family']]=proof
        prepared.append((d,served,rate,key))
    frozen=freeze_run(root,{'parent_identity':parent['identity'],'code_sha256':_sha(__file__),
        'duration_authority':'native audio file metadata, not resampled feature length',
        'inputs':inputs,'dependencies':{p:_sha(ROOT/p) for p in ('convert.py','timing.py','eval/joint_export.py')}})
    rows=[]
    for d,served,rate,key in prepared:
        original,_=old.baseline(d);candidate=in_bpm(original,served['bpm'])
        if not math.isclose(rate,source_rate(original),rel_tol=0,abs_tol=1e-12):
            raise ValueError('requested rate changed')
        if not np.allclose(np.array(candidate['walls']),np.array(served['walls']),rtol=0,atol=1e-10):
            raise ValueError('serving environment changed physical time')
        emitted=export_chart(candidate,root/'charts'/key)
        prior=read_chart(old.PILOT/'development'/key/'b0'/'ExpertPlus.dat',
                         old.PILOT/'development'/key/'b0'/'Info.dat')
        a,b=qa_scene(prior),qa_scene(emitted)
        deltas={}
        for field in ('notes','bombs','walls'):
            aa,bb=np.array(a[field]),np.array(b[field])
            if aa.shape!=bb.shape or not np.allclose(aa,bb,rtol=0,atol=1e-10):
                raise ValueError('physical scene changed: '+field)
            deltas[field]=float(np.max(np.abs(aa-bb))) if aa.size else 0.
        view=reference_view({**candidate,**emitted},d['source'])
        p,q=old.profile(original,d['audio']),old.profile(view,d['audio'])
        for axis in ('rhythm','geometry'):
            if not np.allclose(np.array(p[axis],float),np.array(q[axis],float),rtol=0,atol=1e-10,equal_nan=True):
                raise ValueError('reference-window profile changed: '+axis)
        rows.append({'family':d['family'],'authored_bpm':d['source']['bpm'],'serving_bpm':served['bpm'],
                     'physical_max_abs_difference':deltas,'aligned_profiles_match':True,
                     'chart_sha256':emitted['chart_sha256'],'info_sha256':emitted['info_sha256']})
    old.evaluation_freeze()
    result={'identity':frozen['identity'],'status':'DEPLOYMENT_INPUT_AUDIT_PASS','families':rows,
            'scope':'eight opened development families; coordinate/input check, not a candidate quality result',
            'limitations':['MI seed fixed; upstream variation not tested','constant BPM units at audio origin; phase/meter not inferred']}
    _atomic(root/'report.json',result)
    return result


if __name__=='__main__':
    print(json.dumps(audit(),indent=1))
