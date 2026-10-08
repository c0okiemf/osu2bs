"""Fine-lag cross-role audio-overlap screen on existing prepared pilot inputs."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
from scipy.signal import correlate,correlation_lags
import torch

from eval import joint_pilot as old
from eval.corpus import _fingerprint,FP_CORR
from eval.expressive_manifest import freeze_run
from eval.joint_phrase import ROOT,_atomic,_sha

OUT=ROOT/'experiments/joint-audio-identity-v1'
SPEC=ROOT/'docs/specs/2026-09-30-cross-role-audio-audit-design.md'
HZ=50


def peak_correlation(a,b,max_lag=1500,min_overlap=1500):
    """Pearson normalized separately on each overlap; positive lag delays a."""
    a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if len(a)<min_overlap or len(b)<min_overlap:return {'correlation':None,'lag_samples':None,'overlap_samples':0}
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('nonfinite envelope')
    # Center first for numerical stability, then normalize each lag's overlap.
    a=a-a.mean();b=b-b.mean();n=len(a);m=len(b)
    lags=correlation_lags(n,m);allowed=np.abs(lags)<=max_lag
    lags=lags[allowed];dot=correlate(a,b,method='fft')[allowed]
    lo_a=np.maximum(lags,0);lo_b=np.maximum(-lags,0);length=np.minimum(n-lo_a,m-lo_b)
    def sums(v,lo):
        c=np.r_[0.,np.cumsum(v)];q=np.r_[0.,np.cumsum(v*v)]
        return c[lo+length]-c[lo],q[lo+length]-q[lo]
    sa,qa=sums(a,lo_a);sb,qb=sums(b,lo_b)
    va=np.maximum(qa-sa*sa/length,0);vb=np.maximum(qb-sb*sb/length,0);den=np.sqrt(va*vb)
    valid=(length>=min_overlap)&(den>1e-15)
    score=np.full(len(lags),-np.inf);score[valid]=(dot[valid]-sa[valid]*sb[valid]/length[valid])/den[valid]
    if not valid.any():return {'correlation':None,'lag_samples':None,'overlap_samples':0}
    ix=int(np.argmax(score))
    return {'correlation':float(np.clip(score[ix],-1,1)),'lag_samples':int(lags[ix]),'overlap_samples':int(length[ix])}


def run():
    parent=old.freeze();records=parent['config']['records'];bindings={};prepared={}
    assert Counter(r['fit_role'] for r in records)=={'train':316,'validation':32,'development':8}
    for r in records:
        path=old.OUT/'data'/(r['family'].replace(':','_')+'.pt');rp=path.with_suffix('.json');receipt=json.loads(rp.read_text())
        if receipt['identity']!=parent['identity'] or _sha(path)!=receipt['sha256']:raise ValueError('prepared audio identity changed')
        for name in ('audio','info'):
            s=r['sources'][name]
            if _sha(s['path'])!=s['sha256']:raise ValueError('original audio/metadata changed')
        bindings[r['family']]={'fit_role':r['fit_role'],'prepared_sha256':receipt['sha256'],'receipt_sha256':_sha(rp),'sources':r['sources']}
        prepared[r['family']]=path
    frozen=freeze_run(OUT,{'parent_identity':parent['identity'],'inputs':bindings,'code_sha256':_sha(__file__),'spec_sha256':_sha(SPEC),
        'corpus_code_sha256':_sha(ROOT/'eval/corpus.py'),'sampling_hz':HZ,'first_seconds':90,'maximum_lag_seconds':30,
        'minimum_overlap_seconds':30,'screening_threshold':FP_CORR,'independent_decode':'existing corpus._fingerprint: 4kHz original-file RMS, 2048 frame, 80 hop'})
    envelopes={};metadata={}
    for r in records:
        data=torch.load(prepared[r['family']],weights_only=False,mmap=True);a=data['audio']
        times=np.arange(0,min(90.,a['duration_s']),1/HZ)
        envelopes[r['family']]=np.interp(times,a['times'],a['rms'])
        info=json.loads(Path(r['sources']['info']['path']).read_text(encoding='utf-8-sig'))
        metadata[r['family']]={'title':info.get('_songName'),'artist':info.get('_songAuthorName'),'fit_role':r['fit_role'],
                               'audio_sha256':r['sources']['audio']['sha256'],'audio_path':r['sources']['audio']['path']}
        del data
    scores=[];flagged=[];fingerprints={}
    for i,a in enumerate(records):
        for b in records[i+1:]:
            if a['fit_role']==b['fit_role']:continue
            fa,fb=a['family'],b['family'];peak=peak_correlation(envelopes[fa],envelopes[fb])
            row={'families':[fa,fb],'roles':[a['fit_role'],b['fit_role']],**peak};scores.append(row)
            if peak['correlation'] is None or peak['correlation']<FP_CORR:continue
            for f in (fa,fb):
                if f not in fingerprints:fingerprints[f]=_fingerprint(Path(metadata[f]['audio_path']))
            native=peak_correlation(fingerprints[fa],fingerprints[fb]);confirmed=native['correlation'] is not None and native['correlation']>=FP_CORR
            flagged.append({**row,'metadata':[metadata[fa],metadata[fb]],'independent_original_decode':native,'corroborated_audio_overlap':confirmed})
            print('audio overlap',fa,fb,peak,native,flush=True)
    groups=[]
    for r in flagged:
        if not r['corroborated_audio_overlap']:continue
        group=set(r['families']);others=[]
        for previous in groups:
            if group&previous:group|=previous
            else:others.append(previous)
        groups=others+[group]
    # Merge transitively after all edges, including groups joined by later edges.
    changed=True
    while changed:
        changed=False
        for i in range(len(groups)):
            for j in range(i+1,len(groups)):
                if groups[i]&groups[j]:groups[i]|=groups.pop(j);changed=True;break
            if changed:break
    roles={r['family']:r['fit_role'] for r in records};quarantine={'train':set(),'validation':set()}
    for g in groups:
        held={roles[f] for f in g}
        if held&{'validation','development'}:quarantine['train'].update(f for f in g if roles[f]=='train')
        if 'development' in held:quarantine['validation'].update(f for f in g if roles[f]=='validation')
    report={'identity':frozen['identity'],'pairs_screened':len(scores),'flagged_pairs':flagged,'groups':[sorted(g) for g in groups],
        'scores':scores,'quarantine':{k:sorted(v) for k,v in quarantine.items()},
        'claim':'cross-role near-copy audio screen with independent original-file envelope corroboration; not exhaustive recording identity or quality proof'}
    _atomic(OUT/'report.json',report)
    _atomic(OUT/'quarantine.json',{'identity':frozen['identity'],'report_sha256':_sha(OUT/'report.json'),'exclude_from_future_fit_roles':report['quarantine'],
        'scope':'future experiment-owned inventories only; old models and B0 retain their historical exposure; no old artifact altered'})
    return {k:v for k,v in report.items() if k!='scores'}


if __name__=='__main__':print(json.dumps(run(),indent=1))
