"""Supplement corpus held-outs with the exact eight generated-map panel recordings."""
import hashlib
import time
from eval.corpus_scale import ROOT,OUT,read,envelope,_sha,_atomic
from eval.joint_audio_identity import peak_correlation

def main():
    panel=read(ROOT/'eval/clean_rhythm_panel.json')['entries']
    refs=[]
    for e in panel:
        assert _sha(ROOT/e['audio'])==e['audio_sha256']
        refs.append((e['song'],envelope(ROOT/e['audio'])))
    path=OUT/'panel-audio-progress.json'
    done=read(path) if path.exists() else {}
    while True:
        manifest=read(OUT/'manifest.json')
        todo=[r for r in manifest['families'] if r['id'] not in done]
        for r in todo:
            dest=OUT/'maps'/r['id']
            ip=next(p for p in dest.iterdir() if p.name.lower()=='info.dat');info=read(ip)
            name=info.get('_songFilename') or info.get('audio',{}).get('songFilename')
            ap=dest/name if name and (dest/name).is_file() else next(p for p in dest.iterdir() if p.suffix.lower() in ('.ogg','.egg'))
            env=envelope(ap)
            scores=[(peak_correlation(env,ref)['correlation'],name) for name,ref in refs]
            score,name=max((v,n) for v,n in scores if v is not None)
            done[r['id']]=dict(max_correlation=score,song=name,excluded=score>=.95,audio_sha256=_sha(ap))
            if len(done)%25==0:
                _atomic(path,done);print('panel recordings screened',len(done),flush=True)
        _atomic(path,done)
        if manifest['status']!='ACQUIRING' and (OUT/'prepared.json').exists(): break
        time.sleep(20)
    # The loader loads every Standard difficulty. A family with a plain chart
    # can also contain modded charts, so conservatively exclude that family.
    modded={r['id'] for r in manifest['families'] if any(
        d.get('me') or d.get('ne') for d in r['metadata']['versions'][0]['diffs']
        if d.get('characteristic')=='Standard')}
    prepared=read(OUT/'prepared.json')
    ready=[read(p) for p in (OUT/'tensors').glob('*.json')]
    ready=[r for r in ready if r['status']=='READY' and not done[r['id']]['excluded']
           and r['id'] not in modded]
    ready.sort(key=lambda r:hashlib.sha256(('corpus-scale-20261002:'+r['id']).encode()).hexdigest())
    prepared.update(families=ready[:2000],available_families=len(ready),status='READY' if len(ready)>=2000 else 'SHORTFALL')
    excluded={k:v for k,v in done.items() if v['excluded']}
    prepared['panel_audio_exclusions']=excluded
    prepared['modded_family_exclusions']=sorted(modded)
    _atomic(OUT/'prepared.json',prepared)
    report=dict(status='PASS' if prepared['status']=='READY' else 'SHORTFALL',screened=len(done),
                excluded=excluded,modded_family_exclusions=sorted(modded),
                prepared_sha256=_sha(OUT/'prepared.json'),panel=panel,
                method='first 90 seconds, 50Hz RMS, lag +/-30s, >=30s overlap, Pearson >=.95')
    _atomic(OUT/'panel-audio-audit.json',report)
    print(report,flush=True)

if __name__=='__main__': main()
