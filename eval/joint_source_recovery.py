"""Recover approved sources only through already-bound, independently matching files."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from eval.corpus import _fingerprint,FP_CORR
from eval.expressive_manifest import APPROVED_ROOTS,freeze_run
from eval.joint_phrase import ROOT,_atomic,_sha,read_source,verify_sources,UnsupportedSource
from eval.joint_export import export_chart,assert_same
from eval.visibility import resolve_authored

OUT=ROOT/'experiments/joint-approved-source-recovery-v1'
SPEC=ROOT/'docs/specs/2026-09-30-approved-source-recovery-design.md'
READINESS=ROOT/'experiments/joint-phrase-v1/readiness'


def aligned_audio(approved,canonical,fingerprints):
    a=sf.info(approved['path']);b=sf.info(canonical['path'])
    duration_delta=abs(a.duration-b.duration);tolerance=1/a.samplerate+1/b.samplerate
    if approved['sha256']==canonical['sha256']:
        return {'ok':True,'kind':'exact_audio_bytes','duration_delta_s':duration_delta}
    if duration_delta>tolerance:
        return {'ok':False,'kind':'duration_mismatch','duration_delta_s':duration_delta,'tolerance_s':tolerance}
    for record in (approved,canonical):
        if record['sha256'] not in fingerprints:fingerprints[record['sha256']]=_fingerprint(Path(record['path']))
    x,y=(fingerprints[r['sha256']] for r in (approved,canonical));n=min(len(x),len(y))
    corr=float(np.corrcoef(x[:n],y[:n])[0,1]) if n>=1500 and np.std(x[:n])>1e-12 and np.std(y[:n])>1e-12 else None
    return {'ok':corr is not None and corr>=FP_CORR,'kind':'zero_lag_transcode_corroboration',
            'correlation':corr,'overlap_s':n/50,'duration_delta_s':duration_delta,'tolerance_s':tolerance}


def counterpart_proof(approved,candidate,fingerprints):
    verify_sources(approved);verify_sources(candidate)
    if approved['family']!=candidate['family']:return {'ok':False,'reason':'different_family'}
    if approved['sources']['chart']['sha256']!=candidate['sources']['chart']['sha256']:return {'ok':False,'reason':'different_chart_bytes'}
    infos=[json.loads(Path(r['sources']['info']['path']).read_text(encoding='utf-8-sig')) for r in (approved,candidate)]
    if infos[0]!=infos[1]:return {'ok':False,'reason':'different_info_content'}
    binding=resolve_authored(candidate['sources']['chart']['path'],candidate['sources']['chart']['sha256'])
    if binding.get('status')!='verified' or binding.get('difficulty')!='ExpertPlus' or binding.get('rank')!=9:
        return {'ok':False,'reason':'canonical_binding_unverified','binding':binding}
    audio=aligned_audio(approved['sources']['audio'],candidate['sources']['audio'],fingerprints)
    return {'ok':audio['ok'],'reason':None if audio['ok'] else 'audio_alignment_unverified','audio':audio,
            'chart_bytes_identical':True,'parsed_info_identical':True,'binding':binding}


def inventory():
    ready=json.loads((READINESS/'run.json').read_text());corpus=json.loads((ROOT/'eval/corpus_manifest.json').read_text());rows=[]
    for r in ready['config']['records']:
        p=json.loads((READINESS/'partial'/(r['family'].replace(':','_')+'.json')).read_text())
        if p.get('reason')!='authored_binding:no_info_entry_binds_this_filename':continue
        if not r['approved'] or not any(root in r['sources']['chart']['path'] for root in APPROVED_ROOTS):raise ValueError('recovery scope is not approved provenance')
        verify_sources(r);info=json.loads(Path(r['sources']['info']['path']).read_text(encoding='utf-8-sig'))
        entries=[e for s in info.get('_difficultyBeatmapSets',[]) if s.get('_beatmapCharacteristicName')=='Standard' for e in s.get('_difficultyBeatmaps',[]) if e.get('_difficulty')=='ExpertPlus']
        if len(entries)!=1 or entries[0]['_beatmapFilename']!='ExpertPlusStandard.dat' or Path(r['sources']['chart']['path']).name!='ExpertPlus.dat':raise ValueError('unexpected alias inventory')
        candidates=[]
        for m in corpus['maps']:
            if m['family']!=r['family']:continue
            directory=Path(m['dir']);chart=directory/'ExpertPlusStandard.dat'
            if not chart.exists():continue
            ip=next((p for p in directory.iterdir() if p.name.lower()=='info.dat'),None)
            ap=directory/m['audio_file'] if m.get('audio_file') else None
            if ip is None or ap is None or not ap.exists():continue
            sources={name:{'path':str(path),'sha256':_sha(path)} for name,path in (('chart',chart),('info',ip),('audio',ap))}
            candidates.append({**r,'sources':sources})
        rows.append({'approved':r,'candidates':sorted(candidates,key=lambda c:c['sources']['chart']['path'])})
    if len(rows)!=53:raise ValueError('approved recovery inventory changed')
    return ready,rows


def run():
    ready,rows=inventory()
    frozen=freeze_run(OUT,{'readiness_identity':ready['identity'],'corpus_sha256':_sha(ROOT/'eval/corpus_manifest.json'),
        'inputs':rows,'code_sha256':_sha(__file__),'spec_sha256':_sha(SPEC),
        'dependencies':{p:_sha(ROOT/p) for p in ('eval/joint_phrase.py','eval/joint_export.py','eval/map_reader.py','eval/visibility.py','eval/corpus.py')},
        'audio_threshold':FP_CORR,'audio_shift_allowed':False,'future_fit_roles_assigned':False})
    results=[];fingerprints={}
    for row in rows:
        approved=row['approved'];family=approved['family'];key=family.replace(':','_')
        proofs=[{'candidate':c,'proof':counterpart_proof(approved,c,fingerprints)} for c in row['candidates']]
        valid=[p for p in proofs if p['proof']['ok']]
        result={'family':family,'approved_record':approved,'counterpart_proofs':proofs,'status':'unresolved','reason':'no_verified_local_counterpart'}
        if valid:
            canonical=valid[0]['candidate'];composite={**approved,'sources':{**canonical['sources'],'audio':approved['sources']['audio']}}
            result.update(composite_record=composite,equivalent_verified_counterparts=len(valid))
            try:
                source=read_source(composite);dest=OUT/'exports'/key
                actual=export_chart(source,dest,njs=source['authored']['njs'],offset=source['authored']['offset_beats'])
                assert_same(source,actual,source['authored']['njs'],source['authored']['offset_beats'])
                _atomic(OUT/'sources'/(key+'.json'),source)
                result.update(status='recovered_supported',reason=None,source_sha256=_sha(OUT/'sources'/(key+'.json')),
                    exports={name:_sha(dest/name) for name in ('ExpertPlus.dat','Info.dat')},roundtrip_exact=True)
            except UnsupportedSource as exc:result.update(status='binding_recovered_other_scope_exclusion',reason=str(exc))
        _atomic(OUT/'partial'/(key+'.json'),result);results.append(result)
    report={'identity':frozen['identity'],'summary':dict(Counter(r['status'] for r in results)),
        'reasons':dict(Counter(r['reason'] for r in results if r['reason'])),'families':results,
        'claim':'approved chart bytes and metadata independently authenticated; approved audio unchanged; no future fit-role assignment or quality claim'}
    _atomic(OUT/'report.json',report);return {k:v for k,v in report.items() if k!='families'}


if __name__=='__main__':print(json.dumps(run(),indent=1))
