"""Paired full-band/common-band self-retrieval diagnostic on corroborated train audio."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from eval import joint_expanded as expanded,joint_recovered_inventory as recovered,joint_pilot as old
from eval.expressive_manifest import freeze_run
from eval.joint_model import audio_features,_torch_save
from eval.joint_phrase import ROOT,_atomic,_sha,verify_sources

OUT=ROOT/'experiments/joint-audio-view-audit-v1'
SPEC=ROOT/'docs/specs/2026-09-30-audio-view-audit-design.md'


def render_common(path,destination):
    import librosa
    y,sr=librosa.load(str(path),sr=14800,mono=True,res_type='soxr_hq')
    destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
    sf.write(destination,y,sr,format='WAV',subtype='FLOAT')
    original=sf.info(path);rendered=sf.info(destination)
    if rendered.samplerate!=14800 or rendered.channels!=1 or abs(original.duration-rendered.duration)>1/original.samplerate+1/14800:raise ValueError('common-band render timing changed')
    return {'sha256':_sha(destination),'samplerate':rendered.samplerate,'channels':rendered.channels,
            'duration_delta_s':rendered.duration-original.duration,'recipe':'librosa soxr_hq mono14800; float WAV; no shift or trim'}


def self_rank(x,query,own):
    distance=np.square(x-query).mean(1);d=distance[own]
    return {'distance':float(d),'best_tied_rank':int(np.sum(distance<d))+1,'worst_tied_rank':int(np.sum(distance<=d)),
            'stable_raw_rank':int(np.flatnonzero(np.argsort(distance,kind='stable')==own)[0])+1}


def prepare():
    parent,banks=expanded.prepare();inventory=json.loads((recovered.OUT/'inventory.json').read_text())
    rows=json.loads((recovered.OUT/'run.json').read_text())['config']['new_sources'];pairs=[]
    for row in rows:
        if inventory['roles']['sources'][row['family']]['future_role']!='train':continue
        proof=next(p for p in row['counterpart_proofs'] if p['proof']['ok'])
        if proof['proof']['audio']['kind']!='zero_lag_transcode_corroboration':continue
        approved=row['approved_record'];canonical=proof['candidate'];verify_sources(approved);verify_sources(canonical)
        if sf.info(approved['sources']['audio']['path']).samplerate!=14800 or sf.info(canonical['sources']['audio']['path']).samplerate<=14800:raise ValueError('unexpected paired audio formats')
        pairs.append({'family':row['family'],'approved':approved,'canonical':canonical,'proof':proof['proof']})
    if len(pairs)!=10:raise ValueError('paired train inventory changed')
    records=json.loads((old.OUT/'run.json').read_text())['config']['records'];census={}
    for name,rs in (('old_approved_train',[r for r in records if r['fit_role']=='train' and r['approved']]),
                    ('old_general_train',[r for r in records if r['fit_role']=='train' and not r['approved']]),
                    ('development',[r for r in records if r['fit_role']=='development'])):
        census[name]=dict(Counter(f"{sf.info(r['sources']['audio']['path']).samplerate}Hz/{sf.info(r['sources']['audio']['path']).channels}ch" for r in rs))
    frozen=freeze_run(OUT,{'parent_identity':parent['identity'],'bank_receipt':banks['banks']['ordered'],'paired_train_sources':pairs,
        'census':census,'code_sha256':_sha(__file__),'spec_sha256':_sha(SPEC),'audio_recipe_sha256':_sha(ROOT/'eval/joint_model.py'),
        'common_band_hz':14800,'max_window_end_seconds':90,'quality_claim':False})
    return frozen,pairs


def run():
    torch.set_num_threads(2);frozen,pairs=prepare();bank=torch.load(expanded.OUT/'ordered.pt',weights_only=False);rows=[];receipts={}
    for pair in pairs:
        family=pair['family'];key=family.replace(':','_');data=torch.load(recovered.OUT/'data'/(key+'.pt'),weights_only=False)
        audio_path=pair['canonical']['sources']['audio']['path'];views={};receipts[family]={}
        for view in ('full_band','common_band'):
            root=OUT/'features'/key;path=root/(view+'.pt');rp=path.with_suffix('.json')
            if not rp.exists():
                root.mkdir(parents=True,exist_ok=True);render=None;input_path=audio_path
                if view=='common_band':
                    input_path=root/'common14800.wav';render=render_common(audio_path,input_path)
                value=audio_features(input_path);_torch_save(path,value)
                _atomic(rp,{'identity':frozen['identity'],'sha256':_sha(path),'render':render})
            receipt=json.loads(rp.read_text())
            if receipt['identity']!=frozen['identity'] or receipt['sha256']!=_sha(path):raise ValueError('paired audio view changed')
            if receipt['render'] and receipt['render']['sha256']!=_sha(root/'common14800.wav'):raise ValueError('common rendering changed')
            views[view]=torch.load(path,weights_only=False);receipts[family][view]=receipt
        for ix,e in enumerate(bank['entries']):
            if e['family']!=family or (e['start']+8)*60/data['source']['bpm']>90:continue
            def query(audio):return (old.phrase_descriptor(audio,e['start'],e['start']+8,data['source']['bpm'],e['descriptor'][-1])-bank['mean'])/bank['sd']
            cached=query(data['audio']);np.testing.assert_array_equal(cached,bank['x'][ix])
            rows.append({'family':family,'start_beat':e['start'],'cached_approved':self_rank(bank['x'],cached,ix),
                         **{name:self_rank(bank['x'],query(value),ix) for name,value in views.items()}})
        print('paired audio view',family,flush=True)
    summaries={}
    for family in sorted({r['family'] for r in rows}):
        selected=[r for r in rows if r['family']==family];summaries[family]={'windows':len(selected)}
        for view in ('full_band','common_band'):
            summaries[family][view]={'mean_self_distance':float(np.mean([r[view]['distance'] for r in selected])),
                'median_best_tied_rank':float(np.median([r[view]['best_tied_rank'] for r in selected])),
                'top_six_share':float(np.mean([r[view]['best_tied_rank']<=6 for r in selected]))}
    aggregate={view:{key:float(np.mean([f[view][key] for f in summaries.values()])) for key in ('mean_self_distance','median_best_tied_rank','top_six_share')}
               for view in ('full_band','common_band')}
    report={'identity':frozen['identity'],'aggregate_family_mean':aggregate,'families':summaries,'windows':rows,'receipts':receipts,
            'claim':'paired TRAIN audio-view consistency against fixed mixed bank; not deployable target-derived matching, quality or candidate selection'}
    _atomic(OUT/'report.json',report);return {k:v for k,v in report.items() if k not in ('windows','receipts')}


if __name__=='__main__':print(json.dumps(run(),indent=1))
