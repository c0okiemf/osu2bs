"""Scope of the frozen arc objective; descriptor counterfactuals are not maps."""
from collections import defaultdict
import copy
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from eval import hand_state_geometry_fit as fit,hand_state_geometry_data as data
from eval import phrase_native as native,phrase_native_refine as typed,phrase_native_position as fixed,phrase_native_phase as phase
from eval import joint_pilot as old,joint_uniform_audio as uniform,joint_continuity as continuity,phrase_state as codec
from eval.expressive_manifest import freeze_run
from eval.joint_phrase import ROOT,_sha,_atomic,read_source,UnsupportedSource
from eval.joint_deployment import reference_view

OUT=ROOT/'experiments/arc-objective-audit-v1'
SPEC=ROOT/'docs/specs/2026-09-30-arc-objective-audit-design.md'
ROOTS={'native':native.OUT,'typed':typed.OUT,'position':fixed.OUT,'phase':phase.OUT}
RATE_BINS=((0.,1.),(1.,2.),(2.,4.),(4.,float('inf')))


def zero_arc(profile):
    result=copy.deepcopy(profile)
    for row in result['geometry']:
        if row[1] is not None and np.isfinite(row[1]):row[1]=0.
    return result


def marginal(values):
    x=np.sort(np.asarray(values,float));n=len(x)
    if not n:return {'n':0,'nonzero':0,'median':None,'mean':None,'zero_absolute_loss':None,'iid_pair_absolute_loss':None}
    if not np.isfinite(x).all() or (x<0).any():raise ValueError('invalid arc marginal')
    return {'n':n,'nonzero':int((x>0).sum()),'median':float(np.median(x)),'mean':float(x.mean()),
        'zero_absolute_loss':float(np.abs(x).mean()),
        'iid_pair_absolute_loss':float(2*np.dot(2*np.arange(n)-n+1,x)/(n*n))}


def diagnose(reference,profile,scales):
    original=old.errors(reference,profile,scales);counter=old.errors(reference,zero_arc(profile),scales)
    assert original['rhythm']==counter['rhythm'] and original['geometry']['supported_windows']==counter['geometry']['supported_windows']
    a,b=np.array(reference['geometry'],float)[:,1],np.array(profile['geometry'],float)[:,1]
    valid=np.isfinite(a)&np.isfinite(b);components=original['geometry']['components']
    return {'original':original,'zero_arc_descriptor_only':counter,
        'arc_marginal_wasserstein':float(np.mean(np.abs(np.sort(a[valid])-np.sort(b[valid])))/scales['geometry'][1]) if valid.any() else None,
        'arc_share_of_component_sum':components[1]/sum(components) if all(x is not None for x in components) and sum(components)>0 else None,
        'paired_arc_windows':int(valid.sum()),'reference_arc':marginal(a[valid]),'candidate_arc':marginal(b[valid])}


def variant_inventory(sources):
    manifest=json.loads((ROOT/'eval/corpus_manifest.json').read_text());groups=defaultdict(list)
    for row in manifest['maps']:
        if row['family'] in sources and sources[row['family']]['role']=='train' and row.get('charts',{}).get('ExpertPlus'):
            groups[row['family']].append(row)
    result=[]
    for family,rows in sorted(groups.items()):
        if len({r['charts']['ExpertPlus']['sha256'] for r in rows})<2:continue
        for row in sorted(rows,key=lambda r:r['dir']):
            root=Path(row['dir']);chart=row['charts']['ExpertPlus'];infos=sorted(p for p in root.glob('*') if p.name.lower()=='info.dat')
            paths={'chart':root/chart['file'],'audio':root/row['audio_file']}
            if len(infos)==1:paths['info']=infos[0]
            sources_here={k:{'path':str(p),'sha256':_sha(p)} for k,p in paths.items() if p.is_file()}
            reason=None if len(sources_here)==3 else 'missing_chart_audio_or_unique_info'
            if 'chart' in sources_here and sources_here['chart']['sha256']!=chart['sha256']:reason='metadata_chart_hash_changed'
            result.append({'family':family,'dir':row['dir'],'metadata_chart_sha256':chart['sha256'],
                'sources':sources_here,'inventory_reason':reason})
    return result


def freeze():
    fitted,parent,prepared=fit.freeze();selection=json.loads((fit.OUT/'selection.json').read_text())
    audit=json.loads((fit.OUT/'validation-audit.json').read_text())
    if audit['selection_sha256']!=_sha(fit.OUT/'selection.json') or selection['status']!='VALIDATION_NEGATIVE':raise ValueError('start from audited negative validation')
    files=[ROOT/'eval/corpus_manifest.json',data.DATA/'prepared.json',data.DATA/'artifact-audit.json',
        fit.OUT/'selection.json',fit.OUT/'validation-audit.json',uniform.OUT/'bank-receipt.json']
    for root in ROOTS.values():
        files.extend([root/'report.json',root/'artifact-audit.json']);files.extend(sorted((root/'development').glob('*/song.json')))
    files.extend(sorted((data.DATA/'validation').glob('*/control/*/record.json')))
    files.extend(sorted((fit.OUT/'validation'/str(selection['selected'])).glob('*/*/record.json')))
    scales=old.evaluation_freeze()['config']['scales']
    return freeze_run(OUT,{'fit_identity':fitted['identity'],'data_identity':parent['identity'],'sources':parent['config']['sources'],
        'historical_files':{str(p.relative_to(ROOT)):_sha(p) for p in files},'variants':variant_inventory(parent['config']['sources']),
        'scales':scales,'validation_scales':parent['config']['validation_scales'],'selected':selection['selected'],
        'protected':fitted['config']['protected'],'code_sha256':_sha(__file__),'spec_sha256':_sha(SPEC),
        'dependencies':{p:_sha(ROOT/p) for p in ('eval/joint_pilot.py','eval/joint_phrase.py','eval/joint_deployment.py',
            'eval/expression_profile.py','eval/joint_continuity.py','eval/map_reader.py','eval/visibility.py')}})


def saved_profiles(frozen):
    development={};validation={};scales=frozen['config']['scales']
    for path in sorted((native.OUT/'development').glob('*/song.json')):
        song=json.loads(path.read_text());family=song['family'];key=family.replace(':','_')
        records={'b0':song['b0'],'retrieval':song['arms']['retrieval']['selected']}
        for name,root in ROOTS.items():records[name]=json.loads((root/'development'/key/'song.json').read_text())['arms']['ordered']['selected']
        development[family]={}
        for name,r in records.items():
            d=diagnose(r['reference_profile'],r['profile'],scales);assert d['original']==r['errors'],(family,name)
            development[family][name]=d
    selected=frozen['config']['selected']
    for family,item in frozen['config']['sources'].items():
        if item['role']!='validation':continue
        key=family.replace(':','_');validation[family]={}
        for name,root in [('control',data.DATA/'validation'/key/'control'),('conditional',fit.OUT/'validation'/str(selected)/key)]:
            validation[family][name]=[]
            for seed in range(2):
                r=json.loads((root/str(seed)/'record.json').read_text());assert r['ok']
                d=diagnose(r['reference_profile'],r['profile'],frozen['config']['validation_scales'])
                assert d['original']==r['errors'];validation[family][name].append(d)
    def aggregate(rows):
        return {'geometry_mean':float(np.mean([r['original']['geometry']['mean'] for r in rows])),
            'zero_arc_descriptor_geometry_mean':float(np.mean([r['zero_arc_descriptor_only']['geometry']['mean'] for r in rows])),
            'arc_pointwise_mean':float(np.mean([r['original']['geometry']['components'][1] for r in rows])),
            'arc_marginal_mean':float(np.mean([r['arc_marginal_wasserstein'] for r in rows])),
            'zero_arc_descriptor_improves_rows':sum(r['zero_arc_descriptor_only']['geometry']['mean']<r['original']['geometry']['mean'] for r in rows)}
    dev_aggregate={name:aggregate([rows[name] for rows in development.values()]) for name in ('b0','retrieval',*ROOTS)}
    for name,root in ROOTS.items():
        historical=json.loads((root/'report.json').read_text())['aggregate']['ordered']['geometry']['mean']
        assert np.isclose(dev_aggregate[name]['geometry_mean'],historical,rtol=0,atol=1e-12)
    val_aggregate={name:aggregate([r for rows in validation.values() for r in rows[name]]) for name in ('control','conditional')}
    return {'development':development,'validation':validation,'development_aggregate':dev_aggregate,'validation_aggregate':val_aggregate}


def human_marginals(frozen):
    approved={e['family'] for e in torch.load(uniform.OUT/'retrieval.pt',weights_only=False)['entries']};points={}
    for family,item in frozen['config']['sources'].items():
        if item['role']!='train':continue
        if _sha(ROOT/item['payload'])!=item['payload_sha256']:raise ValueError('human payload changed')
        sample=torch.load(ROOT/item['payload'],weights_only=False);p=old.profile(sample['source'],sample['audio'])
        points[family]=[(r[0],g[1]) for r,g in zip(p['rhythm'],p['geometry'])]
    assert len(points)==336 and len(approved)==45 and approved<=points.keys()
    result={}
    for cohort,families in [('train336',set(points)),('approved45',approved)]:
        rows=[v for f in sorted(families) for v in points[f]];supported=[a for _,a in rows if a is not None]
        buckets={}
        for lo,hi in RATE_BINS:
            values=[a for rate,a in rows if lo<=rate<hi and a is not None]
            buckets[f'{lo:g}..{hi:g}']={'total_windows':sum(lo<=rate<hi for rate,a in rows),**marginal(values)}
        per_family={f:marginal([a for _,a in points[f] if a is not None]) for f in sorted(families)}
        means=[v['mean'] for v in per_family.values() if v['mean'] is not None]
        result[cohort]={'families':len(families),'total_windows':len(rows),'pooled':marginal(supported),'event_rate_bins':buckets,
            'family_mean_arc':float(np.mean(means)),'per_family':per_family}
    return result


def geometry_profile(source):
    # Audio diagnostics are deliberately excluded; these errors use chart geometry.
    audio={'times':np.array([0.,source['duration_beats']*60/source['bpm']]),'rms':np.zeros(2),'onset':np.zeros(2)}
    p=old.profile(source,audio);return {k:p[k] for k in ('rhythm','geometry')}


def human_variants(frozen):
    rows=[];groups=defaultdict(dict)
    for index,entry in enumerate(frozen['config']['variants']):
        row={'index':index,'family':entry['family'],'dir':entry['dir'],'status':entry['inventory_reason']}
        if row['status'] is not None:rows.append(row);continue
        try:source=read_source(entry)
        except (UnsupportedSource,json.JSONDecodeError) as exc:
            row['status']='unsupported';row['reason']=str(exc);rows.append(row);continue
        key=(entry['family'],entry['sources']['audio']['sha256'])
        geometry_id=codec.digest({k:source[k] for k in ('notes','bombs','walls','bpm','duration_beats')})
        row.update(status='supported',geometry_id=geometry_id,audio_sha256=key[1],notes=len(source['notes']))
        if geometry_id in groups[key]:row['geometry_alias_of']=groups[key][geometry_id][0]
        else:groups[key][geometry_id]=(index,source)
        rows.append(row)
    comparisons=[]
    for (family,audio_sha),members in sorted(groups.items()):
        for (ia,a),(ib,b) in itertools.combinations(members.values(),2):
            ab=diagnose(geometry_profile(a),geometry_profile(reference_view(b,a)),frozen['config']['scales'])
            ba=diagnose(geometry_profile(b),geometry_profile(reference_view(a,b)),frozen['config']['scales'])
            comparisons.append({'family':family,'audio_sha256':audio_sha,'source_indices':[ia,ib],
                'b_against_a':ab,'a_against_b':ba,'preference_quality':'unknown; human-authored does not certify either alternative'})
    return {'inventory':rows,'aligned_distinct_pairs':comparisons,'aligned_pair_count':len(comparisons)}


def run():
    torch.set_num_threads(2);frozen=freeze();profiles=saved_profiles(frozen);humans=human_marginals(frozen);variants=human_variants(frozen)
    result={'identity':frozen['identity'],'status':'OBJECTIVE_AUDIT_COMPLETE_NO_RELEASE_DECISION',**profiles,
        'human_marginals':humans,'human_variants':variants,'directional_control':continuity.controls(),
        'scope':'saved-profile and empirical arithmetic diagnosis; zeroed descriptors and reordered marginals are not generated maps or quality labels'}
    for path,sha in frozen['config']['historical_files'].items():assert _sha(ROOT/path)==sha,path
    for path,sha in frozen['config']['protected'].items():assert _sha(ROOT/path)==sha,path
    freeze();path=OUT/'report.json'
    if path.exists():assert json.loads(path.read_text())==codec.canonical(result)
    _atomic(path,result)
    return {k:result[k] for k in ('identity','status','development_aggregate','validation_aggregate')}|{'aligned_human_pairs':variants['aligned_pair_count']}


if __name__=='__main__':print(json.dumps(run(),indent=1))
