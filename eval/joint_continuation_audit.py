"""Read-only availability/ranking of native continuation at selected donor seams."""
from collections import Counter
import json

import numpy as np
import torch

from eval import joint_compatible as compatible,joint_joins as native,joint_ordered as ordered,joint_pilot as old
from eval.expressive_manifest import freeze_run
from eval.joint_export import read_chart
from eval.joint_phrase import ROOT,_atomic,_sha

OUT=ROOT/'experiments/joint-continuation-audit-v1'
SPEC=ROOT/'docs/specs/2026-09-30-native-continuation-audit-design.md'


def eligibility(entry,start,end,duration,bpm,previous,last,exits):
    notes=[(b+start,h,c,l,d) for b,h,c,l,d in entry['notes'] if b<end-start]
    reasons=[]
    if end<duration and compatible.terminal_only(entry,notes):reasons.append('nonterminal_unknown_exit')
    if not old._join_ok(previous,notes,bpm):reasons.append('direction_join')
    reasons.extend(sorted({c['status'] for c in native.join_checks(last,exits,notes,entry) if c['status']!='pass'}))
    return notes,reasons


def first_six(order,bank,prior,start,end,duration,bpm,previous,last,exits):
    choices=[]
    for ix in order:
        e=bank['entries'][ix]
        if (e['family'],e['start'])==tuple(prior):continue
        _,reasons=eligibility(e,start,end,duration,bpm,previous,last,exits)
        if not reasons:choices.append(int(ix))
        if len(choices)==6:break
    return choices


def run():
    parent=compatible.freeze_experiment();bank=torch.load(native.OUT/'retrieval.pt',weights_only=False)
    inputs=ordered.inputs();lookup={(e['family'],e['start']):i for i,e in enumerate(bank['entries'])}
    artifacts={str(p.relative_to(ROOT)):_sha(p) for p in (compatible.OUT/'development').rglob('*')
               if p.is_file() and p.name in ('song.json','record.json','ExpertPlus.dat','Info.dat')}
    frozen=freeze_run(OUT,{'parent_identity':parent['identity'],'parent_report_sha256':_sha(compatible.OUT/'report.json'),
        'artifacts':artifacts,'inputs':{f:i['proof'] for f,i in inputs.items()},'code_sha256':_sha(__file__),'spec_sha256':_sha(SPEC)})
    rows=[]
    for family,item in inputs.items():
        root=compatible.OUT/'development'/family.replace(':','_');song=json.loads((root/'song.json').read_text())
        chosen=song['arms']['ordered'];assert not chosen['fallback'];record=chosen['selected']
        target=root/'ordered'/str(chosen['selected_seed']);src=read_chart(target/'ExpertPlus.dat',target/'Info.dat')
        src['duration_beats']=item['source']['duration_beats'];compatible.audit_ledger(src,record['donors'],bank)
        last=[None,None];exits=[None,None];prefix=[]
        for i,key in enumerate(record['donors']):
            start=float(i*8);end=min(start+8,src['duration_beats']);entry=bank['entries'][lookup[tuple(key)]]
            previous=compatible.last_directional_groups(prefix)
            if i:
                prior=record['donors'][i-1];query=(old.phrase_descriptor(item['data']['audio'],start,end,src['bpm'],item['rate']*60/src['bpm'])-bank['mean'])/bank['sd']
                distance=np.square(bank['x']-query).mean(1);order=np.argsort(distance,kind='stable')
                ranks=np.empty(len(order),dtype=int);ranks[order]=np.arange(1,len(order)+1)
                selected_ix=lookup[tuple(key)];assert int(ranks[selected_ix])==record['selected_raw_ranks'][i]
                choices=first_six(order,bank,prior,start,end,src['duration_beats'],src['bpm'],previous,last,exits)
                assert choices[chosen['selected_seed']%len(choices)]==selected_ix
                continuation=(prior[0],prior[1]+8);ci=lookup.get(continuation)
                row={'family':family,'start_beat':start,'previous_donor':prior,'selected_donor':key,
                     'exact_native_selected':tuple(key)==continuation,'source_family_changed':prior[0]!=key[0],
                     'selected_raw_rank':int(ranks[selected_ix]),'selected_distance':float(distance[selected_ix]),'native_available':ci is not None}
                if ci is not None:
                    _,reasons=eligibility(bank['entries'][ci],start,end,src['duration_beats'],src['bpm'],previous,last,exits)
                    row.update(native_donor=continuation,native_raw_rank=int(ranks[ci]),native_distance=float(distance[ci]),
                               native_rejections=reasons,native_in_first_six=ci in choices,native_eligible=not reasons,
                               distance_delta=float(distance[ci]-distance[selected_ix]))
                rows.append(row)
            notes=[(b+start,h,c,l,d) for b,h,c,l,d in entry['notes'] if b<end-start]
            native.advance(last,exits,notes,entry);prefix.extend(notes)
        assert sorted(prefix)==sorted(map(tuple,src['notes']))
    available=[r for r in rows if r['native_available']];eligible=[r for r in available if r['native_eligible']]
    def summary(rs):
        av=[r for r in rs if r['native_available']]
        return {'boundaries':len(rs),'native_available':len(av),'native_eligible':sum(r['native_eligible'] for r in av),
                'native_in_first_six':sum(r['native_in_first_six'] for r in av),
                'native_selected':sum(r['exact_native_selected'] for r in rs),'source_changes':sum(r['source_family_changed'] for r in rs),
                'native_rejections':dict(Counter(reason for r in av for reason in r['native_rejections']))}
    result={'identity':frozen['identity'],'summary':summary(rows),'families':{f:summary([r for r in rows if r['family']==f]) for f in inputs},
        'eligible_native_raw_rank_quantiles':np.quantile([r['native_raw_rank'] for r in eligible],[.25,.5,.75,.9,1]).tolist(),
        'eligible_native_distance_delta_quantiles':np.quantile([r['distance_delta'] for r in eligible],[.25,.5,.75,.9,1]).tolist(),
        'rows':rows,'claim':'descriptive native-continuation availability; all actual raw ranks, choices and exports reproduced; no quality decision'}
    _atomic(OUT/'report.json',result);return {k:v for k,v in result.items() if k!='rows'}


if __name__=='__main__':print(json.dumps(run(),indent=1))
