"""Matched approved24/approved45 retrieval with frozen quarantined similarity moments."""
import json
import time
from unittest.mock import patch

import numpy as np
import torch

from eval import joint_quarantine as quarantine,joint_approved as approved,joint_recovered_inventory as recovered
from eval import joint_compatible as compatible,joint_joins as native,joint_workload as workload,joint_ordered as ordered,joint_pilot as old
from eval.expressive_manifest import freeze_run
from eval.joint_model import _torch_save
from eval.joint_phrase import ROOT,_atomic,_sha
from eval.joint_export import read_chart
from eval.joint_continuity import describe

OUT=ROOT/'experiments/joint-phrase-v14/expanded-approved'
SPEC=ROOT/'docs/specs/2026-09-30-expanded-approved-pool-design.md'


def make_banks(base,approved_families,new_data,roles):
    original=approved.subset_bank(base,approved_families)
    if any(d['fit_role']!='train' or roles[d['family']]['future_role']!='train' for d in new_data):raise ValueError('nontraining recovered donor')
    if {d['family'] for d in new_data}&{e['family'] for e in base['entries']}:raise ValueError('recovered donor already in base')
    added=native.enrich_bank(workload.corrected_bank(old.retrieval_bank(new_data)),{d['family']:d['source'] for d in new_data})
    raw=np.stack([e['descriptor'] for e in added['entries']])
    expanded={**original,'entries':original['entries']+added['entries'],
              'x':np.concatenate([original['x'],(raw-original['mean'])/original['sd']])}
    return {'approved24':original,'ordered':expanded}


def freeze_experiment():
    parent,bank=quarantine.prepare();recovered.prepare()
    inventory=json.loads((recovered.OUT/'inventory.json').read_text())
    if inventory['report_sha256']!=_sha(recovered.OUT/'report.json'):raise ValueError('recovered inventory report changed')
    return freeze_run(OUT,{'parent_identity':parent['identity'],'bank_receipt':bank,'inventory_sha256':_sha(recovered.OUT/'inventory.json'),
        'recovered_report_sha256':inventory['report_sha256'],'spec_sha256':_sha(SPEC),'code_sha256':_sha(__file__),
        'approved_helper_sha256':_sha(approved.__file__),'excluded_train_families':parent['config']['excluded_train_families'],
        'quarantined_control_report_sha256':_sha(quarantine.OUT/'report.json'),'historical_approved_report_sha256':_sha(approved.OUT/'report.json'),
        'historical_approved_artifacts':{str(p.relative_to(ROOT)):_sha(p) for p in (approved.OUT/'development').rglob('*')
            if p.is_file() and p.name in ('song.json','record.json','ExpertPlus.dat','Info.dat')}})


def bank_inputs():
    base=torch.load(quarantine.OUT/'retrieval.pt',weights_only=False)
    records=json.loads((old.OUT/'run.json').read_text())['config']['records']
    families={r['family'] for r in approved.approved_records(records)}
    inventory=json.loads((recovered.OUT/'inventory.json').read_text());data=[]
    for family in sorted(inventory['new_payload_receipts']):
        if inventory['roles']['sources'][family]['future_role']!='train':continue
        d=torch.load(recovered.OUT/'data'/(family.replace(':','_')+'.pt'),weights_only=False)
        data.append({**d,'fit_role':'train'})
    if len(families)!=24 or len(data)!=21:raise ValueError('approved comparison inventory changed')
    return base,families,data,inventory['roles']['sources']


def prepare():
    frozen=freeze_experiment();rp=OUT/'bank-receipts.json'
    if not rp.exists():
        banks=make_banks(*bank_inputs());receipts={}
        for arm,bank in banks.items():
            path=OUT/(arm+'.pt');_torch_save(path,bank)
            receipts[arm]={'sha256':_sha(path),'entries':len(bank['entries']),'families':len({e['family'] for e in bank['entries']})}
        _atomic(rp,{'identity':frozen['identity'],'banks':receipts})
    receipt=json.loads(rp.read_text())
    if receipt['identity']!=frozen['identity'] or any(_sha(OUT/(arm+'.pt'))!=r['sha256'] for arm,r in receipt['banks'].items()):raise ValueError('approved comparison bank changed')
    return frozen,receipt


def decide(songs,frozen,controls,evidence):
    with patch.object(quarantine,'OUT',OUT):report=quarantine.decide(songs,frozen,controls,evidence)
    report['approved_coverage_ablation']=report.pop('quarantine_ablation')
    for d in report['approved_coverage_ablation'].values():
        d['approved24_mean']=d.pop('overlap_exposed_control_mean');d['approved45_mean']=d.pop('quarantined_bank_mean')
    report['arm_semantics'].update(ordered='approved45 with recovered train sources',approved24='matched original approved24 with the same fixed v12 moments')
    report['quarantine_contract']['similarity_normalizers']='fixed v12 moments for both arms; no refit on added sources'
    report['matched_control']={'nonfallback':sum(not s['arms']['approved24']['fallback'] for s in songs),
        'complete_attempts':sum(r['ok'] for s in songs for r in s['arms']['approved24']['attempts']),
        'generation_seconds':sum(r['elapsed_s'] for s in songs for r in s['arms']['approved24']['attempts'])}
    report['approval_provenance']={'control_sources':24,'expanded_sources':45,'recovered_train_sources':21,'new_validation_sources_used':0,
                                  'scope':'original approval and canonical equality proofs; assembled outputs are not human-approved'}
    report['donor_diversity']={arm:{s['family']:{'unique_phrases':len(set(map(tuple,s['arms'][arm]['selected'].get('donors',[])))),
        'native_continuations':sum(a[0]==b[0] and a[1]+8==b[1] for a,b in zip(s['arms'][arm]['selected'].get('donors',[]),s['arms'][arm]['selected'].get('donors',[])[1:]))}
        for s in songs} for arm in ('approved24','ordered')}
    report['budget']='six approved24 and six approved45 attempts per family (96 total); fixed moments/search; no fit or validation outputs'
    _atomic(OUT/'report.json',report);return report


def run(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700);torch.set_num_threads(4)
    frozen,receipt=prepare();banks={a:torch.load(OUT/(a+'.pt'),weights_only=False) for a in ('approved24','ordered')}
    scales=old.evaluation_freeze()['config']['scales'];kit=old.machine_tools();songs=[];controls={};evidence={}
    for family,item in ordered.inputs().items():
        key=family.replace(':','_');dest=OUT/'development'/key
        prior=json.loads((quarantine.OUT/'development'/key/'song.json').read_text());arms={};audits_by_arm={}
        for arm,bank in banks.items():
            records=[];audits=[]
            for seed in range(6):
                target=dest/arm/str(seed);rec=old.cached_record(target/'record.json',frozen['identity'])
                if rec is None:
                    if time.monotonic()>=deadline:return {'status':'INCOMPLETE','family':family,'arm':arm,'next_seed':seed,'families':len(songs)}
                    started=time.monotonic();attempt=compatible.retrieve(item['source'],item['data']['audio'],bank,item['rate'],seed);attempt['elapsed_s']=time.monotonic()-started
                    previous=approved.OUT/'development'/key/'ordered'/str(seed)/'record.json' if arm=='approved24' else dest/'approved24'/str(seed)/'record.json'
                    rec=ordered.measure(attempt,item['data'],target,frozen['identity'],scales,kit,previous)
                    print(f'approved coverage {family} {arm} s{seed}: {rec["ok"]} {rec.get("machine",{}).get("verdict")}',flush=True)
                audit=None
                if rec['ok']:
                    src=read_chart(target/'ExpertPlus.dat',target/'Info.dat');src['duration_beats']=item['source']['duration_beats']
                    audit=compatible.audit_ledger(src,rec['donors'],bank)
                    if audit['compressed'] or audit['unknown'] or audit['nonterminal_unknown_exits']:raise ValueError('native context contract failed')
                records.append(rec);audits.append(audit)
            arms[arm]=ordered.select(records,item['b0']);audits_by_arm[arm]=audits
        song={**prior,'identity':frozen['identity'],'arms':{'retrieval':prior['arms']['retrieval'],**arms}}
        _atomic(dest/'song.json',song);songs.append(song)
        controls[family]={**prior,'arms':{**prior['arms'],'ordered':arms['approved24']}}
        sources={}
        for arm in banks:
            chosen=arms[arm];p=old.PILOT/'development'/key/'b0' if chosen['fallback'] else dest/arm/str(chosen['selected_seed'])
            sources[arm]=read_chart(p/'ExpertPlus.dat',p/'Info.dat');sources[arm]['duration_beats']=item['source']['duration_beats']
        evidence[family]={'attempts':audits_by_arm['ordered'],'selected_seed':arms['ordered']['selected_seed'],
            'approved24_attempts':audits_by_arm['approved24'],
            'matched_approved24_control_audit':None if arms['approved24']['fallback'] else compatible.audit_ledger(sources['approved24'],arms['approved24']['selected']['donors'],banks['approved24']),
            'continuity_control':describe(sources['approved24']),'continuity_candidate':describe(sources['ordered'])}
    prepare();return decide(songs,frozen,controls,evidence)


if __name__=='__main__':print(json.dumps(run(),indent=1))
