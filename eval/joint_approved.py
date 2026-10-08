"""Isolate existing approved-source provenance in compatible literal retrieval."""
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import torch

from eval import joint_compatible as compatible,joint_joins as native,joint_ordered as ordered,joint_pilot as old
from eval.expressive_manifest import APPROVED_ROOTS,REJECTED_ROOTS,freeze_run
from eval.joint_model import _torch_save
from eval.joint_phrase import ROOT,_atomic,_sha
from eval.joint_export import read_chart
from eval.joint_continuity import describe

OUT=ROOT/'experiments/joint-phrase-v11/approved-donors'
SPEC=ROOT/'docs/specs/2026-09-30-approved-donor-pool-design.md'


def approved_records(records):
    result=[]
    for r in records:
        if r['fit_role']!='train' or not r['approved']:continue
        path=str(Path(r['sources']['chart']['path']).resolve())
        if not any(root in path for root in APPROVED_ROOTS) or any(root in path for root in REJECTED_ROOTS):
            raise ValueError('approved family lacks canonical approved chart provenance')
        result.append(r)
    return result


def subset_bank(bank,families):
    indices=[i for i,e in enumerate(bank['entries']) if e['family'] in families]
    if not indices or {bank['entries'][i]['family'] for i in indices}!=set(families):
        raise ValueError('approved donor inventory mismatch')
    result={**bank,'entries':[bank['entries'][i] for i in indices]}
    for name in ('x','temporal_x'):
        if name in bank:result[name]=bank[name][indices].copy()
    return result


def freeze_experiment():
    parent=compatible.freeze_experiment();report=json.loads((compatible.OUT/'report.json').read_text())
    if report['identity']!=parent['identity'] or not report['gates']['complete_panel']:raise ValueError('compatible control incomplete')
    records=approved_records(json.loads((old.OUT/'run.json').read_text())['config']['records'])
    if len(records)!=24:raise ValueError('approved source inventory changed')
    for r in records:
        for src in r['sources'].values():
            if _sha(src['path'])!=src['sha256']:raise ValueError('approved source changed')
    return freeze_run(OUT,{'parent_identity':parent['identity'],'spec_sha256':_sha(SPEC),'code_sha256':_sha(__file__),
            'provenance_code_sha256':_sha(ROOT/'eval/expressive_manifest.py'),'approved_records':records,
            'control_report_sha256':_sha(compatible.OUT/'report.json'),
            'control_artifacts':{str(p.relative_to(ROOT)):_sha(p) for p in (compatible.OUT/'development').rglob('*')
                if p.is_file() and p.name in ('song.json','record.json','ExpertPlus.dat','Info.dat')}})


def prepare():
    frozen=freeze_experiment();path=OUT/'retrieval.pt';rp=OUT/'retrieval-receipt.json'
    if not rp.exists():
        bank=subset_bank(torch.load(native.OUT/'retrieval.pt',weights_only=False),
                         {r['family'] for r in frozen['config']['approved_records']})
        _torch_save(path,bank);_atomic(rp,{'identity':frozen['identity'],'sha256':_sha(path),'entries':len(bank['entries'])})
    receipt=json.loads(rp.read_text())
    if receipt['identity']!=frozen['identity'] or receipt['sha256']!=_sha(path):raise ValueError('approved bank changed')
    return frozen,receipt


def decide(songs,frozen,controls,evidence):
    with patch.object(compatible,'OUT',OUT):report=compatible.decide(songs,frozen,controls,evidence)
    report['approved_pool_ablation']=report.pop('compatibility_search_ablation')
    for d in report['approved_pool_ablation'].values():
        d['full_pool_control_mean']=d.pop('native_shortlist_control_mean')
        d['approved_pool_mean']=d.pop('compatible_search_mean')
    report['arm_semantics']['ordered']='approved-source pool; unchanged compatible search, workload units and native joins'
    families={r['family'] for r in frozen['config']['approved_records']}
    donors=[key for s in songs for r in s['arms']['ordered']['attempts'] for key in r['donors']]
    report['donor_provenance']={'approved_fit_train_families':len(families),'phrase_selections':len(donors),
        'all_selected_donors_approved':bool(donors) and all(key[0] in families for key in donors),
        'scope':'source approval only; assembled charts have no human approval; general sources are unlabeled'}
    report['budget']='six new attempts per family; cached v10 full-pool, v7 original retrieval and B0 controls; no fit'
    _atomic(OUT/'report.json',report);return report


def run(deadline_seconds=2700):
    deadline=time.monotonic()+min(deadline_seconds,2700);torch.set_num_threads(4)
    frozen,receipt=prepare();bank=torch.load(OUT/'retrieval.pt',weights_only=False)
    full=torch.load(native.OUT/'retrieval.pt',weights_only=False)
    scales=old.evaluation_freeze()['config']['scales'];kit=old.machine_tools();songs=[];controls={};evidence={}
    for family,item in ordered.inputs().items():
        key=family.replace(':','_');dest=OUT/'development'/key;root=compatible.OUT/'development'/key
        control=json.loads((root/'song.json').read_text());controls[family]=control;records=[];audits=[]
        for seed in range(6):
            target=dest/'ordered'/str(seed);rec=old.cached_record(target/'record.json',frozen['identity'])
            if rec is None:
                if time.monotonic()>=deadline:return {'status':'INCOMPLETE','family':family,'next_seed':seed,'families':len(songs)}
                started=time.monotonic();attempt=compatible.retrieve(item['source'],item['data']['audio'],bank,item['rate'],seed)
                attempt['elapsed_s']=time.monotonic()-started
                rec=ordered.measure(attempt,item['data'],target,frozen['identity'],scales,kit,root/'ordered'/str(seed)/'record.json')
                print(f'approved pool {family} s{seed}: {rec["ok"]} {rec.get("machine",{}).get("verdict")}',flush=True)
            audit=None
            if rec['ok']:
                src=read_chart(target/'ExpertPlus.dat',target/'Info.dat');src['duration_beats']=item['source']['duration_beats']
                audit=compatible.audit_ledger(src,rec['donors'],bank)
                if audit['compressed'] or audit['unknown'] or audit['nonterminal_unknown_exits']:raise ValueError('native context contract failed')
            records.append(rec);audits.append(audit)
        chosen=ordered.select(records,item['b0']);song={**control,'identity':frozen['identity'],
                    'arms':{'retrieval':control['arms']['retrieval'],'ordered':chosen}}
        _atomic(dest/'song.json',song);songs.append(song)
        before=control['arms']['ordered'];prevpath=old.PILOT/'development'/key/'b0' if before['fallback'] else root/'ordered'/str(before['selected_seed'])
        prev=read_chart(prevpath/'ExpertPlus.dat',prevpath/'Info.dat');prev['duration_beats']=item['source']['duration_beats']
        newpath=old.PILOT/'development'/key/'b0' if chosen['fallback'] else dest/'ordered'/str(chosen['selected_seed'])
        evidence[family]={'attempts':audits,'selected_seed':chosen['selected_seed'],
             'full_pool_control_audit':None if before['fallback'] else compatible.audit_ledger(prev,before['selected']['donors'],full),
             'continuity_control':describe(prev),'continuity_candidate':describe(read_chart(newpath/'ExpertPlus.dat',newpath/'Info.dat'))}
    prepare();return decide(songs,frozen,controls,evidence)


if __name__=='__main__':print(json.dumps(run(),indent=1))
