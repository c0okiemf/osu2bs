"""One fixed causal-likelihood selector on unchanged native-budget candidates."""
import json
import math
from pathlib import Path
import tempfile
from unittest.mock import patch

import torch

from eval import phrase_native as parent,joint_diverse_fit as fitted,joint_uniform_audio as uniform,joint_likelihood as likelihood,joint_ordered as ordered,joint_pilot as old
from eval.expressive_manifest import freeze_run
from eval.joint_model import JointModel
from eval.joint_export import read_chart
from eval.joint_phrase import ROOT,_sha,_atomic

OUT=ROOT/'experiments/phrase-native-likelihood-v1'
SPEC=ROOT/'docs/specs/2026-09-30-budget-native-likelihood-design.md'


def freeze():
    native=parent.freeze();selection=json.loads((fitted.OUT/'selection.json').read_text())
    checkpoint=fitted.OUT/f"snapshot-{selection['selected']}.pt"
    native_audit=json.loads((parent.OUT/'artifact-audit.json').read_text());model_audit=json.loads((fitted.OUT/'artifact-audit.json').read_text())
    if native_audit['repeat_report_sha256']!=_sha(parent.OUT/'report.json') or model_audit['repeat_report_sha256']!=_sha(fitted.OUT/'report.json'):raise ValueError('parent not audited')
    if selection['selected']!=3000 or selection['snapshot_sha256']!=_sha(checkpoint):raise ValueError('fixed scorer changed')
    paths={str(p.relative_to(ROOT)):_sha(p) for p in (parent.OUT/'development').rglob('*')
        if p.is_file() and p.name in ('record.json','song.json','ExpertPlus.dat','Info.dat')}
    return freeze_run(OUT,{'native_identity':native['identity'],'parent_report_sha256':_sha(parent.OUT/'report.json'),
        'parent_audit_sha256':_sha(parent.OUT/'artifact-audit.json'),'model_audit_sha256':_sha(fitted.OUT/'artifact-audit.json'),
        'model_selection_sha256':_sha(fitted.OUT/'selection.json'),'checkpoint':str(checkpoint.relative_to(ROOT)),
        'checkpoint_sha256':_sha(checkpoint),'code_sha256':_sha(__file__),'spec_sha256':_sha(SPEC),'artifacts':paths,
        'query_views':native['config']['views'],'protected':native['config']['protected'],
        'negative_geometry_predictor_sha256':_sha(ROOT/'experiments/phrase-geometry-pilot-v1/report.json'),
        'dependencies':{p:_sha(ROOT/p) for p in ('eval/joint_likelihood.py','eval/joint_model.py','eval/joint_ordered.py','eval/joint_pilot.py','eval/joint_export.py')}})


def run():
    torch.set_num_threads(2);frozen=freeze();model=JointModel().eval()
    model.load_state_dict(torch.load(ROOT/frozen['config']['checkpoint'],map_location='cpu',weights_only=True))
    songs=[];all_scores={};before={};original_report=json.loads((parent.OUT/'report.json').read_text())
    for family,item in ordered.inputs().items():
        key=family.replace(':','_');root=parent.OUT/'development'/key;prior=json.loads((root/'song.json').read_text())
        audio=torch.load(uniform.OUT/'audio'/key/'features.pt',weights_only=False);records=[];scores=[]
        for seed in range(6):
            path=root/'ordered'/str(seed);r=old.cached_record(path/'record.json',frozen['config']['native_identity']);assert r is not None
            score=None
            if r['ok']:
                source=read_chart(path/'ExpertPlus.dat',path/'Info.dat');source['duration_beats']=item['source']['duration_beats']
                score=likelihood.geometry_nll(model,source,audio,chunk_size=256)
                assert score['tokens']==len(source['notes']) and math.isfinite(score['mean_nll'])
            records.append(r);scores.append(score)
        seed,chosen=likelihood.select(records,scores,item['b0'])
        selected={'selected_seed':seed,'fallback':seed is None,'selected':chosen,'attempts':records}
        assert seed is None or (records[seed]['machine']['admitted'] and seed==min(
            (s['mean_nll'],i) for i,(r,s) in enumerate(zip(records,scores)) if r['ok'] and r['machine']['admitted'])[1])
        song={**prior,'identity':frozen['identity'],'arms':{'retrieval':prior['arms']['retrieval'],'ordered':selected}}
        dest=OUT/'development'/key/'song.json'
        if dest.exists():assert json.loads(dest.read_text())==song
        _atomic(dest,song);songs.append(song)
        all_scores[family]={'scores':scores,'first_admitted_seed':prior['arms']['ordered']['selected_seed'],'selected_seed':seed}
        before[family]=prior['arms']['ordered']['selected']
        print('native likelihood',family,'seed',seed,flush=True)
    saved=json.loads((OUT/'report.json').read_text()) if (OUT/'report.json').exists() else None
    with tempfile.TemporaryDirectory(prefix='osu2bs-likelihood-report-') as temporary:
        with patch.object(ordered,'OUT',Path(temporary)):report=ordered.decide(songs,frozen)
    report.update(selection='minimum mean raw geometry NLL among admitted; earliest seed tie',
        scores=all_scores,inherited_candidate_pool=str(parent.OUT.relative_to(ROOT)),new_generation_attempts=0,new_optimizer_updates=0,new_qa_writes=0,
        first_admitted_aggregate=original_report['aggregate']['ordered'],
        paired_changes={axis:{s['family']:s['arms']['ordered']['selected']['errors'][axis]['mean']-before[s['family']]['errors'][axis]['mean'] for s in songs}
            for axis in ('rhythm','geometry')},
        inherited_exposure='v19 scorer fit336 eligible train families; checkpoint selected on six opened validation families; production QA/B0 historical exposure unchanged',
        budget='48 existing native candidates; one fixed checkpoint/scoring rule; no new maps, QA or fit')
    if saved is not None:assert saved==report
    _atomic(OUT/'report.json',report);freeze()
    _atomic(OUT/'artifact-audit.json',{'identity':frozen['identity'],'literal_notes_scored':sum(s['tokens'] for r in all_scores.values() for s in r['scores'] if s),
        'all_48_scores_recomputed':True,'protected_production_unchanged':True,'repeat_verified':saved is not None,
        'report_sha256':_sha(OUT/'report.json'),'code_sha256':_sha(__file__)})
    return {k:report[k] for k in ('identity','status','gates','aggregate','paired_changes')}


if __name__=='__main__':print(json.dumps(run(),indent=1))
