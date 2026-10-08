"""Read-only audit of same-hand dots bridged by the frozen center-path arc proxy."""
import bisect
import json

from eval import expression_profile as expression,joint_pilot as old,joint_ordered as ordered,joint_compatible as v10,joint_approved as v11
from eval.expressive_manifest import freeze_run
from eval.joint_deployment import reference_view
from eval.joint_export import read_chart
from eval.joint_phrase import ROOT,_atomic,_sha,phrase_windows

OUT=ROOT/'experiments/joint-arc-dot-audit-v1'
SPEC=ROOT/'docs/specs/2026-09-30-arc-dot-audit-design.md'


def window_ledger(notes,times):
    p=expression.profile(notes,times)
    if p['status']!='ok':return {'supported':False,'legacy_runs':0,'dot_bridged_runs':0,'heads':sum(n[3]!=8 for n in notes)}
    heads=expression._heads(notes,times);t0=heads[0][0]
    heads=[(t-t0,h,c,l,d) for t,h,c,l,d in heads]
    runs=0;bridged=0;examples=[]
    for hand in (0,1):
        seq=[n for n in heads if n[1]==hand]
        dots=sorted(t-t0 for n,t in zip(notes,times) if n[0]==hand and n[3]==8)
        for i in range(len(seq)-3):
            quad=seq[i:i+4]
            if not expression._arcs(quad)['n_runs']:continue
            runs+=1;k=bisect.bisect_left(dots,quad[0][0])
            if k<len(dots) and dots[k]<=quad[-1][0]:
                bridged+=1
                if len(examples)<3:examples.append({'hand':hand,'from_seconds':quad[0][0]+t0,'to_seconds':quad[-1][0]+t0,'dot_seconds':dots[k]+t0})
    assert runs==p['features']['arcs']['n_runs']
    return {'supported':True,'legacy_runs':runs,'dot_bridged_runs':bridged,'heads':len(heads),
            'legacy_runs_per_100_heads':p['features']['arcs']['runs_per_100_heads'],
            'dot_split_runs_per_100_heads':100*(runs-bridged)/len(heads),'examples':examples}


def describe(source):
    rows=[]
    for w in phrase_windows(source['notes'],source['bombs'],source['walls'],source['duration_beats']):
        ns=[(e['beat'],h,c,l,d) for e in w['events'] for h,hs in enumerate(e['hands']) for c,l,d in hs]
        rows.append({'start_beat':w['start'],**window_ledger([(h,c,l,d) for b,h,c,l,d in ns],[b*60/source['bpm'] for b,h,c,l,d in ns])})
    return {'windows':rows,'legacy_runs':sum(r['legacy_runs'] for r in rows),
            'dot_bridged_runs':sum(r['dot_bridged_runs'] for r in rows),'affected_windows':sum(r['dot_bridged_runs']>0 for r in rows)}


def run():
    v11.prepare();records=json.loads((old.OUT/'run.json').read_text())['config']['records']
    paths={};sources={};cohorts={}
    for r in records:
        if r['fit_role'] not in ('train','development'):continue
        path=ROOT/'experiments/joint-phrase-v1/readiness/sources'/(r['family'].replace(':','_')+'.json')
        assert _sha(path)==r['artifact_sha256'];paths[str(path.relative_to(ROOT))]=_sha(path)
        key=r['family']+'/human';sources[key]=json.loads(path.read_text())
        cohorts[key]=('approved_train' if r['approved'] else 'general_train') if r['fit_role']=='train' else 'development_human'
    for item in json.loads((v11.OUT/'report.json').read_text())['families']:
        family=item['family'];key=family.replace(':','_');reference=sources[family+'/human']
        roots={'b0':old.PILOT/'development'/key/'b0'}
        for name,root,arm in (('original_retrieval',ordered.OUT,'retrieval'),('compatible',v10.OUT,'ordered'),('approved',v11.OUT,'ordered')):
            songpath=root/'development'/key/'song.json';paths[str(songpath.relative_to(ROOT))]=_sha(songpath)
            selected=json.loads(songpath.read_text())['arms'][arm]
            roots[name]=roots['b0'] if selected['fallback'] else root/'development'/key/arm/str(selected['selected_seed'])
        for name,root in roots.items():
            for fn in ('ExpertPlus.dat','Info.dat'):paths[str((root/fn).relative_to(ROOT))]=_sha(root/fn)
            src=read_chart(root/'ExpertPlus.dat',root/'Info.dat')
            src['duration_beats']=reference['duration_beats']*src['bpm']/reference['bpm']
            sources[family+'/'+name]=reference_view(src,reference);cohorts[family+'/'+name]=name
    frozen=freeze_run(OUT,{'files':paths,'code_sha256':_sha(__file__),'spec_sha256':_sha(SPEC),
        'dependencies':{p:_sha(ROOT/p) for p in ('eval/expression_profile.py','eval/joint_phrase.py','eval/joint_deployment.py','eval/joint_export.py')},
        'parent_identity':json.loads((v11.OUT/'run.json').read_text())['identity'],
        'report_sha256':_sha(v11.OUT/'report.json')})
    rows={k:{'cohort':cohorts[k],**describe(s)} for k,s in sources.items()};summary={}
    for cohort in sorted(set(cohorts.values())):
        selected=[r for r in rows.values() if r['cohort']==cohort]
        total=sum(r['legacy_runs'] for r in selected);bridged=sum(r['dot_bridged_runs'] for r in selected)
        summary[cohort]={'families':len(selected),'legacy_runs':total,'dot_bridged_runs':bridged,
                         'bridged_share':bridged/total if total else None,'affected_windows':sum(r['affected_windows'] for r in selected)}
    report={'identity':frozen['identity'],'summary':summary,'rows':rows,
            'claim':'legacy arithmetic reproduced; same-hand dot bridging quantified; no rescoring or quality decision'}
    _atomic(OUT/'report.json',report);return {'identity':frozen['identity'],'summary':summary}


if __name__=='__main__':print(json.dumps(run(),indent=1))
