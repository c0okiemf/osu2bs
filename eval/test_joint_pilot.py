import unittest
from unittest import mock
import tempfile
import json
from pathlib import Path
import copy
import numpy as np
import torch
from eval.joint_model import JointModel, N_GAP
from eval.joint_pilot import controls, retrieval_bank, rollout, errors, machine_tools, decision


class PilotTests(unittest.TestCase):
    def test_controls(self):
        self.assertEqual(controls()['status'],'CONTROLS_PASS')

    def test_retrieval_rejects_validation(self):
        with self.assertRaisesRegex(ValueError,'fit-train'):
            retrieval_bank([{'fit_role':'validation'}])

    def test_missing_is_unknown(self):
        ref={'rhythm':[[1]*5],'geometry':[[1]*5]}
        cand={'rhythm':[[0]*5],'geometry':[[None]*5]}
        r=errors(ref,cand,{'rhythm':[1]*5,'geometry':[1]*5})
        self.assertIsNone(r['geometry']['mean'])
        self.assertEqual(r['geometry']['missing_candidate_windows'],[1]*5)

    def test_machine_bank_contract(self):
        sentinel=object()
        with mock.patch('qa.neighbours.bank_from_role',return_value=(sentinel,[])):
            self.assertIs(machine_tools()[0],sentinel)

    def test_decision_requires_learned_nonfallback_and_coverage(self):
        def rec(error):
            return {'errors':{a:{'mean':error,'components':[error]*5,
                            'supported_windows':[10]*5} for a in ('rhythm','geometry')},
                    'machine':{'contradictions':{'status':'NO_CONTRADICTION_FOUND'},
                               'support':{'share_supported':1.}}}
        songs=[]
        for i in range(8):
            songs.append({'family':f'fam:{i}','requested_rate':4.,'reference_rate':5.,'b0':rec(1.),
                          'arms':{a:{'selected':rec(.5 if a=='model' else 1.),'fallback':False,
                                     'selected_seed':0,'attempts':[]} for a in ('model','retrieval')}})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'partial').mkdir()
            families=[{'family':s['family'],'status':'supported','role':'development'} for s in songs]
            (root/'report.json').write_text(json.dumps({'families':families}))
            (root/'run.json').write_text(json.dumps({'config':{'records':families}}))
            for r in families:
                (root/'partial'/(r['family'].replace(':','_')+'.json')).write_text('{}')
            with mock.patch('eval.joint_phrase.OUT',root), mock.patch('eval.joint_pilot.PILOT',root/'pilot'), \
                 mock.patch('eval.joint_pilot.evaluation_freeze'):
                selection={'learned_candidate':True}
                frozen={'identity':'test'}
                self.assertEqual(decision(songs,selection,frozen)['status'],'PILOT_POSITIVE')
                before=(root/'pilot'/'report.json').read_bytes()
                decision(songs,{'learned_candidate':False},frozen,output_root=root/'separate')
                self.assertEqual((root/'pilot'/'report.json').read_bytes(),before)
                self.assertTrue((root/'separate'/'report.json').exists())
                changed=copy.deepcopy(songs)
                changed[0]['arms']['model']['selected']['errors']['geometry']['supported_windows'][0]=0
                r=decision(changed,selection,frozen)
                self.assertFalse(r['gates']['geometry_coverage_vs_b0'])
                self.assertEqual(r['status'],'PILOT_NEGATIVE')
                for s in changed[:3]:
                    s['arms']['model']['fallback']=True
                self.assertFalse(decision(changed,selection,frozen)['gates']['six_nonfallback'])
                selection['learned_candidate']=False
                self.assertFalse(decision(songs,selection,frozen)['gates']['learned_checkpoint'])

    def test_free_rollout_rest_and_invalid_boundary_preserve_prefix(self):
        torch.set_num_threads(2)
        m=JointModel()
        with torch.no_grad():
            for p in m.parameters():
                p.zero_()
            m.gap.bias.fill_(-1000)
            m.gap.bias[0]=1000
        src={'notes':[],'events':[],'walls':[],'bombs':[],'bpm':120,'duration_beats':16}
        audio={'times':np.arange(20.),'x':np.zeros((20,27))}
        r=rollout(m,src,audio,5)
        self.assertFalse(r['ok'])
        self.assertEqual(r['reason'],'empty_chart')
        self.assertEqual(r['actions'],2)
        with torch.no_grad():
            m.gap.bias[0]=-1000
            m.gap.bias[2]=1000  # 1/48 beat then impossible large positive residual
            m.residual.bias[2]=8
        r=rollout(m,src,audio,5)
        self.assertFalse(r['ok'])
        self.assertEqual(r['reason'],'event_outside_window')
        self.assertEqual(r['source']['notes'],[])
        with torch.no_grad():
            m.residual.bias[2]=0
            m.count.bias.fill_(-1000)
            m.count.bias[4]=1000
        r=rollout(m,src,audio,5,max_actions=3)
        r2=rollout(m,src,audio,5,max_actions=3)
        self.assertEqual(r['source']['notes'],r2['source']['notes'])
        self.assertEqual(len(r['source']['notes']),3)
        self.assertEqual(r['reason'],'action_budget_exhausted')


if __name__=='__main__':
    unittest.main()
