"""Clock reuse requires the same authenticated tensor prefix and sources."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from timing import TimeGrid
from eval import flow_training_returns as run


class ClockCacheTest(unittest.TestCase):
    def test_reuse_and_reject_changed_prefix_or_source(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);pool=root/'pool';oldpool=root/'oldpool';cache=root/'cache'
            for p in (pool,oldpool,cache):p.mkdir()
            prefix=dict(families=[dict(id='a')])
            for p in (pool,oldpool):(p/'prepared.json').write_text(json.dumps(prefix))
            (pool/'manifest.json').write_text('{"families": []}')
            source=root/'chart.dat';source.write_text('{}')
            g=TimeGrid.uniform(125,0,2);torch.save(([g],[]),cache/'grids.pt')
            old=dict(identity=dict(addition=1,pool_dir=str(oldpool),pool_sha256=run._sha(oldpool/'prepared.json'),
                inputs={},code={p:run._sha(run.ROOT/p) for p in ('groom.py','timing.py')}),
                sources={str(source):run._sha(source)},grids_sha256=run._sha(cache/'grids.pt'),
                train_charts=1,val_charts=0,ledger=[dict(family='new:a',difficulties=['ExpertPlus'],
                    bpm=120,events=1,max_elapsed_difference_from_old_uniform_ms=0)])
            (cache/'prepared.json').write_text(json.dumps(old))
            m=(torch.zeros(2,13),torch.zeros(2,2),[(0,0,1,0,0,1)],torch.zeros(2),torch.zeros(2),1.,'new:a')
            with patch.object(run,'source_data',return_value=([m],[],{'inputs':{}})), \
                 patch.object(run.shutil,'copytree'),patch.object(run.shutil,'copyfile'), \
                 patch.object(run.groom,'load_map_all',side_effect=AssertionError('must reuse verified clock')):
                r=run.prepare(root/'out',1,pool,cache)
                self.assertEqual(r['reused_clock_rows'],1)
                tr,va=torch.load(root/'out/grids.pt',weights_only=False)
                self.assertEqual(tr[0].times,g.times)
                (pool/'prepared.json').write_text('{"families": [{"id": "different"}]}')
                with self.assertRaisesRegex(ValueError,'tensor prefix changed'):
                    run.prepare(root/'bad-prefix',1,pool,cache)
                (pool/'prepared.json').write_text(json.dumps(prefix));source.write_text('changed')
                with self.assertRaisesRegex(ValueError,'cached timing source changed'):
                    run.prepare(root/'bad-source',1,pool,cache)


if __name__=='__main__':unittest.main()
