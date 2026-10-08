import unittest
import torch
from eval.joint_model import KEYS,EMPTY,CONTEXT_DIM,JointModel,losses as old_losses
from eval.joint_start import crop,losses


class StartTests(unittest.TestCase):
    def row(self):
        return {'context':torch.zeros(160,CONTEXT_DIM),'prev':torch.full((160,6),EMPTY),
                'kind':torch.tensor([2]+[0]*159),'prev_gap':torch.ones(160),
                'gap':torch.zeros(160,dtype=torch.long),'residual':torch.zeros(160),
                'count':torch.zeros(160,dtype=torch.long),'slots':torch.full((160,6),EMPTY)}

    def test_true_start_targets_included_and_no_fake_mid_song_bos(self):
        row=self.row();start=crop(row,0);later=crop(row,16)
        self.assertTrue(start['valid'].all());self.assertEqual(int(start['kind'][0]),2)
        self.assertFalse(later['valid'][:32].any());self.assertTrue(later['valid'][32:].all())
        self.assertEqual(int(later['kind'][0]),0)
        self.assertTrue(torch.equal(later['prev'],row['prev'][16:144]))
        for bad in (-1,33):
            with self.assertRaisesRegex(ValueError,'crop'):crop(row,bad)
        # No global action <32 can reach a valid local index in the old recipe.
        for global_index in range(32):
            self.assertFalse(any(global_index-start>=32 for start in range(global_index+1)))

    def test_rest_only_start_has_direct_early_gradients_and_unchanged_later_loss(self):
        torch.set_num_threads(2);model=JointModel()
        rows=[crop(self.row(),0),crop(self.row(),16)]
        batch={k:torch.stack([r[k] for r in rows]) for k in (*KEYS,'valid')}
        out,_=model(batch);out['gap'].retain_grad()
        value=losses(out,batch);value.backward()
        self.assertGreater(float(out['gap'].grad[0,:32].abs().sum()),0.)
        self.assertEqual(float(out['gap'].grad[1,:32].abs().sum()),0.)
        self.assertTrue(torch.isfinite(value))
        batch['valid'][:,:32]=False
        self.assertTrue(torch.allclose(losses(out,batch),old_losses(out,batch)))
        batch['gap'][:,40:]=5;batch['count'][:,40:]=4;batch['slots'][:,40:,0]=1
        out,_=model(batch)
        self.assertTrue(torch.allclose(losses(out,batch),old_losses(out,batch)))


if __name__=='__main__':unittest.main()
