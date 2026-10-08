import io
from pathlib import Path
import tempfile
import unittest
import zipfile
import torch
import groom

from eval.corpus_scale import eligible, title_key, unpack


class CorpusScaleTests(unittest.TestCase):
    def test_human_and_supported_chart_required(self):
        doc = dict(automapper=False, stats=dict(score=.9, upvotes=60, downvotes=1),
                   metadata=dict(duration=180), versions=[dict(state='Published',
                   diffs=[dict(characteristic='Standard', notes=500, me=False, ne=False)])])
        self.assertTrue(eligible(doc))
        self.assertFalse(eligible({**doc, 'automapper': True}))
        self.assertFalse(eligible({**doc, 'automapper': None}))
        self.assertFalse(eligible({**doc, 'versions': []}))
        self.assertFalse(eligible({**doc, 'stats': dict(score=.9, upvotes=10)}))

    def test_archive_extraction_cannot_escape_or_overwrite_a_member(self):
        def archive(names):
            b = io.BytesIO()
            with zipfile.ZipFile(b, 'w') as z:
                for name in names: z.writestr(name, '{}')
            return b.getvalue()
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / 'map'
            unpack(archive(['Info.dat', '../../ExpertPlus.dat', 'evil.py']), dest)
            self.assertEqual({p.name for p in dest.iterdir()}, {'Info.dat', 'ExpertPlus.dat'})
            self.assertFalse((Path(td) / 'ExpertPlus.dat').exists())
            with self.assertRaises(ValueError):
                unpack(archive(['Info.dat', 'nested/Info.dat']), dest)

    def test_title_normalization_groups_case_and_punctuation(self):
        self.assertEqual(title_key('Song!', 'The Artist'), title_key('SONG', 'the artist'))

    def test_batched_tokens_exactly_match_existing_per_event_encoder(self):
        events = [(0, 0, 1, 0, 0, 1), (0, 1, 7, 3, 1, 2),
                  (2, 0, 4, 1, 2, 3), (12, 1, 2, 2, 0, 1)]
        wl = torch.zeros(16); wr = torch.ones(16); energy = torch.linspace(-2, 2, 16)
        for ev, left, right in [(events, wl, wr), (groom.mirror_events(events), wr, wl)]:
            xs = []; ys = []; last = {}; previous = None; last_any = None
            for i, (s, h, d, c, l, k) in enumerate(ev):
                p = last.get(h)
                dbl = any(e[0] == s and e[1] != h for e in ev[max(0, i-1):i+2])
                xs.append(groom.token_vec(previous, h, s, min(s-p[0], 8) if p else 9,
                    min(s-last_any, 8) if last_any is not None else 9,
                    dbl, bool(left[s]), bool(right[s]), float(energy[s])))
                ac, al = p[2:4] if p else groom.ANCHOR[h]
                ys.append((d, c-ac+3, l-al+2, k-1))
                previous = (h, d, c, l, k); last[h] = (s, d, c, l); last_any = s
            x, y = groom.events_to_xy(ev, left, right, energy)
            self.assertTrue(torch.equal(x, torch.stack(xs)))
            self.assertTrue(torch.equal(y, torch.tensor(ys)))


if __name__ == '__main__': unittest.main()
