"""Selected extended tiers must remain distinct and preserve their own osu clocks."""
import hashlib
import itertools
import json
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import convert
import run


class DifficultyExportTests(unittest.TestCase):
    def test_unique_ordered_slots_for_every_selection_up_to_five(self):
        choices = list(convert.DIFFS) + [f"ExpertPlus{n}" for n in range(2, 7)]
        for count in range(1, 6):
            for names in itertools.combinations(choices, count):
                slots = convert.difficulty_slots(names)
                self.assertEqual(len(slots), count)
                ranks = [rank for _, rank in slots.values()]
                self.assertEqual(ranks, sorted(set(ranks)))
                self.assertTrue(all(rank in (1, 3, 5, 7, 9) for rank in ranks))
        for selection in [",".join(choices[:6]), "ExpertPlus,ExpertPlus"]:
            with self.assertRaises(convert.GenerationError):
                convert.selected_difficulties(selection)

    def test_three_plus_tiers_export_three_independent_labeled_charts(self):
        names = ["ExpertPlus", "ExpertPlus2", "ExpertPlus3"]
        maps = {n: ([dict(t=1000. + i * 100, hand=0, col=1, layer=0, dir=1)],
                    [], convert.diff_spec(n)) for i, n in enumerate(names)}
        with tempfile.TemporaryDirectory() as td:
            convert.export_charts(maps, 120., dict(Title="test", Artist="test"), td)
            info = json.loads((Path(td) / "Info.dat").read_text())
            diffs = info['_difficultyBeatmapSets'][0]['_difficultyBeatmaps']
            self.assertEqual([d['_difficulty'] for d in diffs], ['Hard', 'Expert', 'ExpertPlus'])
            self.assertEqual([d['_customData']['_difficultyLabel'] for d in diffs],
                             ['Expert+', 'Expert++', 'Expert+++'])
            self.assertEqual([d['_beatmapFilename'] for d in diffs], [n + '.dat' for n in names])
            times = [json.loads((Path(td) / d['_beatmapFilename']).read_text())['_notes'][0]['_time']
                     for d in diffs]
            self.assertEqual(times, [2., 2.2, 2.4])
            diffs[0]['_difficulty'] = 'ExpertPlus'
            diffs[0]['_difficultyRank'] = 9
            (Path(td) / 'Info.dat').write_text(json.dumps(info))
            with self.assertRaises(convert.GenerationError):
                convert.package_map(td)

    def test_pinned_model_multi_source_pipeline_and_playlist(self):
        self.assertEqual(hashlib.sha256(run.ONSET_PT.read_bytes()).hexdigest(),
                         'a11fea850d4ff3e6f510fd075beeb7207003421c780b7ce4a0f7a2128a98fcba')
        names = ['ExpertPlus', 'ExpertPlus2', 'ExpertPlus3']
        calls = []
        def upstream(song, work, stars, hydra_dev, env, title):
            work.mkdir(); calls.append(stars)
            bpm = 100 + 10 * len(calls)
            times = [1000 + 31 * len(calls) + j * int(1000 / stars) for j in range(8)]
            osu = work / 'source.osu'
            osu.write_text('[General]\nAudioFilename:song.wav\n[Metadata]\nTitle:test\nArtist:test\n'
                           f'[TimingPoints]\n0,{60000 / bpm},4,2,1,100,1,0\n[HitObjects]\n' +
                           '\n'.join(f'64,64,{t},1,0,0:0:0:0:' for t in times))
            return osu
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); audio = root / 'song.wav'; out = root / 'map'
            subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'lavfi', '-i',
                            'sine=frequency=440:duration=5', str(audio)], check=True)
            with patch.object(run, 'generate_osu', side_effect=upstream):
                self.assertEqual(run.main(audio, out, diffs=','.join(names), playlist='test'), 0)
            self.assertEqual(calls, [5.5, 6.5, 7.5])
            meta = json.loads((out / 'generation.json').read_text())
            self.assertEqual(meta['requested_osu_stars'], dict(zip(names, calls)))
            info = json.loads((out / 'Info.dat').read_text())
            ledgers = json.loads((out / 'source-onsets.json').read_text())
            for name in names:
                chart = json.loads((out / (name + '.dat')).read_text())
                actual = sorted({n['_time'] * 60000 / info['_beatsPerMinute'] for n in chart['_notes']})
                expected = [e['t'] for e in ledgers[name]]
                self.assertEqual(len(actual), len(expected))
                for a, e in zip(actual, expected):self.assertLess(abs(a-e), .003)
                self.assertTrue(Path(meta['per_difficulty'][name]['source_osu']).is_file())
            with zipfile.ZipFile(out.with_suffix('.zip')) as z:
                self.assertEqual(set(z.namelist()), {'Info.dat', 'song.egg', *(n+'.dat' for n in names)})
            playlist = json.loads((root / 'test.bplist').read_text())
            self.assertEqual(len(playlist['songs']), 1)
            h = hashlib.sha1((out/'Info.dat').read_bytes())
            for n in names:h.update((out/(n+'.dat')).read_bytes())
            self.assertEqual(playlist['songs'][0]['hash'], h.hexdigest())
            # Validate selection before starting any costly upstream work.
            with patch.object(run, 'generate_osu') as generate:
                with self.assertRaises(convert.GenerationError):
                    run.main(audio, out, diffs='Easy,Normal,Hard,Expert,ExpertPlus,ExpertPlus2')
                generate.assert_not_called()


if __name__ == '__main__':
    unittest.main()
