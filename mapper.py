"""Mapperatorinator's inference.py with one fix, run from its own folder.

The model sometimes emits an object (circle, slider head/end, spinner) with no
time token; upstream's Group.time defaults to 0, so these became stacks at
0 ms and sliders from 0:00 to their real end (616 of 1449 shipped tiers).
They have no real time, so drop them before the .osu is written.
osu!standard object types only (we always run gamemode=0).
"""
import runpy
import sys
from pathlib import Path

MAPPER = Path(__file__).resolve().parent / "Mapperatorinator"
sys.path.insert(0, str(MAPPER))

from osuT5.osuT5.dataset.data_utils import get_groups  # noqa: E402
from osuT5.osuT5.event import EventType as E  # noqa: E402
from osuT5.osuT5.inference.postprocessor import Postprocessor  # noqa: E402

ENDS = {E.CIRCLE: None, E.SLIDER_HEAD: E.SLIDER_END, E.SPINNER: E.SPINNER_END}
PARTS = {E.BEZIER_ANCHOR, E.PERFECT_ANCHOR, E.CATMULL_ANCHOR, E.RED_ANCHOR,
         E.LAST_ANCHOR, E.SLIDER_END, E.SPINNER_END, *ENDS}


def drop_untimed(events, types_first=False):
    groups, idx = get_groups(events, types_first=types_first)
    timed = [any(events[i].type == E.TIME_SHIFT for i in ix) for ix in idx]
    drop, k = set(), 0
    while k < len(groups):
        if groups[k].event_type not in ENDS:
            k += 1
            continue
        end, j = ENDS[groups[k].event_type], k
        if end is not None:  # head .. matching end; incomplete ones upstream skips anyway
            j = next((m for m in range(k + 1, len(groups))
                      if groups[m].event_type in (end, *ENDS)), None)
            if j is None or groups[j].event_type != end:
                k += 1
                continue
        if not (timed[k] and timed[j]):
            for m in range(k, j + 1):
                if groups[m].event_type in PARTS:
                    drop.update(idx[m])
        k = j + 1
    return [e for i, e in enumerate(events) if i not in drop]


_generate = Postprocessor.generate


def _generate_timed(self, events, *args, **kwargs):
    return _generate(self, drop_untimed(events, self.types_first), *args, **kwargs)


if __name__ == "__main__":
    Postprocessor.generate = _generate_timed
    sys.argv[0] = str(MAPPER / "inference.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
