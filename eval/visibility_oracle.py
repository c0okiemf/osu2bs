"""Independent test-only occlusion oracle for E01.

Deliberately a DIFFERENT construction from the production projected-rectangle
union: sample a uniform grid of points on the target's physical face, trace a
ray from the eye to each point, and test whether any strictly-nearer blocker's
physical square intersects that ray before the target. It must not import
project_face/covered_fraction/_union_area. This is a CPU geometry oracle, not a
renderer.
"""
from eval.visibility import EPS


def ray_fraction(target, blockers, eye, samples=512):
    """Fraction of a uniform samples×samples grid of points on the target face
    whose eye-ray is blocked by a strictly-nearer blocker square."""
    ex, ey, ez = eye
    td = target.z - ez
    if td <= 0:
        raise ValueError("nonpositive target depth")
    near = [b for b in blockers
            if b.id != target.id and 0 < (b.z - ez) < td - EPS]
    if not near:
        return 0.0
    covered = 0
    n = samples
    for i in range(n):
        py = target.y0 + (target.y1 - target.y0) * (i + 0.5) / n
        for j in range(n):
            px = target.x0 + (target.x1 - target.x0) * (j + 0.5) / n
            hit = False
            for b in near:
                s = (b.z - ez) / (target.z - ez)     # in (0,1) since bz<tz
                x = ex + s * (px - ex)
                y = ey + s * (py - ey)
                if b.x0 - EPS <= x <= b.x1 + EPS and b.y0 - EPS <= y <= b.y1 + EPS:
                    hit = True
                    break
            covered += hit
    return covered / (n * n)
