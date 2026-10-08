"""BSOR (BS Open Replay) v1 parser — QA packet 1.

Validated against the published specification
(github.com/BeatLeader/BS-Open-Replay) and real files. Little-endian
throughout; strings are int-length-prefixed UTF-8. Identity fields
(playerID/playerName/platform) are parsed but returned in a SEPARATE
`identity` dict so ingestion can pseudonymize durable records and keep
acquisition identifiers restricted — a parsed replay record never silently
carries discarded identity.

PARSER_VERSION participates in every cache identity.
"""
import struct

PARSER_VERSION = 1
MAGIC = 0x442D3D69

IDENTITY_FIELDS = ("playerID", "playerName", "platform")
INFO_FIELDS = ("version", "gameVersion", "timestamp", "playerID",
               "playerName", "platform", "trackingSystem", "hmd",
               "controller", "hash", "songName", "mapper", "difficulty")


class BsorError(ValueError):
    pass


class _R:
    def __init__(self, data):
        self.d = data
        self.o = 0

    def _take(self, n):
        if self.o + n > len(self.d):
            raise BsorError(f"truncated at offset {self.o} (+{n})")
        b = self.d[self.o:self.o + n]
        self.o += n
        return b

    def i32(self):
        return struct.unpack("<i", self._take(4))[0]

    def i64(self):
        return struct.unpack("<q", self._take(8))[0]

    def f32(self):
        return struct.unpack("<f", self._take(4))[0]

    def u8(self):
        return self._take(1)[0]

    def s(self):
        n = self.i32()
        if n < 0 or n > 1_000_000:
            raise BsorError(f"bad string length {n} at {self.o - 4}")
        return self._take(n).decode("utf-8", errors="replace")

    def vec3(self):
        return (self.f32(), self.f32(), self.f32())

    def quat(self):
        return (self.f32(), self.f32(), self.f32(), self.f32())

    def eof(self):
        return self.o >= len(self.d)


def decode_note_id(note_id):
    """noteID = scoringType*10000 + lineIndex*1000 + lineLayer*100 +
    colorType*10 + cutDirection (scoringType = game value + 2). These are
    ATTRIBUTES, not a unique chart-note index."""
    return {"scoring_type": note_id // 10000 - 2,
            "line_index": (note_id // 1000) % 10,
            "line_layer": (note_id // 100) % 10,
            "color": (note_id // 10) % 10,
            "cut_direction": note_id % 10}


def parse_bsor(data):
    r = _R(data)
    if r.i32() != MAGIC:
        raise BsorError("bad magic")
    version = r.u8()
    if version != 1:
        raise BsorError(f"unsupported BSOR version {version}")
    if r.u8() != 0:
        raise BsorError("expected info section marker")
    info, identity = {}, {}
    for f in INFO_FIELDS:
        v = r.s()
        (identity if f in IDENTITY_FIELDS else info)[f] = v
    info["score"] = r.i32()
    info["mode"] = r.s()
    info["environment"] = r.s()
    info["modifiers"] = r.s()
    info["jumpDistance"] = r.f32()
    info["leftHanded"] = bool(r.u8())
    info["height"] = r.f32()
    info["startTime"] = r.f32()
    info["failTime"] = r.f32()
    info["speed"] = r.f32()

    if r.u8() != 1:
        raise BsorError("expected frames section marker")
    n = r.i32()
    if n < 0 or n > 10_000_000:
        raise BsorError(f"bad frame count {n}")
    frames = []
    for _ in range(n):
        t = r.f32()
        fps = r.i32()
        head = (r.vec3(), r.quat())
        left = (r.vec3(), r.quat())
        right = (r.vec3(), r.quat())
        frames.append((t, fps, head, left, right))

    if r.u8() != 2:
        raise BsorError("expected notes section marker")
    n = r.i32()
    if n < 0 or n > 1_000_000:
        raise BsorError(f"bad note count {n}")
    notes = []
    for _ in range(n):
        nid = r.i32()
        ev = {"note_id": nid, **decode_note_id(nid),
              "event_time": r.f32(), "spawn_time": r.f32(),
              "event_type": r.i32()}
        if ev["event_type"] in (0, 1):
            ev["cut"] = {
                "speedOK": bool(r.u8()), "directionOK": bool(r.u8()),
                "saberTypeOK": bool(r.u8()), "wasCutTooSoon": bool(r.u8()),
                "saberSpeed": r.f32(), "saberDir": r.vec3(),
                "saberType": r.i32(), "timeDeviation": r.f32(),
                "cutDirDeviation": r.f32(), "cutPoint": r.vec3(),
                "cutNormal": r.vec3(), "cutDistanceToCenter": r.f32(),
                "cutAngle": r.f32(), "beforeCutRating": r.f32(),
                "afterCutRating": r.f32()}
        notes.append(ev)

    walls, heights, pauses, offsets = [], [], [], None
    while not r.eof():
        marker = r.u8()
        if marker == 3:
            for _ in range(r.i32()):
                walls.append({"wall_id": r.i32(), "energy": r.f32(),
                              "time": r.f32(), "spawn_time": r.f32()})
        elif marker == 4:
            for _ in range(r.i32()):
                heights.append({"height": r.f32(), "time": r.f32()})
        elif marker == 5:
            for _ in range(r.i32()):
                pauses.append({"duration": r.i64(), "time": r.f32()})
        elif marker == 6:
            offsets = {"left": (r.vec3(), r.quat()),
                       "right": (r.vec3(), r.quat())}
        elif marker == 7:
            r._take(r.i32())                  # userdata: skipped, never kept
        else:
            raise BsorError(f"unknown section marker {marker} at {r.o - 1}")
    return {"parser_version": PARSER_VERSION, "bsor_version": version,
            "info": info, "identity": identity, "frames": frames,
            "notes": notes, "walls": walls, "heights": heights,
            "pauses": pauses, "controller_offsets": offsets}
