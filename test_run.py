"""Any filename/path must survive the trip into hydra overrides verbatim."""
from hydra.core.override_parser.overrides_parser import OverridesParser
from omegaconf import OmegaConf

from run import _hydra_str


def test_hydra_str_roundtrip():
    p = OverridesParser.create()
    for v in [r"C:\Users\Ђорђе\AppData\Local\Temp\tmp1\song.mp3", "C:\\a\\b\\",
              "a'b ${x} \"q\"", "Ann O'Day - Fever", "Позови меня\u00a0с собой (x=1)",
              "Alexandra Căpitănescu", "日本語 = [1,2] {a:b}", "%x% $y @z ~#!&;", ""]:
        got = OmegaConf.create({"t": p.parse_override("title=" + _hydra_str(v)).value()}).t
        assert got == v, (v, got)


if __name__ == "__main__":
    test_hydra_str_roundtrip()
    print("ok")
