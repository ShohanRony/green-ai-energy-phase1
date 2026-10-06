"""D16 regression check: integrate() must split RAPL domains, not sum them, and must
still handle counter wraparound correctly in both scalar (GPU) and split (CPU) modes.
Also covers the item-3 patch: p_core_set() parsing and find_rapl_domain() name matching.
Run directly: python3 test_pilot.py
"""
from pathlib import Path
from unittest.mock import patch

from pilot import find_rapl_domain, integrate, p_core_set

RANGE = 1000.0


def test_scalar_mode_unchanged():
    trace = [(0.0, [10.0]), (1.0, [20.0]), (2.0, [35.0])]
    expected = 1.0 * (10 + 20) / 2 + 1.0 * (20 + 35) / 2  # trapezoid over each (dt=1) pair
    assert integrate(trace, []) == expected


def test_scalar_mode_wraparound():
    trace = [(0.0, [990.0]), (1.0, [5.0])]  # wrapped past RANGE
    assert integrate(trace, [RANGE]) == 5.0 + (RANGE - 990.0)


def test_split_mode_sums_domains_separately():
    # package-0 rises 10->30 (delta 20), psys rises 40->90 (delta 50) -- must NOT be merged.
    trace = [(0.0, [10.0, 40.0]), (1.0, [30.0, 90.0])]
    out = integrate(trace, [RANGE, RANGE], ['package-0', 'psys'])
    assert out == {'package-0': 20.0, 'psys': 50.0}
    assert out['package-0'] != out['package-0'] + out['psys']  # the D16 bug would have merged these


def test_split_mode_wraparound_per_domain():
    trace = [(0.0, [990.0, 500.0]), (1.0, [5.0, 600.0])]  # package wraps, psys doesn't
    out = integrate(trace, [RANGE, RANGE], ['package-0', 'psys'])
    assert out['package-0'] == 5.0 + (RANGE - 990.0)
    assert out['psys'] == 100.0


def test_p_core_set_parses_ranges_and_singles():
    class FakePath:
        def __init__(self, s): self._s = s
        def exists(self): return True
        def read_text(self): return '0-5,8,10-11\n'
    with patch('pilot.Path', FakePath):
        assert p_core_set() == {0, 1, 2, 3, 4, 5, 8, 10, 11}


def test_p_core_set_raises_without_hybrid_sysfs():
    class FakePath:
        def __init__(self, s): self._s = s
        def exists(self): return False
    with patch('pilot.Path', FakePath):
        try:
            p_core_set()
            raise AssertionError('expected RuntimeError')
        except RuntimeError:
            pass


def test_find_rapl_domain_matches_by_name_on_this_machine():
    # Integration check against the real sysfs this harness actually reads from.
    path, rng = find_rapl_domain('package-0')
    assert path is not None and Path(path).parent.name == 'intel-rapl:0'
    assert rng > 0
    missing_path, missing_rng = find_rapl_domain('definitely-not-a-real-domain')
    assert missing_path is None and missing_rng is None


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f'ok  {t.__name__}')
    print(f'{len(tests)} passed')
