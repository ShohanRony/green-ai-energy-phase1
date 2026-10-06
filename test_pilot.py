"""D16 regression check: integrate() must split RAPL domains, not sum them, and must
still handle counter wraparound correctly in both scalar (GPU) and split (CPU) modes.
Run directly: python3 test_pilot.py
"""
from pilot import integrate

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


if __name__ == '__main__':
    tests = [v for k, v in list(globals().items()) if k.startswith('test_')]
    for t in tests:
        t()
        print(f'ok  {t.__name__}')
    print(f'{len(tests)} passed')
