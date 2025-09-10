def _idc_isub_terms(fets, prefix="XM_", terminal="S"):
    """
    Build raw device current terms for .EXTRACT (DC / operating-point values).

    Each item becomes: 'ISUB(<prefix><fet>.<terminal>)'
    Example: ISUB(XM_0_2_1.S)

    Use this for **DC** (or instantaneous) currents.
    """
    return [f"ISUB({prefix}{fet}.{terminal})" for fet in fets]


def _yval_isub_terms(fets, freq="1MEG", prefix="XM_", terminal="S"):
    """
    Build YVAL-wrapped current terms for .EXTRACT at a given frequency.

    Each item becomes: 'YVAL(ISUB(<prefix><fet>.<terminal>), <freq>)'
    Example: YVAL(ISUB(XM_0_2_1.S), 1MEG)

    Use this for **AC small-signal** currents sampled at <freq>.
    """
    return [f"YVAL(ISUB({prefix}{fet}.{terminal}), {freq})" for fet in fets]


def _amp_isub_terms(n, start=1):
    """
    Interleaved amplifier I/O current terms.
    Returns: ['ISUB(XI0ii.AMP_INPUT)', 'ISUB(XI0ii.AMP_OUTPUT)', ...]

    Normally wrap with YVAL(...) if you need AC small-signal currents.
    """
    terms = []
    for i in range(start, start + n):
        terms.append(f"ISUB(XI0{i}{i}.AMP_INPUT)")
        terms.append(f"ISUB(XI0{i}{i}.AMP_OUTPUT)")
    return terms


def _extract_line(kind, terms):
    """
    Compose a single '.EXTRACT FSST ...' line from a list of terms.

    'kind' is not used here but left for future extension (e.g. DC/AC tags).
    """
    return ".EXTRACT FSST " + " ".join(terms)