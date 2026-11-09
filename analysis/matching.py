from __future__ import annotations

from collections.abc import Iterable
from model.result import SimulationResult


def pooled_static_gain(dynamic_results: Iterable[SimulationResult]) -> float:
    numerator = 0.0
    denominator = 0.0
    count = 0
    for result in dynamic_results:
        if result.condition != "dynamic":
            raise ValueError("p_match must be estimated only from dynamic-STF runs.")
        numerator += result.gain_match_numerator
        denominator += result.gain_match_denominator
        count += 1
    if count == 0 or denominator <= 0:
        raise ValueError("Cannot estimate p_match: empty results or zero weighted NMDA gate.")
    return numerator / denominator
