from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pyomo.environ as pyo


@dataclass(frozen=True)
class CostParameters:
    production: float = 1.0
    holding: float = 0.40
    shortage: float = 5.0

    @property
    def critical_fractile(self) -> float:
        if self.shortage <= self.production:
            raise ValueError("shortage cost must exceed production cost")
        return (self.shortage - self.production) / (self.holding + self.shortage)


def realized_cost(q, demand, costs: CostParameters):
    q, demand = np.asarray(q), np.asarray(demand)
    return costs.production * q + costs.holding * np.maximum(q - demand, 0) + costs.shortage * np.maximum(demand - q, 0)


def saa_quantile(samples: np.ndarray, costs: CostParameters) -> np.ndarray:
    return np.quantile(samples, costs.critical_fractile, axis=0, method="higher")


def solve_saa_with_pyomo(samples: np.ndarray, costs: CostParameters):
    """Explicit LP check for the empirical critical-fractile shortcut."""
    samples = np.asarray(samples, dtype=float)
    m = pyo.ConcreteModel(); m.S = pyo.RangeSet(0, len(samples) - 1)
    m.q = pyo.Var(domain=pyo.NonNegativeReals)
    m.excess = pyo.Var(m.S, domain=pyo.NonNegativeReals)
    m.shortage = pyo.Var(m.S, domain=pyo.NonNegativeReals)
    m.excess_def = pyo.Constraint(m.S, rule=lambda M, s: M.excess[s] >= M.q - float(samples[s]))
    m.shortage_def = pyo.Constraint(m.S, rule=lambda M, s: M.shortage[s] >= float(samples[s]) - M.q)
    m.obj = pyo.Objective(expr=costs.production * m.q + (1 / len(samples)) * sum(costs.holding * m.excess[s] + costs.shortage * m.shortage[s] for s in m.S))
    result = pyo.SolverFactory("appsi_highs").solve(m)
    return float(pyo.value(m.q)), str(result.solver.termination_condition)
