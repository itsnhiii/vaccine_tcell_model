"""Parameter and ParameterSet: values that always carry their provenance.

Models must never consume a bare float for a scientifically meaningful
quantity. They consume Parameter objects (grouped into a ParameterSet)
so that source/units/description travel with the number, and so that
two ParameterSets (e.g. Science vs Mayer) can never be silently merged
(master-spec rule 4).
"""

from __future__ import annotations

from dataclasses import dataclass

from .provenance import SourceType


@dataclass(frozen=True)
class Parameter:
    """A single scientific parameter with full provenance.

    Attributes:
        name: identifier matching the symbol used in docs/equations.md
            (e.g. "d_Ag", "K", "k_p").
        value: numeric value.
        units: physical or normalized units, e.g. "1/day" or
            "dimensionless (normalized dose fraction)".
        source: citation or fit target, e.g.
            "Bhagchandani et al. 2024 Table S2" or
            "fitted to Bhagchandani et al. 2024 Fig. 3F (day 21 model value)".
        source_type: one of SourceType's allowed categories.
        description: short human-readable meaning of the parameter.
    """

    name: str
    value: float
    units: str
    source: str
    source_type: SourceType
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.source_type, SourceType):
            raise TypeError(
                f"Parameter {self.name!r}.source_type must be a SourceType "
                f"enum member, got {type(self.source_type).__name__}. "
                f"Allowed: {[s.value for s in SourceType]}"
            )
        if not self.name:
            raise ValueError("Parameter.name must be non-empty")
        if not self.source:
            raise ValueError(
                f"Parameter {self.name!r} must specify a non-empty `source` "
                "(citation, fit target, or explicit assumption note) -- "
                "master-spec rule 6 forbids unsourced parameters."
            )


class ParameterSet:
    """An immutable, named collection of Parameter objects.

    Deliberately has no merge/update-in-place API: combining values from
    two different ParameterSets (e.g. Science + Mayer) must always be an
    explicit, visible construction of a new ParameterSet (or a documented
    IntegratedParameters class), never an implicit dict merge.
    """

    def __init__(self, name: str, parameters: dict[str, Parameter]):
        for key, p in parameters.items():
            if p.name != key:
                raise ValueError(
                    f"ParameterSet {name!r}: dict key {key!r} does not match "
                    f"Parameter.name {p.name!r}"
                )
        self.name = name
        self._parameters = dict(parameters)

    def __getitem__(self, key: str) -> Parameter:
        return self._parameters[key]

    def value(self, key: str) -> float:
        """Convenience accessor for the bare numeric value of a parameter."""
        return self._parameters[key].value

    def __contains__(self, key: str) -> bool:
        return key in self._parameters

    def __iter__(self):
        return iter(self._parameters.values())

    def __len__(self) -> int:
        return len(self._parameters)

    def keys(self):
        return self._parameters.keys()

    def as_dict(self) -> dict[str, float]:
        """Bare {name: value} view, e.g. for passing to an ODE RHS."""
        return {k: p.value for k, p in self._parameters.items()}

    def with_value(
        self,
        name: str,
        value: float,
        *,
        source: str | None = None,
        source_type: SourceType | None = None,
    ) -> "ParameterSet":
        """Return a NEW ParameterSet with `name`'s value replaced.

        This is how parameter sweeps (master spec Section 19,
        analysis.sweeps.sweep_parameter) vary one named parameter while
        holding everything else fixed. It is NOT the forbidden
        cross-paper merge (rule 4) -- it only ever replaces one value
        already present in this same, single-provenance ParameterSet.
        The replaced Parameter's own source/source_type default to
        recording that it is now an explicit override rather than
        silently keeping its old literature/fitted justification.
        """
        if name not in self._parameters:
            raise KeyError(f"{name!r} is not a parameter of {self.name!r}: {sorted(self._parameters)}")
        old = self._parameters[name]
        new_source = source if source is not None else f"Swept/overridden value (was: {old.source})"
        new_source_type = source_type if source_type is not None else SourceType.USER_DEFINED
        new_param = Parameter(
            name=name, value=value, units=old.units,
            source=new_source, source_type=new_source_type, description=old.description,
        )
        new_params = dict(self._parameters)
        new_params[name] = new_param
        return ParameterSet(self.name, new_params)

    def __repr__(self) -> str:
        return f"ParameterSet({self.name!r}, n={len(self._parameters)})"
