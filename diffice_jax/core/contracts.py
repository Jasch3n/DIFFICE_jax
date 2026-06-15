from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class FieldSchema:
    """Network field layout for one region before evaluating an equation."""

    region_kind: str
    outputs: tuple[str, ...]
    derived_fields: tuple[str, ...] = ()

    @property
    def available_fields(self) -> tuple[str, ...]:
        return self.outputs + tuple(f for f in self.derived_fields if f not in self.outputs)


@dataclass(frozen=True)
class EquationContract:
    """Declarative field contract for an equation in one ice-domain regime."""

    name: str
    region_kind: str
    required_fields: tuple[str, ...]
    forbidden_fields: tuple[str, ...]
    derived_fields: tuple[str, ...]
    residual_names: tuple[str, ...]
    output_shapes: Mapping[str, tuple[int | None, ...]]


def validate_field_schema(contract: EquationContract, schema: FieldSchema) -> None:
    """Raise a readable error when network outputs do not satisfy a contract."""

    if schema.region_kind != contract.region_kind:
        raise ValueError(
            f"{contract.name} expects a {contract.region_kind} region, "
            f"got {schema.region_kind}."
        )

    fields = set(schema.available_fields)
    missing = [name for name in contract.required_fields if name not in fields]
    if missing:
        raise ValueError(f"{contract.name} is missing required fields: {missing}.")

    forbidden = [name for name in contract.forbidden_fields if name in fields]
    if forbidden:
        raise ValueError(f"{contract.name} forbids fields: {forbidden}.")


SSA_ISO_FLOATING = EquationContract(
    name="ssa_iso",
    region_kind="floating",
    required_fields=("u", "v", "h", "s", "mu"),
    forbidden_fields=("C",),
    derived_fields=("s",),
    residual_names=("x_momentum", "y_momentum"),
    output_shapes={"residual": (None, 2), "terms": (None, 7)},
)

SSA_ISO_GROUNDED = EquationContract(
    name="ssa_iso",
    region_kind="grounded",
    required_fields=("u", "v", "h", "s", "mu", "C"),
    forbidden_fields=(),
    derived_fields=(),
    residual_names=("x_momentum", "y_momentum"),
    output_shapes={"residual": (None, 2), "terms": (None, 9)},
)

_CONTRACTS = {
    ("ssa_iso", "floating"): SSA_ISO_FLOATING,
    ("ssa_iso", "grounded"): SSA_ISO_GROUNDED,
}


def get_equation_contract(name: str, region_kind: str) -> EquationContract:
    try:
        return _CONTRACTS[(name, region_kind)]
    except KeyError as exc:
        raise ValueError(f"No equation contract for {name!r} in {region_kind!r} regions.") from exc


def validate_contracts(name: str, schemas: Iterable[FieldSchema]) -> None:
    for schema in schemas:
        validate_field_schema(get_equation_contract(name, schema.region_kind), schema)
