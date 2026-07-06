# Domain Glossary

## ISSM Adjoint Shelf Inversion Experiment

A runnable inversion study for one ice shelf using ISSM adjoint methods. It
includes the shelf data sources, spatial domain, mesh and parameterization
choices, rheology-B inversion settings, diagnostics, and generated artifacts.

## Rheology-B Inversion

The phase of an ISSM adjoint shelf inversion experiment that infers the ice
rigidity field represented by rheology B from observed velocity and
regularization terms.

## L-Curve Inversion

A rheology-B inversion run over multiple regularization weights, where the
selected result is chosen from the velocity-misfit and regularization tradeoff.

## Smoke Inversion

A short rheology-B inversion run used to check that a configured shelf
experiment can execute and reduce the objective before running the full
L-curve inversion.

## Surface Elevation Observation Grid

The x/y coordinate grid on which observed surface elevation values are defined.
For XPINN joint inversion, it is independent from the velocity and thickness
observation grids.
