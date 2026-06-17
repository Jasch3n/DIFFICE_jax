from dataclasses import dataclass
from typing import Any, Optional
import warnings

import jax


@dataclass
class ScaleBundle:
    raw_mean: Any
    raw_range: Any
    canonical: Any


@dataclass
class DataInfo:
    scale: ScaleBundle
    data_norm: Any
    data_raw: Any
    idxval_all: Any
    dsize_all: Any

    def to_legacy(self):
        return [
            self.scale.raw_mean,
            self.scale.raw_range,
            self.data_norm,
            self.data_raw,
            self.scale.canonical,
            self.idxval_all,
            self.dsize_all,
        ]


@jax.tree_util.register_pytree_node_class
class PINNNormalizedData:
    def __init__(self, X_star, U_star, X_ct, nn_ct, boundary=None):
        self.X_star = X_star
        self.U_star = U_star
        self.X_ct = X_ct
        self.nn_ct = nn_ct
        self.boundary = boundary

    def tree_flatten(self):
        return ((self.X_star, self.U_star, self.X_ct, self.nn_ct, self.boundary), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def to_legacy(self, info_legacy):
        if self.boundary is not None:
            return (self.X_star, self.U_star, self.X_ct, self.nn_ct, info_legacy, self.boundary)
        return (self.X_star, self.U_star, self.X_ct, self.nn_ct, info_legacy)


@jax.tree_util.register_pytree_node_class
class XPINNRegionData:
    def __init__(self, X_star, U_star, X_ct, nn_ct, data_info, X_md, boundary=None):
        self.X_star = X_star
        self.U_star = U_star
        self.X_ct = X_ct
        self.nn_ct = nn_ct
        self.data_info = data_info
        self.X_md = X_md
        self.boundary = boundary

    def tree_flatten(self):
        return ((self.X_star, self.U_star, self.X_ct, self.nn_ct, self.data_info, self.X_md, self.boundary), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def to_legacy(self):
        info_legacy = self.data_info.to_legacy() if isinstance(self.data_info, DataInfo) else self.data_info
        return (self.X_star, self.U_star, self.X_ct, self.nn_ct, info_legacy, self.X_md, self.boundary)


@jax.tree_util.register_pytree_node_class
class PINNDataset:
    def __init__(self, norm: PINNNormalizedData, info: DataInfo):
        self.norm = norm
        self.info = info

    def to_legacy(self):
        return self.norm.to_legacy(self.info.to_legacy())

    def __iter__(self):
        return iter(self.to_legacy())

    def __getitem__(self, idx):
        return self.to_legacy()[idx]

    def __len__(self):
        return len(self.to_legacy())

    def tree_flatten(self):
        return ((self.norm, self.info), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class XPINNDataset:
    def __init__(self, regions, idxgall, posi_all, idxcrop_all):
        self.regions = regions
        self.idxgall = idxgall
        self.posi_all = posi_all
        self.idxcrop_all = idxcrop_all

    def to_legacy(self):
        data_all = [r.to_legacy() for r in self.regions]
        return data_all, self.idxgall, self.posi_all, self.idxcrop_all

    def __iter__(self):
        return iter(self.to_legacy())

    def __getitem__(self, idx):
        return self.to_legacy()[idx]

    def tree_flatten(self):
        return ((self.regions, self.idxgall, self.posi_all, self.idxcrop_all), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class PINNBatch:
    def __init__(self, smp, col, bd):
        self.smp = smp
        self.col = col
        self.bd = bd

    def to_legacy(self):
        return dict(smp=self.smp, col=self.col, bd=self.bd)

    def __getitem__(self, key):
        return self.to_legacy()[key]

    def __setitem__(self, key, value):
        if key == "smp":
            self.smp = value
        elif key == "col":
            self.col = value
        elif key == "bd":
            self.bd = value
        else:
            raise KeyError(key)

    def tree_flatten(self):
        return ((self.smp, self.col, self.bd), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


@jax.tree_util.register_pytree_node_class
class XPINNBatch:
    def __init__(self, smp, col, bd, md):
        self.smp = smp
        self.col = col
        self.bd = bd
        self.md = md

    def to_legacy(self):
        return dict(smp=self.smp, col=self.col, bd=self.bd, md=self.md)

    def __getitem__(self, key):
        return self.to_legacy()[key]

    def __setitem__(self, key, value):
        if key == "smp":
            self.smp = value
        elif key == "col":
            self.col = value
        elif key == "bd":
            self.bd = value
        elif key == "md":
            self.md = value
        else:
            raise KeyError(key)

    def tree_flatten(self):
        return ((self.smp, self.col, self.bd, self.md), None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)


def ensure_pinn_dataset(data_all):
    if isinstance(data_all, PINNDataset):
        return data_all
    if isinstance(data_all, (list, tuple)) and len(data_all) >= 5:
        warnings.warn("Legacy PINN dataset detected; auto-converting to PINNDataset.", DeprecationWarning, stacklevel=2)
        X_star, U_star, X_ct, nn_ct, data_info = data_all[0:5]
        boundary = data_all[5] if len(data_all) > 5 else None
        if isinstance(data_info, DataInfo):
            info = data_info
        else:
            scale_bundle = ScaleBundle(data_info[0], data_info[1], data_info[4] if len(data_info) > 6 else None)
            info = DataInfo(scale_bundle, data_info[2], data_info[3], data_info[-2], data_info[-1])
        return PINNDataset(PINNNormalizedData(X_star, U_star, X_ct, nn_ct, boundary), info)
    raise TypeError("Unsupported PINN dataset format")


def ensure_xpinn_dataset(data_all, idxgall=None, posi_all=None, idxcrop_all=None):
    if isinstance(data_all, XPINNDataset):
        return data_all

    if idxgall is None and isinstance(data_all, (list, tuple)) and len(data_all) == 4:
        data_all, idxgall, posi_all, idxcrop_all = data_all

    if isinstance(data_all, (list, tuple)) and idxgall is not None:
        warnings.warn("Legacy XPINN dataset detected; auto-converting to XPINNDataset.", DeprecationWarning, stacklevel=2)
        regions = []
        for region in data_all:
            X_star, U_star, X_ct, nn_ct, data_info, X_md = region[0:6]
            boundary = region[6] if len(region) > 6 else None
            if isinstance(data_info, DataInfo):
                info_obj = data_info
            else:
                scale_bundle = ScaleBundle(data_info[0], data_info[1], data_info[4] if len(data_info) > 6 else None)
                info_obj = DataInfo(scale_bundle, data_info[2], data_info[3], data_info[-2], data_info[-1])
            regions.append(XPINNRegionData(X_star, U_star, X_ct, nn_ct, info_obj, X_md, boundary))
        return XPINNDataset(regions, idxgall, posi_all, idxcrop_all)
    raise TypeError("Unsupported XPINN dataset format")


def ensure_pinn_batch(data):
    if isinstance(data, PINNBatch):
        return data
    if isinstance(data, dict):
        warnings.warn("Legacy PINN batch detected; auto-converting to PINNBatch.", DeprecationWarning, stacklevel=2)
        return PINNBatch(data["smp"], data["col"], data["bd"])
    raise TypeError("Unsupported PINN batch format")


def ensure_xpinn_batch(data):
    if isinstance(data, XPINNBatch):
        return data
    if isinstance(data, dict):
        warnings.warn("Legacy XPINN batch detected; auto-converting to XPINNBatch.", DeprecationWarning, stacklevel=2)
        return XPINNBatch(data["smp"], data["col"], data["bd"], data["md"])
    raise TypeError("Unsupported XPINN batch format")
