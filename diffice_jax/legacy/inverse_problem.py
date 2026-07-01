import copy
import json
import os
import pickle
import random as pyrnd
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from jax.tree_util import tree_map
from scipy.io import loadmat
from scipy.interpolate import griddata
from scipy.spatial import cKDTree

from ..data.xpinns.preprocessing import normalize_data as normdata_xpinn
from ..data.xpinns.sampling import data_sample_create as dsample_xpinn
from ..equation.eqn_iso import front_eqn as dbc_iso
from ..equation.eqn_iso import gov_eqn as ssa_iso
from ..model.architecture import pirate_last_layer_least_squares, resolve_architecture
from ..model.xpinns.initialization import init_nets as init_xpinn
from ..model.xpinns.loss import loss_iso_create as loss_iso_xpinn
from ..model.xpinns.networks import solu_create as solu_xpinn
from ..model.xpinns.prediction import predict as predict_xpinn
from ..optimizer.optimization import adam_optimizer as adam_opt
from ..optimizer.optimization import lbfgs_optimizer as lbfgs_opt
from .save_load import save_model
from .config import ensure_canonical_scale


@dataclass
class DIFFICEInverseProblem:
    config_raw: Dict[str, Any]
    config_resolved: Dict[str, Any]
    base_dir: str
    config_path: Optional[str] = None
    bundle_path: Optional[str] = None

    data_all: Any = None
    scale: Any = None
    idxgall: Any = None
    posi_all: Any = None
    idxcrop_all: Any = None
    basal_mask: Any = None
    region_indices: Any = None
    n_sub: Optional[int] = None
    rawdata: Any = None

    n_pt: Any = None
    n_pt2: Any = None
    dataf: Any = None
    dataf_l: Any = None
    eval_f: Any = None
    pred_u: Any = None
    grad_u: Any = None
    solNN: Any = None
    nn_loss: Any = None
    nn_loss_warmup: Any = None

    params: Any = None
    keys: Any = None
    keys_adam: Any = None
    key_lbfgs: Any = None

    loss_warmup: list = field(default_factory=list)
    loss_adam: list = field(default_factory=list)
    loss_lbfgs: Any = None

    prepared: bool = False
    built: bool = False
    solved: bool = False
    run_timestamp: Optional[str] = None
    run_folder_name: Optional[str] = None
    saved_path: Optional[str] = None
    schema_version: str = "1.0"

    @classmethod
    def from_config(
        cls,
        config_path: str | None = None,
        overrides: dict | None = None,
        base_dir: str | None = None,
    ) -> "DIFFICEInverseProblem":
        cfg_raw: Dict[str, Any] = {}
        if config_path is not None:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg_raw = json.load(f)
        if overrides:
            cfg_raw = cls._deep_merge_dicts(cfg_raw, overrides)
        cfg_resolved = cls._resolve_config(cfg_raw)
        resolved_base_dir = base_dir or os.getcwd()
        return cls(
            config_raw=cfg_raw,
            config_resolved=cfg_resolved,
            base_dir=resolved_base_dir,
            config_path=config_path,
        )

    @classmethod
    def from_bundle(cls, path: str, base_dir: str | None = None) -> "DIFFICEInverseProblem":
        with open(path, "rb") as f:
            model_dict = pickle.load(f)

        if model_dict.get("model_type") != "xpinn":
            raise ValueError(f"Expected XPINN bundle, got model_type={model_dict.get('model_type')}")

        bundle_cfg = model_dict.get("config", {})
        config_raw = bundle_cfg.get("config_raw", {})
        config_resolved = bundle_cfg.get("resolved_config", cls._resolve_config(config_raw))
        if not config_resolved:
            config_resolved = cls._resolve_config(bundle_cfg)

        resolved_base_dir = base_dir or os.getcwd()
        obj = cls(
            config_raw=config_raw,
            config_resolved=config_resolved,
            base_dir=resolved_base_dir,
            config_path=bundle_cfg.get("json_config_path"),
            bundle_path=path,
        )
        obj.params = model_dict["params"]
        obj.loss_warmup = bundle_cfg.get("loss_warmup", [])
        obj.loss_adam = bundle_cfg.get("loss_adam", [])
        obj.loss_lbfgs = bundle_cfg.get("loss_lbfgs", None)
        obj.run_timestamp = bundle_cfg.get("timestamp", None)
        obj.run_folder_name = bundle_cfg.get("folder_name", None)
        obj.saved_path = path
        obj.schema_version = str(bundle_cfg.get("inverse_problem_schema_version", "legacy"))

        has_data_snapshot = all(
            key in bundle_cfg for key in ("data_all", "scale", "idxgall", "posi_all", "idxcrop_all", "basal_mask")
        )
        if has_data_snapshot:
            obj.data_all = bundle_cfg["data_all"]
            obj.scale = bundle_cfg["scale"]
            obj.idxgall = bundle_cfg["idxgall"]
            obj.posi_all = bundle_cfg["posi_all"]
            obj.idxcrop_all = bundle_cfg["idxcrop_all"]
            obj.basal_mask = bundle_cfg["basal_mask"]
            obj.region_indices = bundle_cfg.get("region_indices")
            obj.n_sub = bundle_cfg.get("n_sub", len(obj.basal_mask))
            obj.prepared = True
        else:
            obj.prepare()

        obj._build_runtime(init_params=False)
        obj.built = True
        obj.solved = True
        return obj

    def prepare(self) -> "DIFFICEInverseProblem":
        cfg = self.config_resolved
        if cfg["data_path"] is None:
            raise ValueError("No data_path was provided in config or overrides.")

        data_mat = self._resolve_data_path(cfg["data_path"])
        rawdata = loadmat(data_mat)

        if "basal_mask" in rawdata:
            basal_mask_full = [bool(b) for b in rawdata["basal_mask"].flatten()]
        else:
            n_sub_detect = rawdata["xd"].shape[1]
            basal_mask_full = [False] * n_sub_detect

        region_indices = cfg.get("region_indices")
        if region_indices is None:
            region_indices = list(range(1, len(basal_mask_full)))
        if len(region_indices) == 0:
            raise ValueError("region_indices resolved to empty list.")

        # Keep all per-subregion cell arrays aligned after region filtering.
        # Optional fields (e.g., mud/Cd/alpha2d) are used later by RBF diagnostics.
        cell_keys = [
            "xd",
            "yd",
            "ud",
            "vd",
            "xd_h",
            "yd_h",
            "hd",
            "sd",
            "xct",
            "yct",
            "nnct",
            "mud",
            "Cd",
            "alpha2d",
        ]
        for key in cell_keys:
            if key in rawdata and rawdata[key].dtype == object:
                rawdata[key] = rawdata[key][:, region_indices]

        basal_mask = [basal_mask_full[i] for i in region_indices]
        n_sub = len(region_indices)

        x_md_full = rawdata.get("x_md")
        y_md_full = rawdata.get("y_md")
        if x_md_full is not None and y_md_full is not None and n_sub > 1:
            x_md_new = np.empty((1, n_sub - 1), dtype=object)
            y_md_new = np.empty((1, n_sub - 1), dtype=object)
            for k in range(n_sub - 1):
                ri = region_indices[k]
                rj = region_indices[k + 1]
                iface_idx = min(ri, rj)
                if iface_idx >= x_md_full.shape[1]:
                    raise ValueError(f"Cannot find interface between regions {ri} and {rj}")
                x_md_new[0, k] = x_md_full[0, iface_idx]
                y_md_new[0, k] = y_md_full[0, iface_idx]
            rawdata["x_md"] = x_md_new
            rawdata["y_md"] = y_md_new

        idxcrop_orig = rawdata.get("idxcrop")
        idxcrop_h_orig = rawdata.get("idxcrop_h")
        if idxcrop_orig is not None and idxcrop_h_orig is not None:
            idxcrop_new = np.empty((1, n_sub), dtype=object)
            idxcrop_h_new = np.empty((1, n_sub), dtype=object)
            for i, ri in enumerate(region_indices):
                idxcrop_new[0, i] = idxcrop_orig[0, ri]
                idxcrop_h_new[0, i] = idxcrop_h_orig[0, ri]
            rawdata["idxcrop"] = idxcrop_new
            rawdata["idxcrop_h"] = idxcrop_h_new

        data_all, idxgall, posi_all, idxcrop_all = normdata_xpinn(rawdata, basal_mask=basal_mask, diagnostic=False)
        scale = tree_map(lambda x: data_all[x][4][0:2], idxgall)

        gamma_c = cfg["gamma_c"]
        if gamma_c is not None and len(gamma_c) != len(idxgall):
            raise ValueError(f"gamma_c length ({len(gamma_c)}) must match active regions ({len(idxgall)}).")

        sp = cfg["sampling_points"]
        self.n_pt = [
            jnp.array(sp["velocity"]),
            jnp.array(sp["thickness"]),
            jnp.array(sp["collocation"]),
            jnp.array(sp["boundary"]),
            jnp.array(sp["interface"]),
        ]

        self.n_pt2 = None
        if cfg["use_lbfgs"]:
            self.n_pt2 = [
                jnp.array([2 * sp["velocity"][0], sp["velocity"][1]])
                if isinstance(sp["velocity"], list)
                else jnp.array(2 * sp["velocity"]),
                jnp.array([2 * sp["thickness"][0], sp["thickness"][1]])
                if isinstance(sp["thickness"], list)
                else jnp.array(2 * sp["thickness"]),
                jnp.array([2 * sp["collocation"][0], sp["collocation"][1]])
                if isinstance(sp["collocation"], list)
                else jnp.array(2 * sp["collocation"]),
                jnp.array(sp["boundary"]),
                jnp.array(sp["interface"]),
            ]

        self.data_all = data_all
        self.scale = scale
        self.idxgall = idxgall
        self.posi_all = posi_all
        self.idxcrop_all = idxcrop_all
        self.basal_mask = basal_mask
        self.region_indices = region_indices
        self.n_sub = len(idxgall)
        self.rawdata = rawdata
        self.config_resolved["data_source"] = data_mat

        self.prepared = True
        return self

    def build(self) -> "DIFFICEInverseProblem":
        if not self.prepared:
            self.prepare()
        self._init_run_identity()
        self._build_runtime(init_params=True)
        self.built = True
        return self

    def solve(self) -> "DIFFICEInverseProblem":
        if not self.built:
            self.build()

        cfg = self.config_resolved
        self.loss_warmup = []
        self.loss_adam = []
        self.loss_lbfgs = None

        if cfg["lbpinn_config"]["data_warmup_epochs"] > 0 and self.nn_loss_warmup is not None:
            self.params, self.loss_warmup = adam_opt(
                self.keys_adam[0],
                self.nn_loss_warmup,
                self.params,
                self.dataf,
                cfg["lbpinn_config"]["data_warmup_epochs"],
                lr=cfg["lr"],
                adaptive=False,
                eval_f=self.eval_f,
                adapt_period=cfg["sampling_points"]["adapt_period"],
                adapt_burnin=cfg["sampling_points"]["adapt_burnin"],
                use_lbpinn=False,
                n_sub=self.n_sub,
                lbpinn_config=cfg["lbpinn_config"],
                use_grad_adapt=False,
            )

        if cfg["adam_epochs"] > 0:
            eps_init = None
            if cfg["use_lbpinn"]:
                data0 = self.dataf(self.keys_adam[0])
                _, l_info = self.nn_loss(self.params, data0)
                l_d = float(jnp.maximum(l_info[1], 1e-6))
                l_e = float(jnp.maximum(l_info[2], 1e-6))
                l_b = float(jnp.maximum(l_info[3], 1e-6))
                l_m = float(jnp.maximum(l_info[4], 1e-6))
                eps_init = [float(np.sqrt(l_d)), float(np.sqrt(l_e)), float(np.sqrt(l_b)), float(np.sqrt(l_m))]

            self.params, self.loss_adam = adam_opt(
                self.keys_adam[1],
                self.nn_loss,
                self.params,
                self.dataf,
                cfg["adam_epochs"],
                lr=cfg["lr"],
                adaptive=cfg["sampling_points"]["adaptive"],
                eval_f=self.eval_f,
                adapt_period=cfg["sampling_points"]["adapt_period"],
                adapt_burnin=cfg["sampling_points"]["adapt_burnin"],
                use_lbpinn=cfg["use_lbpinn"],
                n_sub=self.n_sub,
                eps_init=eps_init,
                lbpinn_config=cfg["lbpinn_config"],
                use_grad_adapt=cfg["use_grad_adapt"],
                adapt_grad_period=cfg["adapt_grad_period"],
                embedding=cfg["fourier_features"]["enabled"],
                fourier_anneal=cfg["fourier_features"]["anneal"],
            )

        if cfg["use_lbfgs"] and cfg["lbfgs_epochs"] > 0:
            data_l = self.dataf_l(self.key_lbfgs)
            if cfg["use_lbpinn"] and cfg["freeze_lbpinn"] and isinstance(self.params, tuple):
                fixed_log_eps = self.params[0]
                net_params = self.params[1]
                net_params, self.loss_lbfgs = lbfgs_opt(
                    self.nn_loss,
                    net_params,
                    data_l,
                    cfg["lbfgs_epochs"],
                    basal=True,
                    use_lbpinn=True,
                    fixed_log_eps=fixed_log_eps,
                )
                self.params = (fixed_log_eps, net_params)
            else:
                self.params, self.loss_lbfgs = lbfgs_opt(
                    self.nn_loss,
                    self.params,
                    data_l,
                    cfg["lbfgs_epochs"],
                    basal=True,
                    use_lbpinn=cfg["use_lbpinn"],
                    fixed_log_eps=None,
                )

        self.solved = True
        return self

    def predict_fields(self, aniso: bool = False) -> Dict[str, Any]:
        if self.params is None or self.pred_u is None:
            raise RuntimeError("Model is not built/loaded with params.")
        net_params = self._extract_net_params(self.params)
        f_u_idx = lambda x, idx: self.pred_u(net_params, x, idx)
        func_all = [f_u_idx, ssa_iso]
        return predict_xpinn(
            func_all,
            self.data_all,
            self.posi_all,
            self.idxcrop_all,
            self.idxgall,
            aniso=aniso,
            basal_mask=self.basal_mask,
            gamma_c=self.config_resolved["gamma_c"],
        )

    def net(self, x_physical: Any, idx: int, normalize: bool = False) -> jnp.ndarray:
        """
        Evaluate the trained network at PHYSICAL coordinates.

        Args:
            x_physical: array-like with shape (N, 2), physical coordinates.
            idx: active XPINN subregion index (0..n_sub-1).
            normalize: if True, return raw normalized network outputs.
                      if False, return re-dimensionalized physical outputs.

        Returns:
            Network outputs [u, v, h, s, mu] and [C] when grounded.
        """
        if self.pred_u is None or self.params is None:
            raise RuntimeError("Model is not built/loaded with params.")
        x_n, s = self._normalize_coordinates(x_physical, idx)
        net_params = self._extract_net_params(self.params)
        out_n = self.pred_u(net_params, x_n, idx)
        if normalize:
            return out_n

        u = out_n[:, 0:1] * s["u0"] + s["um"]
        v = out_n[:, 1:2] * s["v0"] + s["vm"]
        h = out_n[:, 2:3] * s["h0"]
        surf = out_n[:, 3:4] * s["s0"] + s["sm"]
        mu = out_n[:, 4:5] * s["mu0"]
        cols = [u, v, h, surf, mu]
        if self.basal_mask[idx] and out_n.shape[1] > 5:
            c = out_n[:, 5:6] * s["c0"]
            cols.append(c)
        return jnp.hstack(cols)

    def eqn_residuals(
        self,
        x_physical: Any,
        idx: int,
        boundary_normals: Any | None = None,
        normalize: bool = False,
    ) -> Dict[str, jnp.ndarray]:
        """
        Evaluate equation residuals at PHYSICAL coordinates.

        Args:
            x_physical: array-like with shape (N, 2), physical coordinates.
            idx: active XPINN subregion index (0..n_sub-1).
            boundary_normals: if provided, evaluate front_eqn residuals; otherwise gov_eqn.
            normalize: if True, return normalized residuals/terms as produced by eqn functions.
                      if False, return re-dimensionalized residuals/terms.

        Returns:
            Dict with `residual` and `terms`.
        """
        if self.pred_u is None or self.params is None:
            raise RuntimeError("Model is not built/loaded with params.")

        x_n, s = self._normalize_coordinates(x_physical, idx)
        net_params = self._extract_net_params(self.params)
        net = lambda z: self.pred_u(net_params, z, idx)
        gamma_c = self.config_resolved["gamma_c"]
        gamma_i = gamma_c[idx] if gamma_c is not None else None

        if boundary_normals is None:
            residual_n, terms_n = ssa_iso(
                net, x_n, self.scale[idx], basal=self.basal_mask[idx], gamma_c=gamma_i
            )
            if normalize:
                return {"residual": residual_n, "terms": terms_n}
            fac = s["term0"]
            return {"residual": residual_n * fac, "terms": terms_n * fac}

        nn = jnp.asarray(boundary_normals)
        residual_n, terms_n = dbc_iso(net, x_n, nn, self.scale[idx])
        if normalize:
            return {"residual": residual_n, "terms": terms_n}
        fac = s["term_bd"]
        return {"residual": residual_n * fac, "terms": terms_n * fac}

    def eqn_residual_from_data(
        self,
        coords_phys: Any,
        region_id: int | None = None,
        fields: Dict[str, Any] | None = None,
        mu_source: str = "data",
        c_source: str = "data",
        B: float | None = None,
        D: float | None = None,
        m: float | None = None,
        n: float | None = None,
        normalize: bool = False,
        rbf_mode: str = "local",
        rbf_kwargs: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Evaluate SSA equation residual terms directly from data in PHYSICAL units
        using RBF-based differentiation/interpolation.

        Args:
            coords_phys: array-like, shape (N, 2), physical coordinates where terms are evaluated.
            region_id: optional active region index. If None, infer from region scales.
            fields: optional explicit field dictionary; when absent, load from self.rawdata.
            mu_source: "data" | "param_B" | "nn"
            c_source: "data" | "param_D" | "nn"
            B: viscosity parameter for mu_source="param_B".
            D: basal friction prefactor for c_source="param_D".
            m: friction exponent for c_source="param_D". Defaults to config value or 3.
            n: flow-law exponent for mu_source="param_B". Defaults to config value or 3.
            normalize: if False (default), return physical units; if True, return normalized quantities.
            rbf_mode: "local" (default) or "global".
            rbf_kwargs: optional dict with RBF settings: {"k": int, "eps": float, "reg": float}.
        """
        print(f'[DIFFICEInverseProblem INFO]: Using viscosity from [{mu_source}] and friction from [{c_source}] to evaluate equation residuals from data.')
        xq = np.asarray(coords_phys, dtype=float)
        if xq.ndim != 2 or xq.shape[1] != 2:
            raise ValueError(f"coords_phys must have shape (N, 2), got {xq.shape}")

        idx = self._resolve_region_id(xq, region_id)
        basal = bool(self.basal_mask[idx]) if self.basal_mask is not None else False
        scale_i = ensure_canonical_scale(
            self.scale[idx],
            basal=basal,
            gamma_c=self.config_resolved["gamma_c"],
            mode="xpinn",
        )
        print('[DIFFICEInverseProblem INFO]: scale_i keys = ', scale_i.keys())

        support = self._load_physical_support_fields(idx, fields)
        rbf_kwargs = {} if rbf_kwargs is None else dict(rbf_kwargs)
        k = int(rbf_kwargs.get("k", 35))
        eps = float(rbf_kwargs.get("eps", 0.01))
        reg = float(rbf_kwargs.get("reg", 1e-10))
        support_seed = int(rbf_kwargs.get("subsample_seed", 42))
        max_support_points = rbf_kwargs.get("max_support_points", None)
        interp_method = str(rbf_kwargs.get("interp_method", "linear"))

        vel_xy_full = np.hstack([support["x_vel"], support["y_vel"]])
        support_idx = self._spatial_subsample_indices(vel_xy_full, max_support_points, seed=support_seed)
        xsv = support["x_vel"][support_idx]
        ysv = support["y_vel"][support_idx]
        support_xy = np.hstack([xsv, ysv])

        u_sup = support["u"][support_idx]
        v_sup = support["v"][support_idx]
        hs_sup = self._interp_to_queries(
            np.hstack([support["x_h"], support["y_h"]]),
            np.hstack([support["h"], support["s"]]),
            support_xy,
            method=interp_method,
        )
        h_sup = hs_sup[:, 0:1]
        s_sup = hs_sup[:, 1:2]

        print(
            f"[DIFFICEInverseProblem INFO]: RBF support reduced from {vel_xy_full.shape[0]} "
            f"to {support_xy.shape[0]} points for eqn_residual_from_data."
        )

        vel_sup = self._rbf_apply(
            xsv,
            ysv,
            np.hstack([u_sup, v_sup]),
            xsv,
            ysv,
            ops=("val", "dx", "dy", "dxx", "dyy"),
            mode=rbf_mode,
            k=k,
            eps=eps,
            reg=reg,
        )
        u_x_sup = vel_sup["dx"][:, 0:1]
        u_y_sup = vel_sup["dy"][:, 0:1]
        u_xx_sup = vel_sup["dxx"][:, 0:1]
        u_yy_sup = vel_sup["dyy"][:, 0:1]
        v_x_sup = vel_sup["dx"][:, 1:2]
        v_y_sup = vel_sup["dy"][:, 1:2]
        v_xx_sup = vel_sup["dxx"][:, 1:2]
        v_yy_sup = vel_sup["dyy"][:, 1:2]

        hs_deriv_sup = self._rbf_apply(
            xsv,
            ysv,
            np.hstack([h_sup, s_sup]),
            xsv,
            ysv,
            ops=("val", "dx", "dy"),
            mode=rbf_mode,
            k=k,
            eps=eps,
            reg=reg,
        )
        h_x_sup = hs_deriv_sup["dx"][:, 0:1]
        h_y_sup = hs_deriv_sup["dy"][:, 0:1]
        s_x_sup = hs_deriv_sup["dx"][:, 1:2]
        s_y_sup = hs_deriv_sup["dy"][:, 1:2]

        n_flow = float(
            n
            if n is not None
            else self.config_resolved.get("flow_law_n", self.config_resolved.get("rheology_n", 3.0))
        )
        m_fric = float(m if m is not None else self.config_resolved.get("friction_m", 3.0))

        if mu_source not in {"data", "param_B", "nn"}:
            raise ValueError(f"Unsupported mu_source={mu_source!r}. Use 'data', 'param_B', or 'nn'.")
        if c_source not in {"data", "param_D", "nn"}:
            raise ValueError(f"Unsupported c_source={c_source!r}. Use 'data', 'param_D', or 'nn'.")

        if mu_source == "data":
            if support.get("mu") is None:
                raise ValueError(
                    "mu_source='data' requested but viscosity field is missing. "
                    "Choose mu_source='param_B' (and provide B) or mu_source='nn'."
                )
            mu_sup = self._interp_to_queries(
                np.hstack([support["x_mu"], support["y_mu"]]),
                support["mu"],
                support_xy,
                method=interp_method,
            )
        elif mu_source == "param_B":
            if B is None:
                raise ValueError("mu_source='param_B' requires B.")
            sr_xy_sup = 0.5 * (u_y_sup + v_x_sup)
            sr_eff_sup = np.sqrt(np.maximum(u_x_sup**2 + v_y_sup**2 + sr_xy_sup**2 + u_x_sup * v_y_sup, 1e-30))
            mu_sup = 0.5 * float(B) * np.power(sr_eff_sup, 1.0 / n_flow - 1.0)
        else:  # mu_source == "nn"
            if self.params is None or self.pred_u is None:
                raise RuntimeError("mu_source='nn' requires a built/loaded neural network model.")
            nn_out_sup = np.asarray(self.net(support_xy, idx=idx, normalize=False))
            mu_sup = nn_out_sup[:, 4:5]

        if basal:
            if c_source == "data":
                if support.get("c") is None:
                    raise ValueError(
                        "c_source='data' requested but friction field is missing. "
                        "Choose c_source='param_D' (and provide D, m) or c_source='nn'."
                    )
                c_sup = self._interp_to_queries(
                    np.hstack([support["x_c"], support["y_c"]]),
                    support["c"],
                    support_xy,
                    method=interp_method,
                )
            elif c_source == "param_D":
                if D is None:
                    raise ValueError("c_source='param_D' requires D.")
                speed2_sup = np.maximum(u_sup**2 + v_sup**2, 1e-30)
                c_sup = float(D) * np.power(speed2_sup, (1.0 / m_fric - 1.0) / 2.0)
            else:  # c_source == "nn"
                if self.params is None or self.pred_u is None:
                    raise RuntimeError("c_source='nn' requires a built/loaded neural network model.")
                nn_out_sup = np.asarray(self.net(support_xy, idx=idx, normalize=False))
                if nn_out_sup.shape[1] < 6:
                    raise ValueError("c_source='nn' requested, but this region network does not output C.")
                c_sup = nn_out_sup[:, 5:6]
        else:
            c_sup = np.zeros_like(u_sup)

        Rxx_sup = 4.0 * mu_sup * h_sup * u_x_sup + 2.0 * mu_sup * h_sup * v_y_sup
        Ryy_sup = 4.0 * mu_sup * h_sup * v_y_sup + 2.0 * mu_sup * h_sup * u_x_sup
        Rxy_sup = mu_sup * h_sup * (u_y_sup + v_x_sup)
        stress_grad_sup = self._rbf_apply(
            xsv,
            ysv,
            np.hstack([Rxx_sup, Ryy_sup, Rxy_sup]),
            xsv,
            ysv,
            ops=("dx", "dy"),
            mode=rbf_mode,
            k=k,
            eps=eps,
            reg=reg,
        )
        dRxx_dx_sup = stress_grad_sup["dx"][:, 0:1]
        dRyy_dy_sup = stress_grad_sup["dy"][:, 1:2]
        dRxy_dx_sup = stress_grad_sup["dx"][:, 2:3]
        dRxy_dy_sup = stress_grad_sup["dy"][:, 2:3]

        rho_i = float(self.config_resolved.get("rho", 917.0))
        g = float(self.config_resolved.get("g", 9.8))
        rho_w = float(self.config_resolved.get("rho_w", 1023))
        print('[DIFFICEInverseProblem INFO]: config_resolved keys = ', self.config_resolved.keys())
        gamma_c = 0.33 if basal else 0.0
        visc_x_sup = dRxx_dx_sup + dRxy_dy_sup
        visc_y_sup = dRyy_dy_sup + dRxy_dx_sup

        SIGMA_G = scale_i['term0']
        SIGMA_MU = (1-gamma_c)*SIGMA_G 
        SIGMA_C = gamma_c * SIGMA_G
        # Match eqn_iso convention:
        # - grounded/basal regions use h * grad(surface)
        # - floating regions use h * grad(thickness)
        if basal:
            grav_x_sup = rho_i * g * h_sup * s_x_sup
            grav_y_sup = rho_i * g * h_sup * s_y_sup
        else:
            grav_x_sup = rho_i * (1 - rho_i/rho_w) * g * h_sup * h_x_sup
            grav_y_sup = rho_i * (1 - rho_i/rho_w) * g * h_sup * h_y_sup
        basal_x_sup = c_sup * u_sup
        basal_y_sup = c_sup * v_sup

        residual_x_sup = visc_x_sup - grav_x_sup - basal_x_sup
        residual_y_sup = visc_y_sup - grav_y_sup - basal_y_sup

        query_xy = xq
        field_query = self._interp_to_queries(
            support_xy,
            np.hstack([
                u_sup, v_sup, h_sup, s_sup, mu_sup, c_sup,
                u_x_sup, u_y_sup, u_xx_sup, u_yy_sup,
                v_x_sup, v_y_sup, v_xx_sup, v_yy_sup,
                h_x_sup, h_y_sup, s_x_sup, s_y_sup,
                Rxx_sup, Ryy_sup, Rxy_sup,
                visc_x_sup, visc_y_sup, grav_x_sup, grav_y_sup,
                basal_x_sup, basal_y_sup, residual_x_sup, residual_y_sup,
            ]),
            query_xy,
            method=interp_method,
        )
        (
            u, v, h, s, mu, c,
            u_x, u_y, u_xx, u_yy,
            v_x, v_y, v_xx, v_yy,
            h_x, h_y, s_x, s_y,
            Rxx, Ryy, Rxy,
            visc_x, visc_y, grav_x, grav_y,
            basal_x, basal_y, residual_x, residual_y,
        ) = [field_query[:, i:i+1] for i in range(field_query.shape[1])]
        residual = np.hstack([residual_x, residual_y])

        result = {
            "region_id": idx,
            "coords_physical": jnp.asarray(xq),
            "fields": {
                "u": jnp.asarray(u),
                "v": jnp.asarray(v),
                "h": jnp.asarray(h),
                "s": jnp.asarray(s),
                "mu": jnp.asarray(mu),
                "C": jnp.asarray(c),
            },
            "derivatives": {
                "u_x": jnp.asarray(u_x),
                "u_y": jnp.asarray(u_y),
                "u_xx": jnp.asarray(u_xx),
                "u_yy": jnp.asarray(u_yy),
                "v_x": jnp.asarray(v_x),
                "v_y": jnp.asarray(v_y),
                "v_xx": jnp.asarray(v_xx),
                "v_yy": jnp.asarray(v_yy),
                "h_x": jnp.asarray(h_x),
                "h_y": jnp.asarray(h_y),
                "s_x": jnp.asarray(s_x),
                "s_y": jnp.asarray(s_y),
            },
            "stresses": {
                "Rxx": jnp.asarray(Rxx),
                "Ryy": jnp.asarray(Ryy),
                "Rxy": jnp.asarray(Rxy),
            },
            "terms": {
                "visc_x": jnp.asarray(visc_x),
                "visc_y": jnp.asarray(visc_y),
                "grav_x": jnp.asarray(grav_x),
                "grav_y": jnp.asarray(grav_y),
                "basal_x": jnp.asarray(basal_x),
                "basal_y": jnp.asarray(basal_y),
                "residual_x": jnp.asarray(residual_x),
                "residual_y": jnp.asarray(residual_y),
            },
            "residual": jnp.asarray(residual),
            "sources": {"mu_source": mu_source, "c_source": c_source},
        }
        if not normalize:
            return result
        return self._normalize_data_residual_result(result, scale_i, basal)

    def save(self, path: str | None = None) -> str:
        if self.params is None:
            raise RuntimeError("Cannot save before build/solve/from_bundle.")

        if path is None:
            if self.run_timestamp is None:
                self._init_run_identity()
            save_dir = os.path.join(self.base_dir, "saved", self.run_folder_name)
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, "inference_data.pkl")
        else:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        config_payload = {
            "inverse_problem_schema_version": self.schema_version,
            "timestamp": self.run_timestamp,
            "folder_name": self.run_folder_name,
            "json_config_path": self.config_path,
            "config_raw": self.config_raw,
            "resolved_config": self.config_resolved,
            "data_source": self.config_resolved.get("data_source"),
            "data_all": self.data_all,
            "scale": self.scale,
            "idxgall": self.idxgall,
            "posi_all": self.posi_all,
            "idxcrop_all": self.idxcrop_all,
            "basal_mask": self.basal_mask,
            "region_indices": self.region_indices,
            "n_sub": self.n_sub,
            "loss_warmup": self.loss_warmup,
            "loss_adam": self.loss_adam,
            "loss_lbfgs": self.loss_lbfgs,
            "embedding": self.config_resolved["fourier_features"]["enabled"],
            "use_rwf": self.config_resolved["use_rwf"],
            "use_modified_mlp": self.config_resolved["use_modified_mlp"],
            "architecture": self.config_resolved["network"]["architecture"],
            "pirate_init": self.config_resolved["network"]["pirate_init"],
        }
        save_model(path, self.params, model_type="xpinn", **config_payload)
        self.saved_path = path
        return path

    def summary(self) -> Dict[str, Any]:
        return {
            "prepared": self.prepared,
            "built": self.built,
            "solved": self.solved,
            "config_path": self.config_path,
            "bundle_path": self.bundle_path,
            "saved_path": self.saved_path,
            "data_source": self.config_resolved.get("data_source"),
            "n_sub": self.n_sub,
            "region_indices": self.region_indices,
            "basal_mask": self.basal_mask,
            "network": self.config_resolved.get("network"),
            "features": {
                "fourier_features": self.config_resolved["fourier_features"]["enabled"],
                "use_rwf": self.config_resolved["use_rwf"],
                "use_modified_mlp": self.config_resolved["use_modified_mlp"],
                "architecture": self.config_resolved["network"]["architecture"],
                "pirate_init": self.config_resolved["network"]["pirate_init"],
                "use_grad_adapt": self.config_resolved["use_grad_adapt"],
                "use_lbpinn": self.config_resolved["use_lbpinn"],
            },
            "epochs": {
                "adam": self.config_resolved["adam_epochs"],
                "lbfgs": self.config_resolved["lbfgs_epochs"],
                "warmup": self.config_resolved["lbpinn_config"].get("data_warmup_epochs", 0),
            },
            "backend": jax.default_backend(),
        }

    def _build_runtime(self, init_params: bool) -> None:
        cfg = self.config_resolved
        if not self.prepared:
            raise RuntimeError("Call prepare() before build runtime.")

        seed = cfg["seed"]
        key = random.PRNGKey(seed)
        np.random.seed(seed)
        self.keys = random.split(key, 4)
        self.keys_adam = random.split(self.keys[1], 5)
        self.key_lbfgs = self.keys[2]

        if init_params or self.params is None:
            self.params = init_xpinn(
                self.keys[0],
                cfg["network"]["n_hidden_layers"],
                cfg["network"]["n_units_per_layer"],
                n_sub=self.n_sub,
                basal_mask=self.basal_mask,
                embedding=cfg["fourier_features"]["enabled"],
                embed_n=cfg["fourier_features"]["embed_n"],
                embed_std=cfg["fourier_features"]["embed_std"],
                use_rwf=cfg["use_rwf"],
                use_modified_mlp=cfg["use_modified_mlp"],
                architecture=cfg["network"]["architecture"],
            )
            self.params = self._maybe_initialize_pirate_last_layers(self.params)

        self.pred_u, self.grad_u = solu_xpinn(
            self.scale,
            embedding=cfg["fourier_features"]["enabled"],
            basal_mask=self.basal_mask,
            use_rwf=cfg["use_rwf"],
            use_modified_mlp=cfg["use_modified_mlp"],
            architecture=cfg["network"]["architecture"],
        )
        self.solNN = (self.pred_u, self.grad_u)
        self.eval_f = lambda params, x, idx: ssa_iso(
            lambda z: self.pred_u(self._extract_net_params(params), z, idx),
            x,
            self.scale[idx],
            basal=self.basal_mask[idx],
            gamma_c=cfg["gamma_c"][idx] if cfg["gamma_c"] is not None else None,
        )

        self.dataf = dsample_xpinn(self.data_all, self.idxgall, self.n_pt)
        if cfg["use_lbfgs"] and self.n_pt2 is not None:
            self.dataf_l = dsample_xpinn(self.data_all, self.idxgall, self.n_pt2)

        data_sampled = self.dataf(self.keys_adam[0])
        eqn_all = (ssa_iso, dbc_iso)
        self.nn_loss = loss_iso_xpinn(
            self.solNN,
            eqn_all,
            self.scale,
            self.idxgall,
            cfg["loss_weights"],
            basal_mask=self.basal_mask,
            gamma_c=cfg["gamma_c"],
            diagnostic=False,
            loss_flags=cfg["loss_flags"],
            lbpinn_config=cfg["lbpinn_config"],
        )
        self.nn_loss.lref = self.nn_loss(self.params, data_sampled)[0]

        loss_flags_warmup = {"data": True, "eqn": False, "bd": False, "match": True}
        self.nn_loss_warmup = loss_iso_xpinn(
            self.solNN,
            eqn_all,
            self.scale,
            self.idxgall,
            [1.0, 0.0, 0.0, 1.0],
            basal_mask=self.basal_mask,
            gamma_c=cfg["gamma_c"],
            diagnostic=False,
            loss_flags=loss_flags_warmup,
            lbpinn_config=cfg["lbpinn_config"],
        )
        self.nn_loss_warmup.lref = self.nn_loss_warmup(self.params, data_sampled)[0]

    def _maybe_initialize_pirate_last_layers(self, params):
        cfg = self.config_resolved
        pirate_cfg = cfg["network"]["pirate_init"]
        if cfg["network"]["architecture"] != "pirate" or not pirate_cfg.get("enabled", False):
            return params
        if pirate_cfg.get("source", "none") != "observed_fields":
            return params

        sample_size = int(pirate_cfg.get("sample_size", 0) or 0)
        ridge = float(pirate_cfg.get("ridge", 1e-6))
        embedding = cfg["fourier_features"]["enabled"]
        use_rwf = cfg["use_rwf"]

        net_u = list(params["net_u"])
        net_mu = list(params["net_mu"])
        net_c = list(params["net_c"])

        for idx, region in enumerate(self.data_all.regions):
            x_vel = region.X_star[0]
            x_h = region.X_star[1]
            u_star = region.U_star

            def maybe_take(x, y):
                if sample_size > 0 and x.shape[0] > sample_size:
                    return x[:sample_size], y[:sample_size]
                return x, y

            x_uv, y_uv = maybe_take(x_vel, u_star[0])
            x_hh, y_h = maybe_take(x_h, u_star[1])
            x_hs, y_s = maybe_take(x_h, u_star[2])
            obs_u = [
                (x_uv, y_uv[:, 0:1], 0),
                (x_uv, y_uv[:, 1:2], 1),
                (x_hh, y_h, 2),
                (x_hs, y_s, 3),
            ]
            net_u[idx] = pirate_last_layer_least_squares(
                net_u[idx],
                observations=obs_u,
                embedding=embedding,
                use_rwf=use_rwf,
                ridge=ridge,
            )

            if len(u_star) > 3:
                x_mu, y_mu = maybe_take(x_vel, u_star[3])
                net_mu[idx] = pirate_last_layer_least_squares(
                    net_mu[idx],
                    observations=[(x_mu, y_mu, 0)],
                    embedding=embedding,
                    use_rwf=use_rwf,
                    ridge=ridge,
                )
            if net_c[idx] is not None and len(u_star) > 4:
                x_c, y_c = maybe_take(x_vel, u_star[4])
                net_c[idx] = pirate_last_layer_least_squares(
                    net_c[idx],
                    observations=[(x_c, y_c, 0)],
                    embedding=embedding,
                    use_rwf=use_rwf,
                    ridge=ridge,
                )

        out = dict(params)
        out["net_u"] = net_u
        out["net_mu"] = net_mu
        out["net_c"] = net_c
        return out

    def _normalize_coordinates(self, x_physical: Any, idx: int):
        x_phys = jnp.asarray(x_physical)
        if x_phys.ndim != 2 or x_phys.shape[1] != 2:
            raise ValueError(f"x_physical must have shape (N, 2), got {x_phys.shape}")
        s = ensure_canonical_scale(
            self.scale[idx],
            basal=self.basal_mask[idx],
            gamma_c=self.config_resolved["gamma_c"],
            mode="xpinn",
        )
        x_n = jnp.empty_like(x_phys)
        x_n = x_n.at[:, 0].set((x_phys[:, 0] - s["lxm"]) / s["lx0"])
        x_n = x_n.at[:, 1].set((x_phys[:, 1] - s["lym"]) / s["ly0"])
        return x_n, s

    @staticmethod
    def _extract_net_params(params: Any) -> Any:
        if isinstance(params, tuple) and len(params) == 2 and isinstance(params[0], list):
            return params[1]
        return params

    def _resolve_data_path(self, data_path: str) -> str:
        if os.path.isabs(data_path):
            return data_path
        return os.path.join(self.base_dir, data_path)

    def _resolve_region_id(self, coords_phys: np.ndarray, region_id: int | None) -> int:
        if self.n_sub is None:
            self.n_sub = len(self.scale) if self.scale is not None else 1
        if region_id is not None:
            idx = int(region_id)
            if idx < 0 or idx >= int(self.n_sub):
                raise ValueError(f"region_id={region_id} out of range [0, {int(self.n_sub) - 1}]")
            return idx
        if int(self.n_sub) == 1:
            return 0
        qmean = np.mean(np.asarray(coords_phys), axis=0)
        centers = []
        for i in range(int(self.n_sub)):
            s = ensure_canonical_scale(
                self.scale[i],
                basal=self.basal_mask[i],
                gamma_c=self.config_resolved["gamma_c"],
                mode="xpinn",
            )
            centers.append([float(s["lxm"]), float(s["lym"])])
        centers = np.asarray(centers)
        return int(np.argmin(np.sum((centers - qmean[None, :]) ** 2, axis=1)))

    def _load_physical_support_fields(self, idx: int, fields: Dict[str, Any] | None) -> Dict[str, np.ndarray]:
        if fields is not None:
            return self._coerce_physical_fields_dict(fields)
        if self.rawdata is None:
            raise RuntimeError("No rawdata available. Provide `fields` or call prepare()/from_bundle with data snapshot.")

        x_vel, y_vel, u, v = self._extract_raw_field_quad("xd", "yd", "ud", "vd", idx)
        x_h, y_h, h, s = self._extract_raw_field_quad("xd_h", "yd_h", "hd", "sd", idx)
        x_vel, y_vel, u, v = self._drop_nan_rows(x_vel, y_vel, u, v)
        x_h, y_h, h, s = self._drop_nan_rows(x_h, y_h, h, s)
        out = {
            "x_vel": x_vel,
            "y_vel": y_vel,
            "u": u,
            "v": v,
            "x_h": x_h,
            "y_h": y_h,
            "h": h,
            "s": s,
        }

        mu_vals = self._extract_raw_optional("mud", idx)
        if mu_vals is not None:
            # Some legacy datasets keep mud on a different grid than velocity points.
            # Prefer velocity coordinates when lengths match; otherwise fall back to
            # thickness coordinates if compatible.
            if mu_vals.shape[0] == x_vel.shape[0]:
                x_mu_src, y_mu_src = x_vel, y_vel
            elif mu_vals.shape[0] == x_h.shape[0]:
                x_mu_src, y_mu_src = x_h, y_h
            else:
                raise ValueError(
                    f"mud length mismatch for region {idx}: "
                    f"len(mud)={mu_vals.shape[0]}, len(x_vel)={x_vel.shape[0]}, len(x_h)={x_h.shape[0]}. "
                    "Check region indexing or provide explicit `fields` with x_mu/y_mu."
                )
            x_mu, y_mu, mu = self._drop_nan_rows(x_mu_src, y_mu_src, mu_vals)
            out["x_mu"] = x_mu
            out["y_mu"] = y_mu
            out["mu"] = mu
        else:
            out["x_mu"] = x_vel
            out["y_mu"] = y_vel
            out["mu"] = None

        c_vals = self._extract_raw_optional("alpha2d", idx)
        if c_vals is None:
            c_vals = self._extract_raw_optional("Cd", idx)
        if c_vals is not None:
            # Same grid-compatibility handling as viscosity.
            if c_vals.shape[0] == x_vel.shape[0]:
                x_c_src, y_c_src = x_vel, y_vel
            elif c_vals.shape[0] == x_h.shape[0]:
                x_c_src, y_c_src = x_h, y_h
            else:
                raise ValueError(
                    f"friction length mismatch for region {idx}: "
                    f"len(C)={c_vals.shape[0]}, len(x_vel)={x_vel.shape[0]}, len(x_h)={x_h.shape[0]}. "
                    "Check region indexing or provide explicit `fields` with x_c/y_c."
                )
            x_c, y_c, c = self._drop_nan_rows(x_c_src, y_c_src, c_vals)
            out["x_c"] = x_c
            out["y_c"] = y_c
            out["c"] = c
        else:
            out["x_c"] = x_vel
            out["y_c"] = y_vel
            out["c"] = None
        return out

    def _coerce_physical_fields_dict(self, fields: Dict[str, Any]) -> Dict[str, np.ndarray]:
        x_base = self._as_col(fields.get("x"))
        y_base = self._as_col(fields.get("y"))

        def _xy(prefix: str, required: bool = True):
            xv = self._as_col(fields.get(f"x_{prefix}"))
            yv = self._as_col(fields.get(f"y_{prefix}"))
            if xv is None:
                xv = x_base
            if yv is None:
                yv = y_base
            if required and (xv is None or yv is None):
                raise ValueError(f"Missing coordinates for field group '{prefix}'.")
            return xv, yv

        x_vel, y_vel = _xy("vel", required=True)
        u = self._as_col(fields.get("u"))
        v = self._as_col(fields.get("v"))
        if u is None or v is None:
            raise ValueError("`fields` must include velocity fields `u` and `v`.")

        x_h, y_h = _xy("h", required=False)
        if x_h is None or y_h is None:
            x_h, y_h = x_vel, y_vel
        h = self._as_col(fields.get("h"))
        s = self._as_col(fields.get("s"))
        if h is None or s is None:
            raise ValueError("`fields` must include thickness/surface fields `h` and `s`.")

        x_mu, y_mu = _xy("mu", required=False)
        if x_mu is None or y_mu is None:
            x_mu, y_mu = x_vel, y_vel
        mu = self._as_col(fields.get("mu"))

        x_c, y_c = _xy("c", required=False)
        if x_c is None or y_c is None:
            x_c, y_c = x_vel, y_vel
        c = self._as_col(fields.get("c"))
        if c is None:
            c = self._as_col(fields.get("alpha2"))

        x_vel, y_vel, u, v = self._drop_nan_rows(x_vel, y_vel, u, v)
        x_h, y_h, h, s = self._drop_nan_rows(x_h, y_h, h, s)

        out = {
            "x_vel": x_vel,
            "y_vel": y_vel,
            "u": u,
            "v": v,
            "x_h": x_h,
            "y_h": y_h,
            "h": h,
            "s": s,
            "x_mu": x_mu,
            "y_mu": y_mu,
            "mu": None,
            "x_c": x_c,
            "y_c": y_c,
            "c": None,
        }
        if mu is not None:
            x_mu, y_mu, mu = self._drop_nan_rows(x_mu, y_mu, mu)
            out["x_mu"], out["y_mu"], out["mu"] = x_mu, y_mu, mu
        if c is not None:
            x_c, y_c, c = self._drop_nan_rows(x_c, y_c, c)
            out["x_c"], out["y_c"], out["c"] = x_c, y_c, c
        return out

    def _normalize_data_residual_result(self, result: Dict[str, Any], scale_i: Dict[str, Any], basal: bool) -> Dict[str, Any]:
        s = scale_i
        out = copy.deepcopy(result)
        fields = out["fields"]
        fields["u"] = (fields["u"] - s["um"]) / s["u0"]
        fields["v"] = (fields["v"] - s["vm"]) / s["v0"]
        fields["h"] = fields["h"] / s["h0"]
        fields["s"] = (fields["s"] - s["sm"]) / s["s0"]
        fields["mu"] = fields["mu"] / s["mu0"]
        if basal and not np.isnan(float(s["c0"])):
            fields["C"] = fields["C"] / s["c0"]

        d = out["derivatives"]
        d["u_x"] = d["u_x"] * s["lx0"] / s["u0"]
        d["u_y"] = d["u_y"] * s["ly0"] / s["u0"]
        d["u_xx"] = d["u_xx"] * (s["lx0"] ** 2) / s["u0"]
        d["u_yy"] = d["u_yy"] * (s["ly0"] ** 2) / s["u0"]
        d["v_x"] = d["v_x"] * s["lx0"] / s["v0"]
        d["v_y"] = d["v_y"] * s["ly0"] / s["v0"]
        d["v_xx"] = d["v_xx"] * (s["lx0"] ** 2) / s["v0"]
        d["v_yy"] = d["v_yy"] * (s["ly0"] ** 2) / s["v0"]
        d["h_x"] = d["h_x"] * s["lx0"] / s["h0"]
        d["h_y"] = d["h_y"] * s["ly0"] / s["h0"]
        d["s_x"] = d["s_x"] * s["lx0"] / s["s0"]
        d["s_y"] = d["s_y"] * s["ly0"] / s["s0"]

        for key in out["stresses"]:
            out["stresses"][key] = out["stresses"][key] / s["term0"]
        for key in out["terms"]:
            out["terms"][key] = out["terms"][key] / s["term0"]
        out["residual"] = out["residual"] / s["term0"]
        return out

    def _extract_raw_field_quad(self, x_key: str, y_key: str, a_key: str, b_key: str, idx: int):
        x = self._extract_raw_cell(x_key, idx)
        y = self._extract_raw_cell(y_key, idx)
        a = self._extract_raw_cell(a_key, idx)
        b = self._extract_raw_cell(b_key, idx)
        if any(v is None for v in (x, y, a, b)):
            missing = [k for k, v in [(x_key, x), (y_key, y), (a_key, a), (b_key, b)] if v is None]
            raise ValueError(f"Missing required rawdata fields for region {idx}: {missing}")
        return self._as_col(x), self._as_col(y), self._as_col(a), self._as_col(b)

    def _extract_raw_optional(self, key: str, idx: int):
        val = self._extract_raw_cell(key, idx)
        if val is None:
            return None
        return self._as_col(val)

    def _extract_raw_cell(self, key: str, idx: int):
        if self.rawdata is None or key not in self.rawdata:
            return None
        arr = self.rawdata[key]
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            if arr.ndim < 2:
                return None
            candidates = [idx, idx + 1, idx - 1]
            for j in candidates:
                if 0 <= j < arr.shape[1]:
                    cell = arr[0, j]
                    if cell is None:
                        continue
                    if np.size(cell) == 0:
                        continue
                    return np.asarray(cell)
            return None
        if np.size(arr) == 0:
            return None
        return np.asarray(arr)

    @staticmethod
    def _as_col(x: Any):
        if x is None:
            return None
        a = np.asarray(x, dtype=float)
        return a.reshape(-1, 1)

    @staticmethod
    def _drop_nan_rows(*arrs: np.ndarray):
        if len(arrs) == 0:
            return ()
        n = arrs[0].shape[0]
        mask = np.ones((n,), dtype=bool)
        for a in arrs:
            if a is None:
                continue
            mask &= np.isfinite(a.reshape(-1))
        return tuple(a[mask] if a is not None else None for a in arrs)

    @staticmethod
    def _spatial_subsample_indices(xy: np.ndarray, max_points: int | None, seed: int = 42):
        n = xy.shape[0]
        if max_points is None or max_points <= 0 or n <= max_points:
            return np.arange(n)
        rng = np.random.default_rng(seed)
        return np.sort(rng.choice(n, size=int(max_points), replace=False))

    @staticmethod
    def _interp_to_queries(
        xy_support: np.ndarray,
        values: np.ndarray,
        xy_query: np.ndarray,
        method: str = "linear",
    ) -> np.ndarray:
        xy_support = np.asarray(xy_support, dtype=float)
        xy_query = np.asarray(xy_query, dtype=float)
        vals = np.asarray(values, dtype=float)
        if vals.ndim == 1:
            vals = vals[:, None]

        cols = []
        for j in range(vals.shape[1]):
            interp = griddata(xy_support, vals[:, j], xy_query, method=method)
            if interp is None:
                interp = np.full((xy_query.shape[0],), np.nan)
            interp = np.asarray(interp, dtype=float).reshape(-1)
            if np.any(~np.isfinite(interp)):
                nearest = griddata(xy_support, vals[:, j], xy_query, method="nearest")
                nearest = np.asarray(nearest, dtype=float).reshape(-1)
                interp = np.where(np.isfinite(interp), interp, nearest)
            cols.append(interp[:, None])
        return np.hstack(cols)

    @staticmethod
    def _rbf_apply(
        x_support: np.ndarray,
        y_support: np.ndarray,
        values: np.ndarray,
        x_query: np.ndarray,
        y_query: np.ndarray,
        ops: tuple[str, ...],
        mode: str,
        k: int,
        eps: float,
        reg: float,
    ) -> Dict[str, np.ndarray]:
        xs = np.asarray(x_support, dtype=float).reshape(-1, 1)
        ys = np.asarray(y_support, dtype=float).reshape(-1, 1)
        vals = np.asarray(values, dtype=float)
        if vals.ndim == 1:
            vals = vals[:, None]
        if xs.shape[0] != vals.shape[0]:
            raise ValueError("Support coordinate and value lengths do not match.")
        xq = np.asarray(x_query, dtype=float).reshape(-1, 1)
        yq = np.asarray(y_query, dtype=float).reshape(-1, 1)

        coords = np.hstack([xs, ys])
        queries = np.hstack([xq, yq])
        n = coords.shape[0]
        if n < 4:
            raise ValueError("RBF differentiation needs at least 4 support points.")
        mode_l = mode.lower()
        if mode_l not in {"local", "global"}:
            raise ValueError(f"Unsupported rbf_mode={mode!r}. Use 'local' or 'global'.")

        if mode_l == "global":
            k_eff = n
            tree = None
        else:
            k_eff = min(max(k, 8), n)
            tree = cKDTree(coords)

        out = {op: np.zeros((queries.shape[0], vals.shape[1]), dtype=float) for op in ops}
        eps2 = eps * eps
        eps4 = eps2 * eps2

        for i, q in enumerate(queries):
            if tree is None:
                idxs = np.arange(n, dtype=int)
            else:
                _, idxs = tree.query(q, k=k_eff)
                idxs = np.atleast_1d(idxs).astype(int)
            xst = coords[idxs, :]
            vst = vals[idxs, :]

            dxij = xst[:, 0:1] - xst[:, 0][None, :]
            dyij = xst[:, 1:2] - xst[:, 1][None, :]
            rij = np.sqrt(dxij * dxij + dyij * dyij)
            Phi = np.sqrt(1.0 + eps2 * rij * rij)
            P = np.hstack([np.ones((xst.shape[0], 1)), xst[:, 0:1], xst[:, 1:2]])
            A = np.block(
                [
                    [Phi + reg * np.eye(xst.shape[0]), P],
                    [P.T, np.zeros((3, 3))],
                ]
            )

            dxq = q[0] - xst[:, 0:1]
            dyq = q[1] - xst[:, 1:2]
            rq = np.sqrt(dxq * dxq + dyq * dyq)
            qq = np.sqrt(1.0 + eps2 * rq * rq)
            qq3 = np.maximum(qq**3, 1e-30)

            rhs_cols = []
            for op in ops:
                if op == "val":
                    rhs_r = qq
                    rhs_p = np.array([[1.0], [q[0]], [q[1]]], dtype=float)
                elif op == "dx":
                    rhs_r = eps2 * dxq / qq
                    rhs_p = np.array([[0.0], [1.0], [0.0]], dtype=float)
                elif op == "dy":
                    rhs_r = eps2 * dyq / qq
                    rhs_p = np.array([[0.0], [0.0], [1.0]], dtype=float)
                elif op == "dxx":
                    rhs_r = eps2 / qq - eps4 * (dxq**2) / qq3
                    rhs_p = np.zeros((3, 1), dtype=float)
                elif op == "dyy":
                    rhs_r = eps2 / qq - eps4 * (dyq**2) / qq3
                    rhs_p = np.zeros((3, 1), dtype=float)
                elif op == "dxy":
                    rhs_r = -eps4 * dxq * dyq / qq3
                    rhs_p = np.zeros((3, 1), dtype=float)
                else:
                    raise ValueError(f"Unsupported operator '{op}'.")
                rhs_cols.append(np.vstack([rhs_r, rhs_p]))
            B = np.hstack(rhs_cols)

            try:
                W = np.linalg.solve(A, B)
            except np.linalg.LinAlgError:
                W = np.linalg.lstsq(A, B, rcond=None)[0]
            wd = W[: xst.shape[0], :]
            est = wd.T @ vst
            for j, op in enumerate(ops):
                out[op][i : i + 1, :] = est[j : j + 1, :]
        return out

    def _init_run_identity(self) -> None:
        if self.run_timestamp is None:
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.run_folder_name is None:
            pyrnd.seed(time.time())
            name = self.config_resolved.get("name", "xpinn")
            self.run_folder_name = f"{self.run_timestamp}_{name}_xpinn_rndi{pyrnd.randint(1, 9999)}"

    @staticmethod
    def _deep_merge_dicts(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        out = copy.deepcopy(base)
        for k, v in updates.items():
            if isinstance(v, dict) and isinstance(out.get(k), dict):
                out[k] = DIFFICEInverseProblem._deep_merge_dicts(out[k], v)
            else:
                out[k] = v
        return out

    @staticmethod
    def _resolve_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
        resolved = copy.deepcopy(cfg)
        resolved.setdefault("name", "xpinn")
        resolved.setdefault("lr", 1e-3)
        resolved.setdefault("USE_LBFGS", False)
        resolved.setdefault("use_lbpinn", False)
        resolved.setdefault("freeze_lbpinn", True)
        resolved.setdefault("adam_epochs", 10000)
        resolved.setdefault("lbfgs_epochs", 0)

        if "loss_weights" in resolved:
            loss_weights = resolved["loss_weights"]
        elif "lw" in resolved:
            loss_weights = resolved["lw"]
        else:
            loss_weights = [1.0, 0.05, 0.1, 0.1]
        resolved["loss_weights"] = loss_weights

        resolved.setdefault("use_rwf", False)
        resolved.setdefault("use_modified_mlp", False)
        resolved.setdefault("use_grad_adapt", False)
        resolved.setdefault("adapt_grad_period", 1000)

        ff = resolved.get("fourier_features", {})
        ff.setdefault("enabled", False)
        ff.setdefault("embed_n", 128)
        ff.setdefault("embed_std", 2)
        ff_anneal = ff.get("anneal", {})
        ff_anneal.setdefault("enabled", False)
        ff_anneal.setdefault("alpha_start", 0.1)
        ff_anneal.setdefault("hold_frac", 0.1)
        ff_anneal.setdefault("ramp_end_frac", 0.7)
        ff["anneal"] = ff_anneal
        resolved["fourier_features"] = ff

        nw = resolved.get("network", {})
        nw.setdefault("n_hidden_layers", 6)
        nw.setdefault("n_units_per_layer", 30)
        nw["architecture"] = resolve_architecture(
            architecture=nw.get("architecture"),
            use_modified_mlp=resolved["use_modified_mlp"],
        )
        pirate_init = nw.get("pirate_init", {})
        pirate_init.setdefault("enabled", False)
        pirate_init.setdefault("source", "none")
        pirate_init.setdefault("sample_size", 0)
        pirate_init.setdefault("ridge", 1e-6)
        nw["pirate_init"] = pirate_init
        resolved["network"] = nw
        resolved["use_modified_mlp"] = nw["architecture"] == "modified_mlp"

        sp = resolved.get("sampling_points", {})
        sp.setdefault("adaptive", False)
        sp.setdefault("adapt_period", 500)
        sp.setdefault("adapt_burnin", 20000)
        sp.setdefault("velocity", 1200)
        sp.setdefault("thickness", 1200)
        sp.setdefault("collocation", 1200)
        sp.setdefault("boundary", 200)
        sp.setdefault("interface", 200)
        resolved["sampling_points"] = sp

        resolved.setdefault("gamma_c", None)
        resolved.setdefault("data_path", None)
        resolved.setdefault("region_indices", None)
        resolved.setdefault("loss_flags", None)
        resolved.setdefault("lbpinn_config", {})
        resolved.setdefault("seed", 2134)
        resolved["lbpinn_config"].setdefault("data_warmup_epochs", 0)
        if resolved["loss_flags"] is None:
            resolved["loss_flags"] = {"data": True, "eqn": True, "bd": True, "match": True}
        if resolved["loss_flags"] and not resolved["loss_flags"].get("eqn", True):
            resolved["sampling_points"]["adaptive"] = False

        resolved["use_lbfgs"] = resolved["USE_LBFGS"]
        return resolved
