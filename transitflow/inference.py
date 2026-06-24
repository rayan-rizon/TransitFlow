"""Amortized inference: detection probability, posterior samples, densities.

Wraps a trained :class:`~transitflow.models.transitflow.TransitFlow` to deliver,
in a single forward pass per object:

* ``p(d=1 | x)`` from the detection head,
* posterior samples ``theta ~ p(theta | d=1, x)`` by integrating the
  probability-flow ODE (FMPE) or sampling the NPE flow,
* exact ``log p(theta | x)`` for NLL / coverage, and
* an importance-sampling efficiency diagnostic (Gebhard 2024) flagging
  simulator misspecification.
"""

from __future__ import annotations

import numpy as np
import torch

from .flow_matching import log_prob as fm_log_prob
from .flow_matching import sample_ode
from .priors import TransitPrior, kipping_to_quadratic
from .simulator import SimConfig
from .transit_model import transit_duration, transit_flux
from .views import make_views


class TransitFlowInference:
    def __init__(self, model, prior: TransitPrior, sim_cfg: SimConfig,
                 device=None, ode_steps: int = 50, ode_method: str = "rk4"):
        self.model = model.eval()
        self.prior = prior
        self.sim_cfg = sim_cfg
        self.device = device or next(model.parameters()).device
        self.ode_steps = ode_steps
        self.ode_method = ode_method

    # ------------------------------------------------------------------ #
    def _to_t(self, a):
        return torch.as_tensor(np.asarray(a, dtype=np.float32), device=self.device)

    @torch.no_grad()
    def embed(self, global_view, local_view, sigma_feat=None,
              periodogram=None) -> torch.Tensor:
        g = self._to_t(global_view)
        l = self._to_t(local_view)
        if g.ndim == 1:
            g, l = g[None], l[None]
        nf = None
        if self.model.cfg.use_noise_feature:
            nf = self._to_t(sigma_feat) if sigma_feat is not None else \
                torch.zeros(g.shape[0], device=self.device)
        pg = None
        if self.model.cfg.use_periodogram:
            if periodogram is None:
                raise ValueError("model expects a periodogram input")
            pg = self._to_t(periodogram)
            if pg.ndim == 1:
                pg = pg[None]
        return self.model.embed(g, l, nf, pg)

    @torch.no_grad()
    def detect(self, global_view, local_view, sigma_feat=None,
               periodogram=None) -> np.ndarray:
        e = self.embed(global_view, local_view, sigma_feat, periodogram)
        return torch.sigmoid(self.model.detect_logits(e)).cpu().numpy()

    @torch.no_grad()
    def posterior_samples(self, global_view, local_view, sigma_feat=None,
                          n_samples: int = 2000, return_std: bool = False,
                          periodogram=None):
        """Return physical posterior samples ``(B, n_samples, 7)``."""
        e = self.embed(global_view, local_view, sigma_feat, periodogram)
        if self.model.head_type == "fmpe":
            std = sample_ode(self.model.velocity_fn(), e, n_samples,
                             n_steps=self.ode_steps, method=self.ode_method)
        else:
            std = self.model.posterior.sample(e, n_samples)
        std_np = std.cpu().numpy()
        phys = self.prior.std_to_physical(std_np.reshape(-1, std_np.shape[-1]))
        phys = phys.reshape(std_np.shape)
        if return_std:
            return phys, std_np
        return phys

    @torch.no_grad()
    def detect_and_characterize(self, global_view, local_view, sigma_feat=None,
                                n_samples: int = 2000, periodogram=None) -> dict:
        e = self.embed(global_view, local_view, sigma_feat, periodogram)
        p_det = torch.sigmoid(self.model.detect_logits(e)).cpu().numpy()
        if self.model.head_type == "fmpe":
            std = sample_ode(self.model.velocity_fn(), e, n_samples,
                             n_steps=self.ode_steps, method=self.ode_method)
        else:
            std = self.model.posterior.sample(e, n_samples)
        std_np = std.cpu().numpy()
        phys = self.prior.std_to_physical(std_np.reshape(-1, std_np.shape[-1]))
        phys = phys.reshape(std_np.shape)
        return {"p_detect": p_det, "samples": phys, "samples_std": std_np}

    def log_prob_std(self, theta_std, e) -> np.ndarray:
        """Exact ``log q(theta | x)`` in standardized space."""
        ts = torch.as_tensor(np.asarray(theta_std, dtype=np.float32),
                             device=self.device)
        if ts.ndim == 1:
            ts = ts[None]
        if self.model.head_type == "fmpe":
            lp = fm_log_prob(self.model.velocity_fn(), ts, e, n_steps=self.ode_steps)
        else:
            lp = self.model.posterior.log_prob(ts, e)
        return lp.detach().cpu().numpy()

    # ------------------------------------------------------------------ #
    # Importance-sampling efficiency diagnostic (approximate)
    # ------------------------------------------------------------------ #
    def importance_diagnostic(self, global_view, local_view, fold_P, fold_t0,
                              sigma_feat=None, n_samples: int = 500) -> dict:
        """Approximate IS efficiency as a misspecification flag.

        Uses a Gaussian likelihood on the *local* view: each posterior draw is
        rendered to a noiseless local view (same folding + normalization as the
        simulator) and compared to the observed local view with a noise level
        estimated from its out-of-window scatter.  Returns the IS effective
        sample size fraction (ESS/N); low values flag a simulator gap.

        This is a single-object diagnostic (B == 1 view inputs).
        """
        gv = np.asarray(global_view, dtype=np.float64).reshape(-1)
        lv = np.asarray(local_view, dtype=np.float64).reshape(-1)
        e = self.embed(gv.astype(np.float32), lv.astype(np.float32), sigma_feat)
        phys, std = self.posterior_samples(gv.astype(np.float32),
                                           lv.astype(np.float32), sigma_feat,
                                           n_samples=n_samples, return_std=True)
        phys = phys[0]                     # (n, 7)
        std = std[0]
        logq = self.log_prob_std(std, e.repeat(std.shape[0], 1))  # (n,)
        logprior = self.prior.log_prob_std(std)                   # (n,)

        # render predicted local views for all samples (vectorized)
        cfg = self.sim_cfg
        t = np.linspace(0.0, cfg.baseline_days, cfg.n_raw)
        P, t0p, RpRs, aRs, b = (phys[:, 0], phys[:, 1], phys[:, 2],
                                phys[:, 3], phys[:, 4])
        u1, u2 = kipping_to_quadratic(phys[:, 5], phys[:, 6])
        t0_abs = t0p * P
        flux = transit_flux(t, P, t0_abs, RpRs, aRs, b, u1, u2,
                            n_radial=cfg.n_radial, engine=cfg.engine)
        dur = transit_duration(P, RpRs, aRs, b)
        preds = np.empty((phys.shape[0], cfg.n_local), dtype=np.float64)
        for i in range(phys.shape[0]):
            _, li = make_views(t, flux[i], float(fold_P), float(fold_t0),
                               float(dur[i]), n_global=cfg.n_global,
                               n_local=cfg.n_local, n_durations=cfg.n_durations,
                               normalize=True)
            preds[i] = li
        # noise estimate from observed out-of-transit edges of the local view
        edge = np.concatenate([lv[:cfg.n_local // 5], lv[-cfg.n_local // 5:]])
        sig = max(np.std(edge), 1e-3)
        loglik = -0.5 * np.sum((lv[None, :] - preds) ** 2, axis=1) / sig ** 2
        logw = loglik + logprior - logq
        logw = logw - np.max(logw)
        w = np.exp(logw)
        w = np.where(np.isfinite(w), w, 0.0)
        if w.sum() <= 0:
            return {"ess_fraction": 0.0, "n_samples": n_samples}
        ess = (w.sum() ** 2) / np.sum(w ** 2)
        return {"ess_fraction": float(ess / len(w)), "n_samples": n_samples,
                "weights": w / w.sum()}
