"""Inferencia métrica en hiperesfera (prototipos de clase / kNN).

Para entrenamiento Slice-MS puro: la cabeza softmax no se entrena; la
clasificación usa similitud coseno a prototipos L2-normalizados (práctica
estándar en deep metric learning).

Sub-prototipos (K>1): cada clase se representa con K sub-centros en S^{d-1}
(concepto sub-center de Deng et al. ECCV 2020, aplicado a la inferencia y NO a
la pérdida). Modela varianza intra-clase multimodal (p.ej. M+ con varios modos
visuales). La predicción usa la máxima similitud coseno entre el embedding y
cualquier sub-centro de la clase.

Domain-aware: cada subcentro puede estar ligado a un `domain_bucket`; la
actualización EMA y la inferencia usan solo subcentros del dominio del tile.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class ClassPrototypeBank:
    """Prototipos de clase en S^{d-1} con actualización EMA por batch.

    Soporta K asimétrico por clase (`num_subcenters_per_class`) y subcentros
    condicionados por dominio (`domain_aware`).
    """

    num_classes: int
    embed_dim: int
    device: torch.device
    momentum: float = 0.05
    scale: float = 10.0
    num_subcenters: int = 1
    num_subcenters_per_class: Optional[tuple[int, ...]] = None
    domain_aware: bool = False
    prototypes: torch.Tensor = field(init=False)
    initialized: torch.Tensor = field(init=False)
    subcenter_domains: list[list[Optional[str]]] = field(init=False)

    def __post_init__(self) -> None:
        if self.num_subcenters_per_class:
            k_per = tuple(max(1, int(k)) for k in self.num_subcenters_per_class)
            if len(k_per) != self.num_classes:
                raise ValueError(
                    f"num_subcenters_per_class debe tener {self.num_classes} entradas, "
                    f"recibido {len(k_per)}"
                )
            self.num_subcenters_per_class = k_per
            k_max = max(k_per)
        else:
            k_max = max(1, int(self.num_subcenters))
            self.num_subcenters_per_class = (k_max,) * self.num_classes
        self.num_subcenters = k_max
        self.prototypes = torch.zeros(self.num_classes, k_max, self.embed_dim, device=self.device)
        self.initialized = torch.zeros(self.num_classes, k_max, dtype=torch.bool, device=self.device)
        self.subcenter_domains = [[None] * k_max for _ in range(self.num_classes)]

    def k_for_class(self, c: int) -> int:
        return int(self.num_subcenters_per_class[c])

    def _active_indices(self, c: int) -> list[int]:
        k_lim = self.k_for_class(c)
        return [k for k in range(k_lim) if bool(self.initialized[c, k].item())]

    # ------------------------------------------------------------------ update
    def _farthest_seed_indices(
        self,
        feats: torch.Tensor,
        *,
        k: int,
        existing: Optional[torch.Tensor] = None,
    ) -> list[int]:
        """Selecciona seeds deterministas maximizando cobertura angular."""
        n = feats.size(0)
        if n == 0 or k <= 0:
            return []
        seeds: list[int] = []
        if existing is None or existing.numel() == 0:
            center = F.normalize(feats.mean(dim=0, keepdim=True), dim=-1)
            sims = (feats @ center.t()).squeeze(1)
            seeds.append(int(sims.argmin().item()))
        seed_mat = feats[seeds] if seeds else existing
        while len(seeds) < min(k, n):
            refs = seed_mat if existing is None or existing.numel() == 0 else torch.cat([existing, seed_mat], dim=0)
            sims = feats @ refs.t()
            nearest = sims.max(dim=1).values
            if seeds:
                nearest[torch.tensor(seeds, device=feats.device)] = 1.0
            nxt = int(nearest.argmin().item())
            if nxt in seeds:
                break
            seeds.append(nxt)
            seed_mat = feats[seeds]
        return seeds

    def _slot_for_domain(self, c: int, domain: Optional[str]) -> int:
        """Slot preferido para un dominio (0..k_for_class-1)."""
        k_lim = self.k_for_class(c)
        if k_lim <= 0:
            return 0
        if domain:
            for k in range(k_lim):
                if self.subcenter_domains[c][k] == domain:
                    return k
            for k in range(k_lim):
                if not bool(self.initialized[c, k].item()):
                    return k
        for k in range(k_lim):
            if not bool(self.initialized[c, k].item()):
                return k
        return 0

    def _subcenters_for_domain(self, c: int, domain: Optional[str]) -> list[int]:
        k_lim = self.k_for_class(c)
        active = [k for k in range(k_lim) if bool(self.initialized[c, k].item())]
        if not active:
            return []
        if not self.domain_aware or domain is None:
            return active
        matching = [k for k in active if self.subcenter_domains[c][k] == domain]
        return matching if matching else active

    def _seed_subcenters(
        self,
        feats: torch.Tensor,
        c: int,
        *,
        domain: Optional[str] = None,
    ) -> None:
        """Inicializa sub-centros de la clase c por farthest-point sampling."""
        n = feats.size(0)
        k_lim = self.k_for_class(c)
        if k_lim == 1 or n == 1:
            slot = 0
            self.prototypes[c, slot] = F.normalize(feats.mean(dim=0), dim=0)
            self.initialized[c, slot] = True
            if self.domain_aware and domain:
                self.subcenter_domains[c][slot] = domain
            return

        if self.domain_aware and domain:
            domain_slots = [
                k for k in range(k_lim) if self.subcenter_domains[c][k] == domain
            ]
            empty = [
                k
                for k in range(k_lim)
                if not bool(self.initialized[c, k].item())
                and self.subcenter_domains[c][k] in (domain, None)
            ]
            if not domain_slots and empty:
                n_seed = min(len(empty), len(feats), k_lim)
                seeds = self._farthest_seed_indices(feats, k=n_seed)
                seed_mat = feats[seeds]
                assign = (feats @ seed_mat.t()).argmax(dim=1) if n_seed > 1 else None
                for j, slot in enumerate(empty[:n_seed]):
                    if assign is not None:
                        grp = feats[assign == j]
                        if grp.size(0) == 0:
                            grp = feats[seeds[min(j, len(seeds) - 1)] : seeds[min(j, len(seeds) - 1)] + 1]
                    else:
                        grp = feats
                    self.prototypes[c, slot] = F.normalize(grp.mean(dim=0), dim=0)
                    self.initialized[c, slot] = True
                    self.subcenter_domains[c][slot] = domain
                return
            if domain_slots and empty and len(domain_slots) < k_lim:
                n_add = min(len(empty), k_lim - len(domain_slots), len(feats))
                seeds = self._farthest_seed_indices(feats, k=n_add)
                for j, slot in enumerate(empty[:n_add]):
                    self.prototypes[c, slot] = F.normalize(feats[seeds[j]], dim=0)
                    self.initialized[c, slot] = True
                    self.subcenter_domains[c, slot] = domain
                return
            if domain_slots:
                return
            if empty:
                slot = empty[0]
                self.prototypes[c, slot] = F.normalize(feats.mean(dim=0), dim=0)
                self.initialized[c, slot] = True
                self.subcenter_domains[c][slot] = domain
            return

        seeds = self._farthest_seed_indices(feats, k=k_lim)
        seed_mat = feats[seeds]
        assign = (feats @ seed_mat.t()).argmax(dim=1)
        for j, _ in enumerate(seeds):
            if j >= k_lim:
                break
            grp = feats[assign == j]
            if grp.size(0) == 0:
                continue
            self.prototypes[c, j] = F.normalize(grp.mean(dim=0), dim=0)
            self.initialized[c, j] = True

    @torch.no_grad()
    def update(
        self,
        embed: torch.Tensor,
        labels: torch.Tensor,
        *,
        domain_buckets: Optional[list[str]] = None,
    ) -> None:
        """EMA de sub-centros por clase, re-normalizados."""
        embed = F.normalize(embed.detach().float(), dim=-1)
        labels = labels.long().view(-1)
        n = embed.size(0)
        if domain_buckets is not None and len(domain_buckets) != n:
            domain_buckets = None

        for c in labels.unique().tolist():
            mask = labels == c
            feats = embed[mask]
            if feats.size(0) == 0:
                continue

            if self.domain_aware and domain_buckets is not None:
                doms = [domain_buckets[i] for i in torch.where(mask)[0].tolist()]
                unique_doms = sorted(set(doms))
                for dom in unique_doms:
                    dom_mask = torch.tensor(
                        [d == dom for d in doms], device=feats.device, dtype=torch.bool
                    )
                    feats_d = feats[dom_mask]
                    if feats_d.size(0) == 0:
                        continue
                    self._update_class_domain(feats_d, c, dom)
                continue

            self._update_class_domain(feats, c, None)

    def _update_class_domain(
        self,
        feats: torch.Tensor,
        c: int,
        domain: Optional[str],
    ) -> None:
        init_k = self._active_indices(c)
        if not init_k:
            self._seed_subcenters(feats, c, domain=domain)
            return

        k_lim = self.k_for_class(c)
        missing = [k for k in range(k_lim) if not bool(self.initialized[c, k].item())]
        if missing and feats.size(0) > 1:
            if self.domain_aware and domain:
                self._seed_subcenters(feats, c, domain=domain)
            else:
                existing = F.normalize(self.prototypes[c, init_k].float(), dim=-1)
                seeds = self._farthest_seed_indices(feats, k=len(missing), existing=existing)
                for k_idx, feat_idx in zip(missing, seeds):
                    self.prototypes[c, k_idx] = F.normalize(feats[feat_idx], dim=0)
                    self.initialized[c, k_idx] = True
                    if self.domain_aware and domain:
                        self.subcenter_domains[c][k_idx] = domain
            init_k = self._active_indices(c)

        if self.domain_aware and domain:
            domain_slots = [
                k
                for k in range(k_lim)
                if bool(self.initialized[c, k].item())
                and self.subcenter_domains[c][k] == domain
            ]
            slots = domain_slots if domain_slots else init_k
        else:
            slots = init_k

        protos = self.prototypes[c, slots]
        assign = (feats @ F.normalize(protos.float(), dim=-1).t()).argmax(dim=1)
        m = self.momentum
        for local_j, k_idx in enumerate(slots):
            grp = feats[assign == local_j]
            if grp.size(0) == 0:
                continue
            mean = F.normalize(grp.mean(dim=0), dim=0)
            updated = F.normalize((1 - m) * self.prototypes[c, k_idx] + m * mean, dim=0)
            self.prototypes[c, k_idx] = updated
            if self.domain_aware and domain and self.subcenter_domains[c][k_idx] is None:
                self.subcenter_domains[c][k_idx] = domain

    # ------------------------------------------------------------------ query
    def ready_class_indices(self) -> list[int]:
        """Clases con al menos un sub-centro inicializado."""
        return [i for i in range(self.num_classes) if bool(self.initialized[i].any().item())]

    def is_ready(self, required: list[int] | None = None) -> bool:
        """Listo si todas las clases *operativas* tienen al menos un sub-centro."""
        if required is None:
            required = [0, 1, 2]
        return len(required) > 0 and all(
            bool(self.initialized[i].any().item()) for i in required
        )

    def logits(
        self,
        embed: torch.Tensor,
        *,
        required: list[int] | None = None,
        domain_buckets: Optional[list[str]] = None,
    ) -> torch.Tensor:
        """Máx similitud coseno a sub-centros, escalada → pseudo-logits (B, num_classes)."""
        embed = F.normalize(embed.float(), dim=-1)
        req = required if required is not None else [0, 1, 2]
        b = embed.size(0)
        out = torch.full((b, self.num_classes), -1e9, device=embed.device, dtype=embed.dtype)
        if not self.is_ready(req):
            return torch.zeros(b, self.num_classes, device=embed.device)
        proto = F.normalize(self.prototypes.float(), dim=-1)
        doms = domain_buckets if domain_buckets is not None else [None] * b
        if len(doms) != b:
            doms = [None] * b
        for i in range(b):
            for cls in req:
                slots = self._subcenters_for_domain(cls, doms[i])
                if not slots:
                    continue
                sims = embed[i] @ proto[cls, slots].t()
                out[i, cls] = sims.max() * self.scale
        return out

    def score_details(
        self,
        embed: torch.Tensor,
        *,
        required: list[int] | None = None,
        domain_buckets: Optional[list[str]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Devuelve logits por clase, sub-centro ganador por clase y similitudes."""
        embed = F.normalize(embed.float(), dim=-1)
        req = required if required is not None else [0, 1, 2]
        b = embed.size(0)
        logits = torch.full((b, self.num_classes), -1e9, device=embed.device, dtype=embed.dtype)
        winner = torch.full((b, self.num_classes), -1, device=embed.device, dtype=torch.long)
        best = torch.full((b, self.num_classes), float("nan"), device=embed.device, dtype=embed.dtype)
        if not self.is_ready(req):
            return torch.zeros_like(logits), winner, best
        proto = F.normalize(self.prototypes.float(), dim=-1)
        doms = domain_buckets if domain_buckets is not None else [None] * b
        if len(doms) != b:
            doms = [None] * b
        for i in range(b):
            for cls in req:
                slots = self._subcenters_for_domain(cls, doms[i])
                if not slots:
                    continue
                sims = embed[i] @ proto[cls, slots].t()
                val, local = sims.max(dim=0)
                logits[i, cls] = val * self.scale
                winner[i, cls] = slots[int(local.item())]
                best[i, cls] = val
        return logits, winner, best

    def representative_prototypes(self) -> torch.Tensor:
        """(num_classes, embed_dim): media L2-norm de sub-centros (para análisis)."""
        out = torch.zeros(self.num_classes, self.embed_dim, device=self.device)
        proto = F.normalize(self.prototypes.float(), dim=-1)
        for i in range(self.num_classes):
            active = self._active_indices(i)
            if not active:
                continue
            out[i] = F.normalize(proto[i, active].mean(dim=0), dim=0)
        return out

    # ------------------------------------------------------------------ state
    def state_dict(self) -> dict:
        return {
            "prototypes": self.prototypes.cpu(),
            "initialized": self.initialized.cpu(),
            "num_subcenters": self.num_subcenters,
            "num_subcenters_per_class": list(self.num_subcenters_per_class),
            "subcenter_domains": self.subcenter_domains,
            "domain_aware": self.domain_aware,
            "momentum": self.momentum,
            "scale": self.scale,
        }

    def load_state_dict(self, state: dict) -> None:
        protos = state["prototypes"]
        init = state["initialized"]
        if protos.ndim == 2:
            protos = protos.unsqueeze(1)
            init = init.unsqueeze(1)
        self.num_subcenters = int(state.get("num_subcenters", protos.size(1)))
        if "num_subcenters_per_class" in state:
            self.num_subcenters_per_class = tuple(int(k) for k in state["num_subcenters_per_class"])
        self.domain_aware = bool(state.get("domain_aware", self.domain_aware))
        self.prototypes = protos.to(self.device).float()
        self.initialized = init.to(self.device).bool()
        raw_domains = state.get("subcenter_domains")
        if raw_domains is not None:
            self.subcenter_domains = [list(row) for row in raw_domains]
        else:
            k_max = self.prototypes.size(1)
            self.subcenter_domains = [[None] * k_max for _ in range(self.num_classes)]
        self.momentum = float(state.get("momentum", self.momentum))
        self.scale = float(state.get("scale", self.scale))


def prototype_bank_from_gate4(
    gate4_config: object,
    *,
    num_classes: int,
    device: torch.device,
) -> ClassPrototypeBank:
    """Construye banco de prototipos alineado con Gate4SliceMSConfig."""
    k_by = getattr(gate4_config, "proto_subcenters_per_class", ()) or ()
    return ClassPrototypeBank(
        num_classes,
        int(gate4_config.embed_dim),
        device,
        num_subcenters=int(gate4_config.proto_subcenters_max),
        num_subcenters_per_class=k_by if k_by else None,
        domain_aware=bool(getattr(gate4_config, "domain_aware_subcenters", False)),
    )


def knn_predict(
    query_embed: np.ndarray,
    ref_embed: np.ndarray,
    ref_labels: np.ndarray,
    *,
    k: int = 5,
) -> np.ndarray:
    """kNN mayoría en espacio coseno (embeddings ya normalizados recomendado)."""
    q = query_embed.astype(np.float32)
    r = ref_embed.astype(np.float32)
    q_norm = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    r_norm = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-8)
    sim = q_norm @ r_norm.T
    k = min(k, r.shape[0])
    nn_idx = np.argpartition(-sim, kth=min(k - 1, sim.shape[1] - 1), axis=1)[:, :k]
    preds = np.empty(len(q), dtype=np.int64)
    for i in range(len(q)):
        votes = ref_labels[nn_idx[i]]
        vals, counts = np.unique(votes, return_counts=True)
        preds[i] = vals[counts.argmax()]
    return preds
