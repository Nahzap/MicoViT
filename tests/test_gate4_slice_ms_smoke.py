"""Smoke tests — Gate 4 clases + Slice Multi-Similarity Loss (G4 plan).

Ejecutar:
    F:\\MicorizaeVision\\.venv\\Scripts\\python.exe -m tests.test_gate4_slice_ms_smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def smoke_gate4_classes() -> None:
    from micorizae.phase_d_stage1.gate_classes import (
        GATE_CLASS_NAMES,
        GATE_CLASS_TO_IDX,
        decode_gate_indices,
        encode_gate_indices,
        stage1_to_gate_label,
    )

    assert len(GATE_CLASS_NAMES) == 4
    assert GATE_CLASS_NAMES == ("Background", "Mminus", "Mplus", "Unknown")
    assert stage1_to_gate_label("Unreadable") == "Unknown"
    assert stage1_to_gate_label("Mplus") == "Mplus"
    assert GATE_CLASS_TO_IDX["Unknown"] == 3

    labels = np.array(["Background", "Mminus", "Mplus", "Unreadable"])
    idx = encode_gate_indices(labels)
    assert idx.tolist() == [0, 1, 2, 3]
    assert decode_gate_indices(idx) == list(GATE_CLASS_NAMES)


def smoke_slice_encoder() -> None:
    from micorizae.phase_d_stage1.gate4.slice_encoder import SliceMSEncoder

    dev = _device()
    enc = SliceMSEncoder(in_dim=384, embed_dim=128, num_slices=4).to(dev)
    x = torch.randn(8, 384, device=dev)
    e = enc(x)
    assert e.shape == (8, 128)
    assert torch.allclose(e.norm(dim=-1), torch.ones(8, device=dev), atol=1e-4)
    chunks = enc.slice_chunks(e)
    assert len(chunks) == 4
    assert all(c.shape == (8, 32) for c in chunks)


def smoke_multi_similarity_loss() -> None:
    from micorizae.phase_d_stage1.gate4.slice_ms_loss import MultiSimilarityLoss

    dev = _device()
    loss_fn = MultiSimilarityLoss(alpha=2.0, beta=50.0, base=0.5).to(dev)
    raw = nn.Parameter(torch.randn(16, 32, device=dev))
    embed = F.normalize(raw, dim=-1)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], device=dev)
    loss = loss_fn(embed, labels)
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0
    loss.backward()
    assert raw.grad is not None


def smoke_slice_ms_loss() -> None:
    from micorizae.phase_d_stage1.gate4.slice_ms_loss import SliceMultiSimilarityLoss

    dev = _device()
    loss_fn = SliceMultiSimilarityLoss(num_slices=4, alpha=2.0, beta=50.0, base=0.5).to(dev)
    embed = nn.Parameter(F.normalize(torch.randn(16, 128, device=dev), dim=-1))
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3] * 2, device=dev)
    loss = loss_fn(embed, labels)
    assert torch.isfinite(loss) and loss.item() > 0

    opt = torch.optim.Adam([embed], lr=0.05)
    l0 = loss_fn(F.normalize(embed, dim=-1), labels).item()
    for _ in range(5):
        opt.zero_grad()
        e = F.normalize(embed, dim=-1)
        l = loss_fn(e, labels)
        l.backward()
        opt.step()
    l1 = loss_fn(F.normalize(embed, dim=-1), labels).item()
    assert l1 <= l0 * 1.05 or l1 < l0, f"MS loss should not explode: {l0} -> {l1}"


def smoke_probe_model() -> None:
    from micorizae.phase_d_stage1.gate4.probe_model import GateSliceProbeModel

    dev = _device()
    model = GateSliceProbeModel(in_dim=384, embed_dim=128, num_slices=4, num_classes=4).to(dev)
    feat = torch.randn(6, 384, device=dev)
    logits, embed = model.forward_from_features(feat, return_embed=True)
    assert logits.shape == (6, 4)
    assert embed.shape == (6, 128)
    assert model.forward_from_features(feat).shape == (6, 4)


def smoke_combined_loss() -> None:
    from micorizae.phase_d_stage1.gate4.config import Gate4SliceMSConfig
    from micorizae.phase_d_stage1.gate4.probe_model import GateSliceProbeModel
    from micorizae.phase_d_stage1.gate4.training import Gate4CombinedLoss

    dev = _device()
    cfg = Gate4SliceMSConfig()
    model = GateSliceProbeModel(
        in_dim=384,
        embed_dim=cfg.embed_dim,
        num_slices=cfg.num_slices,
        num_classes=4,
    ).to(dev)
    loss_fn = Gate4CombinedLoss(cfg).to(dev)
    feat = torch.randn(12, 384, device=dev)
    labels = torch.tensor([0, 1, 2, 3] * 3, device=dev)
    logits, embed = model.forward_from_features(feat, return_embed=True)
    out = loss_fn(logits, embed, labels)
    assert "total" in out and "gate" in out and "slice_ms" in out
    assert torch.isfinite(out["total"])
    out["total"].backward()


def smoke_gate_run_live_publisher() -> None:
    import tempfile

    import pandas as pd

    from micorizae.common.run_outputs import RunOutputs
    from micorizae.phase_d_stage1.gate_run_live import GateRunLivePublisher
    from micorizae.phase_d_stage1.gate_training_protocol import GateTrainProtocol
    from micorizae.phase_d_stage1.train_gpu import GPUTrainHistory

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "live_run"
        run = RunOutputs(run_id="live_run", root=root).ensure()
        train_df = pd.DataFrame({"stage1": ["Mplus", "Background", "Mminus"]})
        val_df = pd.DataFrame({"stage1": ["Background", "Mplus"]})
        info = {"n_train_tiles": 3, "n_val_tiles": 2, "n_train_images": 1, "n_val_images": 1}
        cfg = {"epochs_max": 2, "balance_mode": "balanced_4class", "embed_cache": True}
        ckpt = Path(tmp) / "ckpt"
        ckpt.mkdir()
        pub = GateRunLivePublisher.create(
            run=run,
            train_df=train_df,
            val_df=val_df,
            info_split=info,
            train_config=cfg,
            protocol=GateTrainProtocol(),
            ckpt_dir=ckpt,
        )
        pub.publish_startup()
        assert (root / "images" / "pre_training" / "class_distribution.png").exists()
        assert (root / "training_status.json").exists()

        hist = GPUTrainHistory()
        hist.epochs.append(1)
        hist.train_loss.append(1.2)
        hist.val_loss.append(1.0)
        hist.val_f1.append(0.4)
        hist.val_acc.append(0.5)
        pub.publish_epoch(
            epoch=1,
            epochs_total=2,
            history=hist,
            epoch_details=[{"epoch": 1, "macro_f1": 0.4}],
            pretrain={"macro_f1": 0.2},
            ckpt_metric="macro_f1",
            improved=True,
            last_val={"acc": 0.5, "macro_f1": 0.4, "per_class_recall": {}, "per_class_specificity": {}},
        )
        assert (root / "reports" / "live_metrics.json").exists()
        assert (root / "training_metrics.csv").exists()
        assert (root / "evaluation_metrics_summary_val.json").exists()


def smoke_format_g1_status_nan() -> None:
    from micorizae.phase_d_stage1.gate_training_protocol import GateTrainProtocol, format_g1_status

    metrics = {
        "evangelisti_g1_pass": False,
        "per_class_recall": {
            "Background": 0.99,
            "Mminus": 0.17,
            "Mplus": 0.18,
            "Unknown": float("nan"),
        },
        "per_class_specificity": {
            "Background": 0.26,
            "Mminus": 0.99,
            "Mplus": 0.98,
            "Unknown": float("nan"),
        },
    }
    line = format_g1_status(metrics, GateTrainProtocol())
    assert "Unknown:Sens=N/A" in line
    assert "Unknown" in line and "Sens=0.000" not in line.split("Unknown")[1].split("|")[0]
    assert "[3/4 clases eval]" in line


def smoke_plan_epoch_gate4() -> None:
    import pandas as pd

    from micorizae.phase_d_stage1.gpu_pipeline import plan_epoch_gate

    df = pd.DataFrame(
        {
            "image_path": ["img/a.jpg"] * 20,
            "stage1": ["Mplus"] * 5 + ["Mminus"] * 5 + ["Background"] * 8 + ["Unreadable"] * 2,
        }
    )
    plan = plan_epoch_gate(df, balance_mode="balanced_4class", max_bg_per_image=10, seed=0)
    assert plan.n_tiles > 0
    stages = set()
    for _, sub in plan.items:
        stages.update(sub["stage1"].unique())
    assert "Mplus" in stages and "Mminus" in stages


def smoke_g1_stratified_sampler() -> None:
    import pandas as pd

    from micorizae.phase_d_stage1.gate_epoch_sampler import (
        GateEpochSamplerConfig,
        audit_stratified_batches,
        plan_epoch_stratified,
    )

    n = 200
    df = pd.DataFrame(
        {
            "image_path": [f"img/{i // 40}.jpg" for i in range(n)],
            "row": list(range(n)),
            "col": [0] * n,
            "stage1": (["Mplus"] * 50 + ["Mminus"] * 50 + ["Background"] * 100),
        }
    )
    cfg = GateEpochSamplerConfig(samples_per_class=120, min_per_class_per_batch=4)
    plan = plan_epoch_stratified(df, cfg=cfg, seed=0)
    assert plan.stratified_batches is not None
    assert len(plan.stratified_batches) == (120 * 3) // (4 * 3)
    audit = audit_stratified_batches(plan)
    assert audit["Mplus_per_batch_min"] == 4
    assert audit["Background_per_batch_min"] == 4
    assert audit["Mminus_per_batch_min"] == 4


def smoke_dual_eval_plans() -> None:
    import pandas as pd

    from micorizae.phase_d_stage1.gate_tile_dino import _eval_plans_for_epoch
    from micorizae.phase_d_stage1.gate_training_protocol import GateTrainProtocol

    n = 300
    df = pd.DataFrame(
        {
            "image_path": [f"img/{i // 50}.jpg" for i in range(n)],
            "row": list(range(n)),
            "col": [0] * n,
            "stage1": (["Mplus"] * 50 + ["Mminus"] * 50 + ["Background"] * 200),
        }
    )
    protocol = GateTrainProtocol(
        eval_balance_mode="dual",
        eval_stratified_samples_per_class=60,
        checkpoint_eval="stratified",
        stratified_min_per_batch=4,
    )
    holdout, strat, ckpt = _eval_plans_for_epoch(df, batch_size=12, protocol=protocol, seed=1)
    assert holdout.n_tiles == n
    assert strat is not None
    assert strat.n_tiles == 60 * 3
    assert ckpt is strat


def smoke_gate_runflow_config() -> None:
    import importlib.util

    cfg_path = ROOT / "config.py"
    spec = importlib.util.spec_from_file_location("project_config", cfg_path)
    assert spec and spec.loader
    project_cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(project_cfg)

    from micorizae.gate_runflow import gate_train_params_from_config

    params = gate_train_params_from_config(project_cfg)
    assert params.gate4 is not None
    assert params.gate4.enabled is True
    assert params.gate4.num_slices == 4
    assert params.protocol.balance_mode == "g1_stratified"
    assert params.protocol.eval_balance_mode == "dual"
    assert params.protocol.checkpoint_eval == "stratified"
    assert params.protocol.loss_type == "slice_ms_only"
    assert params.gate4.confusable_pairs == ((1, 2), (0, 2))
    assert params.gate4.confusable_base == 0.48
    assert params.gate4.band_start_epoch == 6
    assert params.gate4.proto_subcenters_per_class == (1, 3, 3, 1)
    assert len(params.gate4.confusable_directed_weights) == 3
    assert params.protocol.stratified_root_only_batch_ratio == 0.5
    assert params.protocol.checkpoint_composite_bg_weight == 0.3
    assert "ABS710" in params.protocol.hard_image_substrings
    assert "AFF756" in params.protocol.hard_image_substrings
    assert params.probe is False
    assert params.finetune_mode == "partial_lora"
    assert params.batch_size == 16


def smoke_gate_run_layout() -> None:
    import tempfile

    import pandas as pd

    from micorizae.common.run_outputs import RunOutputs
    from micorizae.phase_d_stage1.gate_run_layout import GateRunLayout, publish_gate_run_layout
    from micorizae.phase_d_stage1.gate_training_protocol import GateTrainProtocol

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "smoke_layout_run"
        run = RunOutputs(run_id="smoke_layout_run", root=root).ensure()

        train_df = pd.DataFrame(
            {
                "stage1": ["Mplus"] * 4 + ["Mminus"] * 3 + ["Background"] * 10 + ["Unreadable"] * 1,
            }
        )
        val_df = pd.DataFrame(
            {
                "stage1": ["Mplus", "Mminus", "Background", "Background", "Unreadable"],
            }
        )
        info_split = {
            "n_train_tiles": len(train_df),
            "n_val_tiles": len(val_df),
            "n_train_images": 2,
            "n_val_images": 1,
            "split_mode": "fixed",
        }
        history = {
            "epochs": [0, 1],
            "train_loss": [1.2, 0.9],
            "val_loss": [1.1, 0.85],
            "val_acc": [0.5, 0.55],
            "val_f1": [0.4, 0.48],
            "best_epoch": 1,
            "checkpoint_metric": "macro_f1",
            "best_val_auroc": 0.48,
            "elapsed_s": [10.0, 9.5],
            "pretrain_baseline": {"macro_f1": 0.35, "balanced_accuracy": 0.33},
            "epoch_details": [
                {
                    "epoch": 1,
                    "balanced_accuracy": 0.45,
                    "min_class_recall": 0.2,
                    "per_class_recall": {"Background": 0.9, "Mminus": 0.2, "Mplus": 0.15, "Unknown": 0.0},
                    "per_class_specificity": {"Background": 0.8, "Mminus": 0.7, "Mplus": 0.75, "Unknown": 0.99},
                }
            ],
        }
        metrics_summary = {
            "test_holdout": {
                "acc": 0.6,
                "macro_f1": 0.42,
                "balanced_accuracy": 0.44,
                "per_class_recall": {"Background": 0.85, "Mminus": 0.25, "Mplus": 0.2, "Unknown": 0.0},
                "per_class_specificity": {"Background": 0.75, "Mminus": 0.72, "Mplus": 0.78, "Unknown": 0.99},
            }
        }
        train_config = {
            "epochs_max": 2,
            "batch_size": 32,
            "balance_mode": "balanced_4class",
            "loss_type": "focal_ce",
            "checkpoint_metric": "macro_f1",
            "gate4": {"enabled": True, "num_slices": 4, "embed_dim": 128, "loss_weight": 0.5},
        }
        protocol = GateTrainProtocol()

        layout = publish_gate_run_layout(
            run=run,
            train_df=train_df,
            val_df=val_df,
            info_split=info_split,
            history=history,
            metrics_summary=metrics_summary,
            train_config=train_config,
            protocol=protocol,
            map_manifest=[{"image": "img/a.jpg", "maps": {"L0_gold_pred_audit": "a__L0_gold_pred_audit.png"}}],
        )

        assert isinstance(layout, GateRunLayout)
        assert (layout.pre_training / "class_distribution.png").exists()
        assert (layout.pre_training / "baseline_metrics.json").exists()
        assert (layout.post_training / "loss_curves.png").exists()
        assert (layout.run_root / "training_protocol.md").exists()
        assert (layout.run_root / "training_report_test.md").exists()
        assert (layout.run_root / "evaluation_metrics_summary_test.json").exists()
        assert (layout.post_test_fullimage / "INDEX.md").exists()


def smoke_dino_attention_maps() -> None:
    dev = _device()
    if dev.type != "cuda":
        print("[gate4 smoke] skip dino attention (no CUDA)")
        return
    from micorizae.phase_d_stage1 import build_branch_a
    from micorizae.phase_d_stage1.gate_dino_attention import (
        dino_attention_config_from_strings,
        extract_dino_cls_patch_attention,
    )

    model = build_branch_a(
        backbone_name="dinov2_vits14", num_classes=4, freeze_backbone=True
    ).to(dev)
    x = torch.randn(2, 3, 252, 252, device=dev)
    cfg = dino_attention_config_from_strings(
        layers_spec="last",
        num_blocks=12,
        num_heads=6,
        head_reduce="mean",
        dino_input_size=252,
    )
    maps = extract_dino_cls_patch_attention(
        model.backbone, x, cfg=cfg, dino_input_size=252
    )
    assert maps.shape == (2, 1, 18, 18)
    assert torch.isfinite(maps).all()
    assert (maps >= 0).all() and (maps <= 1).all()


def smoke_macro_f1_unknown_excluded() -> None:
    from micorizae.phase_d_stage1.gate_training_protocol import compute_gate_metrics

    labels = np.array([0, 0, 1, 1, 2, 2])
    logits = np.zeros((6, 4), dtype=np.float32)
    for i, y in enumerate(labels):
        logits[i, y] = 10.0
    m = compute_gate_metrics(logits, labels)
    assert m["macro_f1"] == 1.0
    assert m["macro_f1_all_classes"] == 0.75
    assert m["n_classes_eval"] == 3


def smoke_ms_only_loss() -> None:
    from micorizae.phase_d_stage1.gate4.config import Gate4SliceMSConfig
    from micorizae.phase_d_stage1.gate4.probe_model import GateSliceProbeModel
    from micorizae.phase_d_stage1.gate4.training import Gate4SliceMSLossOnly
    from micorizae.phase_d_stage1.gate_metric_inference import ClassPrototypeBank

    dev = _device()
    cfg = Gate4SliceMSConfig()
    model = GateSliceProbeModel(in_dim=384, embed_dim=cfg.embed_dim, num_slices=cfg.num_slices).to(dev)
    loss_fn = Gate4SliceMSLossOnly(cfg).to(dev)
    for p in model.gate_head.parameters():
        p.requires_grad = False
    feat = torch.randn(12, 384, device=dev)
    labels = torch.tensor([0, 1, 2] * 4, device=dev)  # sin Unknown (GATE4_INCLUDE_UNKNOWN=False)
    _, embed = model.forward_from_features(feat, return_embed=True)
    out = loss_fn(embed, labels)
    assert out["total"].item() == out["slice_ms"].item()
    assert out["gate"].item() == 0.0
    out["total"].backward()
    assert any(p.grad is not None for p in model.encoder.parameters())

    bank = ClassPrototypeBank(4, cfg.embed_dim, dev)
    bank.update(embed.detach(), labels)
    assert bank.is_ready([0, 1, 2])
    assert not bank.initialized[3].item()
    logits = bank.logits(embed)
    assert logits.shape == (12, 4)


def smoke_ms_hard_mining_confusable() -> None:
    from micorizae.phase_d_stage1.gate4.slice_ms_loss import MultiSimilarityLoss

    dev = _device()
    raw = nn.Parameter(torch.randn(24, 32, device=dev))
    embed = F.normalize(raw, dim=-1)
    labels = torch.tensor([0, 1, 2] * 8, device=dev)

    base = MultiSimilarityLoss(alpha=2.0, beta=50.0, base=0.5).to(dev)
    mined = MultiSimilarityLoss(
        alpha=2.0, beta=50.0, base=0.5, hard_mining=True, mining_margin=0.1
    ).to(dev)
    conf = MultiSimilarityLoss(
        alpha=2.0,
        beta=50.0,
        base=0.5,
        hard_mining=True,
        confusable_pairs=((1, 2),),
        confusable_neg_weight=2.0,
        confusable_base=0.45,
        confusable_band_low=0.40,
        confusable_band_high=0.62,
    ).to(dev)

    for fn in (base, mined, conf):
        loss = fn(embed, labels)
        assert loss.ndim == 0 and torch.isfinite(loss) and loss.item() > 0
    # El sesgo confundible no debe romper el gradiente.
    loss = conf(embed, labels)
    loss.backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()

    # Sin pares confundibles M-/M+ presentes, confusable == hard_mining puro.
    only_bg = torch.zeros(8, dtype=torch.long, device=dev)
    eb = F.normalize(torch.randn(8, 32, device=dev), dim=-1)
    # labels todos iguales => sin negativos => loss 0 (no rompe).
    assert conf(eb, only_bg).item() == 0.0


def smoke_confusable_band_or_mining() -> None:
    """Banda OR debe incluir pares M-/M+ con sim moderada excluidos por semi-hard."""
    from micorizae.phase_d_stage1.gate4.slice_ms_loss import MultiSimilarityLoss

    dev = _device()
    # Intra M- sim=0.9; cruce M-/M+ sim=0.55 (semi-hard con margin 0.1 lo excluye).
    e = torch.zeros(4, 8, device=dev)
    e[0] = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=dev)
    e[1] = torch.tensor([0.9, 0.436, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=dev)
    e[2] = torch.tensor([0.55, 0.835, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=dev)
    e[3] = torch.tensor([0.54, 0.842, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], device=dev)
    embed = F.normalize(e, dim=-1)
    labels = torch.tensor([1, 1, 2, 2], device=dev)

    sim = embed @ embed.t()
    assert torch.allclose(sim[0, 1], torch.tensor(0.9, device=dev), atol=1e-3)
    assert 0.52 < sim[0, 2].item() < 0.58

    semi_only = MultiSimilarityLoss(
        alpha=2.0,
        beta=50.0,
        base=0.5,
        hard_mining=True,
        mining_margin=0.1,
        confusable_pairs=((1, 2),),
    ).to(dev)
    with_band = MultiSimilarityLoss(
        alpha=2.0,
        beta=50.0,
        base=0.5,
        hard_mining=True,
        mining_margin=0.1,
        confusable_pairs=((1, 2),),
        confusable_base=0.48,
        confusable_band_low=0.40,
        confusable_band_high=0.62,
    ).to(dev)

    l_semi = semi_only(embed, labels)
    l_band = with_band(embed, labels)
    assert torch.isfinite(l_semi) and torch.isfinite(l_band)
    assert l_band.item() > l_semi.item(), "banda OR debe activar más negativos confundibles"


def smoke_subcenter_prototypes() -> None:
    from micorizae.phase_d_stage1.gate_metric_inference import ClassPrototypeBank

    dev = _device()
    bank = ClassPrototypeBank(4, 32, dev, num_subcenters=3)
    emb = F.normalize(torch.randn(60, 32, device=dev), dim=-1)
    labels = torch.tensor([0, 1, 2] * 20, device=dev)
    bank.update(emb, labels)
    assert bank.is_ready([0, 1, 2])
    assert not bool(bank.initialized[3].any().item())
    assert bank.prototypes.shape == (4, 3, 32)
    logits = bank.logits(emb)
    assert logits.shape == (60, 4) and torch.isfinite(logits[:, :3]).all()
    rep = bank.representative_prototypes()
    assert rep.shape == (4, 32)

    # Round-trip state_dict.
    bank2 = ClassPrototypeBank(4, 32, dev, num_subcenters=3)
    bank2.load_state_dict(bank.state_dict())
    assert bank2.num_subcenters == 3
    assert torch.allclose(bank2.prototypes, bank.prototypes)

    # Compat: checkpoint antiguo (2-D) carga como K=1.
    old_state = {
        "prototypes": torch.randn(4, 32),
        "initialized": torch.tensor([True, True, True, False]),
        "momentum": 0.05,
        "scale": 10.0,
    }
    bank3 = ClassPrototypeBank(4, 32, dev, num_subcenters=1)
    bank3.load_state_dict(old_state)
    assert bank3.prototypes.shape == (4, 1, 32)
    assert bank3.is_ready([0, 1, 2])


def smoke_sampler_class_weights() -> None:
    import pandas as pd

    from micorizae.phase_d_stage1.gate_epoch_sampler import (
        GateEpochSamplerConfig,
        audit_stratified_batches,
        plan_epoch_stratified,
    )

    n = 300
    df = pd.DataFrame(
        {
            "image_path": [f"img/{i // 50}.jpg" for i in range(n)],
            "row": list(range(n)),
            "col": [0] * n,
            "stage1": (["Mplus"] * 60 + ["Mminus"] * 60 + ["Background"] * 180),
        }
    )
    cfg = GateEpochSamplerConfig(
        samples_per_class=120,
        min_per_class_per_batch=8,
        class_weights=(("Background", 1.0), ("Mminus", 1.5), ("Mplus", 1.5)),
    )
    quota = cfg.quota_per_class()
    assert quota == {"Background": 8, "Mminus": 12, "Mplus": 12}
    plan = plan_epoch_stratified(df, cfg=cfg, seed=0)
    audit = audit_stratified_batches(plan)
    assert audit["Mplus_per_batch_min"] == 12
    assert audit["Mminus_per_batch_min"] == 12
    assert audit["Background_per_batch_min"] == 8


def smoke_annotated_pretrain_viz() -> None:
    import tempfile

    import pandas as pd

    from micorizae.phase_d_stage1.gate_pretrain_viz import (
        plot_tile_samples_annotated,
        write_pre_training_index,
    )

    df = pd.DataFrame(
        {
            "image_path": ["Data/x.jpg"] * 4,
            "row": [0, 0, 1, 1],
            "col": [0, 1, 0, 1],
            "stage1": ["Mplus", "Mminus", "Background", "Mplus"],
            "aug_id": ["", "hflip", "", ""],
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "annot.png"
        manifest = Path(tmp) / "manifest.csv"
        # Sin JPEG real: función debe crear PNG (celdas missing/err OK)
        plot_tile_samples_annotated(
            df,
            out,
            split="train",
            etapa="pre_train",
            n_per_class=2,
            manifest_csv=manifest,
        )
        assert out.exists()
        assert manifest.exists()
        idx = write_pre_training_index(Path(tmp))
        assert idx.exists()


def smoke_bg_only_pooling() -> None:
    from micorizae.phase_d_stage1.gate_bg_pooling import bg_only_pooling_mask
    from micorizae.phase_d_stage1.gate_classes import GATE_CLASS_TO_IDX

    dev = _device()
    sal = torch.rand(4, 32, 32, device=dev)
    labels = torch.tensor(
        [
            GATE_CLASS_TO_IDX["Background"],
            GATE_CLASS_TO_IDX["Mplus"],
            GATE_CLASS_TO_IDX["Mminus"],
            GATE_CLASS_TO_IDX["Background"],
        ],
        device=dev,
    )
    mask = bg_only_pooling_mask(labels, sal, pooling_mode="bg_only")
    assert mask is not None
    assert torch.allclose(mask[0], sal[0])
    assert torch.allclose(mask[1], torch.ones_like(sal[1]))
    assert torch.allclose(mask[2], torch.ones_like(sal[2]))


def smoke_dino_finetune_lora() -> None:
    from micorizae.phase_d_stage1.gate_dino_finetune import LoRALinear, configure_dino_finetune
    from micorizae.phase_d_stage1 import build_branch_a

    dev = _device()
    branch = build_branch_a("dinov2_vits14", num_classes=4, freeze_backbone=True)
    info = configure_dino_finetune(
        branch.backbone, mode="partial_lora", last_n_blocks=2, lora_rank=4
    )
    assert info["last_n_blocks"] == 2
    assert info["lora_adapters"] >= 1
    lin = nn.Linear(8, 8)
    lora = LoRALinear(lin, rank=4, alpha=8.0).to(dev)
    x = torch.randn(2, 8, device=dev)
    y = lora(x)
    assert y.shape == (2, 8)


def run_all_smokes() -> None:
    smoke_gate4_classes()
    smoke_slice_encoder()
    smoke_multi_similarity_loss()
    smoke_slice_ms_loss()
    smoke_probe_model()
    smoke_combined_loss()
    smoke_gate_run_live_publisher()
    smoke_format_g1_status_nan()
    smoke_plan_epoch_gate4()
    smoke_g1_stratified_sampler()
    smoke_dual_eval_plans()
    smoke_gate_runflow_config()
    smoke_gate_run_layout()
    smoke_dino_attention_maps()
    smoke_macro_f1_unknown_excluded()
    smoke_ms_only_loss()
    smoke_ms_hard_mining_confusable()
    smoke_confusable_band_or_mining()
    smoke_subcenter_prototypes()
    smoke_sampler_class_weights()
    smoke_annotated_pretrain_viz()
    smoke_bg_only_pooling()
    smoke_dino_finetune_lora()
    print("[gate4 smoke] OK — 23 tests passed")


if __name__ == "__main__":
    run_all_smokes()
