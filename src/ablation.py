from dataclasses import replace
from typing import Dict, Any, List

from src.config import Config
from src.build_memory import build_and_save_memory_banks
from src.evaluate import evaluate_model
from src.utils import log, save_json


def run_ablation(config: Config) -> List[Dict[str, Any]]:
    """
    Run ablation experiments:
    1. Backbone: ResNet18 vs WideResNet50_2
    2. Memory: Baseline vs Coreset
    3. Features: Layer3 only vs Layer2+Layer3 (controlled via weights)

    """

    config.ensure_dirs()

    ablations = [
        {"name": "resnet18_baseline", "backbone": "resnet18", "coreset": False},
        {"name": "resnet18_coreset", "backbone": "resnet18", "coreset": True},

        {"name": "wide_resnet_baseline", "backbone": "wide_resnet50_2", "coreset": False},
        {"name": "wide_resnet_coreset", "backbone": "wide_resnet50_2", "coreset": True},

        {"name": "layer3_only", "backbone": "wide_resnet50_2", "coreset": True, "layer2_weight": 0.0},

        {"name": "full_model", "backbone": "wide_resnet50_2", "coreset": True, "layer2_weight": 1.0},
    ]

    results = []

    for exp in ablations:
        log(f"Running ablation: {exp['name']}")

        cfg = replace(
            config,
            backbone=exp["backbone"],
            use_coreset=exp["coreset"],
            layer2_weight=exp.get("layer2_weight", config.layer2_weight),
        )

        build_and_save_memory_banks(cfg)

        bank_type = "coreset" if cfg.use_coreset else "baseline"
        metrics = evaluate_model(cfg, bank_type=bank_type)

        result = {
            "experiment": exp["name"],
            "backbone": cfg.backbone,
            "coreset": cfg.use_coreset,
            "image_auroc": metrics["image_auroc"],
            "pixel_auroc": metrics["pixel_auroc"],
            "f1": metrics["best_f1"],
            "threshold": metrics["best_threshold"],
        }

        results.append(result)

    output_path = config.results_dir / f"{config.category}_ablation_results.json"
    save_json(results, output_path)

    log(f"Ablation results saved to {output_path}")

    return results


if __name__ == "__main__":
    cfg = Config()
    results = run_ablation(cfg)

    for r in results:
        print(r)
