import time
from typing import Dict, Any

import numpy as np
from PIL import Image

from src.config import Config
from src.data_loader import MVTecBottleDataset
from src.inference import AnomalyDetector
from src.utils import log, save_json


def benchmark_inference(
    config: Config,
    bank_type: str = "coreset",
    num_samples: int = 50,
) -> Dict[str, Any]:
    """
    Benchmark end-to-end inference latency.

    Measures:
    - total latency per image
    - average latency
    - throughput (images/sec)
    """

    config.ensure_dirs()

    dataset = MVTecBottleDataset(config, split="test")

    num_samples = min(num_samples, len(dataset))

    detector = AnomalyDetector(config=config, bank_type=bank_type)

    log(f"Running benchmark on {num_samples} samples...")

    latencies = []

    for i in range(5):
        sample = dataset[i]
        detector.predict(sample["image"])

    for i in range(num_samples):
        sample = dataset[i]
        image = sample["image"]

        start = time.time()
        _ = detector.predict(image)
        end = time.time()

        latency_ms = (end - start) * 1000.0
        latencies.append(latency_ms)

    latencies = np.array(latencies, dtype=np.float32)

    avg_latency = float(np.mean(latencies))
    std_latency = float(np.std(latencies))
    min_latency = float(np.min(latencies))
    max_latency = float(np.max(latencies))

    throughput = float(1000.0 / avg_latency)

    result = {
        "category": config.category,
        "bank_type": bank_type,
        "num_samples": num_samples,
        "avg_latency_ms": avg_latency,
        "std_latency_ms": std_latency,
        "min_latency_ms": min_latency,
        "max_latency_ms": max_latency,
        "throughput_img_per_sec": throughput,
    }

    save_path = config.benchmarks_dir / f"{config.category}_{bank_type}_benchmark.json"
    save_json(result, save_path)

    log(f"Benchmark saved to {save_path}")
    log(f"Avg latency: {avg_latency:.2f} ms | Throughput: {throughput:.2f} img/sec")

    return result


if __name__ == "__main__":
    cfg = Config()
    benchmark_inference(cfg, bank_type="coreset")
