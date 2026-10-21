"""Evaluate frozen outputs with the upstream dataset, encoders and metrics.

Keep evaluator batch size 32: upstream's motion GRU processes padded frames,
so changing batch padding changes the embeddings even in evaluation mode.
Generation can use microbatches of one without changing evaluator batching.
"""
from __future__ import annotations

import importlib
import random
from pathlib import Path

import numpy as np
import torch

from .runtime import Generator, find_asset, load_file, norms, official_modules, provenance, write_json


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(cfg: dict):
    official_modules(cfg)
    dataset_module = importlib.import_module("motiongpt_humanml.dataset_t2m_eval")
    vector_module = importlib.import_module("motiongpt_humanml.utils.word_vectorizer")
    assets = Path(cfg["paths"]["assets"])
    vocab = find_asset(assets, "evaluator/t2m/glove/our_vab_data.npy").parent
    vectorizer = vector_module.WordVectorizer(str(vocab), "our_vab")
    mean, std = norms(cfg)[0]
    return dataset_module.Text2MotionDatasetEval(
        data_root=cfg["paths"]["dataset"], split=cfg["evaluation"]["split"], mean=mean,
        std=std, w_vectorizer=vectorizer, max_motion_length=196, min_motion_length=40,
        unit_length=4, fps=20, tmpFile=False)


class Evaluator:
    def __init__(self, cfg: dict):
        official_modules(cfg)
        from mGPT.archs.tm2t_evaluator import MovementConvEncoder, MotionEncoderBiGRUCo, TextEncoderBiGRUCo
        self.device = torch.device(cfg["device"])
        self.text = TextEncoderBiGRUCo(300, 15, 512, 512)
        self.movement = MovementConvEncoder(259, 512, 512)
        self.motion = MotionEncoderBiGRUCo(512, 1024, 512)
        checkpoint = torch.load(find_asset(Path(cfg["paths"]["assets"]), "t2m/text_mot_match/model/finest.tar"), map_location="cpu")
        for model, key in [(self.text, "text_encoder"), (self.movement, "movement_encoder"), (self.motion, "motion_encoder")]:
            model.load_state_dict(checkpoint[key], strict=True)
            model.eval().requires_grad_(False).to(self.device)
        self.mean, self.std = norms(cfg)[1]
        source = Path(cfg["paths"]["motiongpt_source"])
        self.utils = load_file("motiongpt_metric_math", source / "mGPT/metrics/utils.py")

    @torch.inference_mode()
    def embed_motion(self, raw: np.ndarray, lengths: list[int]) -> np.ndarray:
        features = torch.from_numpy((raw - self.mean) / self.std).float().to(self.device)
        lengths = np.asarray(lengths)
        order = np.argsort(lengths)[::-1].copy()
        movement = self.movement(features[order, :, :-4])
        # Follow upstream including its length conversion. The pinned GRU
        # does not pack sequences; see tm2t_evaluator.py at the recorded commit.
        embedding = self.motion(movement, torch.tensor(lengths[order] // 4 // 4))
        return embedding.cpu().numpy()[np.argsort(order)]

    @torch.inference_mode()
    def embed_text(self, samples: list[tuple]) -> np.ndarray:
        word = torch.tensor(np.stack([s[3] for s in samples]), dtype=torch.float32, device=self.device)
        pos = torch.tensor(np.stack([s[4] for s in samples]), dtype=torch.float32, device=self.device)
        length = torch.tensor([s[5] for s in samples])
        return self.text(word, pos, length).cpu().numpy()

    def summarize(self, text: np.ndarray, generated: np.ndarray, real: np.ndarray, diversity_pairs: int) -> dict:
        count = len(text)
        if count < 32 or count <= diversity_pairs:
            raise ValueError("Need >=32 retrieval candidates and more samples than diversity_pairs")
        if not all(np.isfinite(x).all() for x in (text, generated, real)):
            raise ValueError("Non-finite embeddings; metrics refused")
        shuffle = torch.randperm(count).numpy()
        text, generated, real = text[shuffle], generated[shuffle], real[shuffle]
        result = {"samples": count, "retrieval_samples": count // 32 * 32,
                  "retrieval_group_size": 32, "diversity_pairs": diversity_pairs}
        for prefix, motion in [("", generated), ("GT_", real)]:
            hits = np.zeros(3)
            matching = 0.
            for start in range(0, count - 31, 32):
                distance = self.utils.euclidean_distance_matrix(
                    torch.from_numpy(text[start:start + 32]), torch.from_numpy(motion[start:start + 32])).nan_to_num()
                matching += distance.trace().item()
                ranks = torch.argsort(distance, dim=1)
                hits += self.utils.calculate_top_k(ranks, top_k=3).sum(axis=0).numpy()
            for k in (1, 2, 3):
                result[f"{prefix}R_precision_top_{k}"] = float(hits[k - 1] / result["retrieval_samples"])
            result[f"{prefix}MM_Dist"] = float(matching / result["retrieval_samples"])
            result[f"{prefix}Diversity"] = float(self.utils.calculate_diversity_np(motion, diversity_pairs))
        mu, cov = self.utils.calculate_activation_statistics_np(generated)
        gt_mu, gt_cov = self.utils.calculate_activation_statistics_np(real)
        result["FID"] = float(self.utils.calculate_frechet_distance_np(gt_mu, gt_cov, mu, cov))
        return result


def evaluate(cfg: dict, output: Path, max_samples: int | None = None, repetitions: int | None = None) -> dict:
    if cfg["evaluation"]["batch_size"] != 32:
        raise ValueError("Pinned official evaluator protocol requires batch_size=32")
    seed_everything(cfg["seed"])
    dataset = load_dataset(cfg)
    total = min(max_samples, len(dataset)) if max_samples else len(dataset)
    selected_indexes = (np.sort(np.random.default_rng(cfg["seed"]).choice(len(dataset), total, replace=False))
                        if max_samples else np.arange(len(dataset)))
    repeats = repetitions or cfg["evaluation"]["repetitions"]
    pairs = cfg["evaluation"]["diversity_pairs"]
    if total < 32 or total <= pairs:
        raise ValueError(f"Need at least {max(32, pairs + 1)} samples; got {total}. Use demo for a tiny smoke test.")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", cfg)
    write_json(output / "provenance.json", provenance(cfg))
    generator = Generator(cfg)
    evaluator = Evaluator(cfg)
    mean, std = norms(cfg)[0]
    runs = []
    for repetition in range(repeats):
        seed_everything(cfg["seed"] + repetition)
        run_dir = output / f"rep_{repetition:02d}"
        run_dir.mkdir()
        text_embeddings, generated_embeddings, real_embeddings = [], [], []
        invalid = 0
        with (run_dir / "samples.jsonl").open("w", encoding="utf-8") as manifest:
            import json
            for start in range(0, total, 32):
                samples = [(int(index), dataset[int(index)]) for index in selected_indexes[start:start + 32]]
                samples.sort(key=lambda pair: pair[1][5], reverse=True)
                indexes, batch = zip(*samples)
                lengths = [int(item[2]) for item in batch]
                max_length = max(lengths)
                ref = np.broadcast_to(mean, (len(batch), max_length, 263)).copy()
                gen = ref.copy()
                output_lengths = []
                generated = []
                microbatch = cfg["evaluation"].get("generation_batch_size", 8)
                if microbatch < 1:
                    raise ValueError("generation_batch_size must be positive")
                for offset in range(0, len(batch), microbatch):
                    generated.extend(generator.generate([item[0] for item in batch[offset:offset + microbatch]], lengths[offset:offset + microbatch]))
                for position, (index, sample) in enumerate(zip(indexes, batch)):
                    record = generated[position]
                    length = min(len(record["features"]), lengths[position])
                    output_lengths.append(length)
                    ref[position, :lengths[position]] = sample[1] * std + mean
                    gen[position, :length] = record["features"][:length]
                    invalid += not record["valid_generation"]
                    filename = f"{index:06d}.npz"
                    np.savez_compressed(run_dir / filename, features=record["features"][:length],
                                        joints=record["joints"][:length], tokens=record["tokens"],
                                        reference_features=ref[position, :lengths[position]])
                    manifest.write(json.dumps({"index": index, "motion_id": dataset.name_list[dataset.pointer + index],
                                               "caption": sample[0], "file": filename, "frames": length,
                                               "reference_frames": lengths[position], "fps": 20,
                                               "valid_generation": record["valid_generation"]}, ensure_ascii=False) + "\n")
                    manifest.flush()
                text_embeddings.append(evaluator.embed_text(list(batch)))
                generated_embeddings.append(evaluator.embed_motion(gen, output_lengths))
                real_embeddings.append(evaluator.embed_motion(ref, lengths))
                print(f"Evaluation rep {repetition + 1}/{repeats}: {min(start + 32, total)}/{total}", flush=True)
        embedded = {"text": np.concatenate(text_embeddings), "generated": np.concatenate(generated_embeddings),
                    "real": np.concatenate(real_embeddings)}
        np.savez_compressed(run_dir / "embeddings.npz", **embedded)
        metrics = evaluator.summarize(embedded["text"], embedded["generated"], embedded["real"], pairs)
        metrics["invalid_generations"] = int(invalid)
        write_json(run_dir / "metrics.json", metrics)
        runs.append(metrics)
        write_json(output / "progress.json", {"completed_repetitions": len(runs), "requested_repetitions": repeats})
    numerical_keys = [key for key in runs[0] if key not in ("samples", "retrieval_samples", "retrieval_group_size", "diversity_pairs", "invalid_generations")]
    summary = {key: {"mean": float(np.mean([run[key] for run in runs])),
                     "ci95": float(1.96 * np.std([run[key] for run in runs]) / np.sqrt(repeats)) if repeats > 1 else None}
               for key in numerical_keys}
    report = {"status": "baseline_evaluated_critic_pending", "dataset": "HumanML3D",
              "subset": max_samples is not None, "samples": total, "repetitions": repeats,
              "subset_selection": "seeded_random_without_replacement" if max_samples else "full_eligible_test_set",
              "paper_protocol_run": max_samples is None and repeats == 20,
              "metrics": summary, "runs": runs,
              "note": "MotionCritic is scored separately from saved samples. These results are not a completed perceptual evaluation."}
    write_json(output / "baseline_metrics.json", report)
    return report
