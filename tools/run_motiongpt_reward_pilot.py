"""Train-only, two-candidate LoRA reward pilot with a matched SFT control.

No test rewards enter the optimizer. The five-condition panel must be complete
before this command starts. A successful pilot proves mechanics, not quality.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from video_to_3d_motion.motion_generation.adapters import (
    adapter_state, adapters_disabled, clipped_reward_loss, frozen_digest, inject_lora, restore_adapter_state)
from video_to_3d_motion.motion_generation.critic import Critic, SmplFitter
from video_to_3d_motion.motion_generation.evaluation import seed_everything
from video_to_3d_motion.motion_generation.runtime import Generator, load_config, provenance, write_json


def base_identity(identity):
    return identity.removeprefix("M")


def select_training_condition(dataset, panel, seed):
    train = (dataset / "train.txt").read_text(encoding="utf-8").split()
    forbidden = {base_identity(row["motion_id"]) for row in panel["samples"]}
    forbidden_captions = {row["caption"].strip().lower() for row in panel["samples"]}
    forbidden.update(base_identity(name) for split in ("test", "val")
                     for name in (dataset / f"{split}.txt").read_text(encoding="utf-8").split())
    for position in np.random.default_rng(seed).permutation(len(train)):
        identity = train[int(position)]
        if base_identity(identity) in forbidden:
            continue
        raw = np.load(dataset / "new_joint_vecs" / f"{identity}.npy", allow_pickle=False)
        if not 100 <= len(raw) < 200:
            continue
        for line in (dataset / "texts" / f"{identity}.txt").read_text(encoding="utf-8").splitlines():
            parts = line.split("#")
            if (len(parts) == 4 and float(parts[2]) == 0 and float(parts[3]) == 0
                    and parts[0].strip().lower() not in forbidden_captions):
                return {"motion_id": identity, "caption": parts[0], "split": "train"}, raw[:len(raw) // 4 * 4]
    raise ValueError("No eligible training-only condition found")


def token_distribution(lm, input_ids, attention_mask, sequence):
    logits = lm(input_ids=input_ids, attention_mask=attention_mask,
                decoder_input_ids=sequence[:, :-1], use_cache=False).logits
    return torch.log_softmax(logits.float(), dim=-1)


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/motiongpt_stage2/pilot.yaml")
    parser.add_argument("--resume", action="store_true", help="Reuse the same saved rollouts and completed SMPL fits")
    args = parser.parse_args()
    config_file = args.config.resolve()
    settings = yaml.safe_load(config_file.read_text(encoding="utf-8"))
    resolve = lambda key: (config_file.parent / settings[key]).resolve()
    cfg = load_config(resolve("stage1_config"))
    panel = json.loads(resolve("panel").read_text(encoding="utf-8"))
    if panel.get("count") != 5 or panel.get("coverage") != 1 or panel.get("status") != "five_test_conditions_frozen":
        raise ValueError("Complete the five held-out scores before starting stage two")
    if settings["train_prompts"] != 1 or settings["candidates_per_prompt"] != 2:
        raise ValueError("This pilot deliberately supports one train prompt and two candidates")
    output = resolve("output")
    output.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        saved_settings = json.loads((output / "config.json").read_text(encoding="utf-8"))
        saved_panel = json.loads((output / "heldout_panel.json").read_text(encoding="utf-8"))
        if saved_settings != settings or saved_panel != panel:
            raise ValueError("Resume requires the same settings and held-out panel")
        if (output / "pilot_result.json").exists():
            print("Pilot already complete; no additional updates performed")
            return
    write_json(output / "config.json", settings)
    current_version = provenance(cfg)
    if args.resume:
        previous_version = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
        for key in ("model_asset_sha256", "dataset_manifest_sha256", "sources"):
            if current_version[key] != previous_version[key]:
                raise ValueError(f"Resume resource changed: {key}")
    write_json(output / ("resume_provenance.json" if args.resume else "provenance.json"), current_version)
    write_json(output / "heldout_panel.json", panel)
    seed_everything(settings["seed"])
    condition, reference = select_training_condition(Path(cfg["paths"]["dataset"]), panel, settings["seed"])
    write_json(output / "training_condition.json", condition)
    generator = Generator(cfg)
    tokenizer, lm = generator.lm.tokenizer, generator.lm.language_model
    device = generator.device
    prompt = "Generate motion: " + condition["caption"]
    encoding = tokenizer([prompt], padding="max_length", max_length=256, truncation=True, return_tensors="pt")
    input_ids, attention_mask = encoding.input_ids.to(device), encoding.attention_mask.to(device)
    rollout_path = output / "rollouts.pt"
    if args.resume and rollout_path.exists():
        rollout = torch.load(rollout_path, map_location="cpu")
        if rollout["condition"] != condition:
            raise ValueError("Saved rollout belongs to a different train condition")
        input_ids = rollout["input_ids"].to(device)
        attention_mask = rollout["attention_mask"].to(device)
        gt_labels = rollout["gt_labels"].to(device)
        candidates = rollout["candidates"]
        rejected = json.loads((output / "sampling_rejections.json").read_text(encoding="utf-8"))
        print("Reusing saved train-only rollouts", flush=True)
    else:
        candidates, rejected = [], []
        for attempt in range(settings["max_candidate_attempts"]):
            if len(candidates) == settings["candidates_per_prompt"]:
                break
            seed_everything(settings["seed"] + attempt)
            with torch.no_grad():
                sequence = lm.generate(input_ids=input_ids, attention_mask=attention_mask,
                                       max_new_tokens=settings["max_new_tokens"], do_sample=True,
                                       num_beams=1, **settings["sampling"])
                text = tokenizer.batch_decode(sequence, skip_special_tokens=True)
                generator.lm.device = device
                try:
                    tokens = generator.lm.motion_string_to_token(text)[0][0].clamp(0, 511)
                except (ValueError, IndexError) as exc:
                    rejected.append({"attempt": attempt, "reason": "unparseable_generation", "error": str(exc)})
                    continue
                frames = len(tokens) * 4
                if len(tokens) <= 1 or frames < cfg["critic"]["window_frames"]:
                    rejected.append({"attempt": attempt, "reason": "invalid_or_too_short", "tokens": len(tokens)})
                    continue
                raw = generator.vae.decode(tokens)[0].cpu().numpy() * generator.std + generator.mean
                joints = generator.motion_process.recover_from_ric(torch.from_numpy(raw), 22).numpy()
                old = token_distribution(lm, input_ids, attention_mask, sequence).cpu()
            index = len(candidates)
            np.savez_compressed(output / f"candidate_{index}.npz", features=raw, joints=joints, tokens=tokens.cpu().numpy())
            candidates.append({"index": index, "attempt": attempt, "frames": len(raw),
                               "sequence": sequence.cpu(), "reference_log_probs": old})
            print(f"Training candidate {index + 1}/2: {len(raw)} frames", flush=True)
        write_json(output / "sampling_rejections.json", rejected)
        if len(candidates) != 2:
            raise RuntimeError("Unable to sample two scoreable training candidates; no optimizer step performed")
        # A supervised control uses the same train condition's real motion tokens.
        with torch.no_grad():
            normalized = torch.from_numpy((reference - generator.mean) / generator.std).float().to(device)[None]
            gt_tokens = generator.vae.encode(normalized)[0]
            gt_text = generator.lm.motion_token_list_to_string(gt_tokens)[0]
        gt_labels = tokenizer([gt_text], return_tensors="pt").input_ids.to(device)
        torch.save({"condition": condition, "input_ids": input_ids.cpu(), "attention_mask": attention_mask.cpu(),
                    "candidates": candidates, "gt_labels": gt_labels.cpu()}, output / "rollouts.pt")
    # Free the language model while the expensive per-frame fits run.
    generator.lm.cpu()
    generator.vae.cpu()
    gc.collect()
    torch.cuda.empty_cache()
    critic, fitter = Critic(cfg), SmplFitter(cfg)
    scores = []
    for candidate in candidates:
        reward_started = time.monotonic()
        fitted_path = output / f"candidate_{candidate['index']}_smpl.npz"
        if args.resume and fitted_path.exists():
            with np.load(fitted_path, allow_pickle=False) as arrays:
                fitted = {key: arrays[key] for key in arrays.files}
        else:
            with np.load(output / f"candidate_{candidate['index']}.npz", allow_pickle=False) as arrays:
                fitted = fitter.fit(arrays["joints"])
            np.savez_compressed(fitted_path, **fitted)
        score = critic.score_smpl(fitted["pose"], fitted["translation"])
        if score["score"] is None:
            raise RuntimeError("Training candidate has no perceptual reward")
        candidate["reward"] = score["score"]
        scores.append({"index": candidate["index"], "motion_id": condition["motion_id"],
                       "caption": condition["caption"], "frames": candidate["frames"], **score,
                       "fit_rmse_mean_m": float(fitted["fit_rmse_m"].mean()),
                       "fit_rmse_max_m": float(fitted["fit_rmse_m"].max()),
                       "elapsed_seconds_this_run": time.monotonic() - reward_started})
        write_json(output / "training_rewards.json", {"split": "train", "scores": scores})
        print(f"Train reward {candidate['index']}: {score['score']:.4f}", flush=True)
    del critic, fitter
    gc.collect()
    torch.cuda.empty_cache()
    generator.lm.to(device)
    lm = generator.lm.language_model
    # Keep dropout disabled so sampling, old-policy and current-policy
    # likelihoods refer to the same deterministic conditional model.
    lm.eval()
    seed_everything(settings["seed"])
    targets = inject_lora(lm, **settings["lora"])
    initial = adapter_state(lm)
    frozen_before = frozen_digest(lm)
    parameters = [parameter for parameter in lm.parameters() if parameter.requires_grad]
    rewards = np.array([candidate["reward"] for candidate in candidates])
    advantages = rewards - rewards.mean()
    if np.max(np.abs(advantages)) < 1e-8:
        raise RuntimeError("Identical rewards: cannot validate a reward-driven update")
    logs, checkpoints = {}, {}
    for branch in ("sft_control", "critic_reward"):
        restore_adapter_state(lm, initial)
        optimizer = torch.optim.AdamW(parameters, lr=settings["learning_rate"], weight_decay=0)
        history = []
        for step in range(settings["update_steps"]):
            optimizer.zero_grad(set_to_none=True)
            loss_value, kl_value, ratio_value = 0., 0., 0.
            if branch == "sft_control":
                loss = lm(input_ids=input_ids, attention_mask=attention_mask, labels=gt_labels, use_cache=False).loss
                loss.backward()
                loss_value = float(loss.detach())
            else:
                for candidate, advantage in zip(candidates, advantages):
                    sequence = candidate["sequence"].to(device)
                    labels = sequence[:, 1:]
                    # Each rollout is generated separately, so there is no
                    # inter-sequence padding. Include every sampled action,
                    # even a literal pad-ID action, and include the final EOS.
                    mask = torch.ones_like(labels, dtype=torch.bool)
                    reference_log_probs = candidate["reference_log_probs"].to(device)
                    old_taken = reference_log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                    log_probs = token_distribution(lm, input_ids, attention_mask, sequence)
                    loss, diagnostics = clipped_reward_loss(log_probs, old_taken, reference_log_probs,
                                                           labels, mask, float(advantage),
                                                           settings["clip_range"], settings["kl_weight"])
                    (loss / len(candidates)).backward()
                    loss_value += float(loss.detach()) / len(candidates)
                    kl_value += float(diagnostics["kl"]) / len(candidates)
                    ratio_value += float(diagnostics["ratio"]) / len(candidates)
            norm = torch.nn.utils.clip_grad_norm_(parameters, settings["max_grad_norm"])
            if not torch.isfinite(norm) or norm <= 0:
                raise RuntimeError(f"Invalid gradient norm in {branch}: {norm}")
            optimizer.step()
            history.append({"step": step + 1, "loss": loss_value, "kl": kl_value,
                            "mean_ratio": ratio_value, "gradient_norm_before_clip": float(norm)})
            print(f"{branch} step {step + 1}: loss={loss_value:.6f}, grad={norm:.6f}", flush=True)
        state = adapter_state(lm)
        delta = sum(float((state[name] - initial[name]).double().square().sum()) for name in state) ** .5
        if delta <= 0 or frozen_digest(lm) != frozen_before:
            raise RuntimeError("Adapter did not update, or frozen model weights changed")
        checkpoint = output / f"{branch}_adapter.pt"
        torch.save({"adapter_state": state, "lora": settings["lora"], "target_modules": targets,
                    "branch": branch, "base_model_sha256": provenance(cfg)["model_asset_sha256"]["motiongpt_s3_h3d.tar"]}, checkpoint)
        checkpoints[branch] = str(checkpoint.resolve())
        logs[branch] = {"updates": history, "adapter_delta_l2": delta, "frozen_weights_unchanged": True}
        write_json(output / "training_progress.json", logs)
        del optimizer
    # Regenerate held-out conditions with a shared per-condition seed. These
    # clips are visual diagnostics; no test reward is used for model selection.
    previews = []
    for branch in ("base", "sft_control", "critic_reward"):
        if branch == "base":
            restore_adapter_state(lm, initial)
        else:
            saved = torch.load(checkpoints[branch], map_location="cpu")
            restore_adapter_state(lm, saved["adapter_state"])
        generator.vae.to(device)
        for index, heldout in enumerate(panel["samples"]):
            seed_everything(settings["seed"] + 1000 + index)
            generated = generator.generate([heldout["caption"]])[0]
            filename = f"heldout_{branch}_{index}.npz"
            np.savez_compressed(output / filename, **generated)
            previews.append({"branch": branch, "motion_id": heldout["motion_id"], "caption": heldout["caption"],
                             "seed": settings["seed"] + 1000 + index, "file": filename,
                             "frames": len(generated["features"]), "valid_generation": generated["valid_generation"]})
    write_json(output / "heldout_previews.json", previews)
    token_changes = {}
    for branch in ("sft_control", "critic_reward"):
        changed = 0
        for index in range(len(panel["samples"])):
            with np.load(output / f"heldout_base_{index}.npz", allow_pickle=False) as base_array, np.load(
                    output / f"heldout_{branch}_{index}.npz", allow_pickle=False) as branch_array:
                changed += not np.array_equal(base_array["tokens"], branch_array["tokens"])
        token_changes[branch] = int(changed)
    report = {"status": "pilot_completed_not_quality_evaluated", "training_split": "train",
              "train_condition": condition, "train_prompts": 1, "training_candidates": 2,
              "candidate_rewards": rewards.tolist(), "centered_advantages": advantages.tolist(),
              "sampling": settings["sampling"], "rejected_generations": rejected,
              "method": "token-level clipped group-relative policy loss + exact KL; no value network; scoreable-candidate pilot",
              "trainable_parameters": sum(parameter.numel() for parameter in parameters),
              "lora_target_modules": targets, "checkpoints": checkpoints, "branches": logs,
              "heldout_conditions": 5, "test_conditions_used_for_training": 0,
              "heldout_previews": str((output / "heldout_previews.json").resolve()),
              "heldout_output_token_changes_vs_base": token_changes,
              "heldout_valid_generations": {branch: sum(row["valid_generation"] for row in previews if row["branch"] == branch)
                                            for branch in ("base", "sft_control", "critic_reward")},
              "post_training_metrics": None,
              "elapsed_seconds_this_run": time.monotonic() - started,
              "peak_allocated_gpu_mb": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None,
              "limitation": "Two updates on one training condition only validate reward/gradient/checkpoint mechanics. No quality improvement or benchmark conclusion is claimed. Test conditions never enter the optimizer."}
    write_json(output / "pilot_result.json", report)
    print(json.dumps({"status": report["status"], "result": str(output / "pilot_result.json")}))


if __name__ == "__main__":
    main()
