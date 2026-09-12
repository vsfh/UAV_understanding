#!/usr/bin/env python3
"""Five-stage UAV synthesis: download -> two LoRAs -> prompts -> crops -> outpaint.

Run from /media/data2/feihong/UAV_understanding:
  /home/feihong/miniconda3/bin/python scripts/uav_synthesis_pipeline.py --stage setup
  .venvs/uav_synthesis/bin/python scripts/uav_synthesis_pipeline.py --stage all --gpu 1

The defaults use 500 balanced, captioned TRAIN pairs and 3,000 new examples per
configured class. --stage inspect is read-only; prepare writes the 500-pair
dataset without downloading weights or training. Each stage can run separately.
Use a new --work-dir and --synthetic-root for a different experiment.

Core reuse (Apache-2.0):
  HF Diffusers v0.40.0 / d035dcd7cc7c88e0a154609b62887d50bba9fdc2:
  examples/dreambooth/train_dreambooth_lora_flux2_klein.py (run unchanged)
  examples/text_to_image/train_text_to_image_lora_sdxl.py
  nikgli/train-lora-sdxl-inpaint / 20e17660a9c83efa3bf668156c87dcc21cf70e7b:
  diffusers/examples/research_projects/dreambooth_inpaint/
  train_dreambooth_inpaint_lora_sdxl.py (9-channel inpainting construction)

The project-specific code pairs the actual description schema with crops,
uses the Perception YAML resize policy, and replaces random inpainting masks
with the real contextual ROI. No generated image is inserted into real splits.
"""

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import subprocess
import sys
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

DIFFUSERS_COMMIT = "d035dcd7cc7c88e0a154609b62887d50bba9fdc2"
KLEIN_MODEL = "black-forest-labs/FLUX.2-klein-base-4B"
KLEIN_REVISION = "a3b4f4849157f664bdbc776fd7453c2783562f4d"
INPAINT_MODEL = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
INPAINT_REVISION = "115134f363124c53c7d878647567d04daf26e41e"
TRAINER_FILE = "train_dreambooth_lora_flux2_klein.py"
TRAINER_URL = f"https://raw.githubusercontent.com/huggingface/diffusers/{DIFFUSERS_COMMIT}/examples/dreambooth/{TRAINER_FILE}"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def setup(args):
    """Install new dependencies in a venv; share the existing CUDA PyTorch."""
    environment = args.project_root / ".venvs/uav_synthesis"
    subprocess.run([sys.executable, "-m", "venv", "--system-site-packages", str(environment)], check=True)
    python = environment / "bin/python"
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade-strategy", "only-if-needed",
                    "torch==2.6.0", "torchvision==0.21.0", "diffusers==0.40.0",
                    "transformers==5.8.0", "peft==0.19.1", "accelerate==1.13.0",
                    "huggingface_hub>=1.23,<2", "safetensors>=0.8", "datasets>=3.6,<5",
                    "pillow", "pyyaml", "sentencepiece", "ftfy", "tensorboard"], check=True)
    print(f"Ready: {python} {Path(__file__).resolve()} --stage all --gpu {args.gpu}")


def configuration(args):
    import yaml
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_root = (args.project_root / config["data"]["root"]).resolve()
    labels = [s.strip() for s in (args.project_root / config["data"]["labels"]).read_text().splitlines()
              if s.strip() and not s.startswith("#")]
    if args.classes:
        assert set(args.classes) <= set(labels), "--classes must belong to the configured ontology"
        labels = [label for label in labels if label in args.classes]
    return config, data_root, labels


def training_pairs(args):
    """Read only the frozen train CSV, using the live description/image_path schema."""
    from PIL import Image
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    config, data_root, labels = configuration(args)
    processor = read_json(args.project_root / config["model"]["path"] / "preprocessor_config.json")
    factor = processor["patch_size"] * processor["merge_size"]
    coco = read_json(args.project_root / config["data"]["bbox_annotations"])
    images = {image["id"]: image for image in coco["images"]}
    annotations = {a["cropped_file_name"]: a for a in coco["annotations"]}
    grouped = {label: [] for label in labels}
    for row in csv.DictReader((data_root / config["protocol"] / "train.csv").open(encoding="utf-8-sig")):
        label = row["source_class"]
        if label not in grouped:
            continue
        crop_rel = Path(row["evidence_path"]).relative_to("data")
        original_rel = Path(row["context_path"]).relative_to("data")
        description_path = data_root / "description" / crop_rel.with_suffix(".json")
        if not description_path.exists():
            continue
        description = read_json(description_path)
        assert description["image_path"] == crop_rel.as_posix(), f"Caption/view mismatch: {description_path}"
        assert description["commercial_event"] == label, f"Caption/class mismatch: {description_path}"
        assert row["split"] == "train"
        annotation = annotations[crop_rel.as_posix()]
        source = images[annotation["image_id"]]
        assert source["file_name"] == original_rel.as_posix()
        width, height = source["width"], source["height"]
        x, y, box_w, box_h = annotation["bbox"]
        box = [x / width, y / height, (x + box_w) / width, (y + box_h) / height]
        assert 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1, row["record_uid"]
        grouped[label].append({
            "record_uid": row["record_uid"], "event_class": label,
            "content_group_id": row["content_group_id"], "session_id": row["session_id"],
            "crop_path": str(data_root / crop_rel), "original_path": str(data_root / original_rel),
            "crop_relative": crop_rel.as_posix(), "original_relative": original_rel.as_posix(),
            "period_directory": crop_rel.parts[0], "description_path": str(description_path),
            "description": description["description"].strip(), "original_size": [width, height],
            "bbox_normalized": box,
        })
    rng = random.Random(args.seed)
    for values in grouped.values():
        rng.shuffle(values)
    # Round-robin without replacement gives 27/28 images per class for 500/18.
    selected = []
    for group in itertools.zip_longest(*grouped.values()):
        selected.extend(pair for pair in group if pair is not None)
        if len(selected) >= args.samples:
            break
    selected = selected[:args.samples]
    assert len(selected) == args.samples, f"Only {len(selected)} captioned training pairs available"
    assert set(p["event_class"] for p in selected) == set(labels), "Sample budget must cover every requested class"
    for pair in selected:
        with Image.open(pair["crop_path"]) as image:
            crop_w, crop_h = image.size
        def size(w, h):
            rh, rw = smart_resize(h, w, factor=factor, min_pixels=config["input"]["min_pixels"],
                                 max_pixels=config["input"]["max_pixels"])
            return [rw, rh]
        pair["target_original_size"] = size(*pair["original_size"])
        pair["target_crop_size"] = size(crop_w, crop_h)
        pair["bbox_xyxy"] = paste_box(pair["bbox_normalized"], pair["target_original_size"])
        pair["inpaint_prompt"] = "UAV aerial photograph. " + pair["description"]
    return selected, {label: len(values) for label, values in grouped.items()}


def paste_box(normalized, size):
    width, height = size
    return [math.floor(normalized[0] * width), math.floor(normalized[1] * height),
            math.ceil(normalized[2] * width), math.ceil(normalized[3] * height)]


def canvas_and_mask(crop, size, box):
    from PIL import Image, ImageDraw
    x1, y1, x2, y2 = box
    patch = crop.convert("RGB").resize((x2 - x1, y2 - y1), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", tuple(size), "white")
    canvas.paste(patch, (x1, y1))
    mask = Image.new("L", tuple(size), 255)
    ImageDraw.Draw(mask).rectangle((x1, y1, x2 - 1, y2 - 1), fill=0)
    return canvas, mask, patch


def inspect_data(args):
    pairs, available = training_pairs(args)
    report = {"available_captioned_train_pairs": available,
              "selected": dict(Counter(p["event_class"] for p in pairs)),
              "original_training_sizes": dict(Counter(str(p["target_original_size"]) for p in pairs)),
              "description_word_range": [min(len(p["description"].split()) for p in pairs),
                                         max(len(p["description"].split()) for p in pairs)],
              "examples": pairs[:2]}
    print(json.dumps(report, ensure_ascii=False, indent=2))


def download(args):
    from huggingface_hub import snapshot_download
    args.hf_cache.mkdir(parents=True, exist_ok=True)
    snapshot_download(KLEIN_MODEL, revision=KLEIN_REVISION, local_dir=args.crop_model_dir,
                      allow_patterns=["model_index.json", "scheduler/*", "text_encoder/*", "tokenizer/*",
                                      "transformer/*", "vae/*", "LICENSE*", "README.md"])
    snapshot_download(INPAINT_MODEL, revision=INPAINT_REVISION, local_dir=args.inpaint_model_dir,
                      allow_patterns=["model_index.json", "scheduler/*", "tokenizer/*", "tokenizer_2/*",
                                      "*/config.json", "*/*.fp16.safetensors", "LICENSE*", "README.md"])
    upstream = args.hf_cache / "upstream/diffusers-0.40.0"
    upstream.mkdir(parents=True, exist_ok=True)
    source = urllib.request.urlopen(TRAINER_URL).read()
    (upstream / TRAINER_FILE).write_bytes(source)
    (upstream / "LICENSE").write_bytes(urllib.request.urlopen(
        f"https://raw.githubusercontent.com/huggingface/diffusers/{DIFFUSERS_COMMIT}/LICENSE").read())
    write_json(args.work_dir / "upstream.json", {
        "crop_model": [KLEIN_MODEL, KLEIN_REVISION], "inpaint_model": [INPAINT_MODEL, INPAINT_REVISION],
        "trainer_url": TRAINER_URL, "trainer_sha256": hashlib.sha256(source).hexdigest(),
        "inpaint_reference": "https://github.com/nikgli/train-lora-sdxl-inpaint/tree/20e17660a9c83efa3bf668156c87dcc21cf70e7b",
    })


def prepare(args):
    from PIL import Image, ImageOps
    manifest = args.work_dir / "train_pairs.json"
    if manifest.exists():
        return read_json(manifest)
    pairs, available = training_pairs(args)
    folders = {name: args.work_dir / "data" / name for name in ("crops", "originals", "conditions", "masks")}
    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)
    with (folders["crops"] / "metadata.jsonl").open("w", encoding="utf-8") as metadata:
        for pair in pairs:
            filename = pair["record_uid"] + ".png"
            crop = ImageOps.exif_transpose(Image.open(pair["crop_path"])).convert("RGB")
            original = ImageOps.exif_transpose(Image.open(pair["original_path"])).convert("RGB")
            assert list(original.size) == pair["original_size"], pair["original_path"]
            crop.resize(tuple(pair["target_crop_size"]), Image.Resampling.LANCZOS).save(folders["crops"] / filename)
            original.resize(tuple(pair["target_original_size"]), Image.Resampling.LANCZOS).save(folders["originals"] / filename)
            canvas, mask, _ = canvas_and_mask(crop, pair["target_original_size"], pair["bbox_xyxy"])
            canvas.save(folders["conditions"] / filename)
            mask.save(folders["masks"] / filename)
            for key, folder in [("crop", "crops"), ("original", "originals"), ("condition", "conditions"), ("mask", "masks")]:
                pair[f"prepared_{key}_path"] = str(folders[folder] / filename)
            metadata.write(json.dumps({"file_name": filename, "text": pair["description"]}, ensure_ascii=False) + "\n")
    write_json(manifest, pairs)
    write_json(args.work_dir / "data_summary.json", {"available": available,
               "selected": dict(Counter(p["event_class"] for p in pairs)),
               "config": str(args.config), "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest()})
    return pairs


def train_crop(args):
    pairs = read_json(args.work_dir / "train_pairs.json")
    buckets = sorted({(p["target_crop_size"][1], p["target_crop_size"][0]) for p in pairs})
    command = [sys.executable, str(args.hf_cache / "upstream/diffusers-0.40.0" / TRAINER_FILE),
        "--pretrained_model_name_or_path", str(args.crop_model_dir),
        "--dataset_name", str(args.work_dir / "data/crops"), "--image_column", "image", "--caption_column", "text",
        "--instance_prompt", "UAV aerial photograph", "--output_dir", str(args.work_dir / "crop_lora"),
        "--rank", str(args.rank), "--lora_alpha", str(args.rank), "--mixed_precision", "bf16",
        "--gradient_checkpointing", "--cache_latents", "--offload", "--train_batch_size", "1",
        "--gradient_accumulation_steps", str(args.gradient_accumulation), "--max_sequence_length", "512",
        "--text_encoder_out_layers", "9", "18", "27", "--use_aspect_ratio_buckets",
        "--aspect_ratio_buckets", ";".join(f"{h},{w}" for h, w in buckets),
        "--learning_rate", str(args.learning_rate), "--lr_scheduler", "constant", "--lr_warmup_steps", "0",
        "--max_train_steps", str(args.crop_steps), "--checkpointing_steps", str(args.checkpoint_every),
        "--checkpoints_total_limit", "3", "--skip_final_inference", "--report_to", "tensorboard", "--seed", str(args.seed)]
    if args.resume:
        command += ["--resume_from_checkpoint", "latest"]
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def resolve_caption_model(path):
    return path if (path / "config.json").exists() else path / "snapshots" / (path / "refs/main").read_text().strip()


def prompt_records(args):
    _, _, labels = configuration(args)
    for label in labels:
        with (args.work_dir / "prompts" / f"{label}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                yield json.loads(line)


def prompts(args):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    config, data_root, labels = configuration(args)
    definitions = read_json(data_root / "definition.json")
    groups = defaultdict(list)
    for pair in read_json(args.work_dir / "train_pairs.json"):
        groups[pair["event_class"]].append(pair)
    path = resolve_caption_model(args.caption_model)
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="sdpa", local_files_only=True).to("cuda").eval()
    for class_index, label in enumerate(labels):
        destination = args.work_dir / "prompts" / f"{label}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        existing = [json.loads(line) for line in destination.read_text().splitlines()] if destination.exists() else []
        seen = {row["description"].casefold() for row in existing}
        count = len(existing)
        source_texts = {p["description"].casefold() for p in groups[label]}
        maximum = min(args.prompt_words, min(len(p["description"].split()) for p in groups[label]) - 1)
        assert maximum >= 12, f"Source descriptions for {label} are too short"
        minimum = min(24, maximum - 4)
        attempts = 0
        with destination.open("a", encoding="utf-8") as stream:
            while count < args.per_class:
                attempts += 1
                if attempts > math.ceil(args.per_class / args.prompts_per_call) * 4:
                    raise RuntimeError(f"Too few valid new descriptions for {label}; saved {count}, rerun prompts to continue")
                seed = args.seed + class_index * 100000 + count + attempts
                rng = random.Random(seed)
                examples = rng.sample(groups[label], min(3, len(groups[label])))
                request = min(args.prompts_per_call, args.per_class - count)
                instruction = (
                    f"Create {request} distinct new English image descriptions for synthetic UAV crop images.\n"
                    f"Class: {label}. Definition: {definitions[label]}\n"
                    f"Each description must contain the exact phrase '{label.replace('_', ' ')}', "
                    f"have {minimum} to {maximum} words, and describe visible content and spatial layout. "
                    "Vary plausible terrain, object placement, scale, textures, and lighting. "
                    "Create new scenes, not summaries or copies of the examples. Avoid text overlays, "
                    "watermarks, people identification, extra event classes, and explanations. "
                    "Return ONLY a JSON array of strings.\nTraining-only examples:\n" +
                    "\n".join(p["description"] for p in examples)
                )
                inputs = processor.apply_chat_template(
                    [{"role": "user", "content": [{"type": "text", "text": instruction}]}],
                    tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
                inputs = {key: value.to("cuda") for key, value in inputs.items()}
                torch.manual_seed(seed)
                with torch.inference_mode():
                    output = model.generate(**inputs, do_sample=True, temperature=0.9, top_p=0.95,
                                            max_new_tokens=1536, repetition_penalty=1.05)
                text = processor.tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
                candidates = json.loads(text[text.index("["):text.rindex("]") + 1])
                for description in candidates:
                    description = " ".join(description.split())
                    key = description.casefold()
                    if not minimum <= len(description.split()) <= maximum or key in seen or key in source_texts:
                        continue
                    if label.replace("_", " ") not in key:
                        continue
                    template = random.Random(args.seed + class_index * 100000 + count).choice(groups[label])
                    stem = f"syn_{label}_{count + 1:06d}"
                    base = Path(template["period_directory"]) / label
                    record = {"synthetic_id": stem, "event_class": label, "description": description,
                              "template_uid": template["record_uid"], "seed": args.seed + class_index * 100000 + count,
                              "crop_relative": (base / "cropped" / (stem + ".png")).as_posix(),
                              "original_relative": (base / "original" / (stem + ".png")).as_posix(),
                              "target_crop_size": template["target_crop_size"],
                              "target_original_size": template["target_original_size"],
                              "bbox_xyxy": template["bbox_xyxy"], "is_synthetic": True}
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                    stream.flush()
                    seen.add(key)
                    count += 1
                    if count == args.per_class:
                        break
                print(f"Descriptions {label}: {count}/{args.per_class}", flush=True)


def crops(args):
    import torch
    from diffusers import Flux2KleinPipeline
    pipe = Flux2KleinPipeline.from_pretrained(args.crop_model_dir, torch_dtype=torch.bfloat16,
                                            local_files_only=True).to("cuda")
    pipe.load_lora_weights(args.work_dir / "crop_lora", weight_name="pytorch_lora_weights.safetensors")
    pipe.vae.enable_slicing()
    records = itertools.islice(prompt_records(args), args.max_images)
    for record in records:
        output = args.synthetic_root / record["crop_relative"]
        if output.exists():
            continue
        width, height = record["target_crop_size"]
        image = pipe(prompt=record["description"], width=width, height=height,
                     num_inference_steps=args.crop_inference_steps, guidance_scale=4.0,
                     text_encoder_out_layers=(9, 18, 27), max_sequence_length=512,
                     generator=torch.Generator("cuda").manual_seed(record["seed"])).images[0]
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output)
        write_json(args.synthetic_root / "description" / Path(record["crop_relative"]).with_suffix(".json"),
                   {"image_path": record["crop_relative"], "commercial_event": record["event_class"],
                    "description": record["description"], "is_synthetic": True,
                    "generator": KLEIN_MODEL, "lora": str(args.work_dir / "crop_lora"),
                    "template_uid": record["template_uid"], "seed": record["seed"]})
        print(f"Crop: {output}", flush=True)


def outpaint(args):
    import torch
    from PIL import Image
    from diffusers import StableDiffusionXLInpaintPipeline
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        args.inpaint_model_dir, variant="fp16", torch_dtype=torch.float16,
        local_files_only=True, add_watermarker=False).to("cuda")
    pipe.load_lora_weights(args.work_dir / "inpaint_lora", weight_name="pytorch_lora_weights.safetensors")
    pipe.vae.enable_slicing()
    for record in itertools.islice(prompt_records(args), args.max_images):
        output = args.synthetic_root / record["original_relative"]
        metadata = args.synthetic_root / "metadata" / (record["synthetic_id"] + ".json")
        if output.exists() and metadata.exists():
            continue
        with Image.open(args.synthetic_root / record["crop_relative"]) as crop:
            canvas, mask, patch = canvas_and_mask(crop, record["target_original_size"], record["bbox_xyxy"])
        width, height = record["target_original_size"]
        result = pipe(prompt="UAV aerial photograph. " + record["description"],
                      negative_prompt="text, watermark, collage, frame, blank white background, duplicated main subject",
                      image=canvas, mask_image=mask, width=width, height=height, strength=0.99,
                      num_inference_steps=args.inpaint_inference_steps, guidance_scale=7.5,
                      generator=torch.Generator("cuda").manual_seed(record["seed"] + 1)).images[0]
        # SDXL's VAE is lossy. Paste the exact scaled crop back into the protected box.
        x1, y1, x2, y2 = record["bbox_xyxy"]
        result.paste(patch, (x1, y1))
        output.parent.mkdir(parents=True, exist_ok=True)
        result.save(output)
        annotation = dict(record, image_path=record["original_relative"],
                          bbox_xywh=[x1, y1, x2 - x1, y2 - y1],
                          bbox_1000=[1000*x1/width, 1000*y1/height, 1000*x2/width, 1000*y2/height],
                          crop_preserved_exactly=True, bbox_source="synthetic_crop_placement", inpaint_model=INPAINT_MODEL,
                          inpaint_lora=str(args.work_dir / "inpaint_lora"))
        write_json(metadata, annotation)
        if args.save_conditions:
            canvas.save(args.synthetic_root / "metadata" / (record["synthetic_id"] + "_canvas.png"))
            mask.save(args.synthetic_root / "metadata" / (record["synthetic_id"] + "_mask.png"))
        print(f"Original and bbox: {output}", flush=True)
    export_annotations(args)


def export_annotations(args):
    config, _, labels = configuration(args)
    source_categories = read_json(args.project_root / config["data"]["bbox_annotations"])["categories"]
    categories = [category for category in source_categories if category["name"] in labels]
    category_ids = {category["name"]: category["id"] for category in categories}
    images, annotations = [], []
    args.synthetic_root.mkdir(parents=True, exist_ok=True)
    with (args.synthetic_root / "synthetic_manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for record in prompt_records(args):
            metadata = args.synthetic_root / "metadata" / (record["synthetic_id"] + ".json")
            if not metadata.exists():
                continue
            record = read_json(metadata)
            index = len(images) + 1
            width, height = record["target_original_size"]
            images.append({"id": index, "file_name": record["original_relative"],
                           "width": width, "height": height, "is_synthetic": True})
            x, y, w, h = record["bbox_xywh"]
            annotations.append({"id": index, "image_id": index, "category_id": category_ids[record["event_class"]],
                                "bbox": [x, y, w, h], "area": w*h, "iscrowd": 0,
                                "cropped_file_name": record["crop_relative"], "is_synthetic": True})
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    write_json(args.synthetic_root / "crop_bboxes_all_photos/crop_bboxes.json",
               {"images": images, "annotations": annotations, "categories": categories})
    print(f"Exported {len(images)} synthetic original/crop/bbox records", flush=True)


def train_inpaint(args):
    """Fine-tune SDXL's 9-channel inpainting UNet with a frozen VAE and CLIPs.

    Adapted from Apache-2.0 Hugging Face Diffusers SDXL LoRA training (v0.40.0)
    and nikgli/train-lora-sdxl-inpaint at commit
    20e17660a9c83efa3bf668156c87dcc21cf70e7b. Original mask/latent construction:
    https://github.com/nikgli/train-lora-sdxl-inpaint/blob/20e17660a9c83efa3bf668156c87dcc21cf70e7b/diffusers/examples/research_projects/dreambooth_inpaint/train_dreambooth_inpaint_lora_sdxl.py
    https://github.com/huggingface/diffusers/blob/v0.40.0/examples/text_to_image/train_text_to_image_lora_sdxl.py
    Copyright 2024-2026 The HuggingFace Inc. team. Apache License 2.0:
    https://www.apache.org/licenses/LICENSE-2.0

    Dataset adaptation: variable-size originals, fixed bbox-preserving masks,
    paired condition images, cached frozen features and a single-GPU loop.
    Mask 1 denotes the background to generate; mask 0 preserves the crop.
    """
    import gc
    import json
    import random
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn.functional as F
    from diffusers import DDPMScheduler, StableDiffusionXLInpaintPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
    from PIL import Image
    from tqdm.auto import tqdm

    work_dir = Path(args.work_dir)
    pairs = json.loads((work_dir / "train_pairs.json").read_text(encoding="utf-8"))
    output_dir = work_dir / "inpaint_lora"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "training_state.pt"
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    sampler = random.Random(args.seed)
    device, dtype = torch.device("cuda"), torch.bfloat16

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        str(args.inpaint_model_dir), torch_dtype=dtype, variant="fp16",
        use_safetensors=True, local_files_only=True,
    )
    pipe.vae.requires_grad_(False).eval().to(device=device, dtype=torch.float32)
    pipe.text_encoder.requires_grad_(False).eval().to(device)
    pipe.text_encoder_2.requires_grad_(False).eval().to(device)
    pipe.vae.enable_slicing()
    cached = []

    def image_tensor(path):
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device) / 127.5 - 1.0

    # Cache on CPU so the training GPU holds only the UNet and its adapters.
    # VAE uses float32: SDXL's standard VAE can overflow in float16.
    with torch.no_grad():
        for pair in tqdm(pairs, desc="Cache outpainting latents and captions"):
            target = image_tensor(pair["prepared_original_path"])
            condition = image_tensor(pair["prepared_condition_path"])
            with Image.open(pair["prepared_mask_path"]) as image:
                mask = torch.from_numpy(np.asarray(image.convert("L"), dtype=np.float32))
            mask = (mask.to(device)[None, None] >= 127.5).float()
            latent = pipe.vae.encode(target).latent_dist.sample() * pipe.vae.config.scaling_factor
            known_latent = pipe.vae.encode(condition * (1 - mask)).latent_dist.sample()
            known_latent = known_latent * pipe.vae.config.scaling_factor
            latent_mask = F.interpolate(mask, size=latent.shape[-2:], mode="nearest")
            prompt, _, pooled_prompt, _ = pipe.encode_prompt(
                prompt=pair["inpaint_prompt"], device=device,
                num_images_per_prompt=1, do_classifier_free_guidance=False,
            )
            width, height = pair["target_original_size"]
            time_ids = torch.tensor([[height, width, 0, 0, height, width]], dtype=dtype)
            cached.append({
                "latent": latent.to(device="cpu", dtype=dtype),
                "known_latent": known_latent.to(device="cpu", dtype=dtype),
                "mask": latent_mask.to(device="cpu", dtype=dtype),
                "prompt": prompt.to(device="cpu", dtype=dtype),
                "pooled_prompt": pooled_prompt.to(device="cpu", dtype=dtype),
                "time_ids": time_ids,
            })

    unet = pipe.unet
    scheduler = DDPMScheduler.from_pretrained(
        str(args.inpaint_model_dir), subfolder="scheduler", local_files_only=True,
    )
    del pipe, target, condition, mask, latent, known_latent, latent_mask, prompt, pooled_prompt
    gc.collect()
    torch.cuda.empty_cache()
    unet.requires_grad_(False).train().to(device=device, dtype=dtype)
    unet.add_adapter(LoraConfig(
        r=args.rank, lora_alpha=args.rank, init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    ))
    parameters = [parameter for parameter in unet.parameters() if parameter.requires_grad]
    for parameter in parameters:
        parameter.data = parameter.data.float()
    unet.enable_gradient_checkpointing()
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-2)
    start_step = 0
    if args.resume and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        set_peft_model_state_dict(unet, state["adapter"])
        optimizer.load_state_dict(state["optimizer"])
        start_step = state["step"]
        sampler.setstate(state["sampler_state"])
        torch.set_rng_state(state["torch_rng_state"])
        torch.cuda.set_rng_state(state["cuda_rng_state"])
        del state

    def save_checkpoint(step):
        adapter = {key: value.detach().cpu() for key, value in get_peft_model_state_dict(unet).items()}
        StableDiffusionXLInpaintPipeline.save_lora_weights(
            output_dir, unet_lora_layers=convert_state_dict_to_diffusers(adapter),
            safe_serialization=True,
        )
        state = {
            "step": step, "adapter": adapter, "optimizer": optimizer.state_dict(),
            "sampler_state": sampler.getstate(), "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
        }
        temporary = checkpoint_path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(checkpoint_path)

    progress = tqdm(range(start_step, args.inpaint_steps), desc="Train outpainting LoRA", initial=start_step,
                    total=args.inpaint_steps)
    for step in progress:
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(args.gradient_accumulation):
            sample = cached[sampler.randrange(len(cached))]
            latent = sample["latent"].to(device)
            noise = torch.randn_like(latent)
            timestep = torch.randint(0, scheduler.config.num_train_timesteps, (1,), device=device).long()
            noisy_latent = scheduler.add_noise(latent, noise, timestep)
            model_input = torch.cat([
                noisy_latent, sample["mask"].to(device), sample["known_latent"].to(device),
            ], dim=1)
            with torch.autocast("cuda", dtype=dtype):
                prediction = unet(
                    model_input, timestep, encoder_hidden_states=sample["prompt"].to(device),
                    added_cond_kwargs={
                        "text_embeds": sample["pooled_prompt"].to(device),
                        "time_ids": sample["time_ids"].to(device),
                    }, return_dict=False,
                )[0]
            if scheduler.config.prediction_type == "epsilon":
                target = noise
            elif scheduler.config.prediction_type == "v_prediction":
                target = scheduler.get_velocity(latent, noise, timestep)
            else:
                raise ValueError(f"Unsupported prediction type: {scheduler.config.prediction_type}")
            loss = F.mse_loss(prediction.float(), target.float(), reduction="mean")
            (loss / args.gradient_accumulation).backward()
            accumulated_loss += loss.detach().item() / args.gradient_accumulation
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        progress.set_postfix(loss=f"{accumulated_loss:.5f}")
        if (step + 1) % args.checkpoint_every == 0:
            save_checkpoint(step + 1)
    save_checkpoint(max(start_step, args.inpaint_steps))
    return output_dir



def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", default="all", choices=["setup", "inspect", "download", "prepare", "train",
                        "train-crop", "train-inpaint", "prompts", "crops", "outpaint", "export", "all"])
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--config", type=Path, default=Path("configs/yaml/perception_qwen.yaml"))
    parser.add_argument("--work-dir", type=Path, default=Path("outputs/uav_synthesis/seed43"))
    parser.add_argument("--synthetic-root", type=Path, default=Path("um7_synthetic"))
    parser.add_argument("--hf-cache", type=Path, default=Path("hf_cache"))
    parser.add_argument("--caption-model", type=Path, default=Path("hf_cache/models--Qwen--Qwen3-VL-4B-Instruct"))
    parser.add_argument("--gpu", default="1", help="One physical GPU ID; models run sequentially")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--per-class", type=int, default=3000)
    parser.add_argument("--classes", nargs="+")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--crop-steps", type=int, default=1500)
    parser.add_argument("--inpaint-steps", type=int, default=1500)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--prompt-words", type=int, default=40)
    parser.add_argument("--prompts-per-call", type=int, default=8)
    parser.add_argument("--crop-inference-steps", type=int, default=50)
    parser.add_argument("--inpaint-inference-steps", type=int, default=30)
    parser.add_argument("--max-images", type=int, default=None, help="Optional small generation run; default all")
    parser.add_argument("--save-conditions", action="store_true", help="Also save white canvases and masks")
    args = parser.parse_args()
    args.project_root = args.project_root.resolve()
    for key in ["config", "work_dir", "synthetic_root", "hf_cache", "caption_model"]:
        setattr(args, key, (args.project_root / getattr(args, key)).resolve())
    args.crop_model_dir = args.hf_cache / "flux2-klein-base-4b"
    args.inpaint_model_dir = args.hf_cache / "sdxl-inpainting-0.1"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["HF_HOME"] = str(args.hf_cache / "huggingface")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    stages = {"setup": setup, "inspect": inspect_data, "download": download, "prepare": prepare,
              "train-crop": train_crop, "train-inpaint": train_inpaint, "prompts": prompts,
              "crops": crops, "outpaint": outpaint, "export": export_annotations}
    if args.stage in ["all", "train"]:
        sequence = ["prepare", "train-crop", "train-inpaint"]
        if args.stage == "all":
            sequence = ["download"] + sequence + ["prompts", "crops", "outpaint"]
        # Separate processes release each model's GPU allocations before the next.
        for stage in sequence:
            command = [sys.executable, str(Path(__file__).resolve())] + sys.argv[1:]
            command += ["--stage", stage]
            subprocess.run(command, check=True)
    else:
        stages[args.stage](args)


if __name__ == "__main__":
    main()
