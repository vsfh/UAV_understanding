"""Batch the pending What, Where and Verify calls of independent image agents.

One process owns one model/device. Distributed evaluation shards the images
outside this module; every GPU then executes real role batches independently.
The controller and its fake-callback tests require only the Python standard
library. Image annotations never enter the controller or model callbacks.
"""
from __future__ import annotations

import copy
import json
import time

from perception_agents_core import (
    VIS, VERDICT_CODES, _VERDICTS, _event, _score, make_messages, valid_bbox,
)


def _agent_steps(max_revisions, mode):
    """Coroutine counterpart of core.run_agents; yielded values are model requests."""
    trace = {"calls": [], "rounds": [], "revisions": 0, "abstained": False}

    def localize(category, feedback):
        box = yield "where", {"category": category, "feedback": copy.deepcopy(feedback)}
        trace["calls"].append({"role": "where", "category": category, "bbox_1000": copy.deepcopy(box)})
        return copy.deepcopy(box)

    def propose(feedback):
        result = yield "what", {"feedback": copy.deepcopy(feedback)}
        trace["calls"].append({"role": "what", "result": copy.deepcopy(result)})
        category = result.get("category")
        category = category if _event(category) else None
        candidate = {"category": category, "bbox_1000": None, "what_score": _score(result["score"])}
        if category is not None:
            candidate["bbox_1000"] = yield from localize(category, feedback)
        return candidate

    def finish(candidate=None, score=0.0, verified=False, abstained=False):
        trace["abstained"] = abstained
        return {
            "category": candidate["category"] if candidate else None,
            "bbox_1000": copy.deepcopy(candidate["bbox_1000"]) if candidate else None,
            "presence_score": score, "valid": True, "verified": verified,
            "abstained": abstained, "trace": trace,
        }

    candidate = yield from propose(None)
    trace["initial_hypothesis"] = copy.deepcopy(candidate)
    trace["initial_category"] = candidate["category"]
    trace["initial_bbox_1000"] = copy.deepcopy(candidate["bbox_1000"])
    if mode == "no_verify":
        if candidate["category"] is None:
            return finish(score=candidate["what_score"])
        if valid_bbox(candidate["bbox_1000"]):
            return finish(candidate, candidate["what_score"])
        return finish(abstained=True)

    for round_index in range(max_revisions + 1):
        result = yield "verify", {"candidate": copy.deepcopy(candidate)}
        trace["calls"].append({"role": "verify", "result": copy.deepcopy(result)})
        verdict = _VERDICTS.get(result.get("verdict"), result.get("verdict"))
        trace["rounds"].append({"round": round_index, "candidate": copy.deepcopy(candidate), "verification": copy.deepcopy(result)})
        if verdict == "no_event":
            return finish(verified=True)
        if verdict == "accept":
            if _event(candidate["category"]) and valid_bbox(candidate["bbox_1000"]):
                return finish(candidate, candidate["what_score"] * _score(result["score"]), verified=True)
            return finish(abstained=True)
        if verdict not in ("relocalize", "reclassify") or mode == "verify_only" or round_index == max_revisions:
            return finish(abstained=True)
        feedback = {"candidate": copy.deepcopy(candidate), "verdict": verdict}
        if verdict == "reclassify":
            candidate = yield from propose(feedback)
        elif _event(candidate["category"]):
            candidate["bbox_1000"] = yield from localize(candidate["category"], feedback)
        else:
            return finish(abstained=True)
        trace["revisions"] += 1
    raise AssertionError("unreachable controller state")


def run_agents_batched(image_paths, what_batch, where_batch, verify_batch, *, max_revisions=2, mode="full"):
    """Resolve pending role requests in batches, preserving input and trace order.

    Each callback receives a list of dictionaries with ``image_path`` plus its
    role arguments, and must return one result per request in the same order.
    Only paths and model hypotheses are passed to callbacks, never GT records.
    """
    if mode not in ("full", "no_verify", "verify_only") or max_revisions < 0:
        raise ValueError("invalid controller mode or revision budget")
    image_paths = list(image_paths)
    agents = [_agent_steps(max_revisions, mode) for _ in image_paths]
    pending = {index: next(agent) for index, agent in enumerate(agents)}
    results = [None] * len(agents)
    callbacks = {"what": what_batch, "where": where_batch, "verify": verify_batch}
    while pending:
        for role, callback in callbacks.items():
            indices = [index for index, request in pending.items() if request[0] == role]
            if not indices:
                continue
            requests = [dict(image_path=image_paths[index], **copy.deepcopy(pending[index][1])) for index in indices]
            replies = callback(requests)
            if len(replies) != len(indices):
                raise ValueError(f"{role} returned {len(replies)} results for {len(indices)} requests")
            for index, reply in zip(indices, replies):
                try:
                    pending[index] = agents[index].send(reply)
                except StopIteration as finished:
                    results[index] = finished.value
                    del pending[index]
    return results


def encode_batch(processor, config, requests, role, category_text):
    """Encode one image per row; Where's supplied answer comes only from What."""
    from PIL import Image, ImageDraw

    processor.tokenizer.padding_side = "left"
    conversations, opened_images = [], []
    try:
        for request in requests:
            candidate = request.get("candidate")
            image = str(request["image_path"])
            if role == "verify":
                with Image.open(image) as source:
                    image = source.convert("RGB")
                opened_images.append(image)
                box = candidate.get("bbox_1000") if candidate else None
                if box is not None:
                    width, height = image.size
                    coordinates = [box[0] * width / 1000, box[1] * height / 1000,
                                   box[2] * width / 1000, box[3] * height / 1000]
                    ImageDraw.Draw(image).rectangle(coordinates, outline=(255, 0, 255),
                                                   width=config["agents"]["marker_width"])
            answer = None
            if role == "where":
                candidate = {"category": request["category"], "bbox_1000": None}
                answer = request["category"] + VIS
            conversations.append(make_messages(image, role, category_text, candidate=candidate,
                                                feedback=request.get("feedback"), answer=answer))
        encoded = processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=role != "where",
            return_dict=True, return_tensors="pt",
            processor_kwargs={"padding": True, "size": {
                "longest_edge": config["input"]["max_pixels"],
                "shortest_edge": config["input"]["min_pixels"],
            }})
    finally:
        for image in opened_images:
            image.close()
    if role == "where":
        encoded["logits_to_keep"] = 1
    return dict(encoded)


def _attach_metrics(result, latency_ms, batch_size):
    trace = result["trace"]
    calls = {role: sum(call["role"] == role for call in trace["calls"])
             for role in ("what", "where", "verify")}
    last = trace["rounds"][-1] if trace["rounds"] else None
    result.update(
        latency_ms=latency_ms, num_calls=sum(calls.values()), agent_calls=calls,
        inference_batch_size=batch_size,
        timing_scope="batched_what_where_verify_and_feedback_wall_time_amortized",
        raw_output=json.dumps(trace, ensure_ascii=False), revisions=trace["revisions"],
        what_presence_score=(last["candidate"] if last else trace["initial_hypothesis"])["what_score"],
        verify_score=last["verification"]["score"] if last else None,
    )
    return result


class BatchedAgentRuntime:
    def __init__(self, model, processor, config, labels):
        import torch
        from clear_uav.table4 import category_block

        self.model, self.processor, self.config = model, processor, config
        self.labels = list(labels)
        self.device = torch.device(config["device"])
        self.batch_size = int(config.get("test", {}).get("batch_size", 4))
        if self.batch_size < 1:
            raise ValueError("test.batch_size must be positive")
        self.category_text = category_block(config)
        tokenizer = processor.tokenizer
        tokenizer.padding_side = "left"
        self.code_tokens = {code: tokenizer.encode(code, add_special_tokens=False)
                            for code in VERDICT_CODES.values()}
        if any(len(tokens) != 1 for tokens in self.code_tokens.values()) or len({tokens[0] for tokens in self.code_tokens.values()}) != 4:
            raise ValueError("Verifier A/B/C/D must be distinct single tokens for score calculation")

    def _inputs(self, requests, role):
        encoded = encode_batch(self.processor, self.config, requests, role, self.category_text)
        return {name: value.to(self.device) if hasattr(value, "to") else value
                for name, value in encoded.items()}

    def _generate(self, requests, role, answers):
        import torch
        from clear_uav.generation_constraints import label_prefix_allowed_tokens

        inputs = self._inputs(requests, role)
        prompt_length = inputs["input_ids"].shape[1]
        tokenizer = self.processor.tokenizer
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            generated = self.model.vlm.generate(
                **inputs, do_sample=False, use_cache=True,
                max_new_tokens=max(len(tokenizer.encode(answer, add_special_tokens=False)) for answer in answers) + 1,
                prefix_allowed_tokens_fn=label_prefix_allowed_tokens(tokenizer, answers, prompt_length=prompt_length),
                return_dict_in_generate=True, output_scores=True,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        sequences = generated.sequences[:, prompt_length:].cpu().tolist()
        first_scores = generated.scores[0].float().clone()
        del generated
        results = []
        for index, tokens in enumerate(sequences):
            # Other rows may decode longer labels; remove only post-EOS padding.
            if tokenizer.eos_token_id in tokens:
                tokens = tokens[:tokens.index(tokenizer.eos_token_id) + 1]
            results.append((tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                            tokenizer.decode(tokens, skip_special_tokens=False), first_scores[index]))
        return results

    def what_batch(self, requests):
        from clear_uav.table4 import event_probability

        generated = self._generate(requests, "what", [label + VIS for label in self.labels] + ["no_event"])
        results = []
        for text, raw, scores in generated:
            category = text.replace(VIS, "").strip()
            results.append({"category": category if category in self.labels else None,
                            "score": float(event_probability(scores, self.processor.tokenizer, self.labels)), "raw": raw})
        return results

    def where_batch(self, requests):
        import torch

        inputs = self._inputs(requests, "where")
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            _, boxes = self.model(inputs)
        return (boxes.float() * 1000).cpu().tolist()

    def verify_batch(self, requests):
        generated = self._generate(requests, "verify", list(VERDICT_CODES.values()))
        reverse = {code: verdict for verdict, code in VERDICT_CODES.items()}
        results = []
        for code, raw, scores in generated:
            probabilities = scores[[self.code_tokens[value][0] for value in VERDICT_CODES.values()]].softmax(0)
            results.append({"verdict": reverse[code], "score": float(probabilities[0]), "raw": raw})
        return results

    def predict_batch(self, image_paths, mode="full"):
        import torch

        # Public inference boundary takes only paths, never annotated samples.
        image_paths = list(image_paths)
        if not image_paths:
            return []
        self.model.eval()
        with torch.inference_mode():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            results = run_agents_batched(
                image_paths, self.what_batch, self.where_batch, self.verify_batch,
                max_revisions=self.config["agents"]["max_revisions"], mode=mode)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            latency = (time.perf_counter() - started) * 1000 / len(image_paths)
        return [_attach_metrics(result, latency, len(image_paths)) for result in results]

    def predict_one(self, image_path, mode="full"):
        return self.predict_batch([image_path], mode)[0]

    def predict(self, samples, mode="full"):
        from tqdm import tqdm

        results = []
        for start in tqdm(range(0, len(samples), self.batch_size), desc=f"agents {mode}", unit="batch"):
            paths = [sample.image_path for sample in samples[start:start + self.batch_size]]
            results.extend(self.predict_batch(paths, mode))
        return results
