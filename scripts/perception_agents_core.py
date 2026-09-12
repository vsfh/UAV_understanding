"""Role prompts, supervised verifier targets and a bounded three-role controller.

Only ``oracle_verdict`` and ``make_candidates`` consume annotations. Inference
uses model callbacks and their hypotheses; a verifier's acceptance is not proof
of correctness against ground truth.
"""
import copy
import json
import math

VIS = "<vis>"
VERDICT_CODES = {"accept": "A", "relocalize": "B", "reclassify": "C", "no_event": "D"}
_VERDICTS = {code: name for name, code in VERDICT_CODES.items()}


def valid_bbox(box):
    """Whether xyxy is finite, ordered and inside the normalized 0..1000 frame."""
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in box):
        return False
    x1, y1, x2, y2 = box
    return 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000


def bbox_iou(a, b):
    if not valid_bbox(a) or not valid_bbox(b):
        return 0.0
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection)


def _event(category):
    return isinstance(category, str) and category.strip() not in ("", "no_event")


def oracle_verdict(presence, gt_label, gt_box, candidate_category, candidate_box, accept_iou=0.5):
    """Supervised target; never call this function in model inference."""
    if not presence:
        return "no_event"
    if not _event(candidate_category) or candidate_category != gt_label:
        return "reclassify"
    if not valid_bbox(candidate_box) or bbox_iou(gt_box, candidate_box) < accept_iou:
        return "relocalize"
    return "accept"


def _random_box(rng):
    width, height = rng.uniform(30, 700), rng.uniform(30, 700)
    x, y = rng.uniform(0, 1000 - width), rng.uniform(0, 1000 - height)
    return [x, y, x + width, y + height]


def _jitter(box, rng, translation, scale):
    width, height = box[2] - box[0], box[3] - box[1]
    cx = (box[0] + box[2]) / 2 + rng.uniform(-translation, translation) * width
    cy = (box[1] + box[3]) / 2 + rng.uniform(-translation, translation) * height
    width = min(1000, max(1, width * rng.uniform(*scale)))
    height = min(1000, max(1, height * rng.uniform(*scale)))
    x, y = min(1000 - width, max(0, cx - width / 2)), min(1000 - height, max(0, cy - height / 2))
    return [x, y, x + width, y + height]


def make_candidates(presence, label, bbox_1000, labels, rng):
    """Twelve seeded training hypotheses per image.

    Positive images supply four examples each of accept, relocalize and
    reclassify. Negative images supply no_event targets; balancing these four
    targets across images belongs to the training sampler. Returned dictionaries
    contain hypotheses only, not targets or annotation fields.
    """
    labels = list(labels)
    if not labels:
        raise ValueError("labels must contain event categories")
    if not presence:
        return [{"category": None, "bbox_1000": None}] + [
            {"category": rng.choice(labels), "bbox_1000": [0, 0, 1000, 1000] if i == 0 else _random_box(rng)}
            for i in range(11)
        ]
    if label not in labels or not valid_bbox(bbox_1000):
        raise ValueError("positive candidates require a known label and a valid GT box")
    pools = {name: [] for name in ("accept", "relocalize", "reclassify")}

    def add(category, box):
        verdict = oracle_verdict(True, label, bbox_1000, category, box)
        if len(pools[verdict]) < 4:
            pools[verdict].append({"category": category, "bbox_1000": copy.deepcopy(box)})

    add(label, list(bbox_1000))
    add(label, [0, 0, 1000, 1000])
    add(label, None)
    for _ in range(32):
        add(label, _jitter(bbox_1000, rng, 0.08, (0.9, 1.1)))
        add(label, _jitter(bbox_1000, rng, 2.5, (0.3, 2.5)))
        add(label, _random_box(rng))
    corners = ([0, 0, 1, 1], [999, 999, 1000, 1000])
    bad_box = min(corners, key=lambda box: bbox_iou(bbox_1000, box))
    while len(pools["accept"]) < 4:
        add(label, list(bbox_1000))
    while len(pools["relocalize"]) < 4:
        add(label, list(bad_box))
    wrong = [category for category in labels if category != label]
    add(None, None)
    for box in (bbox_1000, _random_box(rng), None):
        add(rng.choice(wrong) if wrong else None, box)
    candidates = [candidate for pool in pools.values() for candidate in pool]
    rng.shuffle(candidates)
    return candidates


def _hypothesis(candidate):
    candidate = candidate or {}
    return {"category": candidate.get("category"), "bbox_1000": candidate.get("bbox_1000")}


def make_messages(image, role, category_text, candidate=None, feedback=None, answer=None):
    """Build one-image role messages; only hypothesis/verdict enter feedback."""
    instructions = {
        "what": f"Inspect the entire image. Decide whether an event from the list is present and identify its category. Reply with exactly category_id{VIS}, or no_event if none is present.",
        "where": f"Locate the specified event using the entire image, including regions outside any previous box. Reply with exactly the supplied category_id followed by {VIS}; the continuous ROI head predicts the box. Recompute the location from visual evidence.",
        "verify": "Check whether the proposed event category and bounding box are supported by the entire image. A proposed box, when present, is also drawn on the image. Reply with exactly one letter: A = category and location are correct; B = category is correct but location needs recomputing; C = an event is present but the category is wrong or missing; D = no listed event is present. For an empty proposal, choose C if an event is visible and D otherwise. A requires an event category and a valid box.",
    }
    if role not in instructions:
        raise ValueError(f"unknown role: {role}")
    text = instructions[role] + "\nEvent categories:\n" + category_text
    if role in ("where", "verify"):
        text += "\nProposed hypothesis (xyxy coordinates in 0..1000): " + json.dumps(_hypothesis(candidate), ensure_ascii=False)
    if feedback:
        previous = feedback.get("candidate", feedback.get("previous_candidate"))
        clean_feedback = {"candidate": _hypothesis(previous), "verdict": feedback.get("verdict")}
        text += "\nPrevious model feedback (may be mistaken): " + json.dumps(clean_feedback, ensure_ascii=False)
    messages = [
        {"role": "system", "content": "You are a UAV event perception agent. Use the image as evidence. Previous proposals and feedback are fallible model outputs."},
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]},
    ]
    if answer is not None:
        messages.append({"role": "assistant", "content": answer})
    return messages


def _score(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("model confidence must be finite and in [0, 1]")
    return value


def run_agents(what, where, verify, max_revisions=2, mode="full"):
    """Run shared-model role callbacks without annotations or external state.

    A revision is one verifier-requested retry of What+Where or Where. The
    initial proposal and each revision receive at most one verification call.
    Unresolved/invalid hypotheses abstain; the prediction still has a valid
    absent-output schema and ``verified=False``.
    """
    if mode not in ("full", "no_verify", "verify_only") or max_revisions < 0:
        raise ValueError("invalid controller mode or revision budget")
    trace = {"calls": [], "rounds": [], "revisions": 0, "abstained": False}

    def propose(feedback):
        result = what(copy.deepcopy(feedback))
        trace["calls"].append({"role": "what", "result": copy.deepcopy(result)})
        category = result.get("category")
        category = category if _event(category) else None
        candidate = {"category": category, "bbox_1000": None, "what_score": _score(result["score"])}
        if category is not None:
            candidate["bbox_1000"] = localize(category, feedback)
        return candidate

    def localize(category, feedback):
        box = where(category, copy.deepcopy(feedback))
        trace["calls"].append({"role": "where", "category": category, "bbox_1000": copy.deepcopy(box)})
        return copy.deepcopy(box)

    def finish(candidate=None, score=0.0, verified=False, abstained=False):
        trace["abstained"] = abstained
        return {
            "category": candidate["category"] if candidate else None,
            "bbox_1000": copy.deepcopy(candidate["bbox_1000"]) if candidate else None,
            "presence_score": score,
            "valid": True,
            "verified": verified,
            "abstained": abstained,
            "trace": trace,
        }

    candidate = propose(None)
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
        result = verify(copy.deepcopy(candidate))
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
            candidate = propose(feedback)
        elif _event(candidate["category"]):
            candidate["bbox_1000"] = localize(candidate["category"], feedback)
        else:
            return finish(abstained=True)
        trace["revisions"] += 1
    raise AssertionError("unreachable controller state")
