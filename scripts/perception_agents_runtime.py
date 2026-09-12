"""Three sequential roles sharing one Perception + Spatial model on one device."""
from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image, ImageDraw
import torch
from tqdm import tqdm

from train_perception_qwen import read_config, output_path
from train_perception_spatial import build_model as build_spatial
from clear_uav.modeling import assistant_only_labels
from clear_uav.table4 import category_block, event_probability
from clear_uav.generation_constraints import label_prefix_allowed_tokens
from perception_agents_core import VIS, VERDICT_CODES, make_messages, run_agents

SCHEMA_VERSION = 1


def build_model(config, checkpoint=None, is_trainable=False):
    source = Path(str(checkpoint or config['initial_checkpoint']).format(
        protocol=config['protocol'], seed=config['seed']))
    # All three roles start from the user's complete trained Spatial checkpoint.
    if not (source/'spatial_head.pt').is_file():
        raise FileNotFoundError(f'Spatial checkpoint required: {source}')
    if checkpoint is not None:
        schema = json.loads((source/'agent_schema.json').read_text(encoding='utf-8'))
        if schema['schema_version'] != SCHEMA_VERSION:
            raise ValueError('Unsupported agent checkpoint schema')
    return build_spatial(config, checkpoint=source, is_trainable=is_trainable)


def marked_image(image_path, candidate, width):
    """One full-frame image; the proposed rectangle is drawn in memory only."""
    with Image.open(image_path) as source:
        image = source.convert('RGB')
    box = candidate.get('bbox_1000') if candidate else None
    if box is not None:
        w, h = image.size
        coordinates = [box[0]*w/1000, box[1]*h/1000, box[2]*w/1000, box[3]*h/1000]
        ImageDraw.Draw(image).rectangle(coordinates, outline=(255, 0, 255), width=width)
    return image


def encode_role(processor, config, image_path, role, *, candidate=None, feedback=None,
                answer=None, supervised=False):
    image = marked_image(image_path, candidate, config['agents']['marker_width']) if role == 'verify' else str(image_path)
    messages = make_messages(image, role, category_block(config), candidate=candidate,
                             feedback=feedback, answer=answer)
    try:
        encoded = processor.apply_chat_template(
            [messages], tokenize=True, add_generation_prompt=answer is None,
            return_dict=True, return_tensors='pt',
            processor_kwargs={'padding': True, 'size':{
                'longest_edge': config['input']['max_pixels'],
                'shortest_edge': config['input']['min_pixels']}})
    finally:
        if isinstance(image, Image.Image):
            image.close()
    if supervised:
        encoded['labels'] = assistant_only_labels(encoded['input_ids'], encoded['attention_mask'], processor.tokenizer)
    elif answer is not None:
        # The known category is supplied by What, never by test annotations.
        encoded['logits_to_keep'] = 1
    return dict(encoded)


class AgentRuntime:
    def __init__(self, model, processor, config, labels):
        self.model, self.processor, self.config = model, processor, config
        self.labels = list(labels)
        self.device = torch.device(config['device'])
        tokenizer = processor.tokenizer
        self.code_tokens = {v: tokenizer.encode(v, add_special_tokens=False) for v in VERDICT_CODES.values()}
        if any(len(t) != 1 for t in self.code_tokens.values()) or len({t[0] for t in self.code_tokens.values()}) != 4:
            raise ValueError('Verifier A/B/C/D must be distinct single tokens for score calculation')
        self.calls = {}

    def _generate(self, image_path, role, answers, candidate=None, feedback=None):
        inputs = encode_role(self.processor, self.config, image_path, role,
                             candidate=candidate, feedback=feedback)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        length = inputs['input_ids'].shape[1]
        tokenizer = self.processor.tokenizer
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type=='cuda'):
            generated = self.model.vlm.generate(
                **inputs, do_sample=False, use_cache=True,
                max_new_tokens=max(len(tokenizer.encode(a, add_special_tokens=False)) for a in answers)+1,
                prefix_allowed_tokens_fn=label_prefix_allowed_tokens(tokenizer, answers, prompt_length=length),
                return_dict_in_generate=True, output_scores=True,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        tokens = generated.sequences[0, length:]
        text = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        raw = tokenizer.decode(tokens, skip_special_tokens=False)
        scores = generated.scores[0][0].float().clone()
        # Release the KV cache before the next role executes.
        del generated
        self.calls[role] += 1
        return text, raw, scores

    def what(self, image_path, feedback=None):
        text, raw, scores = self._generate(image_path, 'what', [x+VIS for x in self.labels]+['no_event'], feedback=feedback)
        # Tokenizers may or may not strip the added <vis> special token.
        category = text.replace(VIS, '').strip()
        return {'category':category if category in self.labels else None,
                'score':float(event_probability(scores, self.processor.tokenizer, self.labels)), 'raw':raw}

    def where(self, image_path, category, feedback=None):
        inputs = encode_role(self.processor, self.config, image_path, 'where',
                             candidate={'category':category, 'bbox_1000':None}, feedback=feedback,
                             answer=category+VIS)
        inputs = {k:v.to(self.device) for k,v in inputs.items()}
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type=='cuda'):
            _, boxes = self.model(inputs)
        self.calls['where'] += 1
        return (boxes[0].float()*1000).cpu().tolist()

    def verify(self, image_path, candidate):
        code, raw, scores = self._generate(image_path, 'verify', list(VERDICT_CODES.values()), candidate=candidate)
        probabilities = scores[[self.code_tokens[c][0] for c in VERDICT_CODES.values()]].softmax(0)
        verdict = {v:k for k,v in VERDICT_CODES.items()}[code]
        return {'verdict':verdict, 'score':float(probabilities[0]), 'raw':raw}

    @torch.inference_mode()
    def predict_one(self, image_path, mode='full'):
        # This boundary intentionally takes only an image path, never a sample's GT.
        self.model.eval()
        self.calls = {'what':0, 'where':0, 'verify':0}
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        result = run_agents(lambda feedback:self.what(image_path,feedback),
                            lambda category,feedback:self.where(image_path,category,feedback),
                            lambda candidate:self.verify(image_path,candidate),
                            max_revisions=self.config['agents']['max_revisions'],mode=mode)
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        result.update(latency_ms=(time.perf_counter()-started)*1000, num_calls=sum(self.calls.values()),
                      agent_calls=dict(self.calls), inference_batch_size=1,
                      timing_scope='serial_what_where_verify_and_feedback_wall_time',
                      raw_output=json.dumps(result['trace'],ensure_ascii=False))
        trace=result['trace']
        last=trace['rounds'][-1] if trace['rounds'] else None
        result['revisions']=trace['revisions']
        result['what_presence_score']=(last['candidate'] if last else trace['initial_hypothesis'])['what_score']
        result['verify_score']=last['verification']['score'] if last else None
        return result

    def predict(self, samples, mode='full'):
        return [self.predict_one(sample.image_path,mode) for sample in tqdm(samples,desc=f'agents {mode}')]
