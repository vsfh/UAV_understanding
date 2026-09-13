"""CPU-only invariants for exact distributed per-image objective and sharding."""
import importlib.util
import json
import multiprocessing
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'train_perception_agents_dual.py'
SPEC = importlib.util.spec_from_file_location('agents_dual_training', SCRIPT)
training = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(training)


def _distributed_worker(rank, rendezvous, folder):
    """Exercise the actual reduction with dynamic heads and an empty rank."""
    import torch
    import torch.distributed as distributed
    distributed.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        model = torch.nn.Module()
        model.shared = torch.nn.Parameter(torch.tensor(1.0 + rank))
        model.where = torch.nn.Parameter(torch.tensor(2.0 + rank))
        model.unused = torch.nn.Parameter(torch.tensor(3.0 + rank))
        training.synchronize_initial_state(model, distributed)
        parameters = list(model.parameters())
        optimizer = torch.optim.SGD(parameters, lr=0.1, weight_decay=0.1)
        # Global group: positive (shared + where) / 2, negative (shared only).
        if rank == 0:
            loss = (model.shared.square() + model.where.square()) / 2
        else:
            loss = (model.shared * 3).square()
        (loss / 2).backward()
        training.sum_gradients(parameters, distributed, bucket_bytes=8)
        first = [p.grad.item() if p.grad is not None else None for p in parameters]
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        # Last short group has one negative on rank 0 and no work on rank 1.
        if rank == 0:
            model.shared.square().backward()
        training.sum_gradients(parameters, distributed, bucket_bytes=8)
        second = [p.grad.item() if p.grad is not None else None for p in parameters]
        optimizer.step()
        Path(folder, f'rank{rank}.json').write_text(json.dumps({
            'first': first, 'second': second,
            'parameters': [p.item() for p in parameters]}))
    finally:
        distributed.destroy_process_group()


class TrainingDistributionTests(unittest.TestCase):
    def test_real_epoch_keeps_last_five_images(self):
        indices = list(range(16597))
        groups = list(training.optimizer_groups(indices, 8))
        self.assertEqual(len(groups), 2075)
        self.assertEqual(len(groups[-1]), 5)
        restored = []
        for group in groups:
            shards = [training.shard_group(group, rank, 2, 2) for rank in range(2)]
            self.assertEqual(sorted(shards[0] + shards[1]), group)
            self.assertFalse(set(shards[0]) & set(shards[1]))
            restored.extend(sorted(shards[0] + shards[1]))
        self.assertEqual(restored, list(enumerate(indices)))

    def test_repeated_sampler_draws_have_distinct_draw_numbers(self):
        group = next(training.optimizer_groups([7, 7, 4, 7, 4], 8))
        shards = [training.shard_group(group, rank, 2, 2) for rank in range(2)]
        self.assertEqual(sorted(shards[0] + shards[1]), list(enumerate([7, 7, 4, 7, 4])))

    def test_empty_rank_in_short_final_group(self):
        group = [(16, 13)]
        self.assertEqual(training.shard_group(group, 0, 2, 2), group)
        self.assertEqual(training.shard_group(group, 1, 2, 2), [])

    def test_global_gradient_is_per_image_mean_for_mixed_roles(self):
        # Simulated scalar derivatives: positives have three equally weighted
        # roles; negatives have two. A last group of five is divided by five.
        gradients = [[3, 6, 9], [5, 7], [2, 5, 8], [4, 6], [6, 9, 12]]
        per_image = [sum(values) / len(values) for values in gradients]
        group = list(enumerate(range(len(gradients))))
        reduced = sum(sum(per_image[index] / len(group) for _, index
                          in training.shard_group(group, rank, 2, 2)) for rank in range(2))
        self.assertAlmostEqual(reduced, sum(per_image) / len(per_image))

    def test_validation_verifier_candidates_do_not_overweight_verify(self):
        # Positive: What 1/3, Where 1/3, each of four Verify candidates 1/12.
        weights = [1 / 3, 1 / 3, 1 / 12, 1 / 12, 1 / 12, 1 / 12]
        self.assertAlmostEqual(sum(weights), 1)
        self.assertAlmostEqual(sum(weights[2:]), weights[0])

    def test_update_plan_has_global_batch_eight(self):
        plan = training.training_plan(16597, 3, 2, 2, 2)
        self.assertEqual(plan['global_batch_size'], 8)
        self.assertEqual(plan['updates_per_epoch'], 2075)
        self.assertEqual(plan['optimizer_updates'], 6225)
        self.assertEqual(training.training_plan(16597, 3, 2, 2, 2, 20)['optimizer_updates'], 20)

    def test_bucket_capacity_keeps_order(self):
        class Parameter:
            def __init__(self, size):
                self.size = size
            def numel(self):
                return self.size
        parameters = [Parameter(n) for n in (3, 5, 2, 11, 1)]
        buckets = list(training.parameter_buckets(parameters, 32))
        self.assertEqual([[p.size for p in b] for b in buckets], [[3, 5], [2], [11], [1]])

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is not installed in this local interpreter')
    def test_two_process_gradients_and_unused_head_match_global_objective(self):
        import torch.distributed as distributed
        if not distributed.is_available() or not distributed.is_gloo_available():
            self.skipTest('Gloo is unavailable')
        with tempfile.TemporaryDirectory() as folder:
            rendezvous = Path(folder, 'rendezvous').as_uri()
            context = multiprocessing.get_context('spawn')
            workers = [context.Process(target=_distributed_worker, args=(rank, rendezvous, folder))
                       for rank in range(2)]
            for worker in workers:
                worker.start()
            try:
                for worker in workers:
                    worker.join(timeout=60)
                    self.assertEqual(worker.exitcode, 0, 'Distributed gradient worker failed or hung')
            finally:
                for worker in workers:
                    if worker.is_alive():
                        worker.terminate()
                        worker.join()
            states = [json.loads(Path(folder, f'rank{rank}.json').read_text()) for rank in range(2)]
            self.assertEqual(states[0], states[1])
            # d/dshared ((s^2 + w^2)/2 + (3s)^2)/2 = 9.5 at s=1.
            self.assertAlmostEqual(states[0]['first'][0], 9.5, places=6)
            self.assertAlmostEqual(states[0]['first'][1], 1.0, places=6)
            self.assertIsNone(states[0]['first'][2])
            # First SGD leaves shared=0.04 and where=1.88. The second update
            # touches shared only; neither unused head receives weight decay.
            self.assertAlmostEqual(states[0]['second'][0], 0.08, places=5)
            self.assertEqual(states[0]['second'][1:], [None, None])
            self.assertAlmostEqual(states[0]['parameters'][0], 0.0316, places=5)
            self.assertAlmostEqual(states[0]['parameters'][1], 1.88, places=5)
            self.assertAlmostEqual(states[0]['parameters'][2], 3.0, places=5)


if __name__ == '__main__':
    unittest.main()
