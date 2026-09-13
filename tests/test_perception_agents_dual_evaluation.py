"""CPU-only invariants for exact, unpadded multi-GPU agent evaluation."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("agents_dual_evaluation_under_test", SCRIPTS / "test_perception_agents_dual.py")
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


class DualEvaluationTests(unittest.TestCase):
    def test_cli_passes_real_batch_size_to_evaluation(self):
        runtime = SimpleNamespace(read_config=lambda path: {'test': {'batch_size': 1}})
        with patch.dict(sys.modules, {'perception_agents_runtime': runtime}), \
             patch.object(sys, 'argv', ['test_dual', '--batch-size', '4', '--split', 'val']), \
             patch.object(evaluation, 'evaluate') as run:
            evaluation.main()
        self.assertEqual(run.call_args.args, ({'test': {'batch_size': 4}}, 'val', 'full'))

    def samples(self, count):
        return [SimpleNamespace(record_uid=f"uid-{index}", image_path=f"image-{index}.png") for index in range(count)]

    def shards(self, count, world_size):
        return [
            {"rank": rank, "rows": [
                (index, f"uid-{index}", {"category": f"event-{index}", "trace": {"source_index": index}})
                for index in evaluation.shard_indices(count, rank, world_size)
            ]}
            for rank in range(world_size)
        ]

    def test_shards_cover_once_without_distributed_sampler_padding(self):
        for count in (0, 1, 2, 3, 5, 2728, 5338):
            for world_size in (1, 2, 3, 8):
                with self.subTest(count=count, world_size=world_size):
                    parts = [evaluation.shard_indices(count, rank, world_size) for rank in range(world_size)]
                    flat = [index for part in parts for index in part]
                    self.assertEqual(sorted(flat), list(range(count)))
                    self.assertEqual(len(flat), len(set(flat)))
                    self.assertLessEqual(max(map(len, parts)) - min(map(len, parts)), 1)

    def test_rank_and_world_size_are_checked(self):
        for arguments in ((-1, 0, 2), (2, -1, 2), (2, 2, 2), (2, 0, 0)):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                evaluation.shard_indices(*arguments)

    def test_merge_restores_original_order_even_when_ranks_arrive_reversed(self):
        for count, world_size in ((0, 2), (1, 2), (5, 2), (7, 3), (2, 8)):
            with self.subTest(count=count, world_size=world_size):
                predictions = evaluation.merge_predictions(list(reversed(self.shards(count, world_size))), self.samples(count))
                self.assertEqual([row["category"] for row in predictions], [f"event-{index}" for index in range(count)])
                self.assertEqual([row["trace"]["source_index"] for row in predictions], list(range(count)))

    def test_merge_rejects_duplicate_including_padded_tail(self):
        shards = self.shards(5, 2)
        shards[1]["rows"].append(shards[0]["rows"][0])
        with self.assertRaisesRegex(ValueError, "Repeated"):
            evaluation.merge_predictions(shards, self.samples(5))

    def test_merge_rejects_missing_prediction(self):
        shards = self.shards(5, 2)
        shards[0]["rows"].pop()
        with self.assertRaisesRegex(ValueError, "Missing"):
            evaluation.merge_predictions(shards, self.samples(5))

    def test_merge_rejects_uid_misalignment(self):
        shards = self.shards(5, 2)
        index, _, prediction = shards[1]["rows"][0]
        shards[1]["rows"][0] = (index, "uid-from-another-split", prediction)
        with self.assertRaisesRegex(ValueError, "record_uid"):
            evaluation.merge_predictions(shards, self.samples(5))

    def test_merge_rejects_out_of_range_and_boolean_indices(self):
        for index in (-1, 5, True):
            shards = self.shards(5, 2)
            _, uid, prediction = shards[1]["rows"][0]
            shards[1]["rows"][0] = (index, uid, prediction)
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, "Out-of-range"):
                evaluation.merge_predictions(shards, self.samples(5))

    def test_original_calibration_helpers_are_reused(self):
        import test_perception_agents as original
        for name in ("evaluation_configuration", "inference_signature", "validate_calibration", "write_traces", "agent_metrics"):
            self.assertIs(getattr(evaluation, name), getattr(original, name))

    def test_batched_prediction_keeps_rank_indices_and_partial_last_batch(self):
        calls = []

        class Progress:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def update(self, count):
                pass

        class Runtime:
            def predict_batch(self, paths, mode):
                calls.append((paths, mode))
                return [{"path": path, "mode": mode} for path in paths]

        with patch.dict(sys.modules, {"tqdm": SimpleNamespace(tqdm=Progress)}):
            rows = evaluation.predict_shard(Runtime(), self.samples(10), 1, 2, "verify_only", "val", 3)
            empty = evaluation.predict_shard(Runtime(), self.samples(1), 1, 2, "full", "test", 3)
        self.assertEqual([row[0] for row in rows], [1, 3, 5, 7, 9])
        self.assertEqual([row[1] for row in rows], [f"uid-{index}" for index in (1, 3, 5, 7, 9)])
        self.assertEqual([row[2]["path"] for row in rows], [f"image-{index}.png" for index in (1, 3, 5, 7, 9)])
        self.assertEqual([len(paths) for paths, _ in calls], [3, 2])
        self.assertTrue(all(mode == "verify_only" for _, mode in calls))
        self.assertEqual(empty, [])

    def test_batched_prediction_rejects_short_runtime_output(self):
        class Progress:
            def __init__(self, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        runtime = SimpleNamespace(predict_batch=lambda paths, mode: [])
        with patch.dict(sys.modules, {"tqdm": SimpleNamespace(tqdm=Progress)}):
            with self.assertRaisesRegex(ValueError, "different number"):
                evaluation.predict_shard(runtime, self.samples(2), 0, 2, "full", "val", 4)


if __name__ == "__main__":
    unittest.main()
