"""Small console-progress tests; no model or GPU is loaded."""
import ast
from contextlib import redirect_stdout, nullcontext, ExitStack
import io
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clear_uav.run_progress import elapsed, phase, run_with_progress


class ProgressTests(unittest.TestCase):
    def test_elapsed(self):
        self.assertEqual(elapsed(3661), "01:01:01")

    def test_phase_failure_is_not_done(self):
        output = io.StringIO()
        with redirect_stdout(output):
            with self.assertRaises(ValueError):
                with phase("broken phase"):
                    raise ValueError("test")
        self.assertIn("[stopped]", output.getvalue())
        self.assertNotIn("[done]", output.getvalue())

    def test_silent_wait_and_completion(self):
        output = io.StringIO()
        with redirect_stdout(output):
            run_with_progress([sys.executable, "-c", "import time; time.sleep(0.15)"],
                              cwd=ROOT, env=None, label="silent child", index=1,
                              total=2)
        self.assertIn("[task 1/2]", output.getvalue())
        self.assertNotIn("[waiting]", output.getvalue())
        self.assertNotIn("has not exited", output.getvalue())
        self.assertIn("[task 1/2 done]", output.getvalue())

    def test_child_failure_propagates(self):
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.CalledProcessError) as error:
                run_with_progress([sys.executable, "-c", "raise SystemExit(7)"],
                                  cwd=ROOT, env=None, label="bad child", index=1, total=1)
        self.assertEqual(error.exception.returncode, 7)

    def test_loader_phases_preserve_visual_only_keys(self):
        # Exercise the actual edited loader with lightweight stand-ins.
        tree = ast.parse((ROOT / "src/clear_uav/qwen_ground_ms.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == "load_qwen_vision")
        module = ast.Module(body=[ast.ImportFrom(module="__future__",
                            names=[ast.alias(name="annotations")], level=0), function],
                            type_ignores=[])
        model = MagicMock()
        model.rotary_pos_emb.dim = 4
        model.rotary_pos_emb.theta = 10000
        tensor = MagicMock()
        weights = MagicMock()
        weights.keys.return_value = ["model.visual.patch", "model.language.other"]
        weights.get_tensor.return_value = tensor
        context = MagicMock()
        context.__enter__.return_value = weights
        torch = MagicMock()
        torch.device.side_effect = lambda _: nullcontext()
        scope = dict(model_snapshot=lambda _: Path("model"), AutoConfig=MagicMock(),
                     torch=torch, Qwen3_5MoeVisionModel=lambda _: model,
                     safe_open=lambda *a, **kw: context, ExitStack=ExitStack,
                     tqdm=lambda keys, **kw: keys, AutoProcessor=MagicMock(), phase=phase)
        exec(compile(ast.fix_missing_locations(module), "<loader test>", "exec"), scope)
        output = io.StringIO()
        with redirect_stdout(output):
            actual, _ = scope["load_qwen_vision"](Path("model"), "cpu")
        self.assertIs(actual, model)
        weights.get_tensor.assert_called_once_with("model.visual.patch")
        model.load_state_dict.assert_called_once_with({"patch": tensor}, strict=True, assign=True)
        self.assertIn("Open weight file", output.getvalue())
        self.assertIn("Materialize vision weights", output.getvalue())


if __name__ == "__main__":
    unittest.main()
