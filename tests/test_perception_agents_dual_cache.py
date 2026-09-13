"""Changing a repository mount prefix must not bypass GT/stat cache binding."""
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from perception_agents_dual_common import relocated_fingerprint


class PortableCacheTests(unittest.TestCase):
    def test_mount_rebase_matches_original_but_gt_changes_do_not(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            image = root / 'event' / 'original' / 'frame.png'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'fixture bytes')
            sample = SimpleNamespace(image_path=image, record_uid='r1', group_id='g1',
                presence=True, label='event', bbox_1000=(10, 20, 300, 400))
            stat = image.stat()
            # pathlib uses the host separator; the stored record always uses POSIX paths.
            original = Path('/home/feihong/UAV_understanding/um7')
            row = ['r1', 'g1', (original / 'event/original/frame.png').as_posix(),
                   stat.st_size, stat.st_mtime_ns, True, 'event', (10, 20, 300, 400)]
            expected = hashlib.sha256(json.dumps([row], sort_keys=True).encode()).hexdigest()
            self.assertEqual(relocated_fingerprint([sample], root, original), expected)
            sample.bbox_1000 = (11, 20, 300, 400)
            self.assertNotEqual(relocated_fingerprint([sample], root, original), expected)


if __name__ == '__main__':
    unittest.main()
