from contextlib import redirect_stdout
import io
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import download_data as data


def make_archive(path, missing=None, columns=99, extra=None):
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in data.EXPECTED_FILES:
            if name != missing:
                archive.writestr("h3.6m/dataset/"+name, ",".join(["0"]*columns)+"\n")
        if extra is not None:
            archive.writestr(*extra)


class DownloadDataTests(unittest.TestCase):
    def test_subjects_actions_and_columns_match_the_reader(self):
        from runtime import ACTIONS
        self.assertEqual(list(data.ACTIONS), ACTIONS)
        self.assertEqual(len(data.EXPECTED_FILES), 210)

    def test_nested_archive_installs_expected_structure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, output = root/"data.zip", root/"dataset"
            make_archive(archive)
            summary = data.install_archive(archive, output)
            self.assertEqual(summary["files"], 210)
            self.assertEqual(summary["columns"], 99)
            self.assertTrue((output/"S11/walking_2.txt").is_file())
            self.assertFalse((output/"h3.6m").exists())
            self.assertTrue((output/"download_manifest.json").is_file())

    def test_missing_file_does_not_leave_partial_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_archive(root/"data.zip", missing="S5/walking_2.txt")
            with self.assertRaisesRegex(ValueError, "missing 1"):
                data.install_archive(root/"data.zip", root/"output")
            self.assertFalse((root/"output").exists())

    def test_wrong_coordinate_format_is_not_installed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_archive(root/"data.zip", columns=96)
            with self.assertRaisesRegex(ValueError, "99 finite"):
                data.install_archive(root/"data.zip", root/"output")
            self.assertFalse((root/"output").exists())

    def test_archive_traversal_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_archive(root/"data.zip", extra=("../escape.txt", "unwanted"))
            with self.assertRaisesRegex(ValueError, "unsafe archive path"):
                data.install_archive(root/"data.zip", root/"output")
            self.assertFalse((root/"escape.txt").exists())

    def test_archive_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            link = zipfile.ZipInfo("some_link")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            make_archive(root/"data.zip", extra=(link, "../outside"))
            with self.assertRaisesRegex(ValueError, "links are not supported"):
                data.install_archive(root/"data.zip", root/"output")

    def test_existing_dataset_is_checked_without_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_archive(root/"data.zip")
            data.install_archive(root/"data.zip", root/"output")
            with patch.object(data, "download_archive", side_effect=AssertionError("unexpected download")), redirect_stdout(io.StringIO()):
                self.assertEqual(data.main(["--output-dir", str(root/"output"), "--check-only"]), 0)

    def test_existing_files_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root/"output"
            output.mkdir()
            (output/"keep.txt").write_text("keep", encoding="utf-8")
            make_archive(root/"data.zip")
            with self.assertRaises(FileExistsError):
                data.install_archive(root/"data.zip", output)
            self.assertEqual((output/"keep.txt").read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
