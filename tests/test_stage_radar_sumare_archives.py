from __future__ import annotations

import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from nowcasting.cli.stage_radar_sumare_archives import Archive, discover_archives, extract_archive, inspect_archive


def add_member(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(data)
    tar.addfile(member, io.BytesIO(data))


class StageRadarSumareArchivesTests(unittest.TestCase):
    def test_inventory_and_safe_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_root = root / "archives"
            archive_root.mkdir()
            archive_path = archive_root / "2025_01.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                add_member(tar, "2025/01/01/2025_01_01_00_00.png", b"first")
                add_member(tar, "2025/01/01/2025_01_01_00_02.png", b"second")

            archives = discover_archives(archive_root, 2025, 2025)
            self.assertEqual(archives, [Archive(archive_path, 2025, 1)])
            members, report = inspect_archive(archives[0])
            self.assertEqual(report["png_members"], 2)
            self.assertEqual(report["first_png"], "2025/01/01/2025_01_01_00_00.png")

            raw_root = root / "raw"
            extracted = extract_archive(archives[0], members, raw_root, resume=False)
            self.assertEqual(extracted, {"extracted_pngs": 2, "resumed_pngs": 0})
            self.assertEqual((raw_root / "2025/01/01/2025_01_01_00_02.png").read_bytes(), b"second")
            self.assertEqual(extract_archive(archives[0], members, raw_root, resume=True)["resumed_pngs"], 2)

    def test_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "2025_01.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                add_member(tar, "../outside.png", b"unsafe")
            archive = Archive(archive_path, 2025, 1)
            with self.assertRaisesRegex(ValueError, "inseguro"):
                inspect_archive(archive)

    def test_rejects_unexpected_png_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "2025_01.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                add_member(tar, "2025/02/01/2025_02_01_00_00.png", b"wrong month")
            with self.assertRaisesRegex(ValueError, "Layout inesperado"):
                inspect_archive(Archive(archive_path, 2025, 1))

    def test_records_filename_date_that_diverges_from_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "2025_01.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                add_member(tar, "2025/01/02/2025_01_01_00_00.png", b"wrong day")
            _, report = inspect_archive(Archive(archive_path, 2025, 1))
            self.assertEqual(report["directory_date_mismatches"], 1)
            self.assertEqual(report["directory_date_mismatch_examples"], ["2025/01/02/2025_01_01_00_00.png"])

    def test_records_filename_month_that_diverges_from_archive(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            archive_path = root / "2025_01.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                add_member(tar, "2025/01/01/2024_12_31_23_59.png", b"previous month")
            _, report = inspect_archive(Archive(archive_path, 2025, 1))
            self.assertEqual(report["archive_month_mismatches"], 1)
            self.assertEqual(report["archive_month_mismatch_examples"], ["2025/01/01/2024_12_31_23_59.png"])


if __name__ == "__main__":
    unittest.main()
