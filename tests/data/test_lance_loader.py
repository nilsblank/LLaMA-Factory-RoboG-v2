"""Tests for lance:// URI round-trip via lance_utils and mm_plugin."""

from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helper: build a minimal JPEG in memory (no external deps at import time)
# ---------------------------------------------------------------------------

def _make_jpeg_bytes() -> bytes:
    from PIL import Image

    img = Image.new("RGB", (8, 8), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# lance_utils tests
# ---------------------------------------------------------------------------


class TestParseLanceUri(unittest.TestCase):
    def setUp(self):
        from llamafactory.data.lance_utils import _parse_lance_uri

        self._parse = _parse_lance_uri

    def test_basic(self):
        path, col, row = self._parse("lance:///data/foo.lance#image#42")
        self.assertEqual(path, "/data/foo.lance")
        self.assertEqual(col, "image")
        self.assertEqual(row, 42)

    def test_path_with_hash_in_name_uses_rightmost_two(self):
        # rsplit("#", 2) always takes the last two "#" as col/row separators.
        path, col, row = self._parse("lance:///data/foo#bar.lance#image#7")
        self.assertEqual(path, "/data/foo#bar.lance")
        self.assertEqual(col, "image")
        self.assertEqual(row, 7)


class TestGetLanceDatasetCache(unittest.TestCase):
    def test_returns_same_object_on_second_call(self):
        import llamafactory.data.lance_utils as lu

        fake_ds = MagicMock()
        fake_lance = MagicMock()
        fake_lance.dataset.return_value = fake_ds

        # Patch the cache to be empty for this test
        with patch.dict(lu._LANCE_HANDLES, {}, clear=True), \
             patch.dict("sys.modules", {"lance": fake_lance}):
            first = lu._get_lance_dataset("/fake/path.lance")
            second = lu._get_lance_dataset("/fake/path.lance")

        self.assertIs(first, second)
        fake_lance.dataset.assert_called_once_with("/fake/path.lance")


class TestResolveLanceUri(unittest.TestCase):
    def test_round_trip(self):
        import llamafactory.data.lance_utils as lu

        expected_bytes = b"\xff\xd8\xff\xe0fake jpeg"

        # Build a fake lance dataset that returns expected_bytes
        fake_row = {"img": expected_bytes}
        fake_table = MagicMock()
        fake_table.to_pylist.return_value = [fake_row]
        fake_ds = MagicMock()
        fake_ds.take.return_value = fake_table

        with patch.dict(lu._LANCE_HANDLES, {"/fake/path.lance": fake_ds}):
            result = lu.resolve_lance_uri("lance:///fake/path.lance#img#0")

        self.assertEqual(result, expected_bytes)
        fake_ds.take.assert_called_once_with([0], columns=["img"])


# ---------------------------------------------------------------------------
# Integration: lance:// URI round-trip through mm_plugin._regularize_images
# ---------------------------------------------------------------------------


class TestMmPluginLanceImages(unittest.TestCase):
    def test_regularize_images_lance_uri(self):
        """_regularize_images resolves lance:// URI → PIL.Image."""
        from PIL import Image as PILImage

        from llamafactory.data.lance_utils import _LANCE_HANDLES
        import llamafactory.data.lance_utils as lu

        jpeg_bytes = _make_jpeg_bytes()
        fake_row = {"img": jpeg_bytes}
        fake_table = MagicMock()
        fake_table.to_pylist.return_value = [fake_row]
        fake_ds = MagicMock()
        fake_ds.take.return_value = fake_table

        uri = "lance:///fake/path.lance#img#0"

        # Import the BasePlugin subclass that has _regularize_images
        from llamafactory.data.mm_plugin import BasePlugin

        plugin = BasePlugin.__new__(BasePlugin)

        with patch.dict(lu._LANCE_HANDLES, {"/fake/path.lance": fake_ds}):
            result = plugin._regularize_images([uri])

        self.assertIn("images", result)
        self.assertEqual(len(result["images"]), 1)
        self.assertIsInstance(result["images"][0], PILImage.Image)


# ---------------------------------------------------------------------------
# _load_lance_dataset URI generation (loader.py integration)
# ---------------------------------------------------------------------------


class TestLoadLanceDatasetUris(unittest.TestCase):
    def test_binary_column_becomes_uri_strings(self):
        """_load_lance_dataset stores lance:// URIs (utf8) not raw bytes."""
        import pytest
        lance = pytest.importorskip("lance")
        pytest.importorskip("pyarrow")

        import pyarrow as pa

        jpeg_bytes = _make_jpeg_bytes()

        with tempfile.TemporaryDirectory() as tmpdir:
            lance_path = str(Path(tmpdir) / "test.lance")

            # Create a tiny lance dataset with a binary column
            table = pa.table({
                "messages": pa.array(["hello", "world"], type=pa.string()),
                "image": pa.array([jpeg_bytes, jpeg_bytes], type=pa.binary()),
            })
            lance.write_dataset(table, lance_path)

            from llamafactory.data.loader import _load_lance_dataset

            ds = _load_lance_dataset(lance_path)

        # The binary column should now be utf8 URI strings
        image_col = ds["image"]
        self.assertEqual(len(image_col), 2)
        for uri in image_col:
            self.assertIsInstance(uri, str)
            self.assertTrue(uri.startswith("lance://"), f"Expected lance:// URI, got: {uri!r}")
            self.assertIn("#image#", uri)

    def test_uri_resolves_to_original_bytes(self):
        """resolve_lance_uri on a generated URI returns the original bytes."""
        import pytest
        pytest.importorskip("lance")
        pytest.importorskip("pyarrow")

        import pyarrow as pa

        jpeg_bytes = _make_jpeg_bytes()

        with tempfile.TemporaryDirectory() as tmpdir:
            lance_path = str(Path(tmpdir) / "test.lance")
            table = pa.table({"image": pa.array([jpeg_bytes], type=pa.binary())})
            lance.write_dataset(table, lance_path)

            from llamafactory.data.loader import _load_lance_dataset
            from llamafactory.data.lance_utils import resolve_lance_uri
            import llamafactory.data.lance_utils as lu

            ds = _load_lance_dataset(lance_path)
            uri = ds["image"][0]

            # Clear cache so resolve_lance_uri opens fresh
            with patch.dict(lu._LANCE_HANDLES, {}, clear=True):
                resolved = resolve_lance_uri(uri)

        self.assertEqual(resolved, jpeg_bytes)


if __name__ == "__main__":
    unittest.main()
