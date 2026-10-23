from __future__ import annotations

import inspect
import re
import shutil
import sqlite3
from pathlib import Path
from unittest.mock import patch

from mulder.server.tools import artifacts


def test_chrome_history_is_extracted_once(tmp_path: Path) -> None:
    history = tmp_path / "History"
    path = "Users/alice/AppData/Local/Google/Chrome/User Data/Default/History"
    url = "https://example.com/one-visit"
    with sqlite3.connect(history) as connection:
        connection.execute("CREATE TABLE urls (url, title, visit_count, last_visit_time)")
        connection.execute("INSERT INTO urls VALUES (?, 'Example', 1, 13380163200000000)", (url,))

    def extract(_image: str, _offset: int, _inode: str, dest: Path) -> bool:
        shutil.copyfile(history, dest)
        return True

    with (
        patch.object(artifacts, "get_ctx"),
        patch.object(shutil, "which", return_value="icat"),
        patch.object(artifacts, "_resolve_image_and_offset", return_value=("disk.dd", 0)),
        patch.object(
            artifacts,
            "_find_inodes_by_path",
            side_effect=lambda pattern: [("42", path, 0)] if re.search(pattern, path) else [],
        ),
        patch.object(artifacts, "_icat_extract", side_effect=extract) as extraction,
        patch.object(artifacts, "extract_and_index", return_value={}) as index,
    ):
        result = inspect.unwrap(artifacts.parse_browser_history)()

    assert result["result_count"] == 1
    extraction.assert_called_once()
    assert index.call_args.args[0].count(url) == 1
