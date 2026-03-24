from __future__ import annotations

from unittest.mock import MagicMock, patch

from astar_island.api_client import AstarClient


def test_client_builds_bearer_header():
    with patch("astar_island.api_client.requests.Session") as Sess:
        inst = MagicMock()
        inst.headers = {}
        Sess.return_value = inst
        c = AstarClient(token="t")
        assert c.session.headers["Authorization"] == "Bearer t"
