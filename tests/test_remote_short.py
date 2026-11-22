"""A file that ends before the bytes asked for is not a network loss (correction 2026-11-23)."""
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from pedigree import remote, units

BODIES = {"/empty.gguf": (b"", None), "/cut.gguf": (b"GGUF\x03\x00", None),
          "/dropped.gguf": (b"GG", 24), "/whole.gguf": (bytes(range(24)), None)}


class _H(BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        body, claim = BODIES[self.path]
        self.send_response(200)
        self.send_header("Content-Length", str(len(body) if claim is None else claim))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setenv("PEDIGREE_CACHE", str(tmp_path))
    monkeypatch.setattr(remote.time, "sleep", lambda s: None)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


@pytest.mark.parametrize("name", ["empty.gguf", "cut.gguf"])
def test_a_file_shorter_than_the_range_is_not_readable_not_lost(server, name):
    with pytest.raises(remote.FileTooShort) as e:
        remote.fetch_range(f"{server}/{name}", 0, 23, retries=2)
    assert not units._network(e.value)


def test_a_dropped_connection_is_still_a_network_loss(server):
    with pytest.raises(Exception) as e:          # IncompleteRead: retried, then raised
        remote.fetch_range(f"{server}/dropped.gguf", 0, 23, retries=2)
    assert not isinstance(e.value, remote.FileTooShort) and units._network(e.value)


def test_a_whole_range_reads(server):
    assert remote.fetch_range(f"{server}/whole.gguf", 0, 23) == bytes(range(24))
