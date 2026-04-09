"""
Route existence tests.

Every public endpoint must be reachable. A refactor that accidentally deletes
a route (like the /ws deletion that broke live transcription) will fail here.
"""


def test_index_serves_html(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Aurem" in resp.text


def test_static_css(client):
    resp = client.get("/static/style.css")
    assert resp.status_code == 200


def test_static_js(client):
    resp = client.get("/static/app.js")
    assert resp.status_code == 200


# --- API routes exist and return expected shapes ---

def test_get_devices(client):
    resp = client.get("/api/devices")
    assert resp.status_code == 200
    devices = resp.json()
    assert isinstance(devices, list)
    assert len(devices) >= 1
    assert "index" in devices[0]
    assert "type" in devices[0]


def test_get_session_status_idle(client):
    resp = client.get("/api/session/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["active"] is False


def test_get_sessions_empty(client):
    resp = client.get("/api/sessions")
    assert resp.status_code == 200
    assert resp.json() == []


def test_diarization_status(client):
    resp = client.get("/api/diarization/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "available" in data
    assert "hf_token_set" in data


def test_get_ai_config(client):
    resp = client.get("/api/ai-config")
    assert resp.status_code == 200
    # Should never leak full API key
    data = resp.json()
    assert "api_key" not in data


def test_websocket_endpoint_exists(client):
    """
    CRITICAL: Catches the /ws deletion regression.
    If this fails, live transcription is completely broken.
    """
    with client.websocket_connect("/ws") as ws:
        # Connection succeeded — endpoint exists
        ws.close()


def test_stop_without_session(client):
    resp = client.post("/api/session/stop")
    assert resp.status_code == 400
    assert "error" in resp.json()


def test_get_nonexistent_session(client):
    resp = client.get("/api/sessions/doesnotexist")
    assert resp.status_code == 404


def test_delete_nonexistent_session(client):
    resp = client.delete("/api/sessions/doesnotexist")
    assert resp.status_code == 404


def test_export_nonexistent_session(client):
    resp = client.get("/api/sessions/doesnotexist/export/txt")
    assert resp.status_code == 404


def test_file_job_nonexistent(client):
    resp = client.get("/api/transcribe-file/bad-id/status")
    assert resp.status_code == 404


def test_file_result_nonexistent(client):
    resp = client.get("/api/transcribe-file/bad-id/result")
    assert resp.status_code == 404
