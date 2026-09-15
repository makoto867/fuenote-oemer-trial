from fastapi.testclient import TestClient

from app import app, parse_musicxml


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_rejects_non_image() -> None:
    response = client.post("/v1/recognize", files={"file": ("bad.txt", b"not an image", "text/plain")})
    assert response.status_code == 415


def test_parse_musicxml() -> None:
    xml = b"""<?xml version="1.0"?><score-partwise><part id="P1"><measure number="1">
    <note><pitch><step>B</step><alter>-1</alter><octave>4</octave></pitch><duration>8</duration><type>eighth</type><dot/><voice>1</voice></note>
    <note><rest/><duration>8</duration><type>eighth</type></note>
    </measure></part></score-partwise>"""
    parsed = parse_musicxml(xml)
    assert parsed["note_count"] == 1
    assert parsed["event_count"] == 2
    assert parsed["notes"][0]["pitch"] == "Bb4"
    assert parsed["notes"][0]["solfege"] == "シ♭4"
    assert parsed["notes"][0]["dotted"] is True
    assert parsed["notes"][1]["rest"] is True
