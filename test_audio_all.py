"""
test_audio_all.py - Quick end-to-end test of audio upload against the live server.
Usage: .\venv\Scripts\python.exe test_audio_all.py
"""
import requests

FILES = [
    ("dog_bark.mp3",  "audio/mpeg"),
    ("dog_bark2.mp3", "audio/mpeg"),
    ("dog_bark3.mp3", "audio/mpeg"),
]

BASE = "http://127.0.0.1:8000"

def test_file(endpoint, filename, content_type):
    print(f"  -> {endpoint}  file={filename}")
    with open(filename, "rb") as f:
        r = requests.post(
            BASE + endpoint,
            files={"file": (filename, f, content_type)},
            timeout=60,
        )
    print(f"     status : {r.status_code}")
    try:
        data = r.json()
    except Exception as e:
        print(f"     PARSE ERROR: {e}  raw={r.text[:200]}")
        return

    if "error" in data:
        print(f"     ERROR  : {data['error']}")
        return

    t = data.get("type", "?")
    print(f"     type   : {t}")

    if t == "audio":
        vocal = data.get("vocalization", {})
        conf  = vocal.get("confidence", 0)
        label = vocal.get("label", "?")
        print(f"     vocal  : {label}  ({round(conf*100)}%)")
        print(f"     gesture: {data.get('gesture', '?')}")
        emotion = data.get("emotion", {})
        print(f"     emotion: {emotion.get('label', '?')}")
        alerts = data.get("alerts", [])
        for a in alerts:
            print(f"     alert  : [{a.get('severity','?')}] {a.get('message','?')}")
    else:
        print(f"     data keys: {list(data.keys())}")

print("=" * 60)
print("AUDIO UPLOAD END-TO-END TEST")
print("=" * 60)

for filename, ct in FILES:
    print(f"\nFile: {filename}")
    print("-" * 40)
    test_file("/upload",            filename, ct)
    print()
    test_file("/predict_audio_file", filename, ct)

print("\n" + "=" * 60)
print("DONE")
