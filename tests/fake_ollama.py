"""Minimal stand-in for the Ollama HTTP API, used only to test the LLM plumbing
(request format, schema validation, retry, cache) without a real model.
It answers with the rule parser's output; the first call for each text returns invalid JSON
to exercise the retry path."""
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
sys.path.insert(0, ".")
from pipeline import parse_rules

seen = set()

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        self._send({"models": [{"name": "qwen2.5:7b"}]})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["format"]["title"] == "Extraction" and body["options"]["temperature"] == 0
        user = [m for m in body["messages"] if m["role"] == "user"][-1]["content"]
        if "failed validation" in user:
            user = [m for m in body["messages"] if m["role"] == "user"][-2]["content"]
            text = user.split('Message: "', 1)[1].rsplit('"', 1)[0]
            content = parse_rules.parse(text).model_dump_json()
        else:
            text = user.split('Message: "', 1)[1].rsplit('"', 1)[0]
            if text not in seen:
                seen.add(text)
                content = json.dumps({"symptoms": [{"symptom": "nausea", "present": False, "intensity": "severe",
                                                    "evidence": "x"}], "medications": []})
            else:
                content = parse_rules.parse(text).model_dump_json()
        self._send({"message": {"role": "assistant", "content": content}})

HTTPServer(("127.0.0.1", 11555), H).serve_forever()
