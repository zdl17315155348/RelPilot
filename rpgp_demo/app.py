import argparse
import json
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from rpgp_demo.data import EXAMPLES
from rpgp_demo.data import RELATION_KEYWORDS
from rpgp_demo.extractor import RPGPExtractor


STATIC_DIR = Path(__file__).with_name("static")
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
EXTRACTOR = RPGPExtractor()


def build_extract_payload(text: str):
    payload = EXTRACTOR.extract(text).to_dict()
    total_relations = _total_relations()
    predicted_relations = len(payload["predicted_relations"])
    payload["relation_stats"] = {
        "total_relations": total_relations,
        "predicted_relations": predicted_relations,
        "skipped_relations": total_relations - predicted_relations,
        "reduction_ratio": round((total_relations - predicted_relations) / total_relations, 4),
    }
    return payload


def configure_extractor(model_path=None):
    global EXTRACTOR
    EXTRACTOR = RPGPExtractor(joint_model_path=model_path)


def _total_relations():
    if getattr(EXTRACTOR, "joint_predictor", None) is not None:
        return len(EXTRACTOR.joint_predictor.relations)
    return len(RELATION_KEYWORDS)


def get_examples_payload():
    return {"examples": [example.__dict__ for example in EXAMPLES]}


class RPGPRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/examples":
            self._send_json(get_examples_payload())
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if path != "/api/extract":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(body or "{}")
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return
        text = str(data.get("text", "")).strip()
        self._send_json(build_extract_payload(text))

    def _send_json(self, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def run(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT, model_path=None):
    configure_extractor(model_path)
    try:
        server = ThreadingHTTPServer((host, port), RPGPRequestHandler)
    except OSError as exc:
        if exc.errno == 98:
            print(
                f"Port {port} is already in use. Try: python3 -m rpgp_demo.app --port {port + 1}",
                file=sys.stderr,
            )
        raise
    print(f"RPGP demo running at http://{host}:{port}")
    server.serve_forever()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run the RPGP demo server.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind, default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind, default: 8000")
    parser.add_argument("--model", default=None, help="Joint model checkpoint path")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run(args.host, args.port, args.model)
