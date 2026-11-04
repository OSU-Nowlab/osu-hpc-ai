"""Query an OpenAI-compatible server (SGLang's or vLLM's) and check its answers.

Sends --requests chat completions at once, so they spread over the server's data-parallel ranks, each asking for a
product of two numbers; greedy decoding. Standard library only.

    python3 client.py --url http://127.0.0.1:30000 [--requests 32] [--wait 1800]
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def call(url, path, payload=None, timeout=600):
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read() or b"null")


def wait_ready(url, seconds):
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url + "/health", timeout=10)
            return True
        except (urllib.error.URLError, OSError):
            time.sleep(10)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--requests", type=int, default=32)
    ap.add_argument("--wait", type=int, default=0, help="seconds to wait for /health first")
    args = ap.parse_args()
    if args.wait and not wait_ready(args.url, args.wait):
        sys.exit(f"client: {args.url} not ready within {args.wait} s")
    model = call(args.url, "/v1/models")["data"][0]["id"]

    def ask(a, b):
        payload = {"model": model, "temperature": 0, "max_tokens": 16,
                   "messages": [{"role": "user", "content": f"What is {a} * {b}? Reply with the number only."}]}
        for attempt in range(3):
            try:
                text = call(args.url, "/v1/chat/completions", payload)["choices"][0]["message"]["content"]
                return a, b, text.strip(), str(a * b) in text.replace(",", "")
            except urllib.error.HTTPError as error:     # a server that is still warming up answers 5xx
                if error.code < 500 or attempt == 2:
                    raise
                time.sleep(10)

    pairs = [(12 + i, 23 + 3 * i) for i in range(args.requests)]
    start = time.time()
    with ThreadPoolExecutor(len(pairs)) as pool:
        results = list(pool.map(lambda p: ask(*p), pairs))
    correct = sum(r[3] for r in results)
    for a, b, text, ok in results[:3]:
        print(f"client: {a} * {b} -> {text!r}{'' if ok else '  (wrong)'}")
    ok = correct >= 0.9 * len(results)
    print(f"client: {'OK' if ok else 'FAILED'}; {correct}/{len(results)} correct from {model} "
          f"in {time.time() - start:.1f} s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
