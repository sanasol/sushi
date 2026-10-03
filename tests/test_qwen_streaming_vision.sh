#!/usr/bin/env bash
# A streamed Qwen keeps its tower and bills it; --no-vision still opts out.
# Usage: bash tests/test_qwen_streaming_vision.sh PACK BUDGET_GIB [PORT]
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PACK=${1:?usage: $0 PACK BUDGET_GIB [PORT]}
BUDGET=${2:?missing resident budget in GiB}
PORT=${3:-11427}
[[ "$BUDGET" =~ ^[1-9][0-9]*$ && "$PORT" =~ ^[0-9]+$ && "$PORT" -ge 1 && "$PORT" -le 65535 ]] || exit 2
[[ -f "$PACK/config.json" ]] || { echo "SKIP: model not found: $PACK"; exit 0; }
BIN="$ROOT/zig-out/bin/sushi"
[[ -x "$BIN" ]] || { echo 'Build first: zig build -Doptimize=ReleaseFast' >&2; exit 1; }
if lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "port $PORT is already in use" >&2; exit 1
fi
mkdir -p "$HOME/.sushi/runs"
RUN=$(mktemp -d "$HOME/.sushi/runs/qwen-stream-vision.XXXXXX")
PID=
LOCKED=0
OWNER="qwen-stream-vision-$$"
stop() {
    if [[ -n "$PID" ]]; then
        kill "$PID" 2>/dev/null || true
        wait "$PID" 2>/dev/null || true
        PID=
    fi
    if [[ "$LOCKED" == 1 ]]; then
        "$ROOT/scripts/gpu-lock.sh" release "$OWNER" >>"$RUN/lock.log" 2>&1
        LOCKED=0
    fi
}
trap 'stop; echo "Artifacts: $RUN"' EXIT
trap 'tail -40 "$RUN"/*.log >&2' ERR
# A fresh path avoids per-model overrides without touching the user's settings.
python3 - "$PACK" "$RUN/pack" <<'PY'
import json, pathlib, sys
source = pathlib.Path(sys.argv[1]).resolve(strict=True)
cfg = json.loads((source / 'config.json').read_text())
assert cfg['model_type'] == 'qwen4_exp', 'requires a Qwen Flash-Next checkpoint'
dest = pathlib.Path(sys.argv[2]); dest.mkdir()
for child in source.iterdir():
    if child.name != 'model-settings.json': (dest / child.name).symlink_to(child)
PY
for arm in vision text; do
    "$ROOT/scripts/gpu-lock.sh" acquire "$OWNER" >>"$RUN/lock.log" 2>&1
    LOCKED=1
    FLAGS=()
    if [[ "$arm" == text ]]; then FLAGS=(--no-vision); fi
    "$BIN" serve --model "$RUN/pack" --model-dir "$RUN" --host 127.0.0.1 --port "$PORT" \
        --ssd-budget-gb "$BUDGET" --no-mtp --no-pld --kv-quant 8 --ctx-size 4096 \
        --prefill-chunk 512 --max-concurrent 1 --prefix-cache-entries 0 \
        --prefix-cache-mem 0 --prefix-cache-disk 0 ${FLAGS[@]+"${FLAGS[@]}"} \
        >"$RUN/$arm.log" 2>&1 &
    PID=$!
    ready=0
    for ((attempt=0; attempt<600; attempt++)); do
        if curl --connect-timeout 1 --max-time 2 -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ready=1; break; fi
        kill -0 "$PID" 2>/dev/null || exit 1
        sleep 1
    done
    [[ "$ready" == 1 ]]
    python3 - "$PORT" "$arm" "$RUN" "$ROOT/tests/fixtures/house.jpeg" <<'PY'
import base64, json, pathlib, sys, urllib.error, urllib.request
port, arm, run, image = sys.argv[1:]
run = pathlib.Path(run)
base = 'http://127.0.0.1:' + port

def call(path, body=None):
    req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=600) as r: return r.status, json.load(r)
    except urllib.error.HTTPError as e: return e.code, json.load(e)

b64 = base64.b64encode(pathlib.Path(image).read_bytes()).decode()
for phase in ('boot', 'cold'):
    code, listing = call('/v1/models'); assert code == 200, listing
    model = next(m for m in listing['data'] if m.get('loaded'))
    assert model['streaming'] and not model['meta']['mtp_loaded'], model
    assert ('image' in model['input_modalities']) == (arm == 'vision'), model
    request = {'model': model['id'], 'max_tokens': 32, 'temperature': 0, 'enable_thinking': False,
               'messages': [{'role': 'user', 'content': [
                   {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + b64}},
                   {'type': 'text', 'text': 'What is the main subject? One word.'}]}]}
    code, reply = call('/v1/chat/completions', request)
    (run / f'{arm}-{phase}.json').write_text(json.dumps({'model': model, 'status': code, 'response': reply}, indent=2))
    if arm == 'vision':
        assert code == 200, reply
        answer = reply['choices'][0]['message']['content'].lower()
        assert any(word in answer for word in ('house', 'home', 'building')), reply
    else:
        assert code == 400 and 'vision tower' in json.dumps(reply), reply
        request['messages'] = [{'role': 'user', 'content': 'Say hello.'}]
        code, reply = call('/v1/chat/completions', request)
        assert code == 200 and reply['choices'][0]['message']['content'], reply
    if phase == 'boot':
        code, reply = call('/v1/unload-model', {'model': model['id']}); assert code == 200, reply
        code, reply = call('/v1/load-model', {'model': model['id']}); assert code == 200, reply
print(f'PASS: {arm} boot and cold load')
PY
    grep -q '\[expert-stream\] ssd budget' "$RUN/$arm.log"
    stop
done
python3 - "$RUN" <<'PY'
import pathlib, re, sys
run = pathlib.Path(sys.argv[1])
def trunk(arm):
    return [float(x) for x in re.findall(r'ssd budget .*?: trunk ([\d.]+) GB', (run / (arm + '.log')).read_text())]
v, t = trunk('vision'), trunk('text')
assert len(v) == len(t) == 2, (v, t)
assert all(a > b for a, b in zip(v, t)), (v, t)
print('PASS: vision weights billed on boot and cold load; --no-vision excludes them')
PY
