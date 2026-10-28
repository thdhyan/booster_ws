# /// script
# requires-python = ">=3.10"
# dependencies = ["websockets"]
# ///
"""K1 fleet dashboard: connection status + battery for every Booster K1 on the LAN.

Online state: robots broadcast a UDP heartbeat on :9000 once a second (free, no SSH).
Battery: the robot's Studio websocket (:51111) sends a status message each second; we open it
briefly every POLL_S seconds, read the first status message, and close (the same socket also
streams camera JPEGs, so we never keep it open).

Agent: if the k1-chat voice agent runs on a robot (wt-k1-chat `k1voice.sh start <robot> agent`), its console is
http://<ip>:8091/ (VLM inputs/outputs, tool calls). The STOP button here POSTs to that console's /api/estop
(robot -> PREPARE). If the agent is down, use the remote or `k1tools.sh estop <robot>` (ssh fallback).

Run: uv run tools/fleet_dashboard/dashboard.py [--host 0.0.0.0] [--port 8080]
"""
import argparse
import asyncio
import json
import socket
import struct
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import websockets

POLL_S = 10
AGENT_PORT = 8091
OFFLINE_AFTER_S = 5
KNOWN = {  # nickname suffix -> (last known ip), shown as offline until seen
    "A1": None, "A2": "192.168.50.215", "A3": "192.168.50.230",
    "B1": "192.168.50.105", "B2": "192.168.50.62", "B3": "192.168.50.248",
}
state = {}  # name -> dict
lock = threading.Lock()


def pb(b):
    """Minimal protobuf wire parser -> list of (field, wiretype, value)."""
    out, i = [], 0
    while i < len(b):
        tag = s = 0
        while True:
            c = b[i]; i += 1; tag |= (c & 0x7F) << s; s += 7
            if not c & 0x80:
                break
        f, w = tag >> 3, tag & 7
        if w == 0:
            v = s = 0
            while True:
                c = b[i]; i += 1; v |= (c & 0x7F) << s; s += 7
                if not c & 0x80:
                    break
            out.append((f, w, v))
        elif w == 1:
            out.append((f, w, struct.unpack("<d", b[i:i + 8])[0])); i += 8
        elif w == 5:
            out.append((f, w, struct.unpack("<f", b[i:i + 4])[0])); i += 4
        elif w == 2:
            n = s = 0
            while True:
                c = b[i]; i += 1; n |= (c & 0x7F) << s; s += 7
                if not c & 0x80:
                    break
            out.append((f, w, b[i:i + n])); i += n
        else:
            break
    return out


def field(msgs, num):
    return [v for f, _, v in msgs if f == num]


def robot_info(info):
    """Fields of the robot-info submessage (shared by the heartbeat and the status message)."""
    g = lambda n: (field(info, n) or [b""])[0]
    s = lambda n: g(n).decode(errors="replace") if isinstance(g(n), bytes) else ""
    return {"sn": s(1), "firmware": s(3), "ip": s(6) or s(13), "name": s(12)}


def short(name):
    return name.rsplit("_", 1)[-1] if name else ""


def touch(name, **kw):
    with lock:
        d = state.setdefault(name, {"name": name})
        d.update(kw)


def udp_listener():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("", 9000))
    while True:
        data, (src, _) = sock.recvfrom(4096)
        try:
            info = robot_info(pb(data))
        except Exception:
            continue
        if info["name"]:
            touch(short(info["name"]), nickname=info["name"], sn=info["sn"], firmware=info["firmware"],
                  ip=info["ip"] or src, last_seen=time.time())


async def poll_battery(ip, key):
    try:
        async with websockets.connect(f"ws://{ip}:51111", open_timeout=4, max_size=None) as ws:
            t = time.time()
            while time.time() - t < 6:
                m = await asyncio.wait_for(ws.recv(), 4)
                if not isinstance(m, bytes) or field(pb(m), 1) != [10000]:
                    continue
                body = pb(field(pb(m), 3)[0])
                for sub in field(body, 6):
                    p = pb(sub)
                    if field(p, 1) == [b"battery"]:
                        touch(key, battery=round(field(p, 3)[0], 1), battery_at=time.time())
                        return
    except Exception as e:
        touch(key, battery_err=type(e).__name__)


def agent_req(ip, path, method="GET", timeout=1.0):
    req = urllib.request.Request(f"http://{ip}:{AGENT_PORT}{path}", method=method, data=b"" if method == "POST" else None)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


async def poll_agent(ip, key):
    try:
        st = (await asyncio.to_thread(agent_req, ip, "/api/events?since=999999999"))["state"]
        touch(key, agent=True, agent_mode=st.get("mode"), agent_estopped=st.get("estopped"))
    except Exception:
        touch(key, agent=False)


async def battery_loop():
    while True:
        with lock:
            targets = [(d["ip"], k) for k, d in state.items()
                       if d.get("ip") and time.time() - d.get("last_seen", 0) < OFFLINE_AFTER_S]
        await asyncio.gather(*(poll_battery(ip, k) for ip, k in targets), *(poll_agent(ip, k) for ip, k in targets))
        await asyncio.sleep(POLL_S)


def snapshot():
    now = time.time()
    rows = []
    with lock:
        keys = set(state) | set(KNOWN)
        for k in sorted(keys):
            d = dict(state.get(k, {"name": k}))
            d.setdefault("ip", KNOWN.get(k))
            seen = d.get("last_seen")
            d["online"] = bool(seen and now - seen < OFFLINE_AFTER_S)
            d["seen_ago"] = round(now - seen, 1) if seen else None
            d["battery_age"] = round(now - d["battery_at"], 1) if d.get("battery_at") else None
            rows.append(d)
    return rows


PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>K1 Fleet</title><style>
body{font:15px system-ui;margin:20px;background:#111;color:#eee}h1{font-size:20px}
table{border-collapse:collapse;width:100%;max-width:900px}td,th{padding:8px 10px;border-bottom:1px solid #333;text-align:left}
.dot{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px}
.on{background:#3c3}.off{background:#c33}.bar{background:#333;border-radius:4px;width:120px;height:12px;display:inline-block;vertical-align:middle;margin-right:8px}
.bar i{display:block;height:100%;border-radius:4px}.dim{color:#888}a{color:#6cf}
.stop{background:#d22;color:#fff;border:0;border-radius:5px;padding:4px 10px;font-weight:bold;cursor:pointer;margin-left:8px}</style>
<h1>Booster K1 fleet</h1><table><thead><tr><th>Robot<th>Link<th>IP<th>Battery<th>Agent<th>Firmware<th>Last seen</thead><tbody id=b></tbody></table>
<p class=dim id=t></p><script>
async function r(){try{const d=await (await fetch('/api/status')).json();
document.getElementById('b').innerHTML=d.map(x=>{const b=x.battery,c=b==null?'#555':b<20?'#c33':b<40?'#ca3':'#3c3';
return `<tr><td><b>${x.name}</b><td><span class="dot ${x.online?'on':'off'}"></span>${x.online?'online':'offline'}
<td>${x.ip||'-'}<td>${b==null?'<span class=dim>-</span>':`<span class=bar><i style="width:${b}%;background:${c}"></i></span>${b}%`}
<td>${x.agent?`<a href="http://${x.ip}:8091/" target=_blank>console</a> <span class=dim>${x.agent_mode||''}${x.agent_estopped?' E-STOPPED':''}</span><button class=stop onclick="stop('${x.name}')">STOP</button>`:'<span class=dim>-</span>'}
<td class=dim>${x.firmware||'-'}<td class=dim>${x.seen_ago==null?'never':x.seen_ago+' s ago'}`}).join('');
document.getElementById('t').textContent='updated '+new Date().toLocaleTimeString()}catch(e){document.getElementById('t').textContent='server unreachable'}}
async function stop(n){const t=await (await fetch('/api/estop/'+n,{method:'POST'})).text();document.getElementById('t').textContent='STOP '+n+': '+t}
r();setInterval(r,2000)</script>"""


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/status":
            body, ctype = json.dumps(snapshot()).encode(), "application/json"
        elif self.path in ("/", "/index.html"):
            body, ctype = PAGE.encode(), "text/html; charset=utf-8"
        else:
            self.send_error(404); return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        name = self.path.removeprefix("/api/estop/")
        with lock:
            ip = (state.get(name) or {}).get("ip") or KNOWN.get(name)
        if not self.path.startswith("/api/estop/") or not ip:
            self.send_error(404); return
        try:
            code, out = 200, agent_req(ip, "/api/estop", "POST", timeout=2.0)
        except Exception as e:
            code, out = 502, {"error": f"agent unreachable ({type(e).__name__}): use the remote or k1tools.sh estop {name}"}
        body = json.dumps(out).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to share on the network")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    threading.Thread(target=udp_listener, daemon=True).start()
    threading.Thread(target=lambda: asyncio.run(battery_loop()), daemon=True).start()
    print(f"K1 fleet dashboard on http://{a.host}:{a.port}")
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
