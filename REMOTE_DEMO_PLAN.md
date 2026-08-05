# Phase 1.7 — Remote Demo Access (one public link, all admin panels reachable)

## Context

The platform only exists on `localhost` today. The user needs to demo it remotely — confirmed via Q&A: the remote person needs to **click around the admin dashboards themselves** (not just watch a screen share, and not a live voice call — that's explicitly out of scope for this pass), and access needs to work **recurring, over days/weeks**, not just a single throwaway session. The user has **no domain name** and wants to keep it to **one free link** rather than juggling several.

That last constraint is the one that shapes the whole design. A plain tunnel (ngrok/Cloudflare) to one port only gets you *that one app* — every other app's admin panel is a separate origin on a separate port. Research this session (a full Explore-agent sweep of all 9 admin-facing frontend files, confirmed by spot-checks) found that **every single `fetch()` call, `href`, `src`, and iframe `src`** in every app's admin panel uses an absolute leading-slash path (`fetch('/admin/api/...')`) or a hardcoded `http://localhost:PORT/...`. There is no relative-path usage anywhere. That means a single tunnel to the launcher only works for every app if the launcher can **reverse-proxy** the other four apps under its own origin, and each app's frontend needs to know it's being served under a path prefix so its own absolute-path calls still land on the right backend.

The two live-voice-call WebSocket connections (`ws://${location.host}/ws/voice`, bank + tech bots) live only in the **public voice-agent pages** (`index.html`), never in `admin.html`. Since the remote need is admin-panel access only, this plan deliberately does **not** touch `index.html`/WebSocket proxying — that's a clean, separable follow-up if a live remote voice demo is ever needed.

## 1. Reverse proxy — lives in the launcher (`launcher/app/main_launcher.py`)

One new generic route, gated the same way everything else on the launcher is:

```python
@app.api_route("/proxy/{app_key}/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE"])
async def proxy(app_key: str, path: str, request: Request, username: str = Depends(require_superadmin)):
    target = _proxy_target(app_key)   # looks up APPS (bank/tech/meet/portal) then load_registry() for custom bots
    if not target:
        raise HTTPException(404, f"Unknown app '{app_key}'")
    url = f"{target}/{path}"
    if request.url.query:
        url += f"?{request.url.query}"
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=False) as client:
        upstream = await client.request(request.method, url, headers=headers, content=body)
    resp_headers = {k: v for k, v in upstream.headers.items()
                     if k.lower() not in ("content-encoding", "transfer-encoding", "connection", "content-length")}
    content = upstream.content
    content_type = upstream.headers.get("content-type", "")
    if "text/html" in content_type:
        injected = f"<script>window.__BASE_PATH__='/proxy/{app_key}';</script>"
        text = content.decode("utf-8", errors="replace").replace("<head>", "<head>" + injected, 1)
        content = text.encode("utf-8")
    return Response(content=content, status_code=upstream.status_code, headers=resp_headers, media_type=content_type)
```

Key points:
- **Auth forwards for free.** The proxy route itself requires `require_superadmin` (same gate as everything else on the launcher). The browser's `Authorization` header is then forwarded untouched to the backend, which does its own `Depends(verify_admin)` check against the *same shared* `users.db` — so there's no double-login, and this doesn't weaken any existing per-app permission check.
- **No cookie/session complexity** — this whole repo is HTTPBasic-only, no cookies anywhere, so there's nothing to rewrite there.
- **Uploads (KB docs, meeting audio/transcripts, bgnoise, logo images) proxy transparently** — the raw request body and original `Content-Type` (including multipart boundary) are forwarded byte-for-byte, never re-parsed.
- **Only HTML responses get the `<script>` injection**; JSON/audio/image responses pass through completely untouched — no risk of corrupting binary or JSON payloads.
- Reuses `APPS` and `load_registry()`, both already defined in `main_launcher.py` — no new data structures beyond a tiny `_proxy_target()` lookup helper.
- New imports needed: `Request`, `Response` from `fastapi`/`starlette`.

## 2. Frontend: `BASE_PATH`-prefix every absolute admin-panel reference

In each admin-facing HTML file, add one line near the top of its `<script>` block:
```js
const BASE_PATH = window.__BASE_PATH__ || '';
```
Then mechanically prefix every absolute `fetch()`, `.href =`, `.src =`, and attribute reference found in the Phase-1.7 research sweep with `BASE_PATH +` (or `${BASE_PATH}` inside template literals). This is a large but entirely mechanical, low-risk edit — no logic changes, just string prefixing. When `BASE_PATH` is empty (i.e. the page is loaded directly at its own port, not through the proxy), every call resolves exactly as it does today — **local direct-port access is unaffected**.

Files touched (admin-facing only):
- `static/admin.html` (bank, ~19 references)
- `techsupport-voice-bot/static_tech/admin.html` (~21 references)
- `meeting-intelligence/static_meet/admin.html` (~18 references)
- `portal/static_portal/dashboard.html` (~6 references)
- `bot-template/static_bot/admin.html` (so every *future* custom bot is demo-ready with zero per-bot editing, same as the rest of the template genericization)

**Not touched:** `static/index.html`, `techsupport-voice-bot/static_tech/index.html`, `bot-template/static_bot/index.html` (public voice pages — out of scope, see Context).

### The "← back to launcher" logo link (all 5 files)
Every admin page's header logo currently hardcodes `href="http://localhost:8004/"` (added earlier this session). That's wrong when proxied — it needs to go to `/` on whatever origin is currently serving the page. Set it via one line of JS instead of a static attribute:
```js
document.querySelector('.logo').href = BASE_PATH ? '/' : 'http://localhost:8004/';
```

### Portal's 3 iframes (`portal/static_portal/dashboard.html` only)
Portal embeds bank/tech/meet admin panels via `<iframe src="http://localhost:8000/admin">` (and :8001, :8002) — reusing the existing lazy-load pattern already in `switchPane()`/`iframeLoaded` (dashboard.html:157-165). Change the src assignment to always go through the proxy, unconditionally:
```js
const IFRAME_SRC = { bank: '/proxy/bank/admin', tech: '/proxy/tech/admin', meet: '/proxy/meet/admin' };
```
This is unconditional (not gated on `BASE_PATH`) because portal is always reached *through the launcher* in the normal navigation flow (launcher tile → portal), so `/proxy/bank/admin` is always resolving against the launcher's own origin, whether that origin is `localhost:8004` or the public tunnel URL. Direct-port access to portal (`localhost:8003` typed by hand) is a pre-existing power-user edge case; under this change its 3 iframes specifically would stop resolving, while the rest of portal (Users pane, Overview) is unaffected — an acceptable, clearly-scoped trade-off given the "click through the launcher" flow is how this platform has been designed to be used all session.

## 3. Launcher's own cross-app links (`launcher/static_launcher/index.html`)

The launcher's own `fetch()` calls don't need `BASE_PATH` (it's always the proxy's own origin). Only its **outbound links to other apps** change, unconditionally, from hardcoded `http://localhost:PORT/...` to `/proxy/<app>/...`:
- Sidebar nav links + Overview cards: `/proxy/meet/admin`, `/proxy/portal/admin`, `/proxy/portal/admin#users`
- AI Voice Bots tile "Manage" links: `` `/proxy/${b.slug}/admin` `` (this already works for custom bots too, since `_proxy_target()` falls back to the registry)

## 4. Public tunnel: ngrok (free, no domain needed)

- Install `ngrok` (Windows: `winget install ngrok` or manual download from ngrok.com) — will confirm before installing anything, since it's new software going onto the machine.
- Quick/immediate test: `ngrok http 8004` — gives a working random `*.ngrok-free.app` HTTPS URL right away, changes on every restart.
- For the "recurring over weeks" requirement: ngrok's free plan includes **one permanent static domain** per account (claimed once via the ngrok dashboard after a free signup). Once claimed: `ngrok http 8004 --domain=<their-static-domain>.ngrok-free.app` gives a URL that never changes.
- New `start_demo_tunnel.bat` at the repo root wrapping that command, so starting the public link is a single double-click, consistent with `start_all.bat`/`stop_all.bat`.

## 5. Security note (surfaced, not a silent code change)

Before actually sharing a public link:
- The bootstrapped `ADMIN_PASS` in `.env` is currently the literal word `password` — fine for localhost-only use, not fine once a URL is public. Recommend changing it.
- Recommend creating a **separate superadmin account** via Portal → Users for whoever needs remote access, rather than sharing the primary admin credentials — cheap to revoke afterward, uses the multi-user system that's already built. (Note: the launcher's own root page requires `require_superadmin` specifically — a scoped non-superadmin user can't load the launcher's tile grid at all, which is pre-existing behavior from Phase 1.5, not something this plan changes.)
- Treat the tunnel URL itself as sensitive (anyone with the link + valid creds reaches the whole platform) — don't post it anywhere public.

## Verification

1. Restart the launcher, hit `/proxy/bank/health`, `/proxy/tech/health`, `/proxy/meet/api/sysinfo`, `/proxy/portal/health` directly (curl) — confirm each returns the right backend's JSON unmodified.
2. Curl `/proxy/bank/admin` — confirm the HTML comes back with `window.__BASE_PATH__='/proxy/bank'` injected right after `<head>`.
3. In-browser, from the launcher: click into each of Bank/Tech/Meet/Portal via the new proxy links — confirm each admin panel loads *and* its data-driven panes populate correctly (e.g. tech bot's Callers list, meet's Meetings list, bank's Recording Logs) — this proves `BASE_PATH`-prefixed fetches are actually resolving.
4. From Portal (reached via `/proxy/portal/admin`), confirm all 3 embedded iframes load their respective bots' admin panels (nested proxy chain).
5. From within a proxied admin page, click the logo — confirm it returns to `/` (the launcher home) rather than a dead `localhost:8004` link.
6. Confirm local direct-port access (`localhost:8000/admin` etc., not through the proxy) still works exactly as before — `BASE_PATH` is empty there, zero regression.
7. Start `ngrok http 8004`, open the resulting public URL on a different device/network (e.g. phone on mobile data, not the office wifi), log in, and click through the same flows as step 3-4 end-to-end over the public tunnel.

### Critical files

- `launcher/app/main_launcher.py` — new `/proxy/{app_key}/{path:path}` route, `_proxy_target()` helper
- `static/admin.html`, `techsupport-voice-bot/static_tech/admin.html`, `meeting-intelligence/static_meet/admin.html`, `bot-template/static_bot/admin.html` — `BASE_PATH` prefixing, logo-link fix
- `portal/static_portal/dashboard.html` — `BASE_PATH` prefixing, logo-link fix, iframe src fix
- `launcher/static_launcher/index.html` — cross-app links switched to `/proxy/...`
- `start_demo_tunnel.bat` — new
- (separately, unrelated to this plan) update `git remote` to the renamed GitHub repo once the exact new URL/slug is confirmed
