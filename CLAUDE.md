# k1-chat rules

- Independent orphan branch `feat/k1-chatbot` (worktree `~/Projects/wt-k1-chat`). Do not copy files in from `booster_ws`; reference it by path.
- Read `PLAN.md`, `docs/FLEET.md`, `docs/ACCESS.md` first. Skills in `.claude/skills/`.
- Test first (see skill `k1-chat-tdd`). Manual velocity control must always win (`arbiter.py`).
- Never run `booster-cli upgrade`/`upload_log`; never auto-run policies on hardware; free OpenRouter models only.
- Secrets: `.env` is gitignored; never print or commit keys, never copy `~/.ssh/booster_k1`.
- Notes and plans go in `.md` files in this directory; reports go to Google Docs/Drive via composio, not Artifacts.
