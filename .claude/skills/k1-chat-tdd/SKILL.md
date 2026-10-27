---
name: k1-chat-tdd
description: Run and extend the k1-chat tests (red/green TDD) including the velocity-arbiter rules. Use before changing llm, arbiter, audio, vision or brain code.
---

# k1-chat TDD

Run: `env -u PYTHONPATH PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --with pytest --with respx --with httpx --with python-dotenv pytest -q`
(ROS pytest plugins break plain pytest here.) `-m net` = live OpenRouter, `-m robot` = live K1 (needs `K1_IP`).

Rules: write the failing test first; follow `PLAN.md` phases P1..P6; any new motion path goes through `VelocityArbiter` (manual wins, chat clamped/expiring, estop latches); only `:free` OpenRouter models; secrets only in `.env`.
