# Repository Guidelines

## Project Structure & Module Organization
- Root scripts: `manage.sh` (orchestrates Docker workflow), `docker-compose.yml`, `Dockerfile.custom`.
- Trading config and data: `user_data/config.json`, logs in `user_data/logs/*.jsonl` and `freqtrade.log`, state DB `user_data/tradesv3.sqlite*`.
- Strategy code: `user_data/strategies/LLMFunctionStrategy.py` plus helpers under `user_data/strategies/llm_modules/` (context, tools, indicators, learning, utils, visualization).
- Tests: `user_data/tests/`; notebooks and artifacts live under `user_data/notebooks`, `user_data/backtest_results`, `user_data/hyperopt_results`.

## Build, Test, and Development Commands
- `./manage.sh start` – build/pull `freqtradeorg/freqtrade:stable`, bring up the container.
- `./manage.sh restart` – bounce the stack; use after config or strategy changes.
- `./manage.sh logs` / `./manage.sh decisions` / `./manage.sh trades` – tail container logs and LLM decision/trade logs locally.
- `./manage.sh deploy` – rebuild and restart for production-like refresh.
- `./manage.sh clean` – wipe logs/DB (prompted); only run when you intentionally want a reset.
- Tests (host with Docker): `docker compose run --rm freqtrade-llm pytest user_data/tests`.

## Coding Style & Naming Conventions
- Python 3.11; follow PEP 8 with 4-space indents, snake_case for functions/vars, PascalCase for classes, UPPER_SNAKE for constants.
- Keep strategy-facing functions deterministic and pure where possible; prefer dependency injection for IO.
- Log in JSONL format when touching `llm_decisions` or `trade_experience`; preserve existing keys and schema.
- Type hints are encouraged; keep docstrings concise and action-oriented.

## Testing Guidelines
- Use pytest in `user_data/tests/`; mirror module paths in test filenames (e.g., `test_critical_fixes.py`).
- Add focused unit tests for indicator/utility changes; add integration-style tests when altering strategy entry/exit logic.
- Aim for coverage of edge cases: leverage bounds, position adjustments, missing context, and log schema changes.

## Commit & Pull Request Guidelines
- Commit messages follow `type: short description` (`feat`, `fix`, `chore`, `refactor`); use imperative mood.
- PRs should include: summary of behavior change, key commands run (tests/logs), links to related issues/tasks, and screenshots or log excerpts when touching LLM decision outputs.
- Keep diffs small and scoped; note any schema changes in `user_data/logs/*.jsonl` or config fields.

## Security & Configuration Tips
- Never commit real API keys; use `.env.example` and `user_data/config.json` templates for local secrets.
- Before pushes, verify `user_data/logs` for sensitive data and scrub as needed.
- Ensure Docker is running and you have access to required images before invoking `manage.sh start`.
