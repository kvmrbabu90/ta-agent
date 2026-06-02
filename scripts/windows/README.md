# Kubera Windows launcher kit

One-click `.cmd` scripts to run / stop / auto-start the Kubera stack on Windows.
No PowerShell, no terminal commands — just double-click.

## Quick start

1. **Double-click [`start_kubera.cmd`](start_kubera.cmd)** in this folder.
2. Four small windows pop to your taskbar (minimized):
   - **Kubera-API** — the FastAPI backend on `http://localhost:8000`
   - **Kubera-Frontend** — the React/Vite dashboard on `http://localhost:5173`
   - **Kubera-Backup** — snapshots `predictions.sqlite` every 30 min
   - **Kubera-Scheduler** — fires the daily pipeline at 08:35 and 17:00 CT
     (OHLCV refresh + predictions + paper backtest). Without this, the
     Paper Trade and Live WF tabs stop updating with fresh data.
3. Your default browser opens to `http://localhost:5173`. That's it.

To stop everything, **double-click [`stop_kubera.cmd`](stop_kubera.cmd)**.

## Auto-start on Windows login

If you want Kubera to come back on its own after a reboot:

1. **Double-click [`install_autostart.cmd`](install_autostart.cmd)** once.
2. It puts a shortcut in your Windows Startup folder (`shell:startup`).
3. Done. Every time you log in, Kubera starts on its own.

To turn it off, **double-click [`uninstall_autostart.cmd`](uninstall_autostart.cmd)**.

## Make it easier to find

- **Pin to taskbar / desktop:** right-click `start_kubera.cmd` → Send to → Desktop (create shortcut). Now it's a desktop icon you can rename "Kubera" and give a custom icon if you like.
- **Pin to Start:** drag the .cmd onto your Start menu.

## Troubleshooting

### "venv not found"
The launcher expects the project at `C:\dev\ta-agent`. If you moved it, open `start_kubera.cmd` in Notepad and edit the `set "ROOT=..."` line at the top.

### Frontend window says "vite not found" or "npm not found"
Run once from the project's frontend folder:
```
cd C:\dev\ta-agent\services\frontend
npm install
```
This pulls down dependencies. You only need to do it once (or after a major dependency change). It's a one-time setup and not included in the launcher so the launcher stays fast.

### API window crashes repeatedly
The launcher wraps `uvicorn` in an auto-restart loop (`api_loop.cmd`). If you see it cycling every 5 seconds, open the Kubera-API window — the error stack trace will be visible. The most common cause is the DuckDB market data being locked by another process (a backtest, the WF run). Stop that other process and the API will recover on its next restart.

### "Start Kubera" button on the dashboard does nothing
That button launches the Alpaca engine + sync (a separate detached process from this launcher). It needs the API running, so make sure `start_kubera.cmd` ran first. The engine's own logs live at `logs/alpaca_engine.log` and `logs/alpaca_sync.log`.

### Want truly silent windows (no flicker on boot)?
The shortcut created by `install_autostart.cmd` is set to **WindowStyle=7 (minimized)**, so on login you'll briefly see the three windows fly to the taskbar — that's normal. They stay minimized. If you'd rather they be completely hidden, the cleanest path is to install Kubera as a Windows service via [NSSM](https://nssm.cc/) — out of scope here, but the launcher commands are simple to translate.

## What each file does

| File | Purpose |
|---|---|
| [`start_kubera.cmd`](start_kubera.cmd) | Main launcher. Starts API + frontend + backup + scheduler, opens browser. |
| [`start_scheduler.cmd`](start_scheduler.cmd) | Standalone scheduler launcher — double-click to add the scheduler to a running stack without touching anything else. |
| [`api_loop.cmd`](api_loop.cmd) | Auto-restart wrapper for the API; called by `start_kubera.cmd`. |
| [`stop_kubera.cmd`](stop_kubera.cmd) | Kills all Kubera processes (including the Alpaca engine if it's running). |
| [`install_autostart.cmd`](install_autostart.cmd) | One-time setup: launch Kubera on every Windows login. |
| [`uninstall_autostart.cmd`](uninstall_autostart.cmd) | Reverse the above. |
