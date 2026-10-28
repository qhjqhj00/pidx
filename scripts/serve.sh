#!/usr/bin/env bash
# usage: scripts/serve.sh start|stop|restart|status   (PORT=8890 by default)
cd "$(dirname "$0")/.." || exit 1
PORT=${PORT:-8890}; PID=logs/server.pid; mkdir -p logs
start() { nohup .venv/bin/python -m pidx serve --port "$PORT" >> logs/server.log 2>&1 & echo $! > $PID; sleep 2; status; }
stop()  { [ -f $PID ] && kill "$(cat $PID)" 2>/dev/null; rm -f $PID; }
status(){ if [ -f $PID ] && kill -0 "$(cat $PID)" 2>/dev/null; then echo "running pid $(cat $PID) on :$PORT"; else echo stopped; fi; }
case "$1" in start) start;; stop) stop;; restart) stop; sleep 1; start;; *) status;; esac
