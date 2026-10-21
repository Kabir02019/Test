# Lakeside patient scheduling agent

A phone agent for Lakeside Medical Group that verifies a caller, works out which doctor they need, checks the clinic's eligibility rules and books an appointment. The model only talks and chooses read tools; code decides when a booking is written. The same agent runs from a text REPL or over a live voice call.

## Run it

Setup: `pip install "pipecat-ai[deepgram,anthropic,silero,local]==0.0.108"`, then put `DEEPGRAM_API_KEY` and `ANTHROPIC_API_KEY` in `.env`.

- Text mode: `.venv/bin/python -m cli` (type `/session` to see the session, `/quit` to leave)
- Voice mode: `.venv/bin/python -m voice.pipeline` (uses the default microphone and speakers; headphones avoid the agent hearing itself)
- Voice mode with the operator view: `.venv/bin/python -m web`, then open http://127.0.0.1:8765 (the same call, plus a live page showing the conversation, stage, collected fields and activity)
- Scenarios: `.venv/bin/python -m tests.scenarios` (makes real Anthropic API calls)

## Architecture

- **adapter/** is the only code that knows the data is CSV and JSON files. It normalises specialty and location labels on the way in and owns the idempotent writes.
- **agent/tools/** holds the clinic tools (reads, eligibility rules, writes) and the schemas shown to the model. Writes have no schema, so the model cannot call them.
- **agent/state_machine.py** (run by `agent/core.py`) holds the stages, the tools allowed in each, and the exit conditions. It calls `book_appointment` itself, after a recorded confirmation, with a key built from session and slot.
- **agent/safety.py** is plain Python with no model calls: red-flag detection on what the caller says before the model sees it, and a grounding check on every reply before it is spoken.
- **voice/** is pipecat wiring only: speech in, `handle_turn`, speech out. `agent/` never imports it.

## Deliberately not built

- **Reschedule and cancel flows.** The tool methods exist and are exercised by `python -m agent.tools`, but no stage uses them. Booking comes first, and each flow needs its own confirmation step and the late-cancellation fee wording.
- **Holds beyond a basic expiry.** A hold lasts five minutes in memory, and that is all. Extending or sharing holds only matters with more than one process.
- **Real EHR integration.** The adapter loads the files into memory and forgets bookings on restart. It is the one seam where an EHR client would go.
- **Observability.** Logging and print statements only. At this size they are enough to read a conversation, and a tracing stack would be more code to explain than to use.
