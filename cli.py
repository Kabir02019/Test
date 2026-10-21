"""Text REPL against the agent core. No audio.

Run from the project root:  .venv/bin/python -m cli
Commands:  /session  dump the whole session    /quit  leave
"""

import json

from dotenv import load_dotenv

load_dotenv(override=True)

from agent.core import handle_turn, new_session  # noqa: E402  (needs the env loaded first)


def show_tool_call(tool_name: str, arguments: dict, result: dict) -> None:
    """Print a tool call and its result as soon as it finishes."""
    print(f"  [tool] {tool_name}({json.dumps(arguments)})")
    print(f"  [result] {json.dumps(result)}")


def show_stage_change(old_stage: str, new_stage: str) -> None:
    """Print a stage transition as soon as it happens."""
    print(f"  [stage change] {old_stage} -> {new_stage}")


def show_state(session: dict) -> None:
    """Print the session fields that matter after each turn."""
    print(f"  [stage] {session['stage']}")
    print(f"  [patient_id] {session['patient_id']}   [hold_id] {session['hold_id']}")
    print(f"  [collected] {json.dumps(session['collected'])}")


def main() -> None:
    session = new_session()
    print("Lakeside scheduling agent (text mode). Type /session to dump state, /quit to leave.")

    while True:
        try:
            user_text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_text == "":
            continue
        if user_text == "/quit":
            break
        if user_text == "/session":
            print(json.dumps(session, indent=2))
            continue

        reply_text, session = handle_turn(
            session, user_text, on_tool_call=show_tool_call, on_stage_change=show_stage_change
        )
        print(f"agent> {reply_text}")
        show_state(session)


if __name__ == "__main__":
    main()
