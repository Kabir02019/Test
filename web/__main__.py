"""Run the operator view and the voice pipeline together: .venv/bin/python -m web (from the project root).

Holds only the start-up order. The pipeline runs exactly as `python -m voice.pipeline` does,
with the hub's publish method as its event sink.
"""

import asyncio

from web.server import EventHub, start_server
from voice.pipeline import main as run_pipeline

PORT = 8765


async def run_both() -> None:
    """Start the web server on the pipeline's event loop, run the call, then stop the server."""
    hub = EventHub()
    runner = await start_server(hub, port=PORT)
    print(f"Operator view: http://127.0.0.1:{PORT}")
    try:
        await run_pipeline(hub.publish)
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(run_both())
