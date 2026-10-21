"""Operator view server: one static page and one websocket that streams pipeline events to it.

Holds only the event fan-out and the two routes. Does not hold any agent or voice logic;
the pipeline calls EventHub.publish and knows nothing else about this module.
"""

import asyncio
import json
from pathlib import Path

from aiohttp import web

from agent.state_machine import STAGES

PAGE_PATH = Path(__file__).parent / "index.html"
# Replayed to a page that connects (or reloads) mid-call, so it can rebuild what it missed.
MAX_HISTORY = 1000
CLIENT_QUEUE_SIZE = MAX_HISTORY + 500


class EventHub:
    """Fans pipeline events out to connected pages. publish never blocks and is safe from any thread."""

    def __init__(self):
        self.loop = None
        self.history = []
        self.client_queues = set()

    def publish(self, event: dict) -> None:
        """Queue an event for every connected page. Does nothing until the server has started."""
        if self.loop is None:
            return
        event_text = json.dumps(event)
        # call_soon_threadsafe returns immediately; the work happens later on the server's loop.
        self.loop.call_soon_threadsafe(self.deliver, event_text)

    def deliver(self, event_text: str) -> None:
        """Runs on the event loop: remember the event and put it on each page's queue."""
        self.history.append(event_text)
        if len(self.history) > MAX_HISTORY:
            del self.history[0]
        for client_queue in self.client_queues:
            # A page that has fallen far behind loses events rather than slowing the call down.
            if client_queue.full():
                continue
            client_queue.put_nowait(event_text)


async def send_queued_events(socket: web.WebSocketResponse, client_queue: asyncio.Queue) -> None:
    """Write queued events to one page until it goes away."""
    try:
        while True:
            event_text = await client_queue.get()
            await socket.send_str(event_text)
    except ConnectionResetError:
        # The page closed between the queue and the write; the handler below is already cleaning up.
        return


async def serve_page(request: web.Request) -> web.StreamResponse:
    """Serve the single HTML page."""
    return web.FileResponse(PAGE_PATH)


async def serve_socket(request: web.Request) -> web.WebSocketResponse:
    """One page's websocket: a hello with the stage list, the history so far, then live events."""
    hub = request.app["hub"]
    socket = web.WebSocketResponse()
    await socket.prepare(request)

    # The page clears itself on hello, so a reconnect replays cleanly instead of duplicating.
    await socket.send_str(json.dumps({"type": "hello", "stages": list(STAGES.keys())}))

    client_queue = asyncio.Queue(maxsize=CLIENT_QUEUE_SIZE)
    for event_text in hub.history:
        client_queue.put_nowait(event_text)
    hub.client_queues.add(client_queue)
    sender_task = asyncio.create_task(send_queued_events(socket, client_queue))
    try:
        # The page sends nothing; this loop just waits for it to close.
        async for message in socket:
            pass
    finally:
        hub.client_queues.discard(client_queue)
        sender_task.cancel()
    return socket


async def start_server(hub: EventHub, host: str = "127.0.0.1", port: int = 8765) -> web.AppRunner:
    """Start serving on the running event loop. Returns the runner so the caller can shut it down."""
    hub.loop = asyncio.get_running_loop()
    app = web.Application()
    app["hub"] = hub
    app.router.add_get("/", serve_page)
    app.router.add_get("/ws", serve_socket)
    runner = web.AppRunner(app, access_log=None)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    return runner
