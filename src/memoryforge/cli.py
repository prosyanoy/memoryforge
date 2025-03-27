"""Command line entry point: serve / seed / produce / consume."""

from __future__ import annotations

import argparse

import structlog

from .config import Settings
from .services import build_services
from .synthetic import generate_events
from .telemetry import configure_logging


def _seed(svc, customers: int, conversations: int, seed: int) -> int:
    events, _ = generate_events(customers, conversations, seed)
    for e in events:
        svc.queue.publish(e)
    outcomes = svc.worker.drain()
    print(f"seeded {len(events)} events -> {dict(outcomes)}")
    return len(events)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="memoryforge")
    sub = parser.add_subparsers(dest="cmd", required=True)
    serve = sub.add_parser("serve", help="run the HTTP API (and the in-process consumer)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--seed", type=int, default=0, metavar="N", help="seed N synthetic customers first")
    for name, help_ in (("produce", "publish synthetic events to the configured queue"),
                        ("consume", "run only the consumer worker (no HTTP)"),
                        ("seed", "process synthetic events straight into the store")):
        p = sub.add_parser(name, help=help_)
        p.add_argument("--customers", type=int, default=20)
        p.add_argument("--conversations", type=int, default=3)
        p.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)

    settings = Settings()
    configure_logging(settings.log_level, settings.log_format)

    if args.cmd == "serve":
        import uvicorn

        from .api import create_app

        svc = build_services(settings)
        if args.seed:
            _seed(svc, args.seed, 3, 7)
        uvicorn.run(create_app(settings, svc), host=args.host, port=args.port, log_level="warning")
    elif args.cmd == "produce":
        svc = build_services(settings)
        events, _ = generate_events(args.customers, args.conversations, args.seed)
        for e in events:
            svc.queue.publish(e)
        print(f"published {len(events)} events to {settings.queue_backend} queue")
        svc.close()
    elif args.cmd == "seed":
        svc = build_services(settings)
        _seed(svc, args.customers, args.conversations, args.seed)
        print(svc.store.stats())
        svc.close()
    elif args.cmd == "consume":
        svc = build_services(settings)
        structlog.get_logger().info("consumer_started", queue=settings.queue_backend)
        try:
            while True:
                svc.worker.run_once(timeout=1.0)
        except KeyboardInterrupt:
            pass
        finally:
            svc.close()


if __name__ == "__main__":
    main()
