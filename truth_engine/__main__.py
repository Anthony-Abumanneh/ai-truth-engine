"""Command line.

    python -m truth_engine setup                  # download / build every model (once, ~1.5 GB)
    python -m truth_engine serve [--port 8000]
    python -m truth_engine audit [--live] "text to check"
    python -m truth_engine ask   [--live] "question?"
    python -m truth_engine watch "claim"          # save a claim to re-check later
    python -m truth_engine recheck [--live]       # re-check saved claims, report changes
"""
import sys


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    live = "--live" in rest
    words = [a for a in rest if not a.startswith("--")]
    if cmd == "setup":
        return setup()
    if cmd == "serve":
        from .api.server import serve
        port = next((int(a.split("=")[1]) for a in rest if a.startswith("--port=")), 8000)
        host = next((a.split("=", 1)[1] for a in rest if a.startswith("--host=")), "127.0.0.1")
        serve(port=port, host=host)
        return 0
    from .pipeline.orchestrator import Orchestrator
    orch = Orchestrator(live=live)
    if cmd == "audit":
        from .pipeline.audit import render
        t = orch.audit(" ".join(words))
        from .schemas import AuditReport
        print(render(AuditReport(text=t.question, claims=t.claims, verdicts=t.verdicts, sources=t.sources,
                                 routing=t.routing, skipped=t.skipped, verifiers=t.verifiers, errors=t.errors,
                                 latency_s=t.latency_s)))
    elif cmd == "ask":
        t = orch.ask(" ".join(words))
        f = t.final
        print(f"\n  {f.summary}\n  Confidence: {f.confidence.label} ({f.confidence.calibrated:.2f}); {f.decision.value}")
        if f.body:
            print(f"  {f.body[:400]}")
        for c in f.caveats:
            print(f"  ⚑ {c}")
    elif cmd == "watch":
        print("watching:", orch.watch(" ".join(words)))
    elif cmd == "recheck":
        changes = orch.recheck_watched()
        print("\n".join(str(c) for c in changes) or "no changes")
    else:
        print(__doc__)
        return 2
    return 0


def setup() -> int:
    """Everything a fresh clone needs. Each part is skipped if already present."""
    from .pipeline import nli, reranker
    from .science import design_classifier as dc
    steps = [("NLI verifier (DeBERTa-v3-large, 870 MB)", lambda: nli.is_downloaded("large") or nli.download("large")),
             ("evidence reranker (MiniLM, 91 MB)",
              lambda: (reranker.model_dir("msmarco") / "model.safetensors").exists() or reranker.download("msmarco")),
             ("study-design classifier (fetches ~7k PubMed abstracts, trains in seconds)",
              lambda: dc.MODEL_PATH.exists() or dc.train(dc.dataset()))]
    for name, fn in steps:
        print(f"- {name} ...", flush=True)
        fn()
    print("- embedding model (bge-base, 438 MB) ...", flush=True)
    from .retrieval.embeddings import get_embedder
    if get_embedder().name.startswith("hashing"):       # silent fallback would quietly hurt retrieval
        print("Could not load bge-base: is sentence-transformers installed? (pip install -r requirements.txt)")
        return 1
    from .models_registry import write_manifest
    m = write_manifest()
    print(f"Model manifest written: {len(m['files'])} files with SHA-256 checksums.\nDone.")
    return 0


def cli() -> None:
    """Entry point for the `truth-engine` command."""
    sys.exit(main(sys.argv[1:]))


if __name__ == "__main__":
    cli()
