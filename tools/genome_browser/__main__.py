import argparse
import json

from .build import finalize, prepare_parallel
from .hosting import probe, serve, verify
from .search_index import attach_reference, build_reference
from .publication import publication_plan


def main():
    parser = argparse.ArgumentParser(description="Prepare or serve static OpenSpliceAI browser data; no inference")
    subs = parser.add_subparsers(dest="command", required=True)
    p = subs.add_parser("prepare")
    p.add_argument("config")
    p.add_argument("output")
    p.add_argument("--first-shard", type=int, default=0)
    p.add_argument("--last-shard", type=int)
    p.add_argument("--workers", type=int, default=1)
    p = subs.add_parser("finalize")
    p.add_argument("config")
    p.add_argument("output")
    p.add_argument("--base-url", default="")
    p.add_argument("--allow-subset", action="store_true")
    p = subs.add_parser("reference")
    p.add_argument("config")
    p.add_argument("output")
    p.add_argument("--search", action="store_true")
    p.add_argument("--contigs", help="comma-separated; omitted means all reference contigs")
    p = subs.add_parser("attach-reference")
    p.add_argument("output")
    p = subs.add_parser("verify")
    p.add_argument("output")
    p = subs.add_parser("serve")
    p.add_argument("output")
    p.add_argument("--port", type=int, default=8765)
    p = subs.add_parser("probe")
    p.add_argument("manifest_url")
    p = subs.add_parser("publication-plan")
    p.add_argument("output")
    p.add_argument("public_url")
    p.add_argument("plan_directory")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_parallel(args.config, args.output, workers=args.workers, first_shard=args.first_shard, last_shard=args.last_shard)
    elif args.command == "finalize":
        manifest = finalize(args.config, args.output, base_url=args.base_url, allow_subset=args.allow_subset)
        print(json.dumps({"dataset": manifest["id"], "files": len(manifest["files"])}))
    elif args.command == "reference":
        build_reference(args.config, args.output, with_search=args.search,
                        contig_names=args.contigs.split(",") if args.contigs else None)
    elif args.command == "attach-reference":
        attach_reference(args.output)
    elif args.command == "serve":
        serve(args.output, args.port)
    elif args.command == "verify":
        print(json.dumps(verify(args.output)))
    elif args.command == "probe":
        print(json.dumps(probe(args.manifest_url)))
    elif args.command == "publication-plan":
        print(json.dumps(publication_plan(args.output, args.public_url, args.plan_directory)))


if __name__ == "__main__":
    main()
