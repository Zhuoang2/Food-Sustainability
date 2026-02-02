#!/usr/bin/env python3
"""
Data pipeline: SQLite menu batch -> LLM extraction -> MySQL (6 tables).
Supports repeat runs: each run creates a new extraction_run and syncs current snapshot.

Usage:
  python pipeline.py --limit 10
  python pipeline.py --limit 5 --sqlite data/mydb_clean.sqlite

Environment (optional):
  OPENAI_API_KEY       (or place API_Key file in project root)
  OPENAI_BASE_URL      (default: https://aiapi-prod.stanford.edu/v1)
  MYSQL_HOST, MYSQL_USER, MYSQL_PASSWORD, MYSQL_DATABASE  (default: localhost/root//food_sustainability)
"""
import argparse
import logging
import os
import sys

# Ensure project root is on path when run as script
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from pipeline.sqlite_loader import load_menu_batch, build_menu_items_text
from pipeline.llm_extract import extract_ingredients
from pipeline.mysql_writer import run_with_transaction, get_mysql_config


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run ingredient extraction pipeline: SQLite -> LLM -> MySQL"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of menu items to process from SQLite (default: 10)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1,
        help="Items per LLM call (1=one request per item; default: 1)",
    )
    parser.add_argument(
        "--sqlite",
        type=str,
        default=None,
        help="Path to data/mydb_clean.sqlite (default: <project>/data/mydb_clean.sqlite)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="LLM model name (default: gpt-4o)",
    )
    parser.add_argument(
        "--prompt-version",
        type=str,
        default=None,
        help="Optional prompt_version for extraction_runs",
    )
    parser.add_argument(
        "--pipeline-version",
        type=str,
        default=None,
        help="Optional pipeline_version for extraction_runs",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    log = logging.getLogger(__name__)

    project_root = _SCRIPT_DIR
    os.chdir(project_root)
    sqlite_path = args.sqlite or os.path.join(project_root, "data", "mydb_clean.sqlite")

    if not os.path.isfile(sqlite_path):
        log.error("SQLite file not found: %s", sqlite_path)
        return 1

    try:
        log.info("Loading up to %d menu items from %s", args.limit, sqlite_path)
        batch = load_menu_batch(sqlite_path, args.limit)
        if not batch:
            log.warning("No menu items found (limit=%d)", args.limit)
            return 0
        log.info("Loaded %d items", len(batch))

        chunk_size = max(1, args.chunk_size)
        results = []
        for start in range(0, len(batch), chunk_size):
            chunk = batch[start : start + chunk_size]
            menu_text = build_menu_items_text(chunk)
            log.info(
                "Calling LLM for items %d–%d (chunk size=%d)",
                start + 1,
                start + len(chunk),
                chunk_size,
            )
            chunk_results = extract_ingredients(menu_text, model=args.model)
            results.extend(chunk_results)
            log.info("LLM returned %d items for this chunk", len(chunk_results))
        log.info("LLM returned %d item results in total", len(results))

        log.info("Writing to MySQL (one transaction)...")
        run_id = run_with_transaction(
            results,
            model_name=args.model,
            prompt_version=args.prompt_version,
            pipeline_version=args.pipeline_version,
        )
        log.info("Pipeline success: run_id=%s", run_id)
        return 0
    except Exception as e:
        log.exception("Pipeline failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
