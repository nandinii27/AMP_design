"""Download full peptide records from the DBAASP REST API.

Two phases, because the endpoints carry different payloads.

  GET /peptides            search. Returns totalCount plus flattened summaries.
                           No per-species MIC, no haemolysis.
  GET /peptides/{id}       full record, with targetActivities and
                           hemoliticCytotoxicActivities.

Requests run concurrently because the bottleneck is server round trip latency,
not local work. Serial fetching measured about 1.3 records per second, which is
five and a half hours for the full index; eight workers brings that to under an
hour. The worker count is deliberately modest: this is a public research
database and the point is to finish tonight, not to hammer it.

Everything is written as newline delimited JSON and re-running skips whatever is
already on disk, so an interrupted run resumes.

Filtering happens later, in the parser, where the rules are visible. Guessing
the API's enum values risks silently returning a filtered subset that looks
complete.

    uv run python scripts/fetch_dbaasp.py --out data/dbaasp_raw.ndjson
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

DEFAULT_BASE = "https://dbaasp.org"
USER_AGENT = "amp-challenge-2027 research client"


def get_json(url: str, retries: int = 4, backoff: float = 2.0, timeout: int = 60):
    """One GET with retries on transient failure.

    A 404 returns None rather than raising: ids occasionally exist in the search
    index but not the record store, and one missing record should not end a run
    of twenty five thousand.
    """
    last_error = None
    for attempt in range(retries):
        try:
            req = Request(url, headers={"Accept": "application/json", "User-Agent": USER_AGENT})
            with urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as e:
            if e.code == 404:
                return None
            last_error = e
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(backoff * (2 ** attempt))
                continue
            raise
        except (URLError, TimeoutError, json.JSONDecodeError) as e:
            last_error = e
            time.sleep(backoff * (2 ** attempt))
    raise RuntimeError(f"Failed after {retries} attempts: {url} ({last_error})")


def collect_ids(base: str, page_size: int, delay: float, max_records: int | None) -> list[int]:
    ids: list[int] = []
    offset = 0
    total = None

    while True:
        url = f"{base}/peptides?" + urlencode({"limit": page_size, "offset": offset})
        payload = get_json(url)
        if payload is None:
            break

        if total is None:
            total = payload.get("totalCount")
            print(f"Search index reports {total} peptides")

        batch = payload.get("data") or []
        if not batch:
            break

        for item in batch:
            pid = item.get("id")
            if pid is not None:
                ids.append(int(pid))

        offset += len(batch)
        print(f"  ids collected: {len(ids)}", end="\r", flush=True)

        if max_records and len(ids) >= max_records:
            ids = ids[:max_records]
            break
        if total is not None and offset >= total:
            break
        time.sleep(delay)

    print(f"\nCollected {len(ids)} peptide ids")
    return ids


def already_fetched(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done: set[int] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("id") is not None:
                done.add(int(record["id"]))
    return done


def fetch_records(base: str, ids: list[int], out_path: Path, workers: int, delay: float) -> None:
    """Concurrent fetch with a single writer.

    Only the main thread writes, so the output file cannot interleave partial
    lines. Worker threads return parsed records and the writer appends them as
    they complete, which also means the file is valid after any interruption.
    """
    done = already_fetched(out_path)
    if done:
        print(f"Resuming: {len(done)} records already on disk")

    pending = [i for i in ids if i not in done]
    if not pending:
        print("Nothing to fetch.")
        return
    print(f"Fetching {len(pending)} records with {workers} workers")

    throttle = threading.Semaphore(workers)

    def task(pid: int):
        with throttle:
            if delay:
                time.sleep(delay)
            return pid, get_json(f"{base}/peptides/{pid}")

    written = 0
    missing = 0
    failed = 0
    completed = 0
    started = time.time()

    with out_path.open("a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(task, pid) for pid in pending]
        for future in as_completed(futures):
            completed += 1
            try:
                _pid, record = future.result()
            except Exception:
                failed += 1
                record = None

            if record is None:
                missing += 1
            else:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                if written % 200 == 0:
                    fh.flush()

            if completed % 100 == 0 or completed == len(pending):
                elapsed = time.time() - started
                rate = completed / elapsed if elapsed > 0 else 0.0
                remaining = (len(pending) - completed) / rate if rate > 0 else 0.0
                print(
                    f"  {completed}/{len(pending)}  written {written}  "
                    f"missing {missing}  failed {failed}  "
                    f"{rate:.1f}/s  eta {remaining / 60:.0f} min",
                    end="\r",
                    flush=True,
                )

    print(f"\nDone. {written} written, {missing} empty, {failed} failed.")
    if failed:
        print("Re-run the same command to retry the failures.")


def _unit_name(unit) -> str:
    """The API returns unit as an object, not a string.

    Serial trial runs showed {'description': '', 'name': 'uM'}. Treating it as a
    string makes every conversion fail and silently discards every activity
    record, which looks like an empty database rather than a parsing bug.
    """
    if isinstance(unit, dict):
        return str(unit.get("name") or "unspecified")
    return str(unit or "unspecified")


def summarise(out_path: Path) -> None:
    n = with_mic = with_hc50 = modified = unusual = multimer = 0
    units: dict[str, int] = {}
    species: dict[str, int] = {}

    with out_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1

            if str(rec.get("complexity") or "").lower() != "monomer":
                multimer += 1

            activities = rec.get("targetActivities") or []
            hemolytic = rec.get("hemoliticCytotoxicActivities") or []
            if activities:
                with_mic += 1
            if hemolytic:
                with_hc50 += 1
            if rec.get("unusualAminoAcids"):
                unusual += 1

            nterm = rec.get("nTerminus")
            cterm = rec.get("cTerminus")
            nterm = nterm.get("name") if isinstance(nterm, dict) else nterm
            cterm = cterm.get("name") if isinstance(cterm, dict) else cterm
            if (nterm and str(nterm).upper() not in {"FREE", "NONE"}) or (
                cterm and str(cterm).upper() not in {"FREE", "NONE", "COOH"}
            ):
                modified += 1

            for act in list(activities) + list(hemolytic):
                unit = _unit_name(act.get("unit"))
                units[unit] = units.get(unit, 0) + 1

            for act in activities:
                target = act.get("targetSpecies")
                target = target.get("name") if isinstance(target, dict) else target
                if target:
                    key = str(target)
                    species[key] = species.get(key, 0) + 1

    print("\n--- summary ---")
    print(f"records                     {n}")
    print(f"non-monomer                 {multimer}")
    print(f"with target activities      {with_mic}")
    print(f"with haemolysis activities  {with_hc50}")
    print(f"terminally modified         {modified}")
    print(f"unusual amino acids         {unusual}")

    print("\nunits seen:")
    for unit, count in sorted(units.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {unit:24s} {count}")

    print("\ntop target species:")
    for name, count in sorted(species.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {name[:48]:50s} {count}")


def main() -> int:
    p = argparse.ArgumentParser(description="Download DBAASP peptide records")
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--out", type=Path, default=Path("data/dbaasp_raw.ndjson"))
    p.add_argument("--page-size", type=int, default=500)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--delay", type=float, default=0.0, help="per-worker pause between requests")
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--probe", action="store_true")
    p.add_argument("--summarise-only", action="store_true")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.summarise_only:
        summarise(args.out)
        return 0

    if args.probe:
        url = f"{args.base}/peptides?" + urlencode({"limit": 1, "offset": 0})
        print(f"GET {url}\n")
        print(json.dumps(get_json(url), indent=2)[:4000])
        return 0

    ids = collect_ids(args.base, args.page_size, 0.1, args.max_records)
    if not ids:
        print("No ids returned. Check --base.", file=sys.stderr)
        return 1

    fetch_records(args.base, ids, args.out, args.workers, args.delay)
    summarise(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
