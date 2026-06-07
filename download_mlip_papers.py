#!/usr/bin/env python3
"""
download_mlip_papers.py

Batch-downloads a set of open-access arXiv PDFs of the MLIP foundation-model
literature into a single folder.

After requesting Claude to search the literature for arXiv MLIP papers, 
I requested it to create this download script for the sake of this example.

- Script uses standard library, no `pip install` needed.
- Files are named "<topic>_<arxiv_id>.pdf" so the filename itself is metadata.
- Resumable: already-downloaded files are skipped.
- Polite: waits a few seconds between requests (arXiv asks for this).

Usage:
    python download_mlip_papers.py                 # -> ./mlip_pdfs/
    python download_mlip_papers.py /path/to/folder # custom output folder
"""

import os
import sys
import time
import urllib.request
import urllib.error

# Be a good citizen: arXiv throttles/blocks default urllib user-agents,
# and asks bulk downloaders to space out requests.
USER_AGENT = "mlip-rag-papers-downloader/1.0 (mailto:you@example.com)"
DELAY_SECONDS = 3          # pause between downloads
TIMEOUT_SECONDS = 60
MAX_RETRIES = 3

# arxiv_id : friendly_name  (friendly name becomes part of the filename)
PAPERS = {
    # --- Section A: foundation / universal models ---
    "2401.00096": "MACE-MP-0",
    "2312.15211": "MACE-OFF23",
    "2405.04967": "MatterSim",
    "2410.22570": "Orb",
    "2504.06231": "Orb-v3",
    "2402.03789": "SevenNet",
    "2302.14231": "CHGNet",
    "2202.02450": "M3GNet",
    "2306.12059": "EquiformerV2",
    "2502.12147": "eSEN",
    "2410.12771": "OMat24",
    "2505.08762": "OMol25",
    "2506.23971": "UMA",
    "2508.17936": "GRACE",
    "2503.14118": "PET-MAD",
    "2312.15492": "DPA-2",
    "2506.01686": "DPA-3",
    "1610.08935": "ANI-1",
    "2312.03687": "MatterGen",
    # --- Section B: application & benchmark papers ---
    "2308.14920": "Matbench-Discovery",
    "2509.20630": "MLIP-Arena",
    "2503.04070": "MatPES",
    "2403.05729": "uMLIP-systematic-assessment",
    "2412.16551": "Phonons-benchmark",
    "2408.00755": "Thermal-conductivity",
    "2403.04217": "Surfaces-assessment",
    "2508.21663": "Cleavage-surface-benchmark",
    "2010.09990": "OC20",
    "2206.08917": "OC22",
    "2507.11806": "MOFSimBench",
    "2509.06719": "MOF-adsorption-screening",
    "2502.09970": "Solid-ion-conductors",
    "2603.20183": "Electrolyte-solvation-OMol25",  # recent: verify ID if it 404s
    "2511.16569": "LiPS-benchmark",
    "2512.03642": "Migration-barriers",
    "2405.07105": "Systematic-softening-finetuning",
    "2506.21935": "Finetuning-tutorial",
    "2506.07401": "Finetuning-performance-study",
    "2511.05337": "Finetuning-unifies-architectures",
    "2510.22999": "Elastic-property-benchmark",
    "2501.07155": "AlphaNet",
    "2503.05771": "HIENet",
    "2504.19578": "LAMBench",
}


def download(arxiv_id, name, out_dir):
    fname = f"{name}_{arxiv_id}.pdf"
    fpath = os.path.join(out_dir, fname)
    if os.path.exists(fpath) and os.path.getsize(fpath) > 0:
        print(f"  skip (exists): {fname}")
        return "skipped"

    url = f"https://arxiv.org/pdf/{arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as r:
                data = r.read()
            if not data.startswith(b"%PDF"):
                print(f"  WARN: {arxiv_id} did not return a PDF (got HTML?).")
                return "failed"
            with open(fpath, "wb") as fh:
                fh.write(data)
            print(f"  ok: {fname}  ({len(data) // 1024} KB)")
            return "ok"
        except urllib.error.HTTPError as e:
            print(f"  HTTP {e.code} for {arxiv_id} (attempt {attempt})")
            if e.code == 404:
                return "failed"   # bad ID, no point retrying
        except Exception as e:
            print(f"  error for {arxiv_id} (attempt {attempt}): {e}")
        time.sleep(DELAY_SECONDS * attempt)  # back off
    return "failed"


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "mlip_pdfs"
    os.makedirs(out_dir, exist_ok=True)
    print(f"Downloading {len(PAPERS)} PDFs into: {os.path.abspath(out_dir)}\n")

    results = {"ok": 0, "skipped": 0, "failed": []}
    for i, (arxiv_id, name) in enumerate(PAPERS.items(), 1):
        print(f"[{i}/{len(PAPERS)}] {name} ({arxiv_id})")
        status = download(arxiv_id, name, out_dir)
        if status == "ok":
            results["ok"] += 1
            time.sleep(DELAY_SECONDS)        # only pause after a real fetch
        elif status == "skipped":
            results["skipped"] += 1
        else:
            results["failed"].append(f"{name} ({arxiv_id})")

    print("\n" + "=" * 50)
    print(f"Downloaded: {results['ok']}   Skipped: {results['skipped']}   "
          f"Failed: {len(results['failed'])}")
    if results["failed"]:
        print("Failed (grab these manually or re-run to retry):")
        for f in results["failed"]:
            print(f"   - {f}")

if __name__ == "__main__":
    main()