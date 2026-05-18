#!/usr/bin/env python3
"""Download PDF corpus files from a manifest.

Expected manifest path:
    data/corpus_manifest.json

Example manifest:
{
  "corpus_version": "academic-rag-demo-v1",
  "documents": [
    {
      "doc_id": "2005.11401",
      "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
      "source_url": "https://arxiv.org/pdf/2005.11401",
      "filename": "2005.11401.pdf"
    }
  ]
}

The script is intentionally simple:
- reads the manifest,
- downloads missing PDFs,
- skips files that already exist,
- optionally verifies sha256 if present,
- writes files into data/raw/pdfs by default.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "corpus_manifest.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "pdfs"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    documents = manifest.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("Manifest must contain a non-empty 'documents' list.")

    for i, doc in enumerate(documents, start=1):
        if not doc.get("doc_id"):
            raise ValueError(f"Document {i} is missing 'doc_id'.")
        if not doc.get("source_url"):
            raise ValueError(f"Document {i} is missing 'source_url'.")
        if not doc.get("filename"):
            raise ValueError(f"Document {i} is missing 'filename'.")

        filename = str(doc["filename"])
        if "/" in filename or "\\" in filename:
            raise ValueError(
                f"Document {i} has unsafe filename {filename!r}. "
                "Use a plain filename, not a path."
            )
        if not filename.lower().endswith(".pdf"):
            raise ValueError(f"Document {i} filename must end with .pdf: {filename}")

    return manifest


def download_file(url: str, destination: Path, timeout: int = 60) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "rag-mlops-corpus-downloader/1.0",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read()
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP error downloading {url}: {e.code} {e.reason}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"URL error downloading {url}: {e.reason}") from e

    if len(data) < 1024:
        raise RuntimeError(f"Downloaded file from {url} is suspiciously small.")

    if "pdf" not in content_type.lower() and not data.startswith(b"%PDF"):
        raise RuntimeError(
            f"Downloaded content from {url} does not look like a PDF "
            f"(Content-Type={content_type!r})."
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download PDF corpus from manifest.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Path to corpus manifest JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where PDFs should be stored.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Corpus version: {manifest.get('corpus_version', 'unknown')}")
    print(f"Manifest: {args.manifest}")
    print(f"Output directory: {output_dir}")
    print()

    downloaded = 0
    skipped = 0
    verified = 0

    for doc in manifest["documents"]:
        doc_id = str(doc["doc_id"])
        title = str(doc.get("title") or "")
        url = str(doc["source_url"])
        filename = str(doc["filename"])
        expected_sha256 = doc.get("sha256")

        destination = output_dir / filename

        print(f"Document: {doc_id}")
        if title:
            print(f"  Title: {title}")
        print(f"  URL: {url}")
        print(f"  File: {destination}")

        if destination.exists() and not args.force:
            print("  Status: exists, skipping download")
            skipped += 1
        else:
            print("  Status: downloading")
            download_file(url, destination)
            downloaded += 1

        if expected_sha256:
            actual_sha256 = sha256_file(destination)
            if actual_sha256.lower() != str(expected_sha256).lower():
                raise RuntimeError(
                    f"SHA256 mismatch for {destination}\n"
                    f"Expected: {expected_sha256}\n"
                    f"Actual:   {actual_sha256}"
                )
            print("  SHA256: verified")
            verified += 1

        print()

    print("Download summary:")
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped:    {skipped}")
    print(f"  Verified:   {verified}")
    print(f"  Total docs: {len(manifest['documents'])}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        raise SystemExit(1)