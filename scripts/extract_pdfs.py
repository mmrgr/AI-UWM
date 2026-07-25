from __future__ import annotations

import argparse
from pathlib import Path

from pypdf import PdfReader


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract one UTF-8 text file per PDF.")
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/pdfs/text"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for pdf_path in args.pdfs:
        reader = PdfReader(pdf_path)
        pages = []
        for number, page in enumerate(reader.pages, start=1):
            pages.append(f"\n\n===== PAGE {number} =====\n\n{page.extract_text() or ''}")
        output_path = args.output_dir / f"{pdf_path.stem}.txt"
        output_path.write_text("".join(pages), encoding="utf-8")
        print(f"{pdf_path}: {len(reader.pages)} pages -> {output_path}")


if __name__ == "__main__":
    main()
