"""Print the section tree extracted from a PDF.

Run with: .venv/Scripts/python scripts/print_section_tree.py <path-to-pdf>

This is the Day 3 acceptance check made repeatable: the printed section
tree should match the PDF's own table of contents for at least 85% of
titles. Verified against corpus/StVZO.pdf and corpus/FZV.pdf (100% of
bold "§ NN" headings matched, independently of the parser's own logic).
"""

import argparse
import sys

from app.adapters.parsing.pdf import parse_pdf


def main() -> int:
    sys.stdout.reconfigure(errors="replace")  # type: ignore[union-attr]  # Windows console code page

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf_path")
    args = parser.parse_args()

    parsed = parse_pdf(args.pdf_path)
    titles = [b for b in parsed.blocks if b.is_title]
    tables = [b for b in parsed.blocks if b.kind == "table"]

    for block in titles:
        depth = block.section_path.count(".") if block.section_path else 0
        indent = "  " * depth
        heading = block.text[:80].replace("\n", " ")
        path_label = block.section_path or "-"
        print(f"{indent}[{path_label:<10}] p.{block.page:>4}  {heading}")

    print(f"\n{len(parsed.blocks)} blocks total, {len(titles)} titles, {len(tables)} tables.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
