#!/usr/bin/env python3
"""Convert a markdown file to a well-formatted PDF with proper tables."""

import sys
import markdown
from weasyprint import HTML

def md_to_pdf(md_path, pdf_path):
    with open(md_path, "r") as f:
        md_text = f.read()

    # Convert markdown to HTML with table + fenced_code + codehilite extensions
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "toc", "sane_lists"],
    )

    # Wrap in a full HTML document with CSS styling
    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
@page {{
    size: letter;
    margin: 0.85in 0.75in;
    @bottom-center {{
        content: counter(page);
        font-size: 9pt;
        color: #666;
    }}
}}
body {{
    font-family: 'DejaVu Sans', 'Liberation Sans', 'Noto Sans', Arial, Helvetica, sans-serif;
    font-size: 10.5pt;
    line-height: 1.45;
    color: #1a1a1a;
    max-width: 100%;
}}
h1 {{
    font-size: 20pt;
    margin-top: 0.3in;
    margin-bottom: 0.12in;
    color: #111;
    border-bottom: 2.5pt solid #333;
    padding-bottom: 6pt;
    page-break-after: avoid;
}}
h2 {{
    font-size: 15pt;
    margin-top: 0.35in;
    margin-bottom: 0.1in;
    color: #222;
    border-bottom: 1pt solid #bbb;
    padding-bottom: 4pt;
    page-break-after: avoid;
}}
h3 {{
    font-size: 12.5pt;
    margin-top: 0.22in;
    margin-bottom: 0.06in;
    color: #333;
    page-break-after: avoid;
}}
h4 {{
    font-size: 11pt;
    margin-top: 0.15in;
    margin-bottom: 0.04in;
    color: #444;
    page-break-after: avoid;
}}
p {{
    margin: 0.06in 0;
    orphans: 3;
    widows: 3;
}}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 0.1in 0 0.15in 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}}
th, td {{
    border: 1px solid #999;
    padding: 4pt 6pt;
    text-align: left;
    vertical-align: top;
}}
th {{
    background-color: #e8e8e8;
    font-weight: bold;
    color: #222;
}}
tr:nth-child(even) {{
    background-color: #f6f6f6;
}}
code {{
    font-family: 'DejaVu Sans Mono', 'Liberation Mono', 'Noto Mono', 'Courier New', monospace;
    font-size: 9pt;
    background-color: #f0f0f0;
    padding: 1pt 3pt;
    border-radius: 2pt;
}}
pre {{
    background-color: #f4f4f4;
    border: 1px solid #ddd;
    border-radius: 3pt;
    padding: 8pt 10pt;
    font-size: 8.5pt;
    line-height: 1.35;
    overflow-x: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
    page-break-inside: avoid;
    margin: 0.08in 0 0.12in 0;
}}
pre code {{
    background-color: transparent;
    padding: 0;
    font-size: inherit;
}}
blockquote {{
    border-left: 3pt solid #999;
    margin: 0.08in 0 0.08in 0.15in;
    padding: 4pt 0 4pt 10pt;
    color: #444;
    font-style: italic;
    background-color: #fafafa;
}}
strong {{
    color: #111;
}}
hr {{
    border: none;
    border-top: 1pt solid #ccc;
    margin: 0.2in 0;
}}
ul, ol {{
    margin: 0.04in 0;
    padding-left: 0.25in;
}}
li {{
    margin-bottom: 2pt;
}}
</style>
</head>
<body>
{html_body}
</body>
</html>"""

    HTML(string=full_html).write_pdf(pdf_path)
    print(f"PDF written to: {pdf_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python md2pdf.py <input.md> [output.pdf]")
        sys.exit(1)
    md_path = sys.argv[1]
    pdf_path = sys.argv[2] if len(sys.argv) > 2 else md_path.rsplit(".", 1)[0] + ".pdf"
    md_to_pdf(md_path, pdf_path)
