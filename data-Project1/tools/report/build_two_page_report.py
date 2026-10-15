"""Export the current Markdown as two plain pages, preserving all report content."""
from pathlib import Path
import hashlib
import html
import json
import re

from PIL import Image as PillowImage
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

BASE = Path(__file__).resolve().parents[3]
SOURCE = BASE / '实验报告.md'
OUT = BASE / 'experiments/reports/20261014_report/实验报告.pdf'
pdfmetrics.registerFont(TTFont('Body', 'C:/Windows/Fonts/simsun.ttc', subfontIndex=0))
pdfmetrics.registerFont(TTFont('Bold', 'C:/Windows/Fonts/simhei.ttf'))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='Bold', italic='Body', boldItalic='Bold')
WIDTH = A4[0] - 68
body = ParagraphStyle('body', fontName='Body', fontSize=10, leading=13,
                      spaceAfter=4, wordWrap='CJK', textColor=colors.black)
heading = ParagraphStyle('heading', parent=body, fontName='Bold', fontSize=11.2,
                         leading=15, spaceBefore=3, spaceAfter=4, keepWithNext=True)
title = ParagraphStyle('title', parent=heading, fontSize=15, leading=20, spaceAfter=6)
caption = ParagraphStyle('caption', parent=body, fontSize=8.4, leading=10.5, spaceAfter=4)
cell = ParagraphStyle('cell', parent=body, fontSize=9, leading=11, spaceAfter=0)


def inline(text):
    text = html.escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    return re.sub(r'\[([^]]+)\]\((https://[^)]+)\)', r'<link href="\2" color="#000000">\1</link>', text)


def plain(text):
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    return re.sub(r'\[([^]]+)\]\((https://[^)]+)\)', r'\1', text)


source_bytes = SOURCE.read_bytes()
lines = source_bytes.decode('utf-8-sig').splitlines()
story, expected, figures = [], [], []
i = 0
while i < len(lines):
    line = lines[i].strip()
    if not line:
        i += 1
        continue
    if line == '<!-- PAGEBREAK -->':
        story.append(PageBreak())
        i += 1
        continue
    if line.startswith('# '):
        story.append(Paragraph(inline(line[2:]), title))
        expected.append(line[2:])
        i += 1
        continue
    if line.startswith('> '):
        story.append(Paragraph(inline(line[2:]), caption))
        expected.append(line[2:])
        i += 1
        continue
    if re.fullmatch(r'\*\*[^*]+\*\*', line):
        text = line[2:-2]
        if text == '三、训练过程与综合评估':
            story.append(PageBreak())
        story.append(Paragraph(inline(text), heading))
        expected.append(text)
        i += 1
        continue
    if line.startswith('|'):
        data = []
        while i < len(lines) and lines[i].strip().startswith('|'):
            parts = [part.strip() for part in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r':?-+:?', part) for part in parts):
                data.append([Paragraph(inline(part), cell) for part in parts])
                expected.extend(parts)
            i += 1
        weights = [.26, .25, .17, .17, .15] if data[0][0].getPlainText() == '评估范围' else [.16, .19, .19, .29, .17]
        table = Table(data, colWidths=[WIDTH * x for x in weights])
        table.setStyle(TableStyle([
            ('LINEBELOW', (0, 0), (-1, 0), .6, colors.black),
            ('LINEBELOW', (0, 1), (-1, -1), .3, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ('TOPPADDING', (0, 0), (-1, -1), 2.5), ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
        ]))
        story.extend([table, Spacer(1, 4)])
        continue
    match = re.fullmatch(r'!\[([^]]*)\]\((.+)\)', line)
    if match:
        path = Path(match.group(2))
        if not path.is_absolute():
            path = BASE / path
        with PillowImage.open(path) as picture:
            iw, ih = picture.size
        story.extend([Image(str(path), width=WIDTH, height=WIDTH * ih / iw), Paragraph(inline(match.group(1)), caption)])
        figures.append(path.name)
        expected.append(match.group(1))
        i += 1
        continue
    story.append(Paragraph(inline(line), body))
    expected.append(line)
    i += 1

OUT.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=34, rightMargin=34,
                        topMargin=29, bottomMargin=29, title=lines[0].removeprefix('# '))
doc.build(story)
reader = PdfReader(OUT)
normalize = lambda text: re.sub(r'\s+', '', text)
assert len(reader.pages) == 2, f'Expected two pages, got {len(reader.pages)}'
assert len(figures) == 3
assert normalize(''.join(page.extract_text() or '' for page in reader.pages)) == normalize(''.join(plain(text) for text in expected))
assert SOURCE.read_bytes() == source_bytes
print(json.dumps({'pages': len(reader.pages), 'figures': figures, 'all_markdown_text_included': True,
                  'markdown_sha256': hashlib.sha256(source_bytes).hexdigest(),
                  'headers_footers_page_numbers': False, 'text_color': 'black'}, ensure_ascii=False))
