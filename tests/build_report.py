"""把 report.md 排版成 report.pdf (A4), 并打印页数.
依赖: pandoc, 以及 Chrome 或 Edge (用无头模式打印 PDF); 页数检查需要 pypdf.
运行: python -X utf8 tests/build_report.py"""
import os, shutil, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MD = os.path.join(ROOT, 'report.md')
HTML = os.path.join(ROOT, 'results', 'report.html')
PDF = os.path.join(ROOT, 'report.pdf')
CSS = os.path.join(ROOT, 'tests', 'report.css')

BROWSERS = [
    r'C:\Program Files\Google\Chrome\Application\chrome.exe',
    r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe',
    shutil.which('chrome') or '', shutil.which('msedge') or '', shutil.which('chromium') or '',
]

os.makedirs(os.path.dirname(HTML), exist_ok=True)
# --resource-path 让图片按 lab2 根目录解析; --embed-resources 把图片和样式写进 HTML
subprocess.run(['pandoc', MD, '-o', HTML, '--standalone', '--embed-resources',
                '--resource-path', ROOT, '--css', CSS, '--metadata', 'lang=zh-CN'],
               check=True, cwd=ROOT)
browser = next((b for b in BROWSERS if b and os.path.exists(b)), None)
if browser is None:
    sys.exit('没有找到 Chrome/Edge, 已生成 HTML: ' + HTML)
subprocess.run([browser, '--headless=new', '--disable-gpu', '--no-pdf-header-footer',
                f'--print-to-pdf={PDF}', 'file:///' + HTML.replace('\\', '/')],
               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    from pypdf import PdfReader
    print('pages:', len(PdfReader(PDF).pages))
except ImportError:
    pass
print('saved', PDF)
