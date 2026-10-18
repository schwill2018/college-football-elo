"""Create an isolated HTML preview of the FBS tabs without requiring Jekyll."""
from pathlib import Path
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / ".validation/portfolio-staging"
destination = ROOT / ".validation/portfolio-preview"
destination.mkdir(exist_ok=True)
shutil.copytree(source / "assets", destination / "assets", dirs_exist_ok=True)
style = '''body{font:16px/1.6 system-ui,sans-serif;background:#fafafa;color:#34343c;margin:0}main{max-width:1100px;margin:36px auto;padding:0 24px}a{color:#165b9e}h1{font-size:2rem}h2{margin-top:2.5rem}.table-wrapper{overflow-x:auto;margin-bottom:1rem}table{border-collapse:collapse;width:100%;font-size:.92rem}th,td{padding:.6rem .7rem;border-bottom:1px solid #ddd;white-space:nowrap}th{background:#f1f2f4;text-align:left}.text-end{text-align:right}.text-muted{color:#666}.text-danger{color:#b21}.d-flex,.row{display:flex;gap:1rem;flex-wrap:wrap}.align-items-center{align-items:center}.mb-3{margin-bottom:1rem}.form-label{display:block}.form-select{padding:.5rem;font:inherit;min-width:220px}.btn{padding:.4rem .8rem;font:inherit;border:1px solid #888;border-radius:4px;background:transparent;cursor:pointer}.btn:disabled{opacity:.4;cursor:default}[hidden]{display:none!important}nav a{margin-right:16px}@media(max-width:600px){main{padding:0 12px}.row>div{width:100%}.form-select{width:100%}}'''
for name in ["fbs-elo", "fbs-elo-performance"]:
    text = (source / "_tabs" / f"{name}.md").read_text(encoding="utf-8")
    _, frontmatter, body = text.split("---", 2)
    title = re.search(r"^title: (.+)$", frontmatter, re.MULTILINE)[1]
    body = re.sub(r"{{\s*'([^']+)'\s*\|\s*relative_url\s*}}", r"\1", body)
    body = re.sub(r"{%\s*link _tabs/([^ ]+)\.md\s*%}", r"/\1/", body)
    body = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<p><a href="\2">\1</a></p>', body)
    page = destination / name
    page.mkdir(exist_ok=True)
    (page / "index.html").write_text(f'<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{style}</style><main><nav><a href="/fbs-elo/">🏈 FBS ELO</a><a href="/fbs-elo-performance/">FBS ELO Performance</a></nav><h1>{title}</h1>{body}</main></html>', encoding="utf-8")
print(destination)
