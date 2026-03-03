from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXTS = {'.py', '.html', '.css', '.js', '.md', '.txt'}
SKIP_DIRS = {'.git', '.venv', '__pycache__', '.idea'}

bad = []
question_run = "?" * 4
for path in ROOT.rglob('*'):
    if not path.is_file():
        continue
    if any(part in SKIP_DIRS for part in path.parts):
        continue
    if path.suffix.lower() not in EXTS:
        continue
    try:
        text = path.read_text(encoding='utf-8')
    except Exception as exc:
        bad.append((path, f'cannot decode utf-8: {exc}'))
        continue

    if question_run in text:
        bad.append((path, "contains four consecutive question marks"))
    if '' in text and any(m in text for m in ('', '', '', '', '', '')):
        # heuristic for classic mojibake fragments in Cyrillic UI text
        if '' not in text and '' not in text:
            bad.append((path, 'possible mojibake fragments'))

if bad:
    print('Encoding issues found:')
    for p, msg in bad:
        print(f'- {p.relative_to(ROOT)}: {msg}')
    sys.exit(1)

print('OK: no obvious encoding issues')

