import os, re, argparse, json, hashlib
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, 'experiments', 'data_processed')
os.makedirs(PROC, exist_ok=True)

HEADER_KEYS = ('from|subject|organization|lines|reply-to|nntp-posting-host|keywords|'
    'x-disclaimer|x-newsreader|distribution|expires|newsgroups|followup-to|references|'
    'message-id|date|path|sender|in-reply-to|x-gated-by|article-i\\.d\\.|summary|originator|'
    'news-software|disclaimer|x-useragent|x-mailer|nf-id|nf-from|content-type|mime-version|'
    'archive-name|xref|supersedes|content-transfer-encoding|content-length')

def strip_headers(raw: str) -> str:
    parts = re.split(r'\n\s*\n', str(raw), maxsplit=1)
    body = parts[1] if (len(parts) == 2 and len(parts[0]) < 4000
                        and re.search(r'^[A-Za-z][\w.\-]*\s*:', parts[0], flags=re.MULTILINE)) else str(raw)
    return re.sub(r'^[ \t]*(' + HEADER_KEYS + r')\s*:.*$',
                  ' ', body, flags=re.MULTILINE | re.IGNORECASE)

def extract_subject(s: str) -> str:
    m = re.search(r'^Subject:\s*(.*)$', str(s), flags=re.MULTILINE)
    subject = m.group(1).strip() if m else ''
    return re.sub(r'^(Re:\s*)+', '', subject, flags=re.IGNORECASE)

def strip_signature(raw: str, max_tail_lines: int = 30) -> str:
    idxs = [m.start() for m in re.finditer(r'\n--\s*\n', raw)]
    if not idxs:
        return raw
    tail = raw[idxs[-1]:].splitlines()
    if len(tail) <= max_tail_lines:
        return raw[:idxs[-1]]
    return raw

def clean_text(s: str) -> str:
    subject = extract_subject(s)
    raw = strip_headers(str(s))
    raw = re.sub(r'^[ \t]*[>|:]+.*$', ' ', raw, flags=re.MULTILINE)  # quoted lines: >, |, :
    raw = re.sub(r'In article[\s\S]{0,200}?(writes|wrote|said|says)\b:?', ' ', raw, flags=re.IGNORECASE)  # bounded, allows newline/colon; colon optional (usenet 'says' variant)
    raw = re.sub(r'^\s*\S+\s+writes:\s*$', ' ', raw, flags=re.MULTILINE | re.IGNORECASE)
    raw = strip_signature(raw)
    raw = re.sub(r'\S+@\S+', ' ', raw)
    raw = re.sub(r'http\S+|www\.\S+', ' ', raw)
    text = (subject + ' ' + raw).lower()
    text = text.replace('c++', ' cplusplus ')
    text = re.sub(r'\$(\d[\d,]*)', r' dollar\1 ', text)  # $50 -> dollar50 so default token_pattern keeps it
    text = re.sub(r"(?<=[a-z0-9])['\u2019](?=[a-z])", '', text)  # don't -> dont, not don t
    text = re.sub(r'(?<=[a-z0-9])-(?=[a-z0-9])',
                  lambda m: '' if re.match(r'^\d', text[max(0, m.start()-3):m.start()]) or re.match(r'^\d', text[m.end():m.end()+3]) or (m.start() > 0 and m.end() < len(text) and (len(re.match(r'[a-z0-9]*', text[max(0,m.start()-3):m.start()]).group(0)) <= 2 or len(re.match(r'[a-z0-9]*', text[m.end():]).group(0)) <= 2)) else ' ',
                  text)  # fuse dx-50/x-11 style, split ordinary compounds (well-known -> well known)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def strip_uuencode(raw: str) -> tuple[str, int]:
    """Remove confirmed encoded line runs, never a whole labeled document.

    A uuencoded line declares its byte length in the first character. Require
    matching payload length/character range and at least eight consecutive
    lines, so repeated English words or short messages cannot trigger removal.
    Multipart BMP posts need not include the original 'begin' marker.
    """
    lines = raw.splitlines()
    def encoded(line):
        if not line or not all(32 <= ord(c) <= 96 for c in line[1:]):
            return False
        n = (ord(line[0]) - 32) & 63
        return 1 <= n <= 45 and len(line) == 1 + 4 * ((n + 2) // 3)
    result, removed, i = [], 0, 0
    while i < len(lines):
        if not encoded(lines[i]):
            result.append(lines[i]); i += 1
            continue
        j = i
        while j < len(lines) and encoded(lines[j]):
            j += 1
        if j - i >= 8:
            removed += j - i
        else:
            result.extend(lines[i:j])
        i = j
    return ('\n'.join(result) if removed else raw), removed

def clean_text_keepall(s: str) -> str:
    """v7: preserve every sample; only remove confirmed encoded content."""
    raw, _ = strip_uuencode(str(s))
    # Preserve the original header/body structure and all v6 normalization.
    return clean_text(raw)

def is_garbage(t: str) -> bool:
    """Frozen v6 diagnostic for the filter-review table and training ablation.

    This heuristic has confirmed false positives. The v7 preparation path
    records its flags but does not delete any training/validation/test rows.
    """
    toks = t.split()
    if len(toks) < 5:
        return True
    digit_ratio = sum(1 for w in toks if re.search(r'\d', w)) / len(toks)
    if digit_ratio > 0.25 and len(toks) <= 6000:
        return False  # box scores / stat tables / pinout diagrams: repetitive but informative
    if len(toks) > 6000:
        alpha = [w for w in toks if re.search(r'[a-z]', w)]
        if not alpha:
            return True
        from collections import Counter as _C
        top = _C(alpha).most_common(1)[0][1]
        if top / len(alpha) > 0.2 or len(set(alpha)) / len(alpha) < 0.12:
            return True  # multi-part uuencode/BMP dumps (roman bmp N of 14)
        return False
    from collections import Counter
    alpha = [w for w in toks if re.search(r'[a-z]', w)]
    if not alpha:
        return True
    top = Counter(alpha).most_common(1)[0][1]
    return (top / len(alpha) > 0.2) or (len(set(alpha)) / len(alpha) < 0.05 and len(toks) > 500)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare fixed raw split; retain every labeled/test row.')
    parser.add_argument('--output-dir', default=os.path.join(PROC, 'v7_keepall'))
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    out = Path(args.output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(os.path.join(BASE, 'data-Project1', 'train_data.csv'), keep_default_na=False)
    df['original_row'] = range(len(df))
    # Split raw rows before deciding or fitting any data-dependent operation.
    tri, vai = train_test_split(df.index, test_size=0.2, stratify=df['target'], random_state=args.seed)
    df['text_clean_v6'] = df['text'].astype(str).map(clean_text)
    df['legacy_remove'] = df['text_clean_v6'].map(is_garbage)
    df['encoded_lines_removed'] = df['text'].map(lambda s: strip_uuencode(strip_headers(s))[1])
    df['text_clean'] = df['text'].astype(str).map(clean_text_keepall)
    df['split'] = 'train'; df.loc[vai, 'split'] = 'val'
    df.loc[tri].to_csv(out / 'train.csv', index=False)
    df.loc[vai].to_csv(out / 'val.csv', index=False)
    te = pd.read_csv(os.path.join(BASE, 'data-Project1', 'test_data_unlabeled.csv'), keep_default_na=False)
    te['original_row'] = range(len(te))
    te['text_clean'] = te['text'].astype(str).map(clean_text_keepall)
    te.to_csv(out / 'test_clean.csv', index=False)
    check = df.loc[df['legacy_remove'], ['original_row','split','target','text_clean_v6','text_clean','encoded_lines_removed']].copy()
    check['tokens_v6'] = check.text_clean_v6.map(lambda t: len(t.split()))
    check['tokens_v7'] = check.text_clean.map(lambda t: len(t.split()))
    check['retained'] = True
    check_dir = Path(BASE) / 'experiments/runs/20261008_revision/20261008_revision_manifest/checks'
    check_dir.mkdir(parents=True, exist_ok=True)
    check.to_csv(check_dir / 'filter_review_v7.csv', index=False)
    summary = {'version':'v7_keepall','split_seed':args.seed,'raw_rows':len(df),'train_rows':len(tri),'val_rows':len(vai),'test_rows':len(te),'rows_deleted':0,'legacy_flagged_rows':int(df.legacy_remove.sum()),'documents_with_encoded_lines_removed':int((df.encoded_lines_removed>0).sum()),'encoded_lines_removed':int(df.encoded_lines_removed.sum()),'empty_train_text':int((df.loc[tri,'text_clean']=='').sum()),'empty_val_text':int((df.loc[vai,'text_clean']=='').sum()),'ax_v6':sum(t.split().count('ax') for t in df.text_clean_v6),'ax_v7':sum(t.split().count('ax') for t in df.text_clean),'source_sha256':{p:hashlib.sha256(Path(BASE,p).read_bytes()).hexdigest() for p in ['data-Project1/train_data.csv','data-Project1/test_data_unlabeled.csv']},'val_original_rows_sha256':hashlib.sha256(','.join(map(str,df.loc[vai,'original_row'])).encode()).hexdigest()}
    (out / 'data_config.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
