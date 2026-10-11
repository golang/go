#!/usr/bin/env python3
"""Copy the canonical GoX module into cmd/vendor reproducibly; never edit it there."""
from pathlib import Path
import shutil
root=Path(__file__).resolve().parents[2]
source=root/'gox'
dest=root/'src/cmd/vendor/github.com/alfu32/go-shared/gox'
if dest.exists(): shutil.rmtree(dest)
for p in source.rglob('*'):
    rel=p.relative_to(source)
    if p.is_file() and ((p.suffix=='.go' and not p.name.endswith('_test.go') and 'testdata' not in rel.parts and rel.parts[0]!='cmd') or p.name in ('LICENSE','UPSTREAM.md')):
        q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,q)
modules=root/'src/cmd/vendor/modules.txt'
lines=modules.read_text().splitlines()
lines=[s for s in lines if 'github.com/alfu32/go-shared/gox' not in s]
# Remove only our following explicit marker by rebuilding the prior section first.
text='\n'.join(lines)+'\n'
# Existing section is always appended below, including its marker.
if text.endswith('## explicit; go 1.26\n'):text=text[:-len('## explicit; go 1.26\n')]
text+='# github.com/alfu32/go-shared/gox v0.0.0 => ../../gox\n## explicit; go 1.26\n'
for package in ('','/internal/parser','/internal/scanner'):
    text+='github.com/alfu32/go-shared/gox'+package+'\n'
text+='# github.com/alfu32/go-shared/gox => ../../gox\n'
modules.write_text(text)
