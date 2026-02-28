// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package token

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"testing"
)

// equal returns nil if p and q describe the same file set;
// otherwise it returns an error describing the discrepancy.
func equal(p, q *FileSet) error {
	if p == q {
		// avoid deadlock if p == q
		return nil
	}

	// not strictly needed for the test
	p.mutex.Lock()
	q.mutex.Lock()
	defer q.mutex.Unlock()
	defer p.mutex.Unlock()

	if p.base != q.base {
		return fmt.Errorf("different bases: %d != %d", p.base, q.base)
	}

	if len(p.files) != len(q.files) {
		return fmt.Errorf("different number of files: %d != %d", len(p.files), len(q.files))
	}

	for i, f := range p.files {
		g := q.files[i]
		if f.set != p {
			return fmt.Errorf("wrong fileset for %q", f.name)
		}
		if g.set != q {
			return fmt.Errorf("wrong fileset for %q", g.name)
		}
		if f.name != g.name {
			return fmt.Errorf("different filenames: %q != %q", f.name, g.name)
		}
		if f.base != g.base {
			return fmt.Errorf("different base for %q: %d != %d", f.name, f.base, g.base)
		}
		if f.size != g.size {
			return fmt.Errorf("different size for %q: %d != %d", f.name, f.size, g.size)
		}
		for j, l := range f.lines {
			m := g.lines[j]
			if l != m {
				return fmt.Errorf("different offsets for %q", f.name)
			}
		}
		for j, l := range f.infos {
			m := g.infos[j]
			if l.Offset != m.Offset || l.Filename != m.Filename || l.Line != m.Line {
				return fmt.Errorf("different infos for %q", f.name)
			}
		}
	}

	// we don't care about .last - it's just a cache
	return nil
}

func checkSerialize(t *testing.T, p *FileSet) {
	var buf bytes.Buffer
	encode := func(x any) error {
		return gob.NewEncoder(&buf).Encode(x)
	}
	if err := p.Write(encode); err != nil {
		t.Errorf("writing fileset failed: %s", err)
		return
	}
	q := NewFileSet()
	decode := func(x any) error {
		return gob.NewDecoder(&buf).Decode(x)
	}
	if err := q.Read(decode); err != nil {
		t.Errorf("reading fileset failed: %s", err)
		return
	}
	if err := equal(p, q); err != nil {
		t.Errorf("filesets not identical: %s", err)
	}
}

func TestSerialization(t *testing.T) {
	p := NewFileSet()
	checkSerialize(t, p)
	// add some files
	for i := 0; i < 10; i++ {
		f := p.AddFile(fmt.Sprintf("file%d", i), p.Base()+i, i*100)
		checkSerialize(t, p)
		// add some lines and alternative file infos
		line := 1000
		for offs := 0; offs < f.Size(); offs += 40 + i {
			f.AddLine(offs)
			if offs%7 == 0 {
				f.AddLineInfo(offs, fmt.Sprintf("file%d", offs), line)
				line += 33
			}
		}
		checkSerialize(t, p)
	}
}
