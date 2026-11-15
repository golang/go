// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"
	"encoding/json"
	"internal/testenv"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/internal/logos/editmap"
)

func TestEditMapDescribesEject(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir := filepath.Join(testenv.GOROOT(t), "test", "logos", "run")
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	checked := 0
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".lgo") {
			continue
		}
		checked++
		t.Run(e.Name(), func(t *testing.T) {
			src, err := os.ReadFile(filepath.Join(dir, e.Name()))
			if err != nil {
				t.Fatal(err)
			}
			base := syntax.NewFileBase(filepath.Join(dir, e.Name()))
			file := ejectParse(t, base, src, syntax.LogosMode|syntax.CheckBranches)
			pkg, info := ejectCheck(t, file)
			edits := ejectLower(t, pkg, info, file, src)
			out, spans, _, err := ApplyEditsSegments(src, base, edits)
			if err != nil {
				t.Fatal(err)
			}
			checkSpans(t, src, out, spans)
		})
	}
	if checked == 0 {
		t.Fatalf("no .lgo files in %s", dir)
	}
}

func checkSpans(t *testing.T, src, out []byte, spans []editmap.EditSpan) {
	t.Helper()
	prev, pos := 0, 0
	for i, s := range spans {
		if s.Start < prev || s.End < s.Start || s.End > len(src) || s.Len < 0 {
			t.Fatalf("span %d %+v is out of order or out of range (previous end %d, source size %d)", i, s, prev, len(src))
		}
		gap := src[prev:s.Start]
		if pos > len(out) || !bytes.HasPrefix(out[pos:], gap) {
			t.Fatalf("source bytes %d:%d do not appear unchanged at ejected offset %d", prev, s.Start, pos)
		}
		pos += len(gap) + s.Len
		prev = s.End
	}
	if pos > len(out) || !bytes.Equal(src[prev:], out[pos:]) {
		t.Fatalf("the source tail from %d does not match the ejected tail from %d", prev, pos)
	}
}

func TestEjectWritesEditMap(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	dir := filepath.Join(testenv.GOROOT(t), "test", "logos", "run")
	const name = "viewas.lgo"
	src, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		t.Skipf("fixture %s unavailable: %v", name, err)
	}
	base := syntax.NewFileBase(filepath.Join(dir, name))
	file := ejectParse(t, base, src, syntax.LogosMode|syntax.CheckBranches)
	pkg, info := ejectCheck(t, file)
	edits := ejectLower(t, pkg, info, file, src)

	out := t.TempDir()
	srcs := map[*syntax.File][]byte{file: src}
	if err := Eject(out, []*syntax.File{file}, srcs, map[*syntax.File][]Edit{file: edits}); err != nil {
		t.Fatal(err)
	}
	ejected, err := os.ReadFile(filepath.Join(out, "viewas.go"))
	if err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(filepath.Join(out, "viewas.go"+editmap.Suffix))
	if err != nil {
		t.Fatal(err)
	}
	var m editmap.EditMap
	if err := json.Unmarshal(raw, &m); err != nil {
		t.Fatal(err)
	}
	if m.Version != editmap.Version || m.Source != name || m.Size != len(src) || m.Out != len(ejected) {
		t.Errorf("edit map header = %+v, want version %d source %s size %d out %d", m, editmap.Version, name, len(src), len(ejected))
	}
	if len(m.Edits) == 0 {
		t.Fatal("edit map has no spans for a fixture that lowers")
	}
	checkSpans(t, src, ejected, m.Edits)
}
