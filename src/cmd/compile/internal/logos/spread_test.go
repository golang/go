// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// lowerSpreadSrc parses, checks, and lowers one inline .lgo file,
// returning the lowered tree print and the ejected text.
func lowerSpreadSrc(t *testing.T, src string) (printed, ejected string) {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "spread.lgo")
	if err := os.WriteFile(path, []byte(src), 0666); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var perrs []error
	f, err := syntax.ParseFile(path, func(err error) { perrs = append(perrs, err) }, nil, syntax.CheckBranches|syntax.LogosMode)
	if err != nil {
		t.Fatal(errors.Join(append(perrs, err)...))
	}
	imp := newTestImporter(t)
	info := &types2.Info{
		Defs:       make(map[*syntax.Name]types2.Object),
		Uses:       make(map[*syntax.Name]types2.Object),
		Types:      make(map[syntax.Expr]types2.TypeAndValue),
		Selections: make(map[*syntax.SelectorExpr]*types2.Selection),
		Scopes:     make(map[syntax.Node]*types2.Scope),
	}
	var cerrs []error
	conf := types2.Config{
		Importer: imp,
		Error:    func(err error) { cerrs = append(cerrs, err) },
		Logos:    true,
	}
	files := []*syntax.File{f}
	pkg, _ := conf.Check("main", files, info)
	if err := errors.Join(cerrs...); err != nil {
		t.Fatalf("type check: %v", err)
	}
	changed, edits := Lower(pkg, info, files, map[*syntax.File][]byte{f: data})
	if !changed {
		t.Fatalf("Lower reported no change for spread source")
	}
	out, err := ApplyEdits(data, f.Pos().FileBase(), edits[f])
	if err != nil {
		t.Fatal(err)
	}
	var buf strings.Builder
	if _, err := syntax.Fprint(&buf, f, 0); err != nil {
		t.Fatal(err)
	}
	return buf.String(), string(out)
}

// TestSpreadSelfUpdateTuple checks the self-update form lowers to one
// tuple assignment of the explicit fields with no struct temporary.
func TestSpreadSelfUpdateTuple(t *testing.T) {
	printed, ejected := lowerSpreadSrc(t, `package main

type State struct {
	A int
	B int
}

func swap(s State) State {
	s = State{A: s.B, B: s.A, ..s}
	return s
}
`)
	if !strings.Contains(printed, "s.A, s.B = s.B, s.A") {
		t.Errorf("lowered tree misses tuple assignment:\n%s", printed)
	}
	if strings.Contains(printed, "State{") {
		t.Errorf("lowered tree keeps a struct temporary:\n%s", printed)
	}
	if !strings.Contains(ejected, "s.A, s.B = s.B, s.A") {
		t.Errorf("ejected text misses tuple assignment:\n%s", ejected)
	}
}

// TestSpreadSourceEvaluatedOnce checks a call source is captured in one
// temporary and read for every copied field.
func TestSpreadSourceEvaluatedOnce(t *testing.T) {
	printed, ejected := lowerSpreadSrc(t, `package main

type State struct {
	A int
	B int
}

func get() State { return State{A: 1, B: 2} }

func build() State {
	return State{A: 9, ..get()}
}
`)
	for _, out := range []string{printed, ejected} {
		// One call site binds the source; the declaration line holds
		// the only other spelling.
		if n := strings.Count(out, ":= get()"); n != 1 {
			t.Errorf("source evaluated %d times; want 1:\n%s", n, out)
		}
		if n := strings.Count(out, "get()"); n != 2 {
			t.Errorf("found %d get() spellings; want 2 (declaration plus one call):\n%s", n, out)
		}
	}
}

// TestSpreadNoSurvivors checks no SpreadExpr survives lowering.
func TestSpreadNoSurvivors(t *testing.T) {
	printed, _ := lowerSpreadSrc(t, `package main

type State struct {
	A int
	B int
}

func build(s State) State {
	return State{B: 3, ..s}
}
`)
	if strings.Contains(printed, "..") {
		t.Errorf("lowered tree keeps spread syntax:\n%s", printed)
	}
}
