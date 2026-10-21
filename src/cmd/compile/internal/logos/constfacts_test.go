// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"cmd/compile/internal/logos/facts"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// TestEmitConstFacts checks a small .lgo package, calls EmitConstFacts
// directly, and asserts the const lines plus the ReceiverMutated and
// ParamMutated flags, including propagation to a wrapper and merging
// with pre-existing facts.
func TestEmitConstFacts(t *testing.T) {
	const src = `package factstest

type Box struct {
	Items const []int
	Name  string
}

func Sum(xs const []int) int {
	total := 0
	for _, v := range xs {
		total += v
	}
	return total
}

func AppendIt(xs []int, v int) []int {
	return append(xs, v)
}

func Double(xs []int) []int {
	return AppendIt(AppendIt(xs, 1), 2)
}

func (b *Box) Rename(n string) {
	b.Name = n
}

func (b *Box) Count() int {
	return len(b.Items)
}
`
	dir := t.TempDir()
	path := filepath.Join(dir, "facts.lgo")
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	file, err := syntax.ParseFile(path, func(err error) { t.Fatalf("parse: %v", err) }, nil, syntax.LogosMode)
	if err != nil {
		t.Fatal(err)
	}
	imp := newTestImporter(t)
	var checkErrs []error
	conf := types2.Config{
		Importer: imp,
		Error:    func(err error) { checkErrs = append(checkErrs, err) },
		Logos:    true,
	}
	info := &types2.Info{
		Types:      make(map[syntax.Expr]types2.TypeAndValue),
		Defs:       make(map[*syntax.Name]types2.Object),
		Uses:       make(map[*syntax.Name]types2.Object),
		Selections: make(map[*syntax.SelectorExpr]*types2.Selection),
	}
	pkg, _ := conf.Check("factstest", []*syntax.File{file}, info)
	if err := errors.Join(checkErrs...); err != nil {
		t.Fatalf("check: %v", err)
	}

	seed := &facts.Facts{Fns: []facts.FnFact{{Object: "AppendIt", ReturnsNonNil: true}}}
	got := EmitConstFacts(pkg, info, []*syntax.File{file}, seed)
	if got != seed {
		t.Fatal("EmitConstFacts did not return the passed facts")
	}

	consts := make(map[string]bool)
	for _, c := range got.Consts {
		consts[c.Object+" "+c.Path] = true
	}
	for _, want := range []string{"Sum P0", "Box FItems"} {
		if !consts[want] {
			t.Errorf("missing const line %q (have %v)", want, got.Consts)
		}
	}

	fns := make(map[string]facts.FnFact)
	for _, fn := range got.Fns {
		fns[fn.Object] = fn
	}
	checkFn := func(object, mask string, recv bool) {
		t.Helper()
		fn, ok := fns[object]
		if mask == "" && !recv {
			if ok {
				t.Errorf("%s unexpectedly has facts %+v", object, fn)
			}
			return
		}
		if !ok {
			t.Errorf("missing fn facts for %s", object)
			return
		}
		if fn.ParamMutated != mask {
			t.Errorf("%s ParamMutated = %q, want %q", object, fn.ParamMutated, mask)
		}
		if fn.ReceiverMutated != recv {
			t.Errorf("%s ReceiverMutated = %v, want %v", object, fn.ReceiverMutated, recv)
		}
	}
	checkFn("AppendIt", "1", false)
	checkFn("Double", "1", false)
	checkFn("(*Box).Rename", "", true)
	checkFn("Sum", "", false)
	checkFn("(*Box).Count", "", false)
	if fn := fns["AppendIt"]; !fn.ReturnsNonNil {
		t.Error("AppendIt lost its seeded ReturnsNonNil flag")
	}
}
