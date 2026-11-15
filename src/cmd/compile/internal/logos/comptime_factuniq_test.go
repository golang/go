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
	"cmd/internal/goobj"
)

func TestComptimeWriterKeepsDistinctKeys(t *testing.T) {
	const src = `package duptest

import "comptime"

type Pair struct {
	Name string
	Age  int
}

func Dup[T any](v T) int {
	n := 0
	const for _, f := range comptime.Fields[T]() {
		n += f.Index
	}
	return n
}

type DupBox struct {
	N int
}

func (DupBox) Dup[T any](v T) int {
	n := 0
	const for _, f := range comptime.Fields[Pair]() {
		n += f.Index
	}
	return n
}

type DupCarrier struct {
	N int
}

func (DupCarrier) Dup[T any](v T) int {
	n := 0
	const for _, f := range comptime.Fields[Pair]() {
		n += f.Index
	}
	return n
}
`
	dir := t.TempDir()
	path := filepath.Join(dir, "dup.lgo")
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var perrs []error
	file, err := syntax.ParseFile(path, func(err error) { perrs = append(perrs, err) }, nil, syntax.CheckBranches|syntax.LogosMode)
	if err != nil {
		t.Fatal(errors.Join(append(perrs, err)...))
	}
	imp := newTestImporter(t)
	var checkErrs []error
	conf := types2.Config{
		Importer: imp,
		Error:    func(err error) { checkErrs = append(checkErrs, err) },
		Logos:    true,
	}
	info := &types2.Info{
		Defs:       make(map[*syntax.Name]types2.Object),
		Uses:       make(map[*syntax.Name]types2.Object),
		Types:      make(map[syntax.Expr]types2.TypeAndValue),
		Selections: make(map[*syntax.SelectorExpr]*types2.Selection),
		Scopes:     make(map[syntax.Node]*types2.Scope),
		Instances:  make(map[*syntax.Name]types2.Instance),
	}
	files := []*syntax.File{file}
	pkg, _ := conf.Check("duptest", files, info)
	if err := errors.Join(checkErrs...); err != nil {
		t.Fatalf("type check: %v", err)
	}
	shared := &packageLower{info: info, tpkg: pkg}
	fl := &fileLower{pkg: shared, file: file, src: data, orig: len(data), start: lineStarts(data)}
	fl.srcMap = origBuf(data)
	exp := &cfExpander{
		byObj:   make(map[*types2.Func]*cfGenericDef),
		byDecl:  make(map[*syntax.FuncDecl]*cfGenericDef),
		foreign: make(map[string]*cfRecord),
		memo:    make(map[string]string),
		pkgs:    make(map[string]*types2.Package),
	}
	got := fl.registerComptime(exp, &facts.Facts{})
	counts := make(map[string]int)
	for _, c := range got.Comptimes {
		counts[c.Object]++
	}
	for obj, n := range counts {
		if n > 1 {
			t.Errorf("writer emitted %d comptime records for %q", n, obj)
		}
	}
	for _, obj := range []string{"Dup", "DupBox.Dup", "DupCarrier.Dup"} {
		if counts[obj] != 1 {
			t.Errorf("comptime records for %q = %d, want 1", obj, counts[obj])
		}
	}
	fp := goobj.FingerprintType{0, 1, 2, 3, 4, 5, 6, 7}
	if _, err := facts.Decode(facts.Encode(fp, got)); err != nil {
		t.Fatalf("Decode(Encode(writer output)) failed: %v", err)
	}
}
