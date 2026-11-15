// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func TestEraseConstViews(t *testing.T) {
	const src = `package erasetest

func Sum(xs const []int, n int) int {
	total := 0
	for _, v := range xs {
		total += v
	}
	_ = xs[:n]
	return total
}

type Holder struct {
	F const []int
}
`
	dir := t.TempDir()
	path := filepath.Join(dir, "erase.lgo")
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
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
		Types:            make(map[syntax.Expr]types2.TypeAndValue),
		Defs:             make(map[*syntax.Name]types2.Object),
		Uses:             make(map[*syntax.Name]types2.Object),
		Selections:       make(map[*syntax.SelectorExpr]*types2.Selection),
		LogosConstSlices: make(map[syntax.Expr]bool),
	}
	pkg, _ := conf.Check("erasetest", []*syntax.File{file}, info)
	if err := errors.Join(checkErrs...); err != nil {
		t.Fatalf("check: %v", err)
	}
	if len(info.LogosConstSlices) != 1 {
		t.Fatalf("recorded %d const slices, want 1", len(info.LogosConstSlices))
	}

	shared := &packageLower{info: info, tpkg: pkg}
	fl := &fileLower{
		pkg:   shared,
		file:  file,
		src:   data,
		base:  file.Pos().FileBase(),
		start: lineStarts(data),
	}
	fl.eraseConstViews()

	survived := 0
	syntax.Inspect(file, func(n syntax.Node) bool {
		if _, ok := n.(*syntax.ConstType); ok {
			survived++
		}
		return true
	})
	if survived != 0 {
		t.Errorf("%d ConstType nodes survived erasure", survived)
	}
	out, err := applySplices(data, slices.Clone(fl.splices))
	if err != nil {
		t.Fatalf("applySplices: %v", err)
	}
	if !strings.Contains(string(out), ":len(xs)") {
		t.Errorf("missing len max splice, eject text:\n%s", out)
	}
	if strings.Contains(string(out), "const") {
		t.Errorf("eject text still mentions const:\n%s", out)
	}
	maxed := false
	syntax.Inspect(file, func(n syntax.Node) bool {
		if s, ok := n.(*syntax.SliceExpr); ok && s.Index[2] != nil {
			maxed = true
		}
		return true
	})
	if !maxed {
		t.Error("recorded slice expression gained no max bound in the native tree")
	}
}

func TestEraseReadonlyMethod(t *testing.T) {
	const src = `package erasetest

type I interface {
	const Name() string
	Extra() int
}
`
	dir := t.TempDir()
	path := filepath.Join(dir, "readonly.lgo")
	if err := os.WriteFile(path, []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
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
		Types:            make(map[syntax.Expr]types2.TypeAndValue),
		Defs:             make(map[*syntax.Name]types2.Object),
		Uses:             make(map[*syntax.Name]types2.Object),
		Selections:       make(map[*syntax.SelectorExpr]*types2.Selection),
		LogosConstSlices: make(map[syntax.Expr]bool),
	}
	pkg, _ := conf.Check("erasetest", []*syntax.File{file}, info)
	if err := errors.Join(checkErrs...); err != nil {
		t.Fatalf("check: %v", err)
	}
	shared := &packageLower{info: info, tpkg: pkg}
	fl := &fileLower{
		file:  file,
		pkg:   shared,
		src:   data,
		base:  file.Pos().FileBase(),
		start: lineStarts(data),
	}
	fl.eraseConstViews()
	defer LogosClearErasedAnonReadonly()
	if len(erasedAnonReadonly) != 1 {
		t.Fatalf("recorded readonly methods = %d, want 1", len(erasedAnonReadonly))
	}
	kept := 0
	syntax.Inspect(file, func(n syntax.Node) bool {
		if f, ok := n.(*syntax.Field); ok && f.Const {
			kept++
		}
		return true
	})
	if kept != 0 {
		t.Fatalf("%d Const flags survived erasure, want 0", kept)
	}
	if len(fl.splices) != 1 {
		t.Fatalf("splices = %d, want 1", len(fl.splices))
	}
	sp := fl.splices[0]
	if sp.text != "" || sp.end-sp.start != len("const") {
		t.Fatalf("splice = [%d:%d]%q, want 5-byte deletion", sp.start, sp.end, sp.text)
	}
	if string(data[sp.start:sp.end]) != "const" {
		t.Fatalf("deleted span = %q, want %q", data[sp.start:sp.end], "const")
	}
	want := append(bytes.Clone(data[:sp.start]), data[sp.end:]...)
	out, err := applySplices(data, slices.Clone(fl.splices))
	if err != nil {
		t.Fatalf("applySplices: %v", err)
	}
	if string(out) != string(want) {
		t.Fatalf("eject text = %q, want %q", out, want)
	}
	LogosRestoreErasedAnonReadonly()
	if !erasedAnonReadonly[0].Const {
		t.Fatal("restore did not re-arm the readonly flag")
	}
}
