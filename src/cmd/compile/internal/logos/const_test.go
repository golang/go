// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// TestEraseConstViews checks and erases a small .lgo snippet directly:
// no ConstType survives in the tree, the `const ` prefixes gain eject
// splices, and a recorded non-constant slice bound gains a len max.
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
