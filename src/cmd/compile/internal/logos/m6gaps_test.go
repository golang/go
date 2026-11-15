// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

func lowerSnippetSrc(t *testing.T, name, src string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, name)
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
		Instances:  make(map[*syntax.Name]types2.Instance),
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
		t.Fatalf("Lower reported no change for %s source", name)
	}
	out, err := ApplyEdits(data, f.Pos().FileBase(), edits[f])
	if err != nil {
		t.Fatal(err)
	}
	return string(out)
}
