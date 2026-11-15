// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"errors"
	"os"
	"path/filepath"
	"slices"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

const groupConstViewSrc = `package main

import "context"

type Node struct {
	X int
}

func sum(n const *Node) error {
	return nil
}

func run(ctx context.Context) error {
	var p const *Node = &Node{X: 1}
	var q *Node = &Node{X: 2}
	go group (ctx) {
		go sum(p)
	}
	_ = q
	return nil
}

func main() {
}
`

type groupConstViewRun struct {
	file   *syntax.File
	decls  []syntax.Pos
	p      syntax.Pos
	q      syntax.Pos
	goStmt syntax.Pos
}

func lowerGroupConstView(t *testing.T, srcText string) groupConstViewRun {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "groupview.lgo")
	if err := os.WriteFile(path, []byte(srcText), 0666); err != nil {
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
	var run groupConstViewRun
	run.file = f
	syntax.Inspect(f, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.VarDecl:
			for _, name := range n.NameList {
				switch name.Value {
				case "p":
					run.p = name.Pos()
				case "q":
					run.q = name.Pos()
				}
			}
		case *syntax.CallStmt:
			if n.Tok == syntax.Go && !run.goStmt.IsKnown() {
				run.goStmt = n.Pos()
			}
		}
		return true
	})
	if !run.p.IsKnown() || !run.q.IsKnown() || !run.goStmt.IsKnown() {
		t.Fatalf("fixture lost its anchor positions: p=%v q=%v go=%v", run.p, run.q, run.goStmt)
	}
	ConstViewDecls()
	Lower(pkg, info, files, map[*syntax.File][]byte{f: data})
	run.decls = ConstViewDecls()
	return run
}

func groupConstViewNames(f *syntax.File) []syntax.Pos {
	var names []syntax.Pos
	syntax.Inspect(f, func(n syntax.Node) bool {
		if name, ok := n.(*syntax.Name); ok {
			names = append(names, name.Pos())
		}
		return true
	})
	return names
}

func TestGroupConstViewDecls(t *testing.T) {
	run := lowerGroupConstView(t, groupConstViewSrc)

	t.Run("ConstViewDeclRecorded", func(t *testing.T) {
		if !slices.Contains(run.decls, run.p) {
			t.Errorf("declaration of p at %s missing from the side table %v", run.p, run.decls)
		}
	})
	t.Run("PlainPointerNotRecorded", func(t *testing.T) {
		if slices.Contains(run.decls, run.q) {
			t.Errorf("declaration of q at %s must not be in the side table", run.q)
		}
	})
	t.Run("CaptureTemporaryRecorded", func(t *testing.T) {
		if !slices.Contains(run.decls, run.goStmt) {
			t.Errorf("capture temporary of go sum(p) at %s missing from the side table %v", run.goStmt, run.decls)
		}
	})
	t.Run("PositionsSurviveRewrites", func(t *testing.T) {
		if len(run.decls) == 0 {
			t.Fatal("side table is empty")
		}
		names := groupConstViewNames(run.file)
		for _, d := range run.decls {
			if !slices.Contains(names, d) {
				t.Errorf("recorded position %s names no node of the lowered tree", d)
			}
		}
	})
}
