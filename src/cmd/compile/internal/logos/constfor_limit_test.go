// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"strings"
	"testing"
)

func boundaryStmts(t *testing.T, calls, lits int) []syntax.Stmt {
	t.Helper()
	var b strings.Builder
	b.WriteString("package p\nfunc f() {\n")
	for range calls {
		b.WriteString("f()\n")
	}
	for range lits {
		b.WriteString("0\n")
	}
	b.WriteString("}\n")
	fbase := syntax.NewFileBase("boundary.lgo")
	var errs []error
	f, _ := syntax.Parse(fbase, strings.NewReader(b.String()), func(err error) {
		errs = append(errs, err)
	}, nil, syntax.CheckBranches|syntax.LogosMode)
	if len(errs) > 0 || f == nil {
		t.Fatalf("parse boundary snippet: %v", errs)
	}
	for _, decl := range f.DeclList {
		if fn, ok := decl.(*syntax.FuncDecl); ok && fn.Name.Value == "f" {
			return fn.Body.List
		}
	}
	t.Fatal("boundary snippet has no func f")
	return nil
}

func countStmtsNodes(stmts []syntax.Stmt) int {
	n := 0
	for _, s := range stmts {
		syntax.Inspect(s, func(x syntax.Node) bool {
			n++
			return true
		})
	}
	return n
}

func TestComptimeNodeLimitDefaultBoundary(t *testing.T) {
	at := boundaryStmts(t, 682, 1)
	if n := countStmtsNodes(at); n != 4096 {
		t.Fatalf("accept list holds %d nodes, want exactly 4096", n)
	}
	fl := &fileLower{}
	before := base.Errors()
	fl.cfCountNodes(at, at[0].Pos())
	if base.Errors() != before {
		t.Fatalf("4096-node expansion rejected under default limit")
	}
}
