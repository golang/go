// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A test that ensures that ast.FileExports(file) and
// ast.FilterFile(file, ast.IsExported) produce the
// same trimmed AST given the same input file for all
// files under runtime.GOROOT().
//
// The test is here because it requires parsing, and the
// parser imports AST already (avoid import cycle).

package parser

import (
	"bytes"
	"go/ast"
	"go/printer"
	"go/token"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// For a short test run, limit the number of files to a few.
// Set to a large value to test all files under GOROOT.
const maxFiles = 10

type visitor struct {
	t *testing.T
	n int
}

func (v *visitor) VisitDir(path string, f *os.FileInfo) bool {
	return true
}

func str(f *ast.File, fset *token.FileSet) string {
	var buf bytes.Buffer
	printer.Fprint(&buf, fset, f)
	return buf.String()
}

func (v *visitor) VisitFile(path string, f *os.FileInfo) {
	// exclude files that clearly don't make it
	if !f.IsRegular() || len(f.Name) > 0 && f.Name[0] == '.' || !strings.HasSuffix(f.Name, ".go") {
		return
	}

	// should we stop early for quick test runs?
	if v.n <= 0 {
		return
	}
	v.n--

	fset := token.NewFileSet()

	// get two ASTs f1, f2 of identical structure;
	// parsing twice is easiest
	f1, err := ParseFile(fset, path, nil, ParseComments)
	if err != nil {
		v.t.Logf("parse error (1): %s", err)
		return
	}

	f2, err := ParseFile(fset, path, nil, ParseComments)
	if err != nil {
		v.t.Logf("parse error (2): %s", err)
		return
	}

	b1 := ast.FileExports(f1)
	b2 := ast.FilterFile(f2, ast.IsExported)
	if b1 != b2 {
		v.t.Errorf("filtering failed (a): %s", path)
		return
	}

	s1 := str(f1, fset)
	s2 := str(f2, fset)
	if s1 != s2 {
		v.t.Errorf("filtering failed (b): %s", path)
		return
	}
}

func TestFilter(t *testing.T) {
	filepath.Walk(runtime.GOROOT(), &visitor{t, maxFiles}, nil)
}
