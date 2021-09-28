// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build typeparams && go1.18
// +build typeparams,go1.18

package typeparams_test

import (
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"testing"

	"golang.org/x/tools/internal/apidiff"
	"golang.org/x/tools/internal/testenv"
)

func TestAPIConsistency(t *testing.T) {
	testenv.NeedsGoBuild(t) // This is a lie. We actually need the source code.

	// The packages below exclude enabled_*.go, as typeparams.Enabled is
	// permitted to change between versions.
	old := typeCheck(t, []string{"common.go", "typeparams_go117.go"})
	new := typeCheck(t, []string{"common.go", "typeparams_go118.go"})

	report := apidiff.Changes(old, new)
	if len(report.Changes) > 0 {
		t.Errorf("API diff:\n%s", report)
	}
}

func typeCheck(t *testing.T, filenames []string) *types.Package {
	fset := token.NewFileSet()
	var files []*ast.File
	for _, name := range filenames {
		f, err := parser.ParseFile(fset, name, nil, 0)
		if err != nil {
			t.Fatal(err)
		}
		files = append(files, f)
	}
	conf := types.Config{
		Importer: importer.Default(),
	}
	pkg, err := conf.Check("", fset, files, nil)
	if err != nil {
		t.Fatal(err)
	}
	return pkg
}
