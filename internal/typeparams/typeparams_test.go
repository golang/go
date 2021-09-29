// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package typeparams_test

import (
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"strings"
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

	// Temporarily ignore API diff related to Environment, so that we can use a
	// transient alias in go/types to allow renaming this type without ever
	// breaking the x/tools builder.
	// TODO(rfindley): remove this
	var filteredChanges []apidiff.Change
	for _, change := range report.Changes {
		if strings.Contains(change.Message, "Environment") {
			continue
		}
		filteredChanges = append(filteredChanges, change)
	}
	report.Changes = filteredChanges
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
