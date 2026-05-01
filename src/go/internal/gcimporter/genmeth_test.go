// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.genericmethods

package gcimporter_test

import (
	"internal/testenv"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"go/ast"
	"go/parser"
	"go/token"
	"go/types"

	. "go/internal/gcimporter"
)

func TestGenMeth(t *testing.T) {
	testenv.MustHaveGoBuild(t)

	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	tmpdir := t.TempDir()
	testoutdir := filepath.Join(tmpdir, "testdata")
	if err := os.Mkdir(testoutdir, 0700); err != nil {
		t.Fatalf("making output dir: %v", err)
	}

	compile(t, "testdata", "genmeth.go", testoutdir, nil)

	fset := token.NewFileSet()

	genmeth, err := Import(fset, make(map[string]*types.Package), "./testdata/genmeth", tmpdir, nil)
	if err != nil {
		t.Fatal(err)
	}

	check := func(pkgname, src string, imports importMap) (*types.Package, error) {
		f, err := parser.ParseFile(fset, "genmeth.go", src, 0)
		if err != nil {
			return nil, err
		}
		config := &types.Config{
			Importer: imports,
		}
		return config.Check(pkgname, fset, []*ast.File{f}, nil)
	}

	const pSrc = `package p

import "genmeth"

func _() {
	var ex func(int) genmeth.List[int]
	var fl func(genmeth.List[int]) genmeth.List[int]

	var l genmeth.List[int]
	l = l.Map(ex).FlatMap(fl)

	var bl genmeth.BiList[int, any]
	bl = bl.MapKeys(ex).Flip().FlatMapValues(fl).Flip()

	var id func(int) int

	var op genmeth.Option[int]
	var _ int = op.MapIfPresent(id).Get()

	var ol genmeth.OrderedList[int]
	var _ int = ol.Min().Get()

	var b genmeth.Box[int]
	b.Set(42)
	var _ int = b.Get()
}
`

	importer := importMap{
		"genmeth": genmeth,
	}
	if _, err := check("p", pSrc, importer); err != nil {
		t.Errorf("Check failed: %v", err)
	}
}
