// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build typeparams && go1.18
// +build typeparams,go1.18

package gcimporter_test

import (
	"bytes"
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"golang.org/x/tools/go/internal/gcimporter"
)

func TestGenericExport(t *testing.T) {
	const src = `
package generic

type T[A, B any] struct { Left A; Right B }

var X T[int, string] = T[int, string]{1, "hi"}

func ToInt[P interface{ ~int }](p P) int { return int(p) }

var IntID = ToInt[int]

type G[C comparable] int
`
	testExportSrc(t, []byte(src))
}

func testExportSrc(t *testing.T, src []byte) {
	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "g.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}
	conf := types.Config{
		Importer: importer.Default(),
	}
	pkg, err := conf.Check("", fset, []*ast.File{f}, nil)
	if err != nil {
		t.Fatal(err)
	}

	// export
	var b bytes.Buffer
	if err := gcimporter.IExportData(&b, fset, pkg); err != nil {
		t.Fatal(err)
	}

	testPkgData(t, fset, pkg, b.Bytes())
}

func TestImportTypeparamTests(t *testing.T) {
	// Check go files in test/typeparam.
	rootDir := filepath.Join(runtime.GOROOT(), "test", "typeparam")
	list, err := os.ReadDir(rootDir)
	if err != nil {
		t.Fatal(err)
	}

	if isUnifiedBuilder() {
		t.Skip("unified export data format is currently unsupported")
	}

	for _, entry := range list {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".go") {
			// For now, only consider standalone go files.
			continue
		}

		t.Run(entry.Name(), func(t *testing.T) {
			filename := filepath.Join(rootDir, entry.Name())
			src, err := os.ReadFile(filename)
			if err != nil {
				t.Fatal(err)
			}
			if !bytes.HasPrefix(src, []byte("// run")) && !bytes.HasPrefix(src, []byte("// compile")) {
				// We're bypassing the logic of run.go here, so be conservative about
				// the files we consider in an attempt to make this test more robust to
				// changes in test/typeparams.
				t.Skipf("not detected as a run test")
			}
			testExportSrc(t, src)
		})
	}
}
