// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

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

// TODO(rfindley): migrate this to testdata, as has been done in the standard library.
func TestGenericExport(t *testing.T) {
	const src = `
package generic

type Any any

type T[A, B any] struct { Left A; Right B }

var X T[int, string] = T[int, string]{1, "hi"}

func ToInt[P interface{ ~int }](p P) int { return int(p) }

var IntID = ToInt[int]

type G[C comparable] int

func ImplicitFunc[T ~int]() {}

type ImplicitType[T ~int] int

// Exercise constant import/export
const C1 = 42
const C2 int = 42
const C3 float64 = 42
`
	testExportSrc(t, []byte(src))
}

func testExportSrc(t *testing.T, src []byte) {
	// This package only handles gc export data.
	if runtime.Compiler != "gc" {
		t.Skipf("gc-built packages not available (compiler = %s)", runtime.Compiler)
	}

	// Test at both stages of the 1.18 export data format change.
	tests := []struct {
		name    string
		version int
	}{
		{"legacy generics", gcimporter.IExportVersionGenerics},
		{"go1.18", gcimporter.IExportVersionGo1_18},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
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
			data, err := iexport(fset, test.version, pkg)
			if err != nil {
				t.Fatal(err)
			}

			testPkgData(t, fset, test.version, pkg, data)
		})
	}
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

	skip := map[string]string{
		"issue48424.go": "go/types support missing", // TODO: need to implement this if #48424 is accepted
	}

	for _, entry := range list {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".go") {
			// For now, only consider standalone go files.
			continue
		}

		t.Run(entry.Name(), func(t *testing.T) {
			if reason, ok := skip[entry.Name()]; ok {
				t.Skip(reason)
			}

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
