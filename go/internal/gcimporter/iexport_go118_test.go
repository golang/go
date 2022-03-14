// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.18
// +build go1.18

package gcimporter_test

import (
	"bytes"
	"fmt"
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

func (T[P, Q]) m() {}

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

type Constraint[T any] interface {
       m(T)
}

// TODO(rfindley): revert to multiple blanks once the restriction on multiple
// blanks is removed from the type checker.
// type Blanks[_ any, _ Constraint[int]] int
// func (Blanks[_, _]) m() {}
type Blanks[_ any] int
func (Blanks[_]) m() {}
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
	version := gcimporter.IExportVersion
	data, err := iexport(fset, version, pkg)
	if err != nil {
		t.Fatal(err)
	}

	testPkgData(t, fset, version, pkg, data)
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

func TestRecursiveExport_Issue51219(t *testing.T) {
	const srca = `
package a

type Interaction[DataT InteractionDataConstraint] struct {
}

type InteractionDataConstraint interface {
	[]byte |
		UserCommandInteractionData
}

type UserCommandInteractionData struct {
	resolvedInteractionWithOptions
}

type resolvedInteractionWithOptions struct {
	Resolved Resolved
}

type Resolved struct {
	Users ResolvedData[User]
}

type ResolvedData[T ResolvedDataConstraint] map[uint64]T

type ResolvedDataConstraint interface {
	User | Message
}

type User struct{}

type Message struct {
	Interaction *Interaction[[]byte]
}
`

	const srcb = `
package b

import (
	"a"
)

// InteractionRequest is an incoming request Interaction
type InteractionRequest[T a.InteractionDataConstraint] struct {
	a.Interaction[T]
}
`

	const srcp = `
package p

import (
	"b"
)

// ResponseWriterMock mocks corde's ResponseWriter interface
type ResponseWriterMock struct {
	x b.InteractionRequest[[]byte]
}
`

	importer := &testImporter{
		src: map[string][]byte{
			"a": []byte(srca),
			"b": []byte(srcb),
			"p": []byte(srcp),
		},
		pkgs: make(map[string]*types.Package),
	}
	_, err := importer.Import("p")
	if err != nil {
		t.Fatal(err)
	}
}

// testImporter is a helper to test chains of imports using export data.
type testImporter struct {
	src  map[string][]byte         // original source
	pkgs map[string]*types.Package // memoized imported packages
}

func (t *testImporter) Import(path string) (*types.Package, error) {
	if pkg, ok := t.pkgs[path]; ok {
		return pkg, nil
	}
	src, ok := t.src[path]
	if !ok {
		return nil, fmt.Errorf("unknown path %v", path)
	}

	// Type-check, but don't return this package directly.
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path+".go", src, 0)
	if err != nil {
		return nil, err
	}
	conf := types.Config{
		Importer: t,
	}
	pkg, err := conf.Check(path, fset, []*ast.File{f}, nil)
	if err != nil {
		return nil, err
	}

	// Export and import to get the package imported from export data.
	exportdata, err := iexport(fset, gcimporter.IExportVersion, pkg)
	if err != nil {
		return nil, err
	}
	imports := make(map[string]*types.Package)
	fset2 := token.NewFileSet()
	_, pkg2, err := gcimporter.IImportData(fset2, imports, exportdata, pkg.Path())
	if err != nil {
		return nil, err
	}
	t.pkgs[path] = pkg2
	return pkg2, nil
}
