// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gcimporter_test

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"strings"
	"testing"

	"golang.org/x/sync/errgroup"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/gcimporter"
	"golang.org/x/tools/internal/testenv"
)

// TestStd type-checks the standard library using shallow export data.
func TestShallowStd(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode; too slow (https://golang.org/issue/14113)")
	}
	testenv.NeedsTool(t, "go")

	// Load import graph of the standard library.
	// (No parsing or type-checking.)
	cfg := &packages.Config{
		Mode: packages.NeedImports |
			packages.NeedName |
			packages.NeedFiles | // see https://github.com/golang/go/issues/56632
			packages.NeedCompiledGoFiles,
		Tests: false,
	}
	pkgs, err := packages.Load(cfg, "std")
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if len(pkgs) < 200 {
		t.Fatalf("too few packages: %d", len(pkgs))
	}

	// Type check the packages in parallel postorder.
	done := make(map[*packages.Package]chan struct{})
	packages.Visit(pkgs, nil, func(p *packages.Package) {
		done[p] = make(chan struct{})
	})
	packages.Visit(pkgs, nil,
		func(pkg *packages.Package) {
			go func() {
				// Wait for all deps to be done.
				for _, imp := range pkg.Imports {
					<-done[imp]
				}
				typecheck(t, pkg)
				close(done[pkg])
			}()
		})
	for _, root := range pkgs {
		<-done[root]
	}
}

// typecheck reads, parses, and type-checks a package.
// It squirrels the export data in the ppkg.ExportFile field.
func typecheck(t *testing.T, ppkg *packages.Package) {
	if ppkg.PkgPath == "unsafe" {
		return // unsafe is special
	}

	// Create a local FileSet just for this package.
	fset := token.NewFileSet()

	// Parse files in parallel.
	syntax := make([]*ast.File, len(ppkg.CompiledGoFiles))
	var group errgroup.Group
	for i, filename := range ppkg.CompiledGoFiles {
		i, filename := i, filename
		group.Go(func() error {
			f, err := parser.ParseFile(fset, filename, nil, parser.SkipObjectResolution)
			if err != nil {
				return err // e.g. missing file
			}
			syntax[i] = f
			return nil
		})
	}
	if err := group.Wait(); err != nil {
		t.Fatal(err)
	}
	// Inv: all files were successfully parsed.

	// Build map of dependencies by package path.
	// (We don't compute this mapping for the entire
	// packages graph because it is not globally consistent.)
	depsByPkgPath := make(map[string]*packages.Package)
	{
		var visit func(*packages.Package)
		visit = func(pkg *packages.Package) {
			if depsByPkgPath[pkg.PkgPath] == nil {
				depsByPkgPath[pkg.PkgPath] = pkg
				for path := range pkg.Imports {
					visit(pkg.Imports[path])
				}
			}
		}
		visit(ppkg)
	}

	// importer state
	var (
		insert    func(p *types.Package, name string)
		importMap = make(map[string]*types.Package) // keys are PackagePaths
	)
	loadFromExportData := func(imp *packages.Package) (*types.Package, error) {
		data := []byte(imp.ExportFile)
		return gcimporter.IImportShallow(fset, gcimporter.GetPackageFromMap(importMap), data, imp.PkgPath, insert)
	}
	insert = func(p *types.Package, name string) {
		imp, ok := depsByPkgPath[p.Path()]
		if !ok {
			t.Fatalf("can't find dependency: %q", p.Path())
		}
		imported, err := loadFromExportData(imp)
		if err != nil {
			t.Fatalf("unmarshal: %v", err)
		}
		if imported != p {
			t.Fatalf("internal error: inconsistent packages")
		}
		if obj := imported.Scope().Lookup(name); obj == nil {
			t.Fatalf("lookup %q.%s failed", imported.Path(), name)
		}
	}

	cfg := &types.Config{
		Error: func(e error) {
			t.Error(e)
		},
		Importer: importerFunc(func(importPath string) (*types.Package, error) {
			if importPath == "unsafe" {
				return types.Unsafe, nil // unsafe has no exportdata
			}
			imp, ok := ppkg.Imports[importPath]
			if !ok {
				return nil, fmt.Errorf("missing import %q", importPath)
			}
			return loadFromExportData(imp)
		}),
	}

	// Type-check the syntax trees.
	tpkg, _ := cfg.Check(ppkg.PkgPath, fset, syntax, nil)
	postTypeCheck(t, fset, tpkg)

	// Save the export data.
	data, err := gcimporter.IExportShallow(fset, tpkg)
	if err != nil {
		t.Fatalf("internal error marshalling export data: %v", err)
	}
	ppkg.ExportFile = string(data)
}

// postTypeCheck is called after a package is type checked.
// We use it to assert additional correctness properties,
// for example, that the apparent location of "fmt.Println"
// corresponds to its source location: in other words,
// export+import preserves high-fidelity positions.
func postTypeCheck(t *testing.T, fset *token.FileSet, pkg *types.Package) {
	// We hard-code a few interesting test-case objects.
	var obj types.Object
	switch pkg.Path() {
	case "fmt":
		// func fmt.Println
		obj = pkg.Scope().Lookup("Println")
	case "net/http":
		// method (*http.Request).ParseForm
		req := pkg.Scope().Lookup("Request")
		obj, _, _ = types.LookupFieldOrMethod(req.Type(), true, pkg, "ParseForm")
	default:
		return
	}
	if obj == nil {
		t.Errorf("object not found in package %s", pkg.Path())
		return
	}

	// Now check the source fidelity of the object's position.
	posn := fset.Position(obj.Pos())
	data, err := os.ReadFile(posn.Filename)
	if err != nil {
		t.Errorf("can't read source file declaring %v: %v", obj, err)
		return
	}

	// Check line and column denote a source interval containing the object's identifier.
	line := strings.Split(string(data), "\n")[posn.Line-1]

	if id := line[posn.Column-1 : posn.Column-1+len(obj.Name())]; id != obj.Name() {
		t.Errorf("%+v: expected declaration of %v at this line, column; got %q", posn, obj, line)
	}

	// Check offset.
	if id := string(data[posn.Offset : posn.Offset+len(obj.Name())]); id != obj.Name() {
		t.Errorf("%+v: expected declaration of %v at this offset; got %q", posn, obj, id)
	}

	// Check commutativity of Position() and start+len(name) operations:
	// Position(startPos+len(name)) == Position(startPos) + len(name).
	// This important property is a consequence of the way in which the
	// decoder fills the gaps in the sparse line-start offset table.
	endPosn := fset.Position(obj.Pos() + token.Pos(len(obj.Name())))
	wantEndPosn := token.Position{
		Filename: posn.Filename,
		Offset:   posn.Offset + len(obj.Name()),
		Line:     posn.Line,
		Column:   posn.Column + len(obj.Name()),
	}
	if endPosn != wantEndPosn {
		t.Errorf("%+v: expected end Position of %v here; was at %+v", wantEndPosn, obj, endPosn)
	}
}
