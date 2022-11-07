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
// It squirrels the export data in the the ppkg.ExportFile field.
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
		return gcimporter.IImportShallow(fset, importMap, data, imp.PkgPath, insert)
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

	// Save the export data.
	data, err := gcimporter.IExportShallow(fset, tpkg)
	if err != nil {
		t.Fatalf("internal error marshalling export data: %v", err)
	}
	ppkg.ExportFile = string(data)
}
