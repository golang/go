// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gcexportdata

import (
	"fmt"
	"go/token"
	"go/types"
	"os"
)

// NewImporter returns a new instance of the types.Importer interface
// that reads type information from export data files written by gc.
// The Importer also satisfies types.ImporterFrom.
//
// Export data files are located using "go build" workspace conventions
// and the build.Default context.
//
// Use this importer instead of go/importer.For("gc", ...) to avoid the
// version-skew problems described in the documentation of this package,
// or to control the FileSet or access the imports map populated during
// package loading.
//
// Deprecated: Use the higher-level API in golang.org/x/tools/go/packages,
// which is more efficient.
func NewImporter(fset *token.FileSet, imports map[string]*types.Package) types.ImporterFrom {
	return importer{fset, imports}
}

type importer struct {
	fset    *token.FileSet
	imports map[string]*types.Package
}

func (imp importer) Import(importPath string) (*types.Package, error) {
	return imp.ImportFrom(importPath, "", 0)
}

func (imp importer) ImportFrom(importPath, srcDir string, mode types.ImportMode) (_ *types.Package, err error) {
	filename, path := Find(importPath, srcDir)
	if filename == "" {
		if importPath == "unsafe" {
			// Even for unsafe, call Find first in case
			// the package was vendored.
			return types.Unsafe, nil
		}
		return nil, fmt.Errorf("can't find import: %s", importPath)
	}

	if pkg, ok := imp.imports[path]; ok && pkg.Complete() {
		return pkg, nil // cache hit
	}

	// open file
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer func() {
		f.Close()
		if err != nil {
			// add file name to error
			err = fmt.Errorf("reading export data: %s: %v", filename, err)
		}
	}()

	r, err := NewReader(f)
	if err != nil {
		return nil, err
	}

	return Read(r, imp.fset, imp.imports, path)
}
