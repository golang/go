// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package srcimporter implements importing directly
// from source files rather than installed packages.
package srcimporter // import "go/internal/srcimporter"

import (
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"go/types"
	"path/filepath"
)

// An Importer provides the context for importing packages from source code.
type Importer struct {
	ctxt     *build.Context
	fset     *token.FileSet
	sizes    types.Sizes
	packages map[string]*types.Package
}

// NewImporter returns a new Importer for the given context, file set, and map
// of packages. The context is used to resolve import paths to package paths,
// and identifying the files belonging to the package. If the context provides
// non-nil file system functions, they are used instead of the regular package
// os functions. The file set is used to track position information of package
// files; and imported packages are added to the packages map.
func New(ctxt *build.Context, fset *token.FileSet, packages map[string]*types.Package) *Importer {
	return &Importer{
		ctxt:     ctxt,
		fset:     fset,
		sizes:    types.SizesFor(ctxt.GOARCH), // uses go/types default if GOARCH not found
		packages: packages,
	}
}

// Importing is a sentinel taking the place in Importer.packages
// for a package that is in the process of being imported.
var importing types.Package

// Import(path) is a shortcut for ImportFrom(path, "", 0).
func (p *Importer) Import(path string) (*types.Package, error) {
	return p.ImportFrom(path, "", 0)
}

// ImportFrom imports the package with the given import path resolved from the given srcDir,
// adds the new package to the set of packages maintained by the importer, and returns the
// package. Package path resolution and file system operations are controlled by the context
// maintained with the importer. The import mode must be zero but is otherwise ignored.
// Packages that are not comprised entirely of pure Go files may fail to import because the
// type checker may not be able to determine all exported entities (e.g. due to cgo dependencies).
func (p *Importer) ImportFrom(path, srcDir string, mode types.ImportMode) (*types.Package, error) {
	if mode != 0 {
		panic("non-zero import mode")
	}

	// determine package path (do vendor resolution)
	var bp *build.Package
	var err error
	switch {
	default:
		if abs, err := p.absPath(srcDir); err == nil { // see issue #14282
			srcDir = abs
		}
		bp, err = p.ctxt.Import(path, srcDir, build.FindOnly)

	case build.IsLocalImport(path):
		// "./x" -> "srcDir/x"
		bp, err = p.ctxt.ImportDir(filepath.Join(srcDir, path), build.FindOnly)

	case p.isAbsPath(path):
		return nil, fmt.Errorf("invalid absolute import path %q", path)
	}
	if err != nil {
		return nil, err // err may be *build.NoGoError - return as is
	}

	// package unsafe is known to the type checker
	if bp.ImportPath == "unsafe" {
		return types.Unsafe, nil
	}

	// no need to re-import if the package was imported completely before
	pkg := p.packages[bp.ImportPath]
	if pkg != nil {
		if pkg == &importing {
			return nil, fmt.Errorf("import cycle through package %q", bp.ImportPath)
		}
		if pkg.Complete() {
			return pkg, nil
		}
	} else {
		p.packages[bp.ImportPath] = &importing
		defer func() {
			// clean up in case of error
			// TODO(gri) Eventually we may want to leave a (possibly empty)
			// package in the map in all cases (and use that package to
			// identify cycles). See also issue 16088.
			if p.packages[bp.ImportPath] == &importing {
				p.packages[bp.ImportPath] = nil
			}
		}()
	}

	// collect package files
	bp, err = p.ctxt.ImportDir(bp.Dir, 0)
	if err != nil {
		return nil, err // err may be *build.NoGoError - return as is
	}
	var filenames []string
	filenames = append(filenames, bp.GoFiles...)
	filenames = append(filenames, bp.CgoFiles...)

	// parse package files
	// TODO(gri) do this concurrently
	var files []*ast.File
	for _, filename := range filenames {
		filepath := p.joinPath(bp.Dir, filename)
		var file *ast.File
		if open := p.ctxt.OpenFile; open != nil {
			f, err := open(filepath)
			if err != nil {
				return nil, fmt.Errorf("opening package file %s failed (%v)", filepath, err)
			}
			file, err = parser.ParseFile(p.fset, filepath, f, 0)
			f.Close() // ignore Close error - import may still succeed
		} else {
			// Special-case when ctxt doesn't provide a custom OpenFile and use the
			// parser's file reading mechanism directly. This appears to be quite a
			// bit faster than opening the file and providing an io.ReaderCloser in
			// both cases.
			// TODO(gri) investigate performance difference (issue #19281)
			file, err = parser.ParseFile(p.fset, filepath, nil, 0)
		}
		if err != nil {
			return nil, fmt.Errorf("parsing package file %s failed (%v)", filepath, err)
		}
		files = append(files, file)
	}

	// type-check package files
	conf := types.Config{
		IgnoreFuncBodies: true,
		FakeImportC:      true,
		Importer:         p,
		Sizes:            p.sizes,
	}
	pkg, err = conf.Check(bp.ImportPath, p.fset, files, nil)
	if err != nil {
		return nil, fmt.Errorf("type-checking package %q failed (%v)", bp.ImportPath, err)
	}

	p.packages[bp.ImportPath] = pkg
	return pkg, nil
}

// context-controlled file system operations

func (p *Importer) absPath(path string) (string, error) {
	// TODO(gri) This should be using p.ctxt.AbsPath which doesn't
	// exist but probably should. See also issue #14282.
	return filepath.Abs(path)
}

func (p *Importer) isAbsPath(path string) bool {
	if f := p.ctxt.IsAbsPath; f != nil {
		return f(path)
	}
	return filepath.IsAbs(path)
}

func (p *Importer) joinPath(elem ...string) string {
	if f := p.ctxt.JoinPath; f != nil {
		return f(elem...)
	}
	return filepath.Join(elem...)
}
