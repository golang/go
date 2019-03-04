package cache

import (
	"fmt"
	"go/ast"
	"go/scanner"
	"go/types"
	"io/ioutil"
	"log"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/lsp/source"
)

func (v *View) parse(uri source.URI) error {
	imp, pkgs, err := v.collectMetadata(uri)
	if err != nil {
		return err
	}
	for _, meta := range pkgs {
		// If the package comes back with errors from `go list`, don't bother
		// type-checking it.
		for _, err := range meta.Errors {
			switch err.Kind {
			case packages.UnknownError, packages.ListError:
				return err
			}
		}
		// Start prefetching direct imports.
		for importPath := range meta.Imports {
			go imp.Import(importPath)
		}

		// Type-check package.
		pkg, err := imp.typeCheck(meta.PkgPath)
		if pkg == nil || pkg.Types == nil {
			return err
		}

		// Add every file in this package to our cache.
		v.cachePackage(pkg)
	}
	// If we still have not found the package for the file, something is wrong.
	f := v.files[uri]
	if f == nil || f.pkg == nil {
		return fmt.Errorf("no package found for %v", uri)
	}
	return nil
}

func (v *View) cachePackage(pkg *packages.Package) {
	for _, file := range pkg.Syntax {
		// TODO: If a file is in multiple packages, which package do we store?
		if !file.Pos().IsValid() {
			log.Printf("invalid position for file %v", file.Name)
			continue
		}
		tok := v.Config.Fset.File(file.Pos())
		if tok == nil {
			log.Printf("no token.File for %v", file.Name)
			continue
		}
		fURI := source.ToURI(tok.Name())
		f := v.getFile(fURI)
		f.token = tok
		f.ast = file
		f.pkg = pkg
	}
}

type importer struct {
	mu       sync.Mutex
	entries  map[string]*entry
	packages map[string]*packages.Package

	*View
}

type entry struct {
	pkg   *packages.Package
	err   error
	ready chan struct{}
}

func (v *View) collectMetadata(uri source.URI) (*importer, []*packages.Package, error) {
	filename, err := uri.Filename()
	if err != nil {
		return nil, nil, err
	}
	// TODO(rstambler): Enforce here that LoadMode is LoadImports?
	pkgs, err := packages.Load(&v.Config, fmt.Sprintf("file=%s", filename))
	if len(pkgs) == 0 {
		if err == nil {
			err = fmt.Errorf("no packages found for %s", filename)
		}
		return nil, nil, err
	}
	imp := &importer{
		entries:  make(map[string]*entry),
		packages: make(map[string]*packages.Package),
		View:     v,
	}
	for _, pkg := range pkgs {
		if err := imp.addImports(pkg.PkgPath, pkg); err != nil {
			return nil, nil, err
		}
	}
	return imp, pkgs, nil
}

func (imp *importer) addImports(path string, pkg *packages.Package) error {
	if _, ok := imp.packages[path]; ok {
		return nil
	}
	imp.packages[path] = pkg
	for importPath, importPkg := range pkg.Imports {
		if err := imp.addImports(importPath, importPkg); err != nil {
			return err
		}
	}
	return nil
}

func (imp *importer) Import(path string) (*types.Package, error) {
	imp.mu.Lock()
	e, ok := imp.entries[path]
	if ok {
		// cache hit
		imp.mu.Unlock()
		// wait for entry to become ready
		<-e.ready
	} else {
		// cache miss
		e = &entry{ready: make(chan struct{})}
		imp.entries[path] = e
		imp.mu.Unlock()

		// This goroutine becomes responsible for populating
		// the entry and broadcasting its readiness.
		e.pkg, e.err = imp.typeCheck(path)
		close(e.ready)
	}
	if e.err != nil {
		return nil, e.err
	}
	return e.pkg.Types, nil
}

func (imp *importer) typeCheck(pkgPath string) (*packages.Package, error) {
	imp.mu.Lock()
	pkg, ok := imp.packages[pkgPath]
	imp.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("no metadata for %v", pkgPath)
	}
	// TODO(rstambler): Get real TypeSizes from go/packages (golang.org/issues/30139).
	pkg.TypesSizes = &types.StdSizes{}
	pkg.Fset = imp.Config.Fset
	appendError := func(err error) {
		imp.appendPkgError(pkg, err)
	}
	files, errs := imp.parseFiles(pkg.CompiledGoFiles)
	for _, err := range errs {
		appendError(err)
	}
	pkg.Syntax = files
	cfg := &types.Config{
		Error:    appendError,
		Importer: imp,
	}
	pkg.Types = types.NewPackage(pkg.PkgPath, pkg.Name)
	pkg.TypesInfo = &types.Info{
		Types:      make(map[ast.Expr]types.TypeAndValue),
		Defs:       make(map[*ast.Ident]types.Object),
		Uses:       make(map[*ast.Ident]types.Object),
		Implicits:  make(map[ast.Node]types.Object),
		Selections: make(map[*ast.SelectorExpr]*types.Selection),
		Scopes:     make(map[ast.Node]*types.Scope),
	}
	check := types.NewChecker(cfg, imp.Config.Fset, pkg.Types, pkg.TypesInfo)
	check.Files(pkg.Syntax)

	return pkg, nil
}

func (imp *importer) appendPkgError(pkg *packages.Package, err error) {
	if err == nil {
		return
	}
	var errs []packages.Error
	switch err := err.(type) {
	case *scanner.Error:
		errs = append(errs, packages.Error{
			Pos:  err.Pos.String(),
			Msg:  err.Msg,
			Kind: packages.ParseError,
		})
	case scanner.ErrorList:
		// The first parser error is likely the root cause of the problem.
		if err.Len() > 0 {
			errs = append(errs, packages.Error{
				Pos:  err[0].Pos.String(),
				Msg:  err[0].Msg,
				Kind: packages.ParseError,
			})
		}
	case types.Error:
		errs = append(errs, packages.Error{
			Pos:  imp.Config.Fset.Position(err.Pos).String(),
			Msg:  err.Msg,
			Kind: packages.TypeError,
		})
	}
	pkg.Errors = append(pkg.Errors, errs...)
}

// We use a counting semaphore to limit
// the number of parallel I/O calls per process.
var ioLimit = make(chan bool, 20)

// parseFiles reads and parses the Go source files and returns the ASTs
// of the ones that could be at least partially parsed, along with a
// list of I/O and parse errors encountered.
//
// Because files are scanned in parallel, the token.Pos
// positions of the resulting ast.Files are not ordered.
//
func (imp *importer) parseFiles(filenames []string) ([]*ast.File, []error) {
	var wg sync.WaitGroup
	n := len(filenames)
	parsed := make([]*ast.File, n)
	errors := make([]error, n)
	for i, filename := range filenames {
		if imp.Config.Context.Err() != nil {
			parsed[i] = nil
			errors[i] = imp.Config.Context.Err()
			continue
		}

		// First, check if we have already cached an AST for this file.
		f := imp.files[source.ToURI(filename)]
		var fAST *ast.File
		if f != nil {
			fAST = f.ast
		}

		wg.Add(1)
		go func(i int, filename string) {
			ioLimit <- true // wait

			if fAST != nil {
				parsed[i], errors[i] = fAST, nil
			} else {
				// We don't have a cached AST for this file.
				var src []byte
				// Check for an available overlay.
				for f, contents := range imp.Config.Overlay {
					if sameFile(f, filename) {
						src = contents
					}
				}
				var err error
				// We don't have an overlay, so we must read the file's contents.
				if src == nil {
					src, err = ioutil.ReadFile(filename)
				}
				if err != nil {
					parsed[i], errors[i] = nil, err
				} else {
					// ParseFile may return both an AST and an error.
					parsed[i], errors[i] = imp.Config.ParseFile(imp.Config.Fset, filename, src)
				}
			}

			<-ioLimit // signal
			wg.Done()
		}(i, filename)
	}
	wg.Wait()

	// Eliminate nils, preserving order.
	var o int
	for _, f := range parsed {
		if f != nil {
			parsed[o] = f
			o++
		}
	}
	parsed = parsed[:o]

	o = 0
	for _, err := range errors {
		if err != nil {
			errors[o] = err
			o++
		}
	}
	errors = errors[:o]

	return parsed, errors
}

// sameFile returns true if x and y have the same basename and denote
// the same file.
//
func sameFile(x, y string) bool {
	if x == y {
		// It could be the case that y doesn't exist.
		// For instance, it may be an overlay file that
		// hasn't been written to disk. To handle that case
		// let x == y through. (We added the exact absolute path
		// string to the CompiledGoFiles list, so the unwritten
		// overlay case implies x==y.)
		return true
	}
	if strings.EqualFold(filepath.Base(x), filepath.Base(y)) { // (optimisation)
		if xi, err := os.Stat(x); err == nil {
			if yi, err := os.Stat(y); err == nil {
				return os.SameFile(xi, yi)
			}
		}
	}
	return false
}
