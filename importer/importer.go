// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package importer defines the Importer, which loads, parses and
// type-checks packages of Go code plus their transitive closure, and
// retains both the ASTs and the derived facts.
//
// CONCEPTS AND TERMINOLOGY
//
// An AD-HOC package is one specified as a set of source files on the
// command line.  In the simplest case, it may consist of a single file
// such as src/pkg/net/http/triv.go.
//
// EXTERNAL TEST packages are those comprised of a set of *_test.go
// files all with the same 'package foo_test' declaration, all in the
// same directory.  (go/build.Package calls these files XTestFiles.)
//
// An IMPORTABLE package is one that can be referred to by some import
// spec.  Ad-hoc packages and external test packages are non-importable.
// The importer and its clients must be careful not to assume that
// the import path of a package may be used for a name-based lookup.
// For example, a pointer analysis scope may consist of two initial
// (ad-hoc) packages both called "main".
//
// An AUGMENTED package is an importable package P plus all the
// *_test.go files with same 'package foo' declaration as P.
// (go/build.Package calls these files TestFiles.)
// An external test package may depend upon members of the augmented
// package that are not in the unaugmented package, such as functions
// that expose internals.  (See bufio/export_test.go for an example.)
// So, the importer must ensure that for each external test package
// it loads, it also augments the corresponding non-test package.
//
// The import graph over n unaugmented packages must be acyclic; the
// import graph over n-1 unaugmented packages plus one augmented
// package must also be acyclic.  ('go test' relies on this.)  But the
// import graph over n augmented packages may contain cycles, and
// currently, go/types is incapable of handling such inputs, so the
// Importer will only augment (and create an external test package
// for) the first import path specified on the command-line.
//
package importer

import (
	"errors"
	"fmt"
	"go/ast"
	"go/build"
	"go/token"
	"os"
	"strings"
	"sync"

	"code.google.com/p/go.tools/astutil"
	"code.google.com/p/go.tools/go/exact"
	"code.google.com/p/go.tools/go/gcimporter"
	"code.google.com/p/go.tools/go/types"
)

// An Importer's exported methods are not thread-safe.
type Importer struct {
	Fset          *token.FileSet         // position info for all files seen
	config        *Config                // the client configuration, unmodified
	importfn      types.Importer         // client's type import function
	augment       map[string]bool        // packages to be augmented by TestFiles when imported
	allPackagesMu sync.Mutex             // guards 'allPackages' during internal concurrency
	allPackages   []*PackageInfo         // all packages, including non-importable ones
	importedMu    sync.Mutex             // guards 'imported'
	imported      map[string]*importInfo // all imported packages (incl. failures) by import path
}

// importInfo holds internal information about each import path.
type importInfo struct {
	path  string        // import path
	info  *PackageInfo  // results of typechecking (including type errors)
	err   error         // reason for failure to construct a package
	ready chan struct{} // channel close is notification of ready state
}

// Config specifies the configuration for the importer.
type Config struct {
	// TypeChecker contains options relating to the type checker.
	//
	// The supplied IgnoreFuncBodies is not used; the effective
	// value comes from the TypeCheckFuncBodies func below.
	//
	// All callbacks must be thread-safe.
	TypeChecker types.Config

	// TypeCheckFuncBodies is a predicate over package import
	// paths.  A package for which the predicate is false will
	// have its package-level declarations type checked, but not
	// its function bodies; this can be used to quickly load
	// dependencies from source.  If nil, all func bodies are type
	// checked.
	//
	// Must be thread-safe.
	//
	TypeCheckFuncBodies func(string) bool

	// If Build is non-nil, it is used to satisfy imports.
	//
	// If it is nil, binary object files produced by the gc
	// compiler will be loaded instead of source code for all
	// imported packages.  Such files supply only the types of
	// package-level declarations and values of constants, but no
	// code, so this mode will not yield a whole program.  It is
	// intended for analyses that perform intraprocedural analysis
	// of a single package.
	Build *build.Context
}

// New returns a new, empty Importer using configuration options
// specified by config.
//
func New(config *Config) *Importer {
	// Initialize by mutating the caller's copy,
	// so all copies agree on the identity of the map.
	if config.TypeChecker.Packages == nil {
		config.TypeChecker.Packages = make(map[string]*types.Package)
	}

	// Save the caller's effective Import funcion.
	importfn := config.TypeChecker.Import
	if importfn == nil {
		importfn = gcimporter.Import
	}

	imp := &Importer{
		Fset:     token.NewFileSet(),
		config:   config,
		importfn: importfn,
		augment:  make(map[string]bool),
		imported: make(map[string]*importInfo),
	}
	return imp
}

// AllPackages returns a new slice containing all packages loaded by
// importer imp.
//
func (imp *Importer) AllPackages() []*PackageInfo {
	return append([]*PackageInfo(nil), imp.allPackages...)
}

func (imp *Importer) addPackage(info *PackageInfo) {
	imp.allPackagesMu.Lock()
	imp.allPackages = append(imp.allPackages, info)
	imp.allPackagesMu.Unlock()
}

// doImport imports the package denoted by path.
// It implements the types.Importer prototype.
//
// imports is the import map of the importing package, later
// accessible as types.Package.Imports().  If non-nil, doImport will
// update it to include this import.  (It may be nil in recursive
// calls for prefetching.)
//
// It returns an error if a package could not be created
// (e.g. go/build or parse error), but type errors are reported via
// the types.Config.Error callback (the first of which is also saved
// in the package's PackageInfo).
//
// Idempotent and thread-safe.
//
//
// TODO(gri): The imports map (an alias for TypeChecker.Packages) must
// not be concurrently accessed.  Today, the only (non-test) accesses
// of this map are in the client-supplied implementation of the
// importer function, i.e. the function below.  But this is a fragile
// API because if go/types ever starts to access this map, it and its
// clients will have to agree to use the same mutex.
// Two better ideas:
//
// (1) require that the importer functions be stateful and have this
//     map be part of that internal state.
//     Pro: good encapsulation.
//     Con: we can't have different importers collaborate, e.g.
//          we can't use a source importer for some packages and
//          GcImport for others since they'd each have a distinct map.
//
// (2) have there be a single map in go/types.Config, but expose
//     lookup and update behind an interface and pass that interface
//     to the importer implementations.
//     Pro: sharing of the map among importers.
//
//  This is idempotent but still doesn't address the issue of
//  atomicity: when loading packages concurrently, we want to avoid
//  the benign but suboptimal situation of two goroutines both
//  importing "fmt", finding it not present, doing all the work, and
//  double-updating the map.
//  The interface to the map needs to express the idea that when a
//  caller requests an import from the map and finds it not present,
//  then it (and no other goroutine) becomes responsible for loading
//  the package and updating the map; other goroutines should block
//  until then.  That's exactly what doImport0 below does; I think
//  some of that logic should migrate into go/types.check.resolveFiles.
//
func (imp *Importer) doImport(imports map[string]*types.Package, path string) (*types.Package, error) {
	// Package unsafe is handled specially, and has no PackageInfo.
	// TODO(adonovan): a fake empty package would make things simpler.
	if path == "unsafe" {
		return types.Unsafe, nil
	}

	info, err := imp.doImport0(imports, path)
	if err != nil {
		return nil, err
	}

	if imports != nil {
		// Update the package's imports map, whether success or failure.
		//
		// types.Package.Imports() is used by PackageInfo.Imports and
		// thence by ssa.builder.
		// TODO(gri): given that it doesn't specify whether it
		// contains direct or transitive dependencies, it probably
		// shouldn't be exposed.  This package can make its own
		// arrangements to implement PackageInfo.Imports().
		importsMu.Lock()
		imports[path] = info.Pkg
		importsMu.Unlock()
	}

	return info.Pkg, nil
}

var importsMu sync.Mutex // hack; see comment at doImport

// Like doImport, but returns a PackageInfo.
// Precondition: path != "unsafe".
func (imp *Importer) doImport0(imports map[string]*types.Package, path string) (*PackageInfo, error) {
	imp.importedMu.Lock()
	ii, ok := imp.imported[path]
	if !ok {
		ii = &importInfo{path: path, ready: make(chan struct{})}
		imp.imported[path] = ii
	}
	imp.importedMu.Unlock()

	if !ok {
		// Find and create the actual package.
		if imp.config.Build != nil {
			imp.importSource(path, ii)
		} else {
			imp.importBinary(imports, ii)
		}
		if ii.info != nil {
			ii.info.Importable = true
		}

		close(ii.ready) // enter ready state and wake up waiters
	} else {
		<-ii.ready // wait for ready condition
	}

	// Invariant: ii is ready.

	return ii.info, ii.err
}

// importBinary implements package loading from the client-supplied
// external source, e.g. object files from the gc compiler.
//
func (imp *Importer) importBinary(imports map[string]*types.Package, ii *importInfo) {
	pkg, err := imp.importfn(imports, ii.path)
	if pkg != nil {
		ii.info = &PackageInfo{Pkg: pkg}
		imp.addPackage(ii.info)
	} else {
		ii.err = err
	}
}

// importSource implements package loading by parsing Go source files
// located by go/build.
//
func (imp *Importer) importSource(path string, ii *importInfo) {
	which := "g" // load *.go files
	if imp.augment[path] {
		which = "gt" // augment package by in-package *_test.go files
	}
	if files, err := parsePackageFiles(imp.config.Build, imp.Fset, path, which); err == nil {
		// Prefetch the imports asynchronously.
		for path := range importsOf(path, files) {
			go func(path string) { imp.doImport(nil, path) }(path)
		}

		// Type-check the package.
		ii.info = imp.CreatePackage(path, files...)

		// We needn't wait for the prefetching goroutines to
		// finish.  Each one either runs quickly and populates
		// the imported map, in which case the type checker
		// will wait for the map entry to become ready; or, it
		// runs slowly, even after we return, but then becomes
		// just another map waiter, in which case it won't
		// mutate anything.
	} else {
		ii.err = err
	}
}

// CreatePackage creates and type-checks a package from the specified
// list of parsed files, importing their dependencies.  It returns a
// PackageInfo containing the resulting types.Package, the ASTs, and
// other type information.
//
// The order of files determines the package initialization order.
//
// path is the full name under which this package is known, such as
// appears in an import declaration. e.g. "sync/atomic".  It need not
// be unique; for example, it is possible to construct two distinct
// packages both named "main".
//
// The resulting package is accessible via AllPackages() but is not
// importable, i.e. no 'import' spec can resolve to it.
//
// CreatePackage never fails, but the resulting package may contain type
// errors; the first of these is recorded in PackageInfo.Err.
//
func (imp *Importer) CreatePackage(path string, files ...*ast.File) *PackageInfo {
	info := &PackageInfo{
		Files: files,
		Info: types.Info{
			Types:      make(map[ast.Expr]types.Type),
			Values:     make(map[ast.Expr]exact.Value),
			Objects:    make(map[*ast.Ident]types.Object),
			Implicits:  make(map[ast.Node]types.Object),
			Scopes:     make(map[ast.Node]*types.Scope),
			Selections: make(map[*ast.SelectorExpr]*types.Selection),
		},
	}

	// Use a copy of the types.Config so we can vary IgnoreFuncBodies.
	tc := imp.config.TypeChecker
	tc.IgnoreFuncBodies = false
	if f := imp.config.TypeCheckFuncBodies; f != nil {
		tc.IgnoreFuncBodies = !f(path)
	}
	if tc.Error == nil {
		tc.Error = func(e error) { fmt.Fprintln(os.Stderr, e) }
	}
	tc.Import = imp.doImport // doImport wraps the user's importfn, effectively
	info.Pkg, info.Err = tc.Check(path, imp.Fset, files, &info.Info)
	imp.addPackage(info)
	return info
}

// InitialPackagesUsage is a partial usage message that client
// applications may wish to include in their -help output.
const InitialPackagesUsage = `
<args> is a list of arguments denoting a set of initial packages.
Each argument may take one of two forms:

1. A comma-separated list of *.go source files.

   All of the specified files are loaded, parsed and type-checked
   as a single package.  The name of the package is taken from the
   files' package declarations, which must all be equal.  All the
   files must belong to the same directory.

2. An import path.

   The package's directory is found relative to the $GOROOT and
   $GOPATH using similar logic to 'go build', and the *.go files in
   that directory are loaded, parsed and type-checked as a single
   package.

   In addition, all *_test.go files in the directory are then loaded
   and parsed.  Those files whose package declaration equals that of
   the non-*_test.go files are included in the primary package.  Test
   files whose package declaration ends with "_test" are type-checked
   as another package, the 'external' test package, so that a single
   import path may denote two packages.  This behaviour may be
   disabled by prefixing the import path with "notest:",
   e.g. "notest:fmt".

   Due to current limitations in the type-checker, only the first
   import path of the command line will contribute any tests.

A '--' argument terminates the list of packages.
`

// LoadInitialPackages interprets args as a set of packages, loads
// those packages and their dependencies, and returns them.
//
// It is intended for use in command-line interfaces that require a
// set of initial packages to be specified; see InitialPackagesUsage
// message for details.
//
// The second result parameter returns the list of unconsumed
// arguments.
//
// It is an error to specify no packages.
//
// Precondition: LoadInitialPackages cannot be called after any
// previous calls to Load* on the same importer.
//
func (imp *Importer) LoadInitialPackages(args []string) ([]*PackageInfo, []string, error) {
	// The "augmentation" mechanism requires that we mark all
	// packages to be augmented before we import a single one.
	if len(imp.allPackages) > 0 {
		return nil, nil, errors.New("LoadInitialPackages called on non-pristine Importer")
	}

	// We use two passes.  The first parses the files for each
	// non-importable package and discovers the set of importable
	// packages that require augmentation by in-package _test.go
	// files.  The second creates the ad-hoc packages and imports
	// the importable ones.
	//
	// This is necessary to ensure that all packages requiring
	// augmentation are known before before any package is
	// imported.

	// Pass 1: parse the sets of files for each package.
	var pkgs []*initialPkg
	for len(args) > 0 {
		arg := args[0]
		args = args[1:]
		if arg == "--" {
			break // consume "--" and return the remaining args
		}

		if strings.HasSuffix(arg, ".go") {
			// Assume arg is a comma-separated list of *.go files
			// comprising a single package.
			pkg, err := initialPackageFromFiles(imp.Fset, arg)
			if err != nil {
				return nil, nil, err
			}
			pkgs = append(pkgs, pkg)

		} else {
			// Assume arg is a directory name denoting a
			// package, perhaps plus an external test
			// package unless prefixed by "notest:".
			path := strings.TrimPrefix(arg, "notest:")

			if path == "unsafe" {
				continue // ignore; has no PackageInfo
			}

			pkg := &initialPkg{
				path:       path,
				importable: true,
			}
			pkgs = append(pkgs, pkg)

			if path != arg {
				continue // had "notest:" prefix
			}

			if imp.config.Build == nil {
				continue // can't locate *_test.go files
			}

			// TODO(adonovan): due to limitations of the current type
			// checker design, we can augment at most one package.
			if len(imp.augment) > 0 {
				continue // don't attempt a second
			}

			// Load the external test package.
			xtestFiles, err := parsePackageFiles(imp.config.Build, imp.Fset, path, "x")
			if err != nil {
				return nil, nil, err
			}
			if len(xtestFiles) > 0 {
				pkgs = append(pkgs, &initialPkg{
					path:       path + "_test",
					importable: false,
					files:      xtestFiles,
				})
			}

			// Mark the non-xtest package for augmentation with
			// in-package *_test.go files when we import it below.
			imp.augment[pkg.path] = true
		}
	}

	// Pass 2: type-check each set of files to make a package.
	var infos []*PackageInfo
	imports := make(map[string]*types.Package) // keep importBinary happy
	for _, pkg := range pkgs {
		var info *PackageInfo
		if pkg.importable {
			// import package
			var err error
			info, err = imp.doImport0(imports, pkg.path)
			if err != nil {
				return nil, nil, err // e.g. parse error (but not type error)
			}
		} else {
			// create package
			info = imp.CreatePackage(pkg.path, pkg.files...)
		}
		infos = append(infos, info)
	}

	if len(pkgs) == 0 {
		return nil, nil, errors.New("no *.go source files nor packages were specified")
	}

	return infos, args, nil
}

// LoadPackage loads and type-checks the package whose import path is
// path, plus its necessary dependencies.
//
func (imp *Importer) LoadPackage(path string) (*PackageInfo, error) {
	imports := make(map[string]*types.Package) // keep importBinary happy
	return imp.doImport0(imports, path)
}

type initialPkg struct {
	path       string      // the package's import path
	importable bool        // add package to import map false for main and xtests)
	files      []*ast.File // set of files (non-importable packages only)
}

// initialPackageFromFiles returns an initialPkg, given a
// comma-separated list of *.go source files belonging to the same
// directory and possessing the same 'package decl'.
//
func initialPackageFromFiles(fset *token.FileSet, arg string) (*initialPkg, error) {
	filenames := strings.Split(arg, ",")
	for _, filename := range filenames {
		if !strings.HasSuffix(filename, ".go") {
			return nil, fmt.Errorf("not a *.go source file: %q", filename)
		}
	}

	files, err := ParseFiles(fset, ".", filenames...)
	if err != nil {
		return nil, err
	}

	// Take the package name from the 'package decl' in each file,
	// all of which must match.
	pkgname := files[0].Name.Name
	for i, file := range files[1:] {
		if pn := file.Name.Name; pn != pkgname {
			err := fmt.Errorf("can't load package: found packages %s (%s) and %s (%s)",
				pkgname, filenames[0], pn, filenames[i])
			return nil, err
		}
		// TODO(adonovan): check dirnames are equal, like 'go build' does.
	}

	return &initialPkg{
		path:       pkgname,
		importable: false,
		files:      files,
	}, nil
}

// PathEnclosingInterval returns the PackageInfo and ast.Node that
// contain source interval [start, end), and all the node's ancestors
// up to the AST root.  It searches all ast.Files of all packages in the
// Importer imp.  exact is defined as for astutil.PathEnclosingInterval.
//
// The result is (nil, nil, false) if not found.
//
func (imp *Importer) PathEnclosingInterval(start, end token.Pos) (pkg *PackageInfo, path []ast.Node, exact bool) {
	for _, info := range imp.allPackages {
		for _, f := range info.Files {
			if !tokenFileContainsPos(imp.Fset.File(f.Package), start) {
				continue
			}
			if path, exact := astutil.PathEnclosingInterval(f, start, end); path != nil {
				return info, path, exact
			}
		}
	}
	return nil, nil, false
}

// TODO(adonovan): make this a method: func (*token.File) Contains(token.Pos)
func tokenFileContainsPos(f *token.File, pos token.Pos) bool {
	p := int(pos)
	base := f.Base()
	return base <= p && p < base+f.Size()
}
