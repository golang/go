// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.5

package loader

// See doc.go for package documentation and implementation notes.

import (
	"errors"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/buildutil"
)

const trace = false // show timing info for type-checking

// Config specifies the configuration for loading a whole program from
// Go source code.
// The zero value for Config is a ready-to-use default configuration.
type Config struct {
	// Fset is the file set for the parser to use when loading the
	// program.  If nil, it may be lazily initialized by any
	// method of Config.
	Fset *token.FileSet

	// ParserMode specifies the mode to be used by the parser when
	// loading source packages.
	ParserMode parser.Mode

	// TypeChecker contains options relating to the type checker.
	//
	// The supplied IgnoreFuncBodies is not used; the effective
	// value comes from the TypeCheckFuncBodies func below.
	// The supplied Import function is not used either.
	TypeChecker types.Config

	// TypeCheckFuncBodies is a predicate over package paths.
	// A package for which the predicate is false will
	// have its package-level declarations type checked, but not
	// its function bodies; this can be used to quickly load
	// dependencies from source.  If nil, all func bodies are type
	// checked.
	TypeCheckFuncBodies func(path string) bool

	// If Build is non-nil, it is used to locate source packages.
	// Otherwise &build.Default is used.
	//
	// By default, cgo is invoked to preprocess Go files that
	// import the fake package "C".  This behaviour can be
	// disabled by setting CGO_ENABLED=0 in the environment prior
	// to startup, or by setting Build.CgoEnabled=false.
	Build *build.Context

	// The current directory, used for resolving relative package
	// references such as "./go/loader".  If empty, os.Getwd will be
	// used instead.
	Cwd string

	// If DisplayPath is non-nil, it is used to transform each
	// file name obtained from Build.Import().  This can be used
	// to prevent a virtualized build.Config's file names from
	// leaking into the user interface.
	DisplayPath func(path string) string

	// If AllowErrors is true, Load will return a Program even
	// if some of the its packages contained I/O, parser or type
	// errors; such errors are accessible via PackageInfo.Errors.  If
	// false, Load will fail if any package had an error.
	AllowErrors bool

	// CreatePkgs specifies a list of non-importable initial
	// packages to create.  The resulting packages will appear in
	// the corresponding elements of the Program.Created slice.
	CreatePkgs []PkgSpec

	// ImportPkgs specifies a set of initial packages to load.
	// The map keys are package paths.
	//
	// The map value indicates whether to load tests.  If true, Load
	// will add and type-check two lists of files to the package:
	// non-test files followed by in-package *_test.go files.  In
	// addition, it will append the external test package (if any)
	// to Program.Created.
	ImportPkgs map[string]bool

	// FindPackage is called during Load to create the build.Package
	// for a given import path from a given directory.
	// If FindPackage is nil, a default implementation
	// based on ctxt.Import is used.  A client may use this hook to
	// adapt to a proprietary build system that does not follow the
	// "go build" layout conventions, for example.
	//
	// It must be safe to call concurrently from multiple goroutines.
	FindPackage func(ctxt *build.Context, fromDir, importPath string, mode build.ImportMode) (*build.Package, error)
}

// A PkgSpec specifies a non-importable package to be created by Load.
// Files are processed first, but typically only one of Files and
// Filenames is provided.  The path needn't be globally unique.
//
type PkgSpec struct {
	Path      string      // package path ("" => use package declaration)
	Files     []*ast.File // ASTs of already-parsed files
	Filenames []string    // names of files to be parsed
}

// A Program is a Go program loaded from source as specified by a Config.
type Program struct {
	Fset *token.FileSet // the file set for this program

	// Created[i] contains the initial package whose ASTs or
	// filenames were supplied by Config.CreatePkgs[i], followed by
	// the external test package, if any, of each package in
	// Config.ImportPkgs ordered by ImportPath.
	//
	// NOTE: these files must not import "C".  Cgo preprocessing is
	// only performed on imported packages, not ad hoc packages.
	//
	// TODO(adonovan): we need to copy and adapt the logic of
	// goFilesPackage (from $GOROOT/src/cmd/go/build.go) and make
	// Config.Import and Config.Create methods return the same kind
	// of entity, essentially a build.Package.
	// Perhaps we can even reuse that type directly.
	Created []*PackageInfo

	// Imported contains the initially imported packages,
	// as specified by Config.ImportPkgs.
	Imported map[string]*PackageInfo

	// AllPackages contains the PackageInfo of every package
	// encountered by Load: all initial packages and all
	// dependencies, including incomplete ones.
	AllPackages map[*types.Package]*PackageInfo

	// importMap is the canonical mapping of package paths to
	// packages.  It contains all Imported initial packages, but not
	// Created ones, and all imported dependencies.
	importMap map[string]*types.Package
}

// PackageInfo holds the ASTs and facts derived by the type-checker
// for a single package.
//
// Not mutated once exposed via the API.
//
type PackageInfo struct {
	Pkg                   *types.Package
	Importable            bool        // true if 'import "Pkg.Path()"' would resolve to this
	TransitivelyErrorFree bool        // true if Pkg and all its dependencies are free of errors
	Files                 []*ast.File // syntax trees for the package's files
	Errors                []error     // non-nil if the package had errors
	types.Info                        // type-checker deductions.
	dir                   string      // package directory

	checker   *types.Checker // transient type-checker state
	errorFunc func(error)
}

func (info *PackageInfo) String() string { return info.Pkg.Path() }

func (info *PackageInfo) appendError(err error) {
	if info.errorFunc != nil {
		info.errorFunc(err)
	} else {
		fmt.Fprintln(os.Stderr, err)
	}
	info.Errors = append(info.Errors, err)
}

func (conf *Config) fset() *token.FileSet {
	if conf.Fset == nil {
		conf.Fset = token.NewFileSet()
	}
	return conf.Fset
}

// ParseFile is a convenience function (intended for testing) that invokes
// the parser using the Config's FileSet, which is initialized if nil.
//
// src specifies the parser input as a string, []byte, or io.Reader, and
// filename is its apparent name.  If src is nil, the contents of
// filename are read from the file system.
//
func (conf *Config) ParseFile(filename string, src interface{}) (*ast.File, error) {
	// TODO(adonovan): use conf.build() etc like parseFiles does.
	return parser.ParseFile(conf.fset(), filename, src, conf.ParserMode)
}

// FromArgsUsage is a partial usage message that applications calling
// FromArgs may wish to include in their -help output.
const FromArgsUsage = `
<args> is a list of arguments denoting a set of initial packages.
It may take one of two forms:

1. A list of *.go source files.

   All of the specified files are loaded, parsed and type-checked
   as a single package.  All the files must belong to the same directory.

2. A list of import paths, each denoting a package.

   The package's directory is found relative to the $GOROOT and
   $GOPATH using similar logic to 'go build', and the *.go files in
   that directory are loaded, parsed and type-checked as a single
   package.

   In addition, all *_test.go files in the directory are then loaded
   and parsed.  Those files whose package declaration equals that of
   the non-*_test.go files are included in the primary package.  Test
   files whose package declaration ends with "_test" are type-checked
   as another package, the 'external' test package, so that a single
   import path may denote two packages.  (Whether this behaviour is
   enabled is tool-specific, and may depend on additional flags.)

A '--' argument terminates the list of packages.
`

// FromArgs interprets args as a set of initial packages to load from
// source and updates the configuration.  It returns the list of
// unconsumed arguments.
//
// It is intended for use in command-line interfaces that require a
// set of initial packages to be specified; see FromArgsUsage message
// for details.
//
// Only superficial errors are reported at this stage; errors dependent
// on I/O are detected during Load.
//
func (conf *Config) FromArgs(args []string, xtest bool) ([]string, error) {
	var rest []string
	for i, arg := range args {
		if arg == "--" {
			rest = args[i+1:]
			args = args[:i]
			break // consume "--" and return the remaining args
		}
	}

	if len(args) > 0 && strings.HasSuffix(args[0], ".go") {
		// Assume args is a list of a *.go files
		// denoting a single ad hoc package.
		for _, arg := range args {
			if !strings.HasSuffix(arg, ".go") {
				return nil, fmt.Errorf("named files must be .go files: %s", arg)
			}
		}
		conf.CreateFromFilenames("", args...)
	} else {
		// Assume args are directories each denoting a
		// package and (perhaps) an external test, iff xtest.
		for _, arg := range args {
			if xtest {
				conf.ImportWithTests(arg)
			} else {
				conf.Import(arg)
			}
		}
	}

	return rest, nil
}

// CreateFromFilenames is a convenience function that adds
// a conf.CreatePkgs entry to create a package of the specified *.go
// files.
//
func (conf *Config) CreateFromFilenames(path string, filenames ...string) {
	conf.CreatePkgs = append(conf.CreatePkgs, PkgSpec{Path: path, Filenames: filenames})
}

// CreateFromFiles is a convenience function that adds a conf.CreatePkgs
// entry to create package of the specified path and parsed files.
//
func (conf *Config) CreateFromFiles(path string, files ...*ast.File) {
	conf.CreatePkgs = append(conf.CreatePkgs, PkgSpec{Path: path, Files: files})
}

// ImportWithTests is a convenience function that adds path to
// ImportPkgs, the set of initial source packages located relative to
// $GOPATH.  The package will be augmented by any *_test.go files in
// its directory that contain a "package x" (not "package x_test")
// declaration.
//
// In addition, if any *_test.go files contain a "package x_test"
// declaration, an additional package comprising just those files will
// be added to CreatePkgs.
//
func (conf *Config) ImportWithTests(path string) { conf.addImport(path, true) }

// Import is a convenience function that adds path to ImportPkgs, the
// set of initial packages that will be imported from source.
//
func (conf *Config) Import(path string) { conf.addImport(path, false) }

func (conf *Config) addImport(path string, tests bool) {
	if path == "C" || path == "unsafe" {
		return // ignore; not a real package
	}
	if conf.ImportPkgs == nil {
		conf.ImportPkgs = make(map[string]bool)
	}
	conf.ImportPkgs[path] = conf.ImportPkgs[path] || tests
}

// PathEnclosingInterval returns the PackageInfo and ast.Node that
// contain source interval [start, end), and all the node's ancestors
// up to the AST root.  It searches all ast.Files of all packages in prog.
// exact is defined as for astutil.PathEnclosingInterval.
//
// The zero value is returned if not found.
//
func (prog *Program) PathEnclosingInterval(start, end token.Pos) (pkg *PackageInfo, path []ast.Node, exact bool) {
	for _, info := range prog.AllPackages {
		for _, f := range info.Files {
			if f.Pos() == token.NoPos {
				// This can happen if the parser saw
				// too many errors and bailed out.
				// (Use parser.AllErrors to prevent that.)
				continue
			}
			if !tokenFileContainsPos(prog.Fset.File(f.Pos()), start) {
				continue
			}
			if path, exact := astutil.PathEnclosingInterval(f, start, end); path != nil {
				return info, path, exact
			}
		}
	}
	return nil, nil, false
}

// InitialPackages returns a new slice containing the set of initial
// packages (Created + Imported) in unspecified order.
//
func (prog *Program) InitialPackages() []*PackageInfo {
	infos := make([]*PackageInfo, 0, len(prog.Created)+len(prog.Imported))
	infos = append(infos, prog.Created...)
	for _, info := range prog.Imported {
		infos = append(infos, info)
	}
	return infos
}

// Package returns the ASTs and results of type checking for the
// specified package.
func (prog *Program) Package(path string) *PackageInfo {
	if info, ok := prog.AllPackages[prog.importMap[path]]; ok {
		return info
	}
	for _, info := range prog.Created {
		if path == info.Pkg.Path() {
			return info
		}
	}
	return nil
}

// ---------- Implementation ----------

// importer holds the working state of the algorithm.
type importer struct {
	conf  *Config   // the client configuration
	start time.Time // for logging

	progMu sync.Mutex // guards prog
	prog   *Program   // the resulting program

	// findpkg is a memoization of FindPackage.
	findpkgMu sync.Mutex // guards findpkg
	findpkg   map[findpkgKey]findpkgValue

	importedMu sync.Mutex             // guards imported
	imported   map[string]*importInfo // all imported packages (incl. failures) by import path

	// import dependency graph: graph[x][y] => x imports y
	//
	// Since non-importable packages cannot be cyclic, we ignore
	// their imports, thus we only need the subgraph over importable
	// packages.  Nodes are identified by their import paths.
	graphMu sync.Mutex
	graph   map[string]map[string]bool
}

type findpkgKey struct {
	importPath string
	fromDir    string
	mode       build.ImportMode
}

type findpkgValue struct {
	bp  *build.Package
	err error
}

// importInfo tracks the success or failure of a single import.
//
// Upon completion, exactly one of info and err is non-nil:
// info on successful creation of a package, err otherwise.
// A successful package may still contain type errors.
//
type importInfo struct {
	path     string        // import path
	info     *PackageInfo  // results of typechecking (including errors)
	complete chan struct{} // closed to broadcast that info is set.
}

// awaitCompletion blocks until ii is complete,
// i.e. the info field is safe to inspect.
func (ii *importInfo) awaitCompletion() {
	<-ii.complete // wait for close
}

// Complete marks ii as complete.
// Its info and err fields will not be subsequently updated.
func (ii *importInfo) Complete(info *PackageInfo) {
	if info == nil {
		panic("info == nil")
	}
	ii.info = info
	close(ii.complete)
}

type importError struct {
	path string // import path
	err  error  // reason for failure to create a package
}

// Load creates the initial packages specified by conf.{Create,Import}Pkgs,
// loading their dependencies packages as needed.
//
// On success, Load returns a Program containing a PackageInfo for
// each package.  On failure, it returns an error.
//
// If AllowErrors is true, Load will return a Program even if some
// packages contained I/O, parser or type errors, or if dependencies
// were missing.  (Such errors are accessible via PackageInfo.Errors.  If
// false, Load will fail if any package had an error.
//
// It is an error if no packages were loaded.
//
func (conf *Config) Load() (*Program, error) {
	// Create a simple default error handler for parse/type errors.
	if conf.TypeChecker.Error == nil {
		conf.TypeChecker.Error = func(e error) { fmt.Fprintln(os.Stderr, e) }
	}

	// Set default working directory for relative package references.
	if conf.Cwd == "" {
		var err error
		conf.Cwd, err = os.Getwd()
		if err != nil {
			return nil, err
		}
	}

	// Install default FindPackage hook using go/build logic.
	if conf.FindPackage == nil {
		conf.FindPackage = func(ctxt *build.Context, path, fromDir string, mode build.ImportMode) (*build.Package, error) {
			ioLimit <- true
			bp, err := ctxt.Import(path, fromDir, mode)
			<-ioLimit
			if _, ok := err.(*build.NoGoError); ok {
				return bp, nil // empty directory is not an error
			}
			return bp, err
		}
	}

	prog := &Program{
		Fset:        conf.fset(),
		Imported:    make(map[string]*PackageInfo),
		importMap:   make(map[string]*types.Package),
		AllPackages: make(map[*types.Package]*PackageInfo),
	}

	imp := importer{
		conf:     conf,
		prog:     prog,
		findpkg:  make(map[findpkgKey]findpkgValue),
		imported: make(map[string]*importInfo),
		start:    time.Now(),
		graph:    make(map[string]map[string]bool),
	}

	// -- loading proper (concurrent phase) --------------------------------

	var errpkgs []string // packages that contained errors

	// Load the initially imported packages and their dependencies,
	// in parallel.
	// No vendor check on packages imported from the command line.
	infos, importErrors := imp.importAll("", conf.Cwd, conf.ImportPkgs, 0)
	for _, ie := range importErrors {
		conf.TypeChecker.Error(ie.err) // failed to create package
		errpkgs = append(errpkgs, ie.path)
	}
	for _, info := range infos {
		prog.Imported[info.Pkg.Path()] = info
	}

	// Augment the designated initial packages by their tests.
	// Dependencies are loaded in parallel.
	var xtestPkgs []*build.Package
	for importPath, augment := range conf.ImportPkgs {
		if !augment {
			continue
		}

		// No vendor check on packages imported from command line.
		bp, err := imp.findPackage(importPath, conf.Cwd, 0)
		if err != nil {
			// Package not found, or can't even parse package declaration.
			// Already reported by previous loop; ignore it.
			continue
		}

		// Needs external test package?
		if len(bp.XTestGoFiles) > 0 {
			xtestPkgs = append(xtestPkgs, bp)
		}

		// Consult the cache using the canonical package path.
		path := bp.ImportPath
		imp.importedMu.Lock() // (unnecessary, we're sequential here)
		ii, ok := imp.imported[path]
		// Paranoid checks added due to issue #11012.
		if !ok {
			// Unreachable.
			// The previous loop called importAll and thus
			// startLoad for each path in ImportPkgs, which
			// populates imp.imported[path] with a non-zero value.
			panic(fmt.Sprintf("imported[%q] not found", path))
		}
		if ii == nil {
			// Unreachable.
			// The ii values in this loop are the same as in
			// the previous loop, which enforced the invariant
			// that at least one of ii.err and ii.info is non-nil.
			panic(fmt.Sprintf("imported[%q] == nil", path))
		}
		if ii.info == nil {
			// Unreachable.
			// awaitCompletion has the postcondition
			// ii.info != nil.
			panic(fmt.Sprintf("imported[%q].info = nil", path))
		}
		info := ii.info
		imp.importedMu.Unlock()

		// Parse the in-package test files.
		files, errs := imp.conf.parsePackageFiles(bp, 't')
		for _, err := range errs {
			info.appendError(err)
		}

		// The test files augmenting package P cannot be imported,
		// but may import packages that import P,
		// so we must disable the cycle check.
		imp.addFiles(info, files, false)
	}

	createPkg := func(path string, files []*ast.File, errs []error) {
		// TODO(adonovan): fix: use dirname of files, not cwd.
		info := imp.newPackageInfo(path, conf.Cwd)
		for _, err := range errs {
			info.appendError(err)
		}

		// Ad hoc packages are non-importable,
		// so no cycle check is needed.
		// addFiles loads dependencies in parallel.
		imp.addFiles(info, files, false)
		prog.Created = append(prog.Created, info)
	}

	// Create packages specified by conf.CreatePkgs.
	for _, cp := range conf.CreatePkgs {
		files, errs := parseFiles(conf.fset(), conf.build(), nil, ".", cp.Filenames, conf.ParserMode)
		files = append(files, cp.Files...)

		path := cp.Path
		if path == "" {
			if len(files) > 0 {
				path = files[0].Name.Name
			} else {
				path = "(unnamed)"
			}
		}
		createPkg(path, files, errs)
	}

	// Create external test packages.
	sort.Sort(byImportPath(xtestPkgs))
	for _, bp := range xtestPkgs {
		files, errs := imp.conf.parsePackageFiles(bp, 'x')
		createPkg(bp.ImportPath+"_test", files, errs)
	}

	// -- finishing up (sequential) ----------------------------------------

	if len(prog.Imported)+len(prog.Created) == 0 {
		return nil, errors.New("no initial packages were loaded")
	}

	// Create infos for indirectly imported packages.
	// e.g. incomplete packages without syntax, loaded from export data.
	for _, obj := range prog.importMap {
		info := prog.AllPackages[obj]
		if info == nil {
			prog.AllPackages[obj] = &PackageInfo{Pkg: obj, Importable: true}
		} else {
			// finished
			info.checker = nil
			info.errorFunc = nil
		}
	}

	if !conf.AllowErrors {
		// Report errors in indirectly imported packages.
		for _, info := range prog.AllPackages {
			if len(info.Errors) > 0 {
				errpkgs = append(errpkgs, info.Pkg.Path())
			}
		}
		if errpkgs != nil {
			var more string
			if len(errpkgs) > 3 {
				more = fmt.Sprintf(" and %d more", len(errpkgs)-3)
				errpkgs = errpkgs[:3]
			}
			return nil, fmt.Errorf("couldn't load packages due to errors: %s%s",
				strings.Join(errpkgs, ", "), more)
		}
	}

	markErrorFreePackages(prog.AllPackages)

	return prog, nil
}

type byImportPath []*build.Package

func (b byImportPath) Len() int           { return len(b) }
func (b byImportPath) Less(i, j int) bool { return b[i].ImportPath < b[j].ImportPath }
func (b byImportPath) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }

// markErrorFreePackages sets the TransitivelyErrorFree flag on all
// applicable packages.
func markErrorFreePackages(allPackages map[*types.Package]*PackageInfo) {
	// Build the transpose of the import graph.
	importedBy := make(map[*types.Package]map[*types.Package]bool)
	for P := range allPackages {
		for _, Q := range P.Imports() {
			clients, ok := importedBy[Q]
			if !ok {
				clients = make(map[*types.Package]bool)
				importedBy[Q] = clients
			}
			clients[P] = true
		}
	}

	// Find all packages reachable from some error package.
	reachable := make(map[*types.Package]bool)
	var visit func(*types.Package)
	visit = func(p *types.Package) {
		if !reachable[p] {
			reachable[p] = true
			for q := range importedBy[p] {
				visit(q)
			}
		}
	}
	for _, info := range allPackages {
		if len(info.Errors) > 0 {
			visit(info.Pkg)
		}
	}

	// Mark the others as "transitively error-free".
	for _, info := range allPackages {
		if !reachable[info.Pkg] {
			info.TransitivelyErrorFree = true
		}
	}
}

// build returns the effective build context.
func (conf *Config) build() *build.Context {
	if conf.Build != nil {
		return conf.Build
	}
	return &build.Default
}

// parsePackageFiles enumerates the files belonging to package path,
// then loads, parses and returns them, plus a list of I/O or parse
// errors that were encountered.
//
// 'which' indicates which files to include:
//    'g': include non-test *.go source files (GoFiles + processed CgoFiles)
//    't': include in-package *_test.go source files (TestGoFiles)
//    'x': include external *_test.go source files. (XTestGoFiles)
//
func (conf *Config) parsePackageFiles(bp *build.Package, which rune) ([]*ast.File, []error) {
	var filenames []string
	switch which {
	case 'g':
		filenames = bp.GoFiles
	case 't':
		filenames = bp.TestGoFiles
	case 'x':
		filenames = bp.XTestGoFiles
	default:
		panic(which)
	}

	files, errs := parseFiles(conf.fset(), conf.build(), conf.DisplayPath, bp.Dir, filenames, conf.ParserMode)

	// Preprocess CgoFiles and parse the outputs (sequentially).
	if which == 'g' && bp.CgoFiles != nil {
		cgofiles, err := processCgoFiles(bp, conf.fset(), conf.DisplayPath, conf.ParserMode)
		if err != nil {
			errs = append(errs, err)
		} else {
			files = append(files, cgofiles...)
		}
	}

	return files, errs
}

// doImport imports the package denoted by path.
// It implements the types.Importer signature.
//
// It returns an error if a package could not be created
// (e.g. go/build or parse error), but type errors are reported via
// the types.Config.Error callback (the first of which is also saved
// in the package's PackageInfo).
//
// Idempotent.
//
func (imp *importer) doImport(from *PackageInfo, to string) (*types.Package, error) {
	// Package unsafe is handled specially, and has no PackageInfo.
	// (Let's assume there is no "vendor/unsafe" package.)
	if to == "unsafe" {
		return types.Unsafe, nil
	}
	if to == "C" {
		// This should be unreachable, but ad hoc packages are
		// not currently subject to cgo preprocessing.
		// See https://github.com/golang/go/issues/11627.
		return nil, fmt.Errorf(`the loader doesn't cgo-process ad hoc packages like %q; see Go issue 11627`,
			from.Pkg.Path())
	}

	bp, err := imp.findPackage(to, from.dir, buildutil.AllowVendor)
	if err != nil {
		return nil, err
	}

	// Look for the package in the cache using its canonical path.
	path := bp.ImportPath
	imp.importedMu.Lock()
	ii := imp.imported[path]
	imp.importedMu.Unlock()
	if ii == nil {
		panic("internal error: unexpected import: " + path)
	}
	if ii.info != nil {
		return ii.info.Pkg, nil
	}

	// Import of incomplete package: this indicates a cycle.
	fromPath := from.Pkg.Path()
	if cycle := imp.findPath(path, fromPath); cycle != nil {
		cycle = append([]string{fromPath}, cycle...)
		return nil, fmt.Errorf("import cycle: %s", strings.Join(cycle, " -> "))
	}

	panic("internal error: import of incomplete (yet acyclic) package: " + fromPath)
}

// findPackage locates the package denoted by the importPath in the
// specified directory.
func (imp *importer) findPackage(importPath, fromDir string, mode build.ImportMode) (*build.Package, error) {
	// TODO(adonovan): opt: non-blocking duplicate-suppressing cache.
	// i.e. don't hold the lock around FindPackage.
	key := findpkgKey{importPath, fromDir, mode}
	imp.findpkgMu.Lock()
	defer imp.findpkgMu.Unlock()
	v, ok := imp.findpkg[key]
	if !ok {
		bp, err := imp.conf.FindPackage(imp.conf.build(), importPath, fromDir, mode)
		v = findpkgValue{bp, err}
		imp.findpkg[key] = v
	}
	return v.bp, v.err
}

// importAll loads, parses, and type-checks the specified packages in
// parallel and returns their completed importInfos in unspecified order.
//
// fromPath is the package path of the importing package, if it is
// importable, "" otherwise.  It is used for cycle detection.
//
// fromDir is the directory containing the import declaration that
// caused these imports.
//
func (imp *importer) importAll(fromPath, fromDir string, imports map[string]bool, mode build.ImportMode) (infos []*PackageInfo, errors []importError) {
	// TODO(adonovan): opt: do the loop in parallel once
	// findPackage is non-blocking.
	var pending []*importInfo
	for importPath := range imports {
		bp, err := imp.findPackage(importPath, fromDir, mode)
		if err != nil {
			errors = append(errors, importError{
				path: importPath,
				err:  err,
			})
			continue
		}
		pending = append(pending, imp.startLoad(bp))
	}

	if fromPath != "" {
		// We're loading a set of imports.
		//
		// We must record graph edges from the importing package
		// to its dependencies, and check for cycles.
		imp.graphMu.Lock()
		deps, ok := imp.graph[fromPath]
		if !ok {
			deps = make(map[string]bool)
			imp.graph[fromPath] = deps
		}
		for _, ii := range pending {
			deps[ii.path] = true
		}
		imp.graphMu.Unlock()
	}

	for _, ii := range pending {
		if fromPath != "" {
			if cycle := imp.findPath(ii.path, fromPath); cycle != nil {
				// Cycle-forming import: we must not await its
				// completion since it would deadlock.
				//
				// We don't record the error in ii since
				// the error is really associated with the
				// cycle-forming edge, not the package itself.
				// (Also it would complicate the
				// invariants of importPath completion.)
				if trace {
					fmt.Fprintln(os.Stderr, "import cycle: %q", cycle)
				}
				continue
			}
		}
		ii.awaitCompletion()
		infos = append(infos, ii.info)
	}

	return infos, errors
}

// findPath returns an arbitrary path from 'from' to 'to' in the import
// graph, or nil if there was none.
func (imp *importer) findPath(from, to string) []string {
	imp.graphMu.Lock()
	defer imp.graphMu.Unlock()

	seen := make(map[string]bool)
	var search func(stack []string, importPath string) []string
	search = func(stack []string, importPath string) []string {
		if !seen[importPath] {
			seen[importPath] = true
			stack = append(stack, importPath)
			if importPath == to {
				return stack
			}
			for x := range imp.graph[importPath] {
				if p := search(stack, x); p != nil {
					return p
				}
			}
		}
		return nil
	}
	return search(make([]string, 0, 20), from)
}

// startLoad initiates the loading, parsing and type-checking of the
// specified package and its dependencies, if it has not already begun.
//
// It returns an importInfo, not necessarily in a completed state.  The
// caller must call awaitCompletion() before accessing its info field.
//
// startLoad is concurrency-safe and idempotent.
//
func (imp *importer) startLoad(bp *build.Package) *importInfo {
	path := bp.ImportPath
	imp.importedMu.Lock()
	ii, ok := imp.imported[path]
	if !ok {
		ii = &importInfo{path: path, complete: make(chan struct{})}
		imp.imported[path] = ii
		go func() {
			info := imp.load(bp)
			ii.Complete(info)
		}()
	}
	imp.importedMu.Unlock()

	return ii
}

// load implements package loading by parsing Go source files
// located by go/build.
func (imp *importer) load(bp *build.Package) *PackageInfo {
	info := imp.newPackageInfo(bp.ImportPath, bp.Dir)
	info.Importable = true
	files, errs := imp.conf.parsePackageFiles(bp, 'g')
	for _, err := range errs {
		info.appendError(err)
	}

	imp.addFiles(info, files, true)

	imp.progMu.Lock()
	imp.prog.importMap[bp.ImportPath] = info.Pkg
	imp.progMu.Unlock()

	return info
}

// addFiles adds and type-checks the specified files to info, loading
// their dependencies if needed.  The order of files determines the
// package initialization order.  It may be called multiple times on the
// same package.  Errors are appended to the info.Errors field.
//
// cycleCheck determines whether the imports within files create
// dependency edges that should be checked for potential cycles.
//
func (imp *importer) addFiles(info *PackageInfo, files []*ast.File, cycleCheck bool) {
	info.Files = append(info.Files, files...)

	// Ensure the dependencies are loaded, in parallel.
	var fromPath string
	if cycleCheck {
		fromPath = info.Pkg.Path()
	}
	// TODO(adonovan): opt: make the caller do scanImports.
	// Callers with a build.Package can skip it.
	imp.importAll(fromPath, info.dir, scanImports(files), buildutil.AllowVendor)

	if trace {
		fmt.Fprintf(os.Stderr, "%s: start %q (%d)\n",
			time.Since(imp.start), info.Pkg.Path(), len(files))
	}

	// Ignore the returned (first) error since we
	// already collect them all in the PackageInfo.
	info.checker.Files(files)

	if trace {
		fmt.Fprintf(os.Stderr, "%s: stop %q\n",
			time.Since(imp.start), info.Pkg.Path())
	}
}

func (imp *importer) newPackageInfo(path, dir string) *PackageInfo {
	pkg := types.NewPackage(path, "")
	info := &PackageInfo{
		Pkg: pkg,
		Info: types.Info{
			Types:      make(map[ast.Expr]types.TypeAndValue),
			Defs:       make(map[*ast.Ident]types.Object),
			Uses:       make(map[*ast.Ident]types.Object),
			Implicits:  make(map[ast.Node]types.Object),
			Scopes:     make(map[ast.Node]*types.Scope),
			Selections: make(map[*ast.SelectorExpr]*types.Selection),
		},
		errorFunc: imp.conf.TypeChecker.Error,
		dir:       dir,
	}

	// Copy the types.Config so we can vary it across PackageInfos.
	tc := imp.conf.TypeChecker
	tc.IgnoreFuncBodies = false
	if f := imp.conf.TypeCheckFuncBodies; f != nil {
		tc.IgnoreFuncBodies = !f(path)
	}
	tc.Importer = closure{imp, info}
	tc.Error = info.appendError // appendError wraps the user's Error function

	info.checker = types.NewChecker(&tc, imp.conf.fset(), pkg, &info.Info)
	imp.progMu.Lock()
	imp.prog.AllPackages[pkg] = info
	imp.progMu.Unlock()
	return info
}

type closure struct {
	imp  *importer
	info *PackageInfo
}

func (c closure) Import(to string) (*types.Package, error) { return c.imp.doImport(c.info, to) }
