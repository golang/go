// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package loader loads, parses and type-checks packages of Go code
// plus their transitive closure, and retains both the ASTs and the
// derived facts.
//
// THIS INTERFACE IS EXPERIMENTAL AND IS LIKELY TO CHANGE.
//
// The package defines two primary types: Config, which specifies a
// set of initial packages to load and various other options; and
// Program, which is the result of successfully loading the packages
// specified by a configuration.
//
// The configuration can be set directly, but *Config provides various
// convenience methods to simplify the common cases, each of which can
// be called any number of times.  Finally, these are followed by a
// call to Load() to actually load and type-check the program.
//
//      var conf loader.Config
//
//      // Use the command-line arguments to specify
//      // a set of initial packages to load from source.
//      // See FromArgsUsage for help.
//      rest, err := conf.FromArgs(os.Args[1:], wantTests)
//
//      // Parse the specified files and create an ad-hoc package with path "foo".
//      // All files must have the same 'package' declaration.
//      conf.CreateFromFilenames("foo", "foo.go", "bar.go")
//
//      // Create an ad-hoc package with path "foo" from
//      // the specified already-parsed files.
//      // All ASTs must have the same 'package' declaration.
//      conf.CreateFromFiles("foo", parsedFiles)
//
//      // Add "runtime" to the set of packages to be loaded.
//      conf.Import("runtime")
//
//      // Adds "fmt" and "fmt_test" to the set of packages
//      // to be loaded.  "fmt" will include *_test.go files.
//      conf.ImportWithTests("fmt")
//
//      // Finally, load all the packages specified by the configuration.
//      prog, err := conf.Load()
//
//
// CONCEPTS AND TERMINOLOGY
//
// An AD-HOC package is one specified as a set of source files on the
// command line.  In the simplest case, it may consist of a single file
// such as $GOROOT/src/net/http/triv.go.
//
// EXTERNAL TEST packages are those comprised of a set of *_test.go
// files all with the same 'package foo_test' declaration, all in the
// same directory.  (go/build.Package calls these files XTestFiles.)
//
// An IMPORTABLE package is one that can be referred to by some import
// spec.  The Path() of each importable package is unique within a
// Program.
//
// Ad-hoc packages and external test packages are NON-IMPORTABLE.  The
// Path() of an ad-hoc package is inferred from the package
// declarations of its files and is therefore not a unique package key.
// For example, Config.CreatePkgs may specify two initial ad-hoc
// packages both called "main".
//
// An AUGMENTED package is an importable package P plus all the
// *_test.go files with same 'package foo' declaration as P.
// (go/build.Package calls these files TestFiles.)
//
// The INITIAL packages are those specified in the configuration.  A
// DEPENDENCY is a package loaded to satisfy an import in an initial
// package or another dependency.
//
package loader

// 'go test', in-package test files, and import cycles
// ---------------------------------------------------
//
// An external test package may depend upon members of the augmented
// package that are not in the unaugmented package, such as functions
// that expose internals.  (See bufio/export_test.go for an example.)
// So, the loader must ensure that for each external test package
// it loads, it also augments the corresponding non-test package.
//
// The import graph over n unaugmented packages must be acyclic; the
// import graph over n-1 unaugmented packages plus one augmented
// package must also be acyclic.  ('go test' relies on this.)  But the
// import graph over n augmented packages may contain cycles.
//
// First, all the (unaugmented) non-test packages and their
// dependencies are imported in the usual way; the loader reports an
// error if it detects an import cycle.
//
// Then, each package P for which testing is desired is augmented by
// the list P' of its in-package test files, by calling
// (*types.Checker).Files.  This arrangement ensures that P' may
// reference definitions within P, but P may not reference definitions
// within P'.  Furthermore, P' may import any other package, including
// ones that depend upon P, without an import cycle error.
//
// Consider two packages A and B, both of which have lists of
// in-package test files we'll call A' and B', and which have the
// following import graph edges:
//    B  imports A
//    B' imports A
//    A' imports B
// This last edge would be expected to create an error were it not
// for the special type-checking discipline above.
// Cycles of size greater than two are possible.  For example:
//   compress/bzip2/bzip2_test.go (package bzip2)  imports "io/ioutil"
//   io/ioutil/tempfile_test.go   (package ioutil) imports "regexp"
//   regexp/exec_test.go          (package regexp) imports "compress/bzip2"
//
//
// Concurrency
// -----------
//
// Let us define the import dependency graph as follows.  Each node is a
// list of files passed to (Checker).Files at once.  Many of these lists
// are the production code of an importable Go package, so those nodes
// are labelled by the package's import path.  The remaining nodes are
// ad-hoc packages and lists of in-package *_test.go files that augment
// an importable package; those nodes have no label.
//
// The edges of the graph represent import statements appearing within a
// file.  An edge connects a node (a list of files) to the node it
// imports, which is importable and thus always labelled.
//
// Loading is controlled by this dependency graph.
//
// To reduce I/O latency, we start loading a package's dependencies
// asynchronously as soon as we've parsed its files and enumerated its
// imports (scanImports).  This performs a preorder traversal of the
// import dependency graph.
//
// To exploit hardware parallelism, we type-check unrelated packages in
// parallel, where "unrelated" means not ordered by the partial order of
// the import dependency graph.
//
// We use a concurrency-safe blocking cache (importer.imported) to
// record the results of type-checking, whether success or failure.  An
// entry is created in this cache by startLoad the first time the
// package is imported.  The first goroutine to request an entry becomes
// responsible for completing the task and broadcasting completion to
// subsequent requestors, which block until then.
//
// Type checking occurs in (parallel) postorder: we cannot type-check a
// set of files until we have loaded and type-checked all of their
// immediate dependencies (and thus all of their transitive
// dependencies). If the input were guaranteed free of import cycles,
// this would be trivial: we could simply wait for completion of the
// dependencies and then invoke the typechecker.
//
// But as we saw in the 'go test' section above, some cycles in the
// import graph over packages are actually legal, so long as the
// cycle-forming edge originates in the in-package test files that
// augment the package.  This explains why the nodes of the import
// dependency graph are not packages, but lists of files: the unlabelled
// nodes avoid the cycles.  Consider packages A and B where B imports A
// and A's in-package tests AT import B.  The naively constructed import
// graph over packages would contain a cycle (A+AT) --> B --> (A+AT) but
// the graph over lists of files is AT --> B --> A, where AT is an
// unlabelled node.
//
// Awaiting completion of the dependencies in a cyclic graph would
// deadlock, so we must materialize the import dependency graph (as
// importer.graph) and check whether each import edge forms a cycle.  If
// x imports y, and the graph already contains a path from y to x, then
// there is an import cycle, in which case the processing of x must not
// wait for the completion of processing of y.
//
// When the type-checker makes a callback (doImport) to the loader for a
// given import edge, there are two possible cases.  In the normal case,
// the dependency has already been completely type-checked; doImport
// does a cache lookup and returns it.  In the cyclic case, the entry in
// the cache is still necessarily incomplete, indicating a cycle.  We
// perform the cycle check again to obtain the error message, and return
// the error.
//
// The result of using concurrency is about a 2.5x speedup for stdlib_test.

// TODO(adonovan):
// - cache the calls to build.Import so we don't do it three times per
//   test package.
// - Thorough overhaul of package documentation.

import (
	"errors"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/gcimporter"
	"golang.org/x/tools/go/types"
)

const trace = false // show timing info for type-checking

// Config specifies the configuration for a program to load.
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
	//
	// TypeChecker.Packages is lazily initialized during Load.
	TypeChecker types.Config

	// TypeCheckFuncBodies is a predicate over package import
	// paths.  A package for which the predicate is false will
	// have its package-level declarations type checked, but not
	// its function bodies; this can be used to quickly load
	// dependencies from source.  If nil, all func bodies are type
	// checked.
	TypeCheckFuncBodies func(string) bool

	// ImportFromBinary determines whether to satisfy dependencies by
	// loading gc export data instead of Go source code.
	//
	// If false, the entire program---the initial packages and their
	// transitive closure of dependencies---will be loaded from
	// source, parsed, and type-checked.  This is required for
	// whole-program analyses such as pointer analysis.
	//
	// If true, the go/gcimporter mechanism is used instead to read
	// the binary export-data files written by the gc toolchain.
	// They supply only the types of package-level declarations and
	// values of constants, but no code, this option will not yield
	// a whole program.  It is intended for analyses that perform
	// modular analysis of a single package, e.g. traditional
	// compilation.
	//
	// No check is made that the export data files are up-to-date.
	//
	// The initial packages (CreatePkgs and ImportPkgs) are always
	// loaded from Go source, regardless of this flag's setting.
	//
	// NB: there is a bug when loading multiple initial packages with
	// this flag enabled: https://github.com/golang/go/issues/9955.
	ImportFromBinary bool

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

	// ImportPkgs specifies a set of initial packages to load from
	// source.  The map keys are package import paths, used to
	// locate the package relative to $GOROOT.
	//
	// The map value indicates whether to load tests.  If true, Load
	// will add and type-check two lists of files to the package:
	// non-test files followed by in-package *_test.go files.  In
	// addition, it will append the external test package (if any)
	// to Program.Created.
	ImportPkgs map[string]bool

	// FindPackage is called during Load to create the build.Package
	// for a given import path.  If nil, a default implementation
	// based on ctxt.Import is used.  A client may use this hook to
	// adapt to a proprietary build system that does not follow the
	// "go build" layout conventions, for example.
	//
	// It must be safe to call concurrently from multiple goroutines.
	FindPackage func(ctxt *build.Context, importPath string) (*build.Package, error)

	// PackageCreated is a hook called when a types.Package
	// is created but before it has been populated.
	//
	// The package's import Path() and Scope() are defined,
	// but not its Name() since no package declaration has
	// been seen yet.
	//
	// Clients may use this to insert synthetic items into
	// the package scope, for example.
	//
	// It must be safe to call concurrently from multiple goroutines.
	PackageCreated func(*types.Package)
}

// A PkgSpec specifies a non-importable package to be created by Load.
// Files are processed first, but typically only one of Files and
// Filenames is provided.  The path needn't be globally unique.
//
type PkgSpec struct {
	Path      string      // import path ("" => use package declaration)
	Files     []*ast.File // ASTs of already-parsed files
	Filenames []string    // names of files to be parsed
}

// A Program is a Go program loaded from source or binary
// as specified by a Config.
type Program struct {
	Fset *token.FileSet // the file set for this program

	// Created[i] contains the initial package whose ASTs or
	// filenames were supplied by Config.CreatePkgs[i], followed by
	// the external test package, if any, of each package in
	// Config.ImportPkgs ordered by ImportPath.
	Created []*PackageInfo

	// Imported contains the initially imported packages,
	// as specified by Config.ImportPkgs.
	Imported map[string]*PackageInfo

	// ImportMap is the canonical mapping of import paths to
	// packages used by the type-checker (Config.TypeChecker.Packages).
	// It contains all Imported initial packages, but not Created
	// ones, and all imported dependencies.
	ImportMap map[string]*types.Package

	// AllPackages contains the PackageInfo of every package
	// encountered by Load: all initial packages and all
	// dependencies, including incomplete ones.
	AllPackages map[*types.Package]*PackageInfo
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
		// denoting a single ad-hoc package.
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
	if path == "unsafe" {
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

// ---------- Implementation ----------

// importer holds the working state of the algorithm.
type importer struct {
	conf  *Config   // the client configuration
	prog  *Program  // resulting program
	start time.Time // for logging

	// This mutex serializes access to prog.ImportMap (aka
	// TypeChecker.Packages); we also use it for AllPackages.
	//
	// The TypeChecker.Packages map is not really used by this
	// package, but may be used by the client's Import function,
	// and by clients of the returned Program.
	typecheckerMu sync.Mutex

	importedMu sync.Mutex
	imported   map[string]*importInfo // all imported packages (incl. failures) by import path

	// import dependency graph: graph[x][y] => x imports y
	//
	// Since non-importable packages cannot be cyclic, we ignore
	// their imports, thus we only need the subgraph over importable
	// packages.  Nodes are identified by their import paths.
	graphMu sync.Mutex
	graph   map[string]map[string]bool
}

// importInfo tracks the success or failure of a single import.
//
// Upon completion, exactly one of info and err is non-nil:
// info on successful creation of a package, err otherwise.
// A successful package may still contain type errors.
//
type importInfo struct {
	path     string       // import path
	mu       sync.Mutex   // guards the following fields prior to completion
	info     *PackageInfo // results of typechecking (including errors)
	err      error        // reason for failure to create a package
	complete sync.Cond    // complete condition is that one of info, err is non-nil.
}

// awaitCompletion blocks until ii is complete,
// i.e. the info and err fields are safe to inspect without a lock.
// It is concurrency-safe and idempotent.
func (ii *importInfo) awaitCompletion() {
	ii.mu.Lock()
	for ii.info == nil && ii.err == nil {
		ii.complete.Wait()
	}
	ii.mu.Unlock()
}

// Complete marks ii as complete.
// Its info and err fields will not be subsequently updated.
func (ii *importInfo) Complete(info *PackageInfo, err error) {
	ii.mu.Lock()
	ii.info = info
	ii.err = err
	ii.complete.Broadcast()
	ii.mu.Unlock()
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
	// Initialize by setting the conf's copy, so all copies of
	// TypeChecker agree on the identity of the map.
	if conf.TypeChecker.Packages == nil {
		conf.TypeChecker.Packages = make(map[string]*types.Package)
	}

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
		conf.FindPackage = func(ctxt *build.Context, path string) (*build.Package, error) {
			bp, err := ctxt.Import(path, conf.Cwd, 0)
			if _, ok := err.(*build.NoGoError); ok {
				return bp, nil // empty directory is not an error
			}
			return bp, err
		}
	}

	prog := &Program{
		Fset:        conf.fset(),
		Imported:    make(map[string]*PackageInfo),
		ImportMap:   conf.TypeChecker.Packages,
		AllPackages: make(map[*types.Package]*PackageInfo),
	}

	imp := importer{
		conf:     conf,
		prog:     prog,
		imported: make(map[string]*importInfo),
		start:    time.Now(),
		graph:    make(map[string]map[string]bool),
	}

	// -- loading proper (concurrent phase) --------------------------------

	var errpkgs []string // packages that contained errors

	// Load the initially imported packages and their dependencies,
	// in parallel.
	for _, ii := range imp.loadAll("", conf.ImportPkgs) {
		if ii.err != nil {
			conf.TypeChecker.Error(ii.err) // failed to create package
			errpkgs = append(errpkgs, ii.path)
			continue
		}
		prog.Imported[ii.info.Pkg.Path()] = ii.info
	}

	// Augment the designated initial packages by their tests.
	// Dependencies are loaded in parallel.
	var xtestPkgs []*build.Package
	for path, augment := range conf.ImportPkgs {
		if !augment {
			continue
		}

		bp, err := conf.FindPackage(conf.build(), path)
		if err != nil {
			// Package not found, or can't even parse package declaration.
			// Already reported by previous loop; ignore it.
			continue
		}

		// Needs external test package?
		if len(bp.XTestGoFiles) > 0 {
			xtestPkgs = append(xtestPkgs, bp)
		}

		imp.importedMu.Lock()           // (unnecessary, we're sequential here)
		info := imp.imported[path].info // must be non-nil, see above
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
		info := imp.newPackageInfo(path)
		for _, err := range errs {
			info.appendError(err)
		}

		// Ad-hoc packages are non-importable,
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
	for _, obj := range prog.ImportMap {
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
// imports is the type-checker's package canonicalization map.
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
	// TODO(adonovan): move this check into go/types?
	if to == "unsafe" {
		return types.Unsafe, nil
	}

	imp.importedMu.Lock()
	ii := imp.imported[to]
	imp.importedMu.Unlock()
	if ii == nil {
		panic("internal error: unexpected import: " + to)
	}
	if ii.err != nil {
		return nil, ii.err
	}
	if ii.info != nil {
		return ii.info.Pkg, nil
	}

	// Import of incomplete package: this indicates a cycle.
	fromPath := from.Pkg.Path()
	if cycle := imp.findPath(to, fromPath); cycle != nil {
		cycle = append([]string{fromPath}, cycle...)
		return nil, fmt.Errorf("import cycle: %s", strings.Join(cycle, " -> "))
	}

	panic("internal error: import of incomplete (yet acyclic) package: " + fromPath)
}

// loadAll loads, parses, and type-checks the specified packages in
// parallel and returns their completed importInfos in unspecified order.
//
// fromPath is the import path of the importing package, if it is
// importable, "" otherwise.  It is used for cycle detection.
//
func (imp *importer) loadAll(fromPath string, paths map[string]bool) []*importInfo {
	result := make([]*importInfo, 0, len(paths))
	for path := range paths {
		result = append(result, imp.startLoad(path))
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
		for path := range paths {
			deps[path] = true
		}
		imp.graphMu.Unlock()
	}

	for _, ii := range result {
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

	}
	return result
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
// caller must call awaitCompletion() before accessing its info and err
// fields.
//
// startLoad is concurrency-safe and idempotent.
//
// Precondition: path != "unsafe".
//
func (imp *importer) startLoad(path string) *importInfo {
	imp.importedMu.Lock()
	ii, ok := imp.imported[path]
	if !ok {
		ii = &importInfo{path: path}
		ii.complete.L = &ii.mu
		imp.imported[path] = ii

		go imp.load(path, ii)
	}
	imp.importedMu.Unlock()

	return ii
}

func (imp *importer) load(path string, ii *importInfo) {
	var info *PackageInfo
	var err error
	// Find and create the actual package.
	if _, ok := imp.conf.ImportPkgs[path]; ok || !imp.conf.ImportFromBinary {
		info, err = imp.loadFromSource(path)
	} else {
		info, err = imp.importFromBinary(path)
	}
	ii.Complete(info, err)
}

// importFromBinary implements package loading from the client-supplied
// external source, e.g. object files from the gc compiler.
//
func (imp *importer) importFromBinary(path string) (*PackageInfo, error) {
	// Determine the caller's effective Import function.
	importfn := imp.conf.TypeChecker.Import
	if importfn == nil {
		importfn = gcimporter.Import
	}
	imp.typecheckerMu.Lock()
	pkg, err := importfn(imp.conf.TypeChecker.Packages, path)
	if pkg != nil {
		imp.conf.TypeChecker.Packages[path] = pkg
	}
	imp.typecheckerMu.Unlock()
	if err != nil {
		return nil, err
	}
	info := &PackageInfo{Pkg: pkg}
	info.Importable = true
	imp.typecheckerMu.Lock()
	imp.prog.AllPackages[pkg] = info
	imp.typecheckerMu.Unlock()
	return info, nil
}

// loadFromSource implements package loading by parsing Go source files
// located by go/build.
// The returned PackageInfo's typeCheck function must be called.
//
func (imp *importer) loadFromSource(path string) (*PackageInfo, error) {
	bp, err := imp.conf.FindPackage(imp.conf.build(), path)
	if err != nil {
		return nil, err // package not found
	}
	info := imp.newPackageInfo(bp.ImportPath)
	info.Importable = true
	files, errs := imp.conf.parsePackageFiles(bp, 'g')
	for _, err := range errs {
		info.appendError(err)
	}

	imp.addFiles(info, files, true)

	imp.typecheckerMu.Lock()
	imp.conf.TypeChecker.Packages[path] = info.Pkg
	imp.typecheckerMu.Unlock()

	return info, nil
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
	imp.loadAll(fromPath, scanImports(files))

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

func (imp *importer) newPackageInfo(path string) *PackageInfo {
	pkg := types.NewPackage(path, "")
	if imp.conf.PackageCreated != nil {
		imp.conf.PackageCreated(pkg)
	}
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
	}

	// Copy the types.Config so we can vary it across PackageInfos.
	tc := imp.conf.TypeChecker
	tc.IgnoreFuncBodies = false
	if f := imp.conf.TypeCheckFuncBodies; f != nil {
		tc.IgnoreFuncBodies = !f(path)
	}
	tc.Import = func(_ map[string]*types.Package, to string) (*types.Package, error) {
		return imp.doImport(info, to)
	}
	tc.Error = info.appendError // appendError wraps the user's Error function

	info.checker = types.NewChecker(&tc, imp.conf.fset(), pkg, &info.Info)
	imp.typecheckerMu.Lock()
	imp.prog.AllPackages[pkg] = info
	imp.typecheckerMu.Unlock()
	return info
}
