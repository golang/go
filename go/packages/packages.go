// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package packages

// See doc.go for package documentation and implementation notes.

import (
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"log"
	"os"
	"sync"

	"golang.org/x/tools/go/gcexportdata"
)

// An Options holds the options for a call to Metadata, TypeCheck
// or WholeProgram to load Go packages from source code.
type Options struct {
	// Fset is the file set for the parser
	// to use when loading the program.
	Fset *token.FileSet

	// Context may be used to cancel a pending call.
	// Context is optional; the default behavior
	// is equivalent to context.Background().
	Context context.Context

	// The Tests flag causes the result to include any test packages
	// implied by the patterns.
	//
	// For example, under 'go build', the "fmt" pattern ordinarily
	// identifies a single importable package, but with the Tests
	// flag it additionally denotes the "fmt.test" executable, which
	// in turn depends on the variant of "fmt" augmented by its
	// in-packages tests, and the "fmt_test" external test package.
	//
	// For build systems in which test names are explicit,
	// this flag may have no effect.
	Tests bool

	// DisableCgo disables cgo-processing of files that import "C",
	// and removes the 'cgo' build tag, which may affect source file selection.
	// By default, TypeCheck, and WholeProgram queries process such
	// files, and the resulting Package.Srcs describes the generated
	// files seen by the compiler.
	DisableCgo bool

	// TypeChecker contains options relating to the type checker,
	// such as the Sizes function.
	//
	// The following fields of TypeChecker are ignored:
	// - Import: the Loader provides the import machinery.
	// - Error: errors are reported to the Error function, below.
	TypeChecker types.Config

	// Error is called for each error encountered during package loading.
	// Implementations must be concurrency-safe.
	// If nil, the default implementation prints errors to os.Stderr.
	// Errors are additionally recorded in each Package.
	// Error is not used in Metadata mode.
	Error func(error)

	// ParseFile is called to read and parse each file,
	// Implementations must be concurrency-safe.
	// If nil, the default implementation uses parser.ParseFile.
	// A client may supply a custom implementation to,
	// for example, provide alternative contents for files
	// modified in a text editor but unsaved,
	// or to selectively eliminate unwanted function
	// bodies to reduce the load on the type-checker.
	// ParseFile is not used in Metadata mode.
	ParseFile func(fset *token.FileSet, filename string) (*ast.File, error)

	// Env is a list of environment variables to pass through
	// to the build system's metadata query tool.
	// If nil, the current process's environment is used.
	Env []string

	// Dir is the directory in which to run the build system's metadata query tool.
	// If "", the current process's working directory is used.
	Dir string
}

// Metadata returns the metadata for a set of Go packages,
// but does not parse or type-check their source files.
// The returned packages are the roots of a directed acyclic graph,
// the "import graph", whose edges are represented by Package.Imports
// and whose transitive closure includes all dependencies of the
// initial packages.
//
// The packages are denoted by patterns, using the usual notation of the
// build system (currently "go build", but in future others such as
// Bazel). Clients should not attempt to infer the relationship between
// patterns and the packages they denote, as in general it is complex
// and many-to-many. Metadata reports an error if the patterns denote no
// packages.
//
// If Metadata was unable to expand the specified patterns to a set of
// packages, or if there was a cycle in the dependency graph, it returns
// an error. Otherwise it returns a set of loaded Packages, even if
// errors were encountered while loading some of them; such errors are
// recorded in each Package.
//
func Metadata(o *Options, patterns ...string) ([]*Package, error) {
	l := &loader{mode: metadata}
	if o != nil {
		l.Options = *o
	}
	return l.load(patterns...)
}

// TypeCheck returns metadata, syntax trees, and type information
// for a set of Go packages.
//
// In addition to the information returned by the Metadata function,
// TypeCheck loads, parses, and type-checks each of the requested packages.
// These packages are "source packages", and the resulting Package
// structure provides complete syntax and type information.
// Due to limitations of the type checker, any package that transitively
// depends on a source package must also be loaded from source.
//
// For each immediate dependency of a source package that is not itself
// a source package, type information is obtained from export data
// files produced by the Go compiler; this mode may entail a partial build.
// The Package for these dependencies provides complete package-level type
// information (types.Package), but no syntax trees.
//
// The remaining packages, comprising the indirect dependencies of the
// packages with complete export data, may have partial package-level type
// information or perhaps none at all.
//
// For example, consider the import graph A->B->C->D->E.
// If the requested packages are A and C,
// then packages A, B, C are source packages,
// D is a complete export data package,
// and E is a partial export data package.
// (B must be a source package because it
// transitively depends on C, a source package.)
//
// Each package bears a flag, IllTyped, indicating whether it
// or one of its transitive dependencies contains an error.
// A package that is not IllTyped is buildable.
//
// Use this mode for compiler-like tools
// that analyze one package at a time.
//
func TypeCheck(o *Options, patterns ...string) ([]*Package, error) {
	l := &loader{mode: typeCheck}
	if o != nil {
		l.Options = *o
	}
	return l.load(patterns...)
}

// WholeProgram returns metadata, complete syntax trees, and complete
// type information for a set of Go packages and their entire transitive
// closure of dependencies.
// Every package in the returned import graph is a source package,
// as defined by the documentation for TypeCheck
//
// Use this mode for whole-program analysis tools.
//
func WholeProgram(o *Options, patterns ...string) ([]*Package, error) {
	l := &loader{mode: wholeProgram}
	if o != nil {
		l.Options = *o
	}
	return l.load(patterns...)
}

// Package holds the metadata, and optionally syntax trees
// and type information, for a single Go package.
//
// The import graph, Imports, forms a directed acyclic graph over Packages.
// (Cycle-forming edges are not inserted into the map.)
//
// A Package is not mutated once returned.
type Package struct {
	// ID is a unique, opaque identifier for a package,
	// as determined by the underlying workspace.
	//
	// IDs distinguish packages that have the same PkgPath, such as
	// a regular package and the variant of that package built
	// during testing. (IDs also distinguish packages that would be
	// lumped together by the go/build API, such as a regular
	// package and its external tests.)
	//
	// Clients should not interpret the ID string as its
	// structure varies from one build system to another.
	ID string

	// PkgPath is the path of the package as understood
	// by the Go compiler and by reflect.Type.PkgPath.
	//
	// PkgPaths are unique for each package in a given executable
	// program, but are not necessarily unique within a workspace.
	// For example, an importable package (fmt) and its in-package
	// tests (fmt·test) may have the same PkgPath, but those
	// two packages are never linked together.
	PkgPath string

	// Name is the identifier appearing in the package declaration
	// at the start of each source file in this package.
	// The name of an executable is "main".
	Name string

	// Srcs is the list of names of this package's Go
	// source files as presented to the compiler.
	// Names are guaranteed to be absolute.
	//
	// In Metadata queries, or if DisableCgo is set,
	// Srcs includes the unmodified source files even
	// if they use cgo (import "C").
	// In all other queries, Srcs contains the files
	// resulting from cgo processing.
	Srcs []string

	// OtherSrcs is the list of names of non-Go source files that the package
	// contains. This includes assembly and C source files.
	// Names are guaranteed to be absolute.
	OtherSrcs []string

	// Imports maps each import path to its package
	// The keys are import paths as they appear in the source files.
	Imports map[string]*Package

	// syntax and type information (only in TypeCheck and WholeProgram modes)
	Fset     *token.FileSet // source position information
	Files    []*ast.File    // syntax trees for the package's Srcs files
	Errors   []error        // non-nil if the package had errors
	Type     *types.Package // type information about the package
	Info     *types.Info    // type-checker deductions
	IllTyped bool           // this package or a dependency has a parse or type error

	// ---- temporary state ----

	// export holds the path to the export data file
	// for this package, if mode == TypeCheck.
	// The export data file contains the package's type information
	// in a compiler-specific format; see
	// golang.org/x/tools/go/{gc,gccgo}exportdata.
	// May be the empty string if the build failed.
	export string

	indirect      bool              // package is a dependency, not explicitly requested
	imports       map[string]string // nominal form of Imports graph
	importErrors  map[string]error  // maps each bad import to its error
	loadOnce      sync.Once
	color         uint8 // for cycle detection
	mark, needsrc bool  // used in TypeCheck mode only
}

func (lpkg *Package) String() string { return lpkg.ID }

// loader holds the working state of a single call to load.
type loader struct {
	mode mode
	cgo  bool
	Options
	exportMu sync.Mutex // enforces mutual exclusion of exportdata operations
}

// The mode determines which packages are visited
// and the level of information reported about each one.
// Modes are ordered by increasing detail.
type mode uint8

const (
	metadata = iota
	typeCheck
	wholeProgram
)

func (ld *loader) load(patterns ...string) ([]*Package, error) {
	if ld.Context == nil {
		ld.Context = context.Background()
	}

	if ld.mode > metadata {
		if ld.Fset == nil {
			ld.Fset = token.NewFileSet()
		}

		ld.cgo = !ld.DisableCgo

		if ld.Error == nil {
			ld.Error = func(e error) {
				fmt.Fprintln(os.Stderr, e)
			}
		}

		if ld.ParseFile == nil {
			ld.ParseFile = func(fset *token.FileSet, filename string) (*ast.File, error) {
				const mode = parser.AllErrors | parser.ParseComments
				return parser.ParseFile(fset, filename, nil, mode)
			}
		}
	}

	if len(patterns) == 0 {
		return nil, fmt.Errorf("no packages to load")
	}

	// Do the metadata query and partial build.
	// TODO(adonovan): support alternative build systems at this seam.
	list, err := golistPackages(ld.Context, ld.Dir, ld.Env, ld.cgo, ld.mode == typeCheck, ld.Tests, patterns)
	if err != nil {
		return nil, err
	}
	pkgs := make(map[string]*Package)
	var initial []*Package
	for _, pkg := range list {
		pkgs[pkg.ID] = pkg

		// Record the set of initial packages
		// corresponding to the patterns.
		if !pkg.indirect {
			initial = append(initial, pkg)

			if ld.mode == typeCheck {
				pkg.needsrc = true
			}
		}
	}
	if len(pkgs) == 0 {
		return nil, fmt.Errorf("packages not found")
	}

	// Materialize the import graph.

	const (
		white = 0 // new
		grey  = 1 // in progress
		black = 2 // complete
	)

	// visit traverses the import graph, depth-first,
	// and materializes the graph as Packages.Imports.
	//
	// Valid imports are saved in the Packages.Import map.
	// Invalid imports (cycles and missing nodes) are saved in the importErrors map.
	// Thus, even in the presence of both kinds of errors, the Import graph remains a DAG.
	//
	// visit returns whether the package is initial or has a transitive
	// dependency on an initial package. These are the only packages
	// for which we load source code in typeCheck mode.
	var stack []*Package
	var visit func(lpkg *Package) bool
	visit = func(lpkg *Package) bool {
		switch lpkg.color {
		case black:
			return lpkg.needsrc
		case grey:
			panic("internal error: grey node")
		}
		lpkg.color = grey
		stack = append(stack, lpkg) // push

		imports := make(map[string]*Package)
		for importPath, id := range lpkg.imports {
			var importErr error
			imp := pkgs[id]
			if imp == nil {
				// (includes package "C" when DisableCgo)
				importErr = fmt.Errorf("missing package: %q", id)
			} else if imp.color == grey {
				importErr = fmt.Errorf("import cycle: %s", stack)
			}
			if importErr != nil {
				if lpkg.importErrors == nil {
					lpkg.importErrors = make(map[string]error)
				}
				lpkg.importErrors[importPath] = importErr
				continue
			}

			if visit(imp) {
				lpkg.needsrc = true
			}
			imports[importPath] = imp
		}
		lpkg.imports = nil // no longer needed
		lpkg.Imports = imports

		stack = stack[:len(stack)-1] // pop
		lpkg.color = black

		return lpkg.needsrc
	}

	// For each initial package, create its import DAG.
	for _, lpkg := range initial {
		visit(lpkg)
	}

	// Load some/all packages from source, starting at
	// the initial packages (roots of the import DAG).
	if ld.mode != metadata {
		var wg sync.WaitGroup
		for _, lpkg := range initial {
			wg.Add(1)
			go func(lpkg *Package) {
				ld.loadRecursive(lpkg)
				wg.Done()
			}(lpkg)
		}
		wg.Wait()
	}

	return initial, nil
}

// loadRecursive loads, parses, and type-checks the specified package and its
// dependencies, recursively, in parallel, in topological order.
// It is atomic and idempotent.
// Precondition: ld.mode != Metadata.
// In typeCheck mode, only needsrc packages are loaded.
func (ld *loader) loadRecursive(lpkg *Package) {
	lpkg.loadOnce.Do(func() {
		// Load the direct dependencies, in parallel.
		var wg sync.WaitGroup
		for _, imp := range lpkg.Imports {
			wg.Add(1)
			go func(imp *Package) {
				ld.loadRecursive(imp)
				wg.Done()
			}(imp)
		}
		wg.Wait()

		ld.loadPackage(lpkg)
	})
}

// loadPackage loads, parses, and type-checks the
// files of the specified package, if needed.
// It must be called only once per Package,
// after immediate dependencies are loaded.
// Precondition: ld.mode != Metadata.
func (ld *loader) loadPackage(lpkg *Package) {
	if lpkg.PkgPath == "unsafe" {
		// Fill in the blanks to avoid surprises.
		lpkg.Type = types.Unsafe
		lpkg.Fset = ld.Fset
		lpkg.Files = []*ast.File{}
		lpkg.Info = new(types.Info)
		return
	}

	if ld.mode == typeCheck && !lpkg.needsrc {
		return // not a source package
	}

	hardErrors := false
	appendError := func(err error) {
		if terr, ok := err.(types.Error); ok && terr.Soft {
			// Don't mark the package as bad.
		} else {
			hardErrors = true
		}
		ld.Error(err)
		lpkg.Errors = append(lpkg.Errors, err)
	}

	files, errs := ld.parseFiles(lpkg.Srcs)
	for _, err := range errs {
		appendError(err)
	}

	lpkg.Fset = ld.Fset
	lpkg.Files = files

	// Call NewPackage directly with explicit name.
	// This avoids skew between golist and go/types when the files'
	// package declarations are inconsistent.
	lpkg.Type = types.NewPackage(lpkg.PkgPath, lpkg.Name)

	lpkg.Info = &types.Info{
		Types:      make(map[ast.Expr]types.TypeAndValue),
		Defs:       make(map[*ast.Ident]types.Object),
		Uses:       make(map[*ast.Ident]types.Object),
		Implicits:  make(map[ast.Node]types.Object),
		Scopes:     make(map[ast.Node]*types.Scope),
		Selections: make(map[*ast.SelectorExpr]*types.Selection),
	}

	// Copy the prototype types.Config as it must vary across Packages.
	tc := ld.TypeChecker // copy
	if !ld.cgo {
		tc.FakeImportC = true
	}
	tc.Importer = importerFunc(func(path string) (*types.Package, error) {
		if path == "unsafe" {
			return types.Unsafe, nil
		}

		// The imports map is keyed by import path.
		imp := lpkg.Imports[path]
		if imp == nil {
			if err := lpkg.importErrors[path]; err != nil {
				return nil, err
			}
			// There was skew between the metadata and the
			// import declarations, likely due to an edit
			// race, or because the ParseFile feature was
			// used to supply alternative file contents.
			return nil, fmt.Errorf("no metadata for %s", path)
		}
		if imp.Type != nil && imp.Type.Complete() {
			return imp.Type, nil
		}
		if ld.mode == typeCheck && !imp.needsrc {
			return ld.loadFromExportData(imp)
		}
		log.Fatalf("internal error: nil Pkg importing %q from %q", path, lpkg)
		panic("unreachable")
	})
	tc.Error = appendError

	// type-check
	types.NewChecker(&tc, ld.Fset, lpkg.Type, lpkg.Info).Files(lpkg.Files)

	lpkg.importErrors = nil // no longer needed

	// If !Cgo, the type-checker uses FakeImportC mode, so
	// it doesn't invoke the importer for import "C",
	// nor report an error for the import,
	// or for any undefined C.f reference.
	// We must detect this explicitly and correctly
	// mark the package as IllTyped (by reporting an error).
	// TODO(adonovan): if these errors are annoying,
	// we could just set IllTyped quietly.
	if tc.FakeImportC {
	outer:
		for _, f := range lpkg.Files {
			for _, imp := range f.Imports {
				if imp.Path.Value == `"C"` {
					appendError(fmt.Errorf(`%s: import "C" ignored`,
						lpkg.Fset.Position(imp.Pos())))
					break outer
				}
			}
		}
	}

	// Record accumulated errors.
	for _, imp := range lpkg.Imports {
		if imp.IllTyped {
			hardErrors = true
			break
		}
	}

	lpkg.IllTyped = hardErrors
}

// An importFunc is an implementation of the single-method
// types.Importer interface based on a function value.
type importerFunc func(path string) (*types.Package, error)

func (f importerFunc) Import(path string) (*types.Package, error) { return f(path) }

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
func (ld *loader) parseFiles(filenames []string) ([]*ast.File, []error) {
	var wg sync.WaitGroup
	n := len(filenames)
	parsed := make([]*ast.File, n)
	errors := make([]error, n)
	for i, file := range filenames {
		wg.Add(1)
		go func(i int, filename string) {
			ioLimit <- true // wait
			// ParseFile may return both an AST and an error.
			parsed[i], errors[i] = ld.ParseFile(ld.Fset, filename)
			<-ioLimit // signal
			wg.Done()
		}(i, file)
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

// loadFromExportData returns type information for the specified
// package, loading it from an export data file on the first request.
func (ld *loader) loadFromExportData(lpkg *Package) (*types.Package, error) {
	if lpkg.PkgPath == "" {
		log.Fatalf("internal error: Package %s has no PkgPath", lpkg)
	}

	// Because gcexportdata.Read has the potential to create or
	// modify the types.Package for each node in the transitive
	// closure of dependencies of lpkg, all exportdata operations
	// must be sequential. (Finer-grained locking would require
	// changes to the gcexportdata API.)
	//
	// The exportMu lock guards the Package.Pkg field and the
	// types.Package it points to, for each Package in the graph.
	//
	// Not all accesses to Package.Pkg need to be protected by exportMu:
	// graph ordering ensures that direct dependencies of source
	// packages are fully loaded before the importer reads their Pkg field.
	ld.exportMu.Lock()
	defer ld.exportMu.Unlock()

	if tpkg := lpkg.Type; tpkg != nil && tpkg.Complete() {
		return tpkg, nil // cache hit
	}

	lpkg.IllTyped = true // fail safe

	if lpkg.export == "" {
		// Errors while building export data will have been printed to stderr.
		return nil, fmt.Errorf("no export data file")
	}
	f, err := os.Open(lpkg.export)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Read gc export data.
	//
	// We don't currently support gccgo export data because all
	// underlying workspaces use the gc toolchain. (Even build
	// systems that support gccgo don't use it for workspace
	// queries.)
	r, err := gcexportdata.NewReader(f)
	if err != nil {
		return nil, fmt.Errorf("reading %s: %v", lpkg.export, err)
	}

	// Build the view.
	//
	// The gcexportdata machinery has no concept of package ID.
	// It identifies packages by their PkgPath, which although not
	// globally unique is unique within the scope of one invocation
	// of the linker, type-checker, or gcexportdata.
	//
	// So, we must build a PkgPath-keyed view of the global
	// (conceptually ID-keyed) cache of packages and pass it to
	// gcexportdata, then copy back to the global cache any newly
	// created entries in the view map. The view must contain every
	// existing package that might possibly be mentioned by the
	// current package---its reflexive transitive closure.
	//
	// (Yes, reflexive: although loadRecursive processes source
	// packages in topological order, export data packages are
	// processed only lazily within Importer calls. In the graph
	// A->B->C, A->C where A is a source package and B and C are
	// export data packages, processing of the A->B and A->C import
	// edges may occur in either order, depending on the sequence
	// of imports within A. If B is processed first, and its export
	// data mentions C, an imcomplete package for C will be created
	// before processing of C.)
	// We could do export data processing in topological order using
	// loadRecursive, but there's no parallelism to be gained.
	//
	// TODO(adonovan): it would be more simpler and more efficient
	// if the export data machinery invoked a callback to
	// get-or-create a package instead of a map.
	//
	view := make(map[string]*types.Package) // view seen by gcexportdata
	seen := make(map[*Package]bool)         // all visited packages
	var copyback []*Package                 // candidates for copying back to global cache
	var visit func(p *Package)
	visit = func(p *Package) {
		if !seen[p] {
			seen[p] = true
			if p.Type != nil {
				view[p.PkgPath] = p.Type
			} else {
				copyback = append(copyback, p)
			}
			for _, p := range p.Imports {
				visit(p)
			}
		}
	}
	visit(lpkg)

	// Parse the export data.
	// (May create/modify packages in view.)
	tpkg, err := gcexportdata.Read(r, ld.Fset, view, lpkg.PkgPath)
	if err != nil {
		return nil, fmt.Errorf("reading %s: %v", lpkg.export, err)
	}

	// For each newly created types.Package in the view,
	// save it in the main graph.
	for _, p := range copyback {
		p.Type = view[p.PkgPath] // may still be nil
	}

	lpkg.Type = tpkg
	lpkg.IllTyped = false

	return tpkg, nil
}

// All returns a new map containing all the transitive dependencies of
// the specified initial packages, keyed by ID.
func All(initial []*Package) map[string]*Package {
	all := make(map[string]*Package)
	var visit func(p *Package)
	visit = func(p *Package) {
		if all[p.ID] == nil {
			all[p.ID] = p
			for _, imp := range p.Imports {
				visit(imp)
			}
		}
	}
	for _, p := range initial {
		visit(p)
	}
	return all
}
