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

// An Options specifies details about how packages should be loaded.
// The loaders do not modify this struct.
// TODO(rsc): Better name would be Config.
type Options struct {
	// Context specifies the context for the load operation.
	// If the context is cancelled, the loader may stop early
	// and return an ErrCancelled error.
	// If Context is nil, the load cannot be cancelled.
	Context context.Context

	// Dir is the directory in which to run the build system tool
	// that provides information about the packages.
	// If Dir is empty, the tool is run in the current directory.
	Dir string

	// DisableCgo disables cgo-processing of files that import "C",
	// and removes the 'cgo' build tag, which may affect source file selection.
	// By default, TypeCheck, and WholeProgram queries process such
	// files, and the resulting Package.Srcs describes the generated
	// files seen by the compiler.
	// TODO(rsc): Drop entirely. I don't think these are the right semantics.
	DisableCgo bool

	// Env is the environment to use when invoking the build system tool.
	// If Env is nil, the current environment is used.
	// Like in os/exec's Cmd, only the last value in the slice for
	// each environment key is used. To specify the setting of only
	// a few variables, append to the current environment, as in:
	//
	//	opt.Env = append(os.Environ(), "GOOS=plan9", "GOARCH=386")
	//
	Env []string

	// Error is called for each error encountered during package loading.
	// It must be safe to call Error simultaneously from multiple goroutines.
	// In addition to calling Error, the loader will record each error
	// in the corresponding Package's Errors list.
	// If Error is nil, the loader will print errors to os.Stderr.
	// To disable printing of errors, set opt.Error = func(error){}.
	// TODO(rsc): What happens in the Metadata loader? Currently nothing.
	Error func(error)

	// Fset is the token.FileSet to use when parsing loaded source files.
	// If Fset is nil, the loader will create one.
	Fset *token.FileSet

	// ParseFile is called to read and parse each file
	// when preparing a package's type-checked syntax tree.
	// It must be safe to call ParseFile simultaneously from multiple goroutines.
	// If ParseFile is nil, the loader will uses parser.ParseFile.
	//
	// Setting ParseFile to a custom implementation can allow
	// providing alternate file content in order to type-check
	// unsaved text editor buffers, or to selectively eliminate
	// unwanted function bodies to reduce the amount of work
	// done by the type checker.
	ParseFile func(fset *token.FileSet, filename string) (*ast.File, error)

	// If Tests is set, the loader includes not just the packages
	// matching a particular pattern but also any related test packages,
	// including test-only variants of the package and the test executable.
	//
	// For example, when using the go command, loading "fmt" with Tests=true
	// returns four packages, with IDs "fmt" (the standard package),
	// "fmt [fmt.test]" (the package as compiled for the test),
	// "fmt_test" (the test functions from source files in package fmt_test),
	// and "fmt.test" (the test binary).
	//
	// In build systems with explicit names for tests,
	// setting Tests may have no effect.
	Tests bool

	// TypeChecker provides additional configuration for type-checking syntax trees.
	//
	// The TypeCheck loader does not use the TypeChecker configuration
	// for packages that have their type information provided by the
	// underlying build system.
	//
	// The TypeChecker.Error function is ignored:
	// errors are reported using the Error function defined above.
	//
	// The TypeChecker.Importer function is ignored:
	// the loader defines an appropriate importer.
	//
	// The TypeChecker.Sizes are only used by the WholeProgram loader.
	// The TypeCheck loader uses the same sizes as the main build.
	// TODO(rsc): At least, it should. Derive these from runtime?
	TypeChecker types.Config
}

// Metadata loads and returns the Go packages named by the given patterns,
// omitting type information and type-checked syntax trees from all packages.
// TODO(rsc): Better name would be Load.
func Metadata(o *Options, patterns ...string) ([]*Package, error) {
	l := &loader{mode: metadata}
	if o != nil {
		l.Options = *o
	}
	return l.load(patterns...)
}

// TypeCheck loads and returns the Go packages named by the given patterns.
// It includes type information in all packages, including dependencies.
// The packages named by the patterns also have type-checked syntax trees.
// TODO(rsc): Better name would be LoadTyped.
func TypeCheck(o *Options, patterns ...string) ([]*Package, error) {
	l := &loader{mode: typeCheck}
	if o != nil {
		l.Options = *o
	}
	return l.load(patterns...)
}

// WholeProgram loads and returns the Go packages named by the given patterns.
// It includes type information and type-checked syntax trees for all packages,
// including dependencies.
// TODO(rsc): Better name would be LoadAllTyped.
func WholeProgram(o *Options, patterns ...string) ([]*Package, error) {
	l := &loader{mode: wholeProgram}
	if o != nil {
		l.Options = *o
	}
	return l.load(patterns...)
}

// A Package describes a single loaded Go package.
type Package struct {
	// ID is a unique identifier for a package,
	// in a syntax provided by the underlying build system.
	//
	// Because the syntax varies based on the build system,
	// clients should treat IDs as opaque and not attempt to
	// interpret them.
	ID string

	// PkgPath is the import path of the package during a particular build.
	//
	// Analyses that need a unique string to identify a returned Package
	// should use ID, not PkgPath. Although PkgPath does uniquely identify
	// a package in a particular build, the loader may return packages
	// spanning multiple builds (for example, multiple commands,
	// or a package and its tests), so PkgPath is not guaranteed unique
	// across all packages returned by a single load.
	//
	// TODO(rsc): This name should be ImportPath.
	PkgPath string

	// Name is the package name as it appears in the package source code.
	Name string

	// Errors lists any errors encountered while loading the package.
	// TODO(rsc): Say something about the errors or at least their Strings,
	// as far as file:line being at the beginning and so on.
	Errors []error

	// Imports maps import paths appearing in the package's Go source files
	// to corresponding loaded Packages.
	Imports map[string]*Package

	// Srcs lists the absolute file paths of the package's Go source files.
	//
	// If a package has typed syntax trees and the DisableCgo option is false,
	// the cgo-processed output files are listed instead of the original
	// source files that contained import "C" statements.
	// In this case, the file paths may not even end in ".go".
	// Although the original sources are not listed in Srcs, the corresponding
	// syntax tree positions will still refer back to the orignal source code,
	// respecting the //line directives in the cgo-processed output.
	//
	// TODO(rsc): Actually, in TypeCheck mode even the packages without
	// syntax trees (pure dependencies) lose their original sources.
	// We should fix that.
	//
	// TODO(rsc): This should be GoFiles.
	Srcs []string

	// OtherSrcs lists the absolute file paths of the package's non-Go source files,
	// including assembly, C, C++, Fortran, Objective-C, SWIG, and so on.
	//
	// TODO(rsc): This should be OtherFiles.
	OtherSrcs []string

	// Type is the type information for the package.
	// The TypeCheck and WholeProgram loaders set this field for all packages.
	// TODO(rsc): This should be Types.
	Type *types.Package

	// IllTyped indicates whether the package has any type errors.
	// The TypeCheck and WholeProgram loaders set this field for all packages.
	IllTyped bool

	// Files is the package's syntax trees, for the files listed in Srcs.
	//
	// The TypeCheck loader sets Files for packages matching the patterns.
	// The WholeProgram loader sets Files for all packages, including dependencies.
	//
	// TODO(rsc): This should be ASTs or Syntax.
	Files []*ast.File

	// Info is the type-checking results for the package's syntax trees.
	// It is set only when Files is set.
	//
	// TODO(rsc): This should be TypesInfo.
	Info *types.Info

	// Fset is the token.FileSet for the package's syntax trees listed in Files.
	// It is set only when Files is set.
	// All packages loaded together share a single Fset.
	Fset *token.FileSet

	// ---- temporary state ----
	// the Package struct should be pure exported data.

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

// All returns a map, from package ID to package,
// containing the packages in the given list and all their dependencies.
// Each call to All returns a new map.
//
// TODO(rsc): I don't understand why this function exists.
// It might be more useful to return a slice in dependency order.
func All(list []*Package) map[string]*Package {
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
	for _, p := range list {
		visit(p)
	}
	return all
}
