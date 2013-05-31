// Package importer defines the Importer, which loads, parses and
// type-checks packages of Go code plus their transitive closure, and
// retains both the ASTs and the derived facts.
//
// TODO(adonovan): document and test this package better, with examples.
// Currently it's covered by the ssa/ tests.
//
package importer

import (
	"go/ast"
	"go/token"

	"code.google.com/p/go.tools/go/exact"
	"code.google.com/p/go.tools/go/types"
)

// An Importer's methods are not thread-safe.
type Importer struct {
	context  *Context                // the client context
	Fset     *token.FileSet          // position info for all files seen
	Packages map[string]*PackageInfo // keys are import paths
	errors   map[string]error        // cache of errors by import path
}

// Context specifies the supporting context for the importer.
//
type Context struct {
	// TypeChecker contains options relating to the type checker.
	// The Importer will override any user-supplied values for its
	// Expr, Ident, ImplicitObj and Import fields; other fields
	// will be passed through to the type checker.
	TypeChecker types.Context

	// If Loader is non-nil, it is used to satisfy imports.
	//
	// If it is nil, binary object files produced by the gc
	// compiler will be loaded instead of source code for all
	// imported packages.  Such files supply only the types of
	// package-level declarations and values of constants, but no
	// code, so this mode will not yield a whole program.  It is
	// intended for analyses that perform intraprocedural analysis
	// of a single package.
	Loader SourceLoader
}

// SourceLoader is the signature of a function that locates, reads and
// parses a set of source files given an import path.
//
// fset is the fileset to which the ASTs should be added.
// path is the imported path, e.g. "sync/atomic".
//
// On success, the function returns files, the set of ASTs produced,
// or the first error encountered.
//
// The MakeGoBuildLoader utility can be used to construct a
// SourceLoader based on go/build.
//
type SourceLoader func(fset *token.FileSet, path string) (files []*ast.File, err error)

// New returns a new, empty Importer using configuration options
// specified by context.
//
func New(context *Context) *Importer {
	imp := &Importer{
		context:  context,
		Fset:     token.NewFileSet(),
		Packages: make(map[string]*PackageInfo),
		errors:   make(map[string]error),
	}
	imp.context.TypeChecker.Import = imp.doImport
	return imp
}

// doImport loads the typechecker package identified by path
// Implements the types.Importer prototype.
//
func (imp *Importer) doImport(imports map[string]*types.Package, path string) (pkg *types.Package, err error) {
	// Package unsafe is handled specially, and has no PackageInfo.
	if path == "unsafe" {
		return types.Unsafe, nil
	}

	if info, ok := imp.Packages[path]; ok {
		imports[path] = info.Pkg
		pkg = info.Pkg
		return // positive cache hit
	}

	if err = imp.errors[path]; err != nil {
		return // negative cache hit
	}

	// Load the source/binary for 'path', type-check it, construct
	// a PackageInfo and update our map (imp.Packages) and the
	// type-checker's map (imports).
	var info *PackageInfo
	if imp.context.Loader != nil {
		info, err = imp.LoadPackage(path)
	} else if pkg, err = types.GcImport(imports, path); err == nil {
		info = &PackageInfo{Pkg: pkg}
		imp.Packages[path] = info
	}

	if err == nil {
		// Cache success.
		pkg = info.Pkg
		imports[path] = pkg
		return pkg, nil
	}

	// Cache failure
	imp.errors[path] = err
	return nil, err
}

// LoadPackage loads the package of the specified import-path,
// performs type-checking, and returns the corresponding
// PackageInfo.
//
// Not idempotent!
// Precondition: Importer.context.Loader != nil.
// Not thread-safe!
// TODO(adonovan): rethink this API.
//
func (imp *Importer) LoadPackage(importPath string) (*PackageInfo, error) {
	if imp.context.Loader == nil {
		panic("Importer.LoadPackage without a SourceLoader")
	}
	files, err := imp.context.Loader(imp.Fset, importPath)
	if err != nil {
		return nil, err
	}
	return imp.CreateSourcePackage(importPath, files)
}

// CreateSourcePackage invokes the type-checker on files and returns a
// PackageInfo containing the resulting type-checker package, the
// ASTs, and other type information.
//
// The order of files determines the package initialization order.
//
// importPath is the full name under which this package is known, such
// as appears in an import declaration. e.g. "sync/atomic".
//
// The ParseFiles utility may be helpful for parsing a set of Go
// source files.
//
func (imp *Importer) CreateSourcePackage(importPath string, files []*ast.File) (*PackageInfo, error) {
	info := &PackageInfo{
		Files:     files,
		types:     make(map[ast.Expr]types.Type),
		idents:    make(map[*ast.Ident]types.Object),
		constants: make(map[ast.Expr]exact.Value),
		typecases: make(map[*ast.CaseClause]*types.Var),
	}
	tc := imp.context.TypeChecker
	tc.Expr = func(x ast.Expr, typ types.Type, val exact.Value) {
		info.types[x] = typ
		if val != nil {
			info.constants[x] = val
		}
	}
	tc.Ident = func(ident *ast.Ident, obj types.Object) {
		// Invariants:
		// - obj is non-nil.
		// - isBlankIdent(ident) <=> obj.GetType()==nil
		info.idents[ident] = obj
	}
	tc.ImplicitObj = func(node ast.Node, obj types.Object) {
		if cc, ok := node.(*ast.CaseClause); ok {
			info.typecases[cc] = obj.(*types.Var)
		}
	}
	var firstErr error
	info.Pkg, firstErr = tc.Check(importPath, imp.Fset, files...)
	tc.Expr = nil
	tc.Ident = nil
	if firstErr != nil {
		return nil, firstErr
	}
	imp.Packages[importPath] = info
	return info, nil
}
