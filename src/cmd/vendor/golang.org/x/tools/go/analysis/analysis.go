// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package analysis

import (
	"flag"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"reflect"
)

// An Analyzer describes an analysis function and its options.
type Analyzer struct {
	// The Name of the analyzer must be a valid Go identifier
	// as it may appear in command-line flags, URLs, and so on.
	Name string

	// Doc is the documentation for the analyzer.
	// The part before the first "\n\n" is the title
	// (no capital or period, max ~60 letters).
	Doc string

	// URL holds an optional link to a web page with additional
	// documentation for this analyzer.
	URL string

	// Flags defines any flags accepted by the analyzer.
	// The manner in which these flags are exposed to the user
	// depends on the driver which runs the analyzer.
	Flags flag.FlagSet

	// Run applies the analyzer to a package.
	// It returns an error if the analyzer failed.
	//
	// On success, the Run function may return a result
	// computed by the Analyzer; its type must match ResultType.
	// The driver makes this result available as an input to
	// another Analyzer that depends directly on this one (see
	// Requires) when it analyzes the same package.
	//
	// To pass analysis results between packages (and thus
	// potentially between address spaces), use Facts, which are
	// serializable.
	Run func(*Pass) (interface{}, error)

	// RunDespiteErrors allows the driver to invoke
	// the Run method of this analyzer even on a
	// package that contains parse or type errors.
	// The Pass.TypeErrors field may consequently be non-empty.
	RunDespiteErrors bool

	// Requires is a set of analyzers that must run successfully
	// before this one on a given package. This analyzer may inspect
	// the outputs produced by each analyzer in Requires.
	// The graph over analyzers implied by Requires edges must be acyclic.
	//
	// Requires establishes a "horizontal" dependency between
	// analysis passes (different analyzers, same package).
	Requires []*Analyzer

	// ResultType is the type of the optional result of the Run function.
	ResultType reflect.Type

	// FactTypes indicates that this analyzer imports and exports
	// Facts of the specified concrete types.
	// An analyzer that uses facts may assume that its import
	// dependencies have been similarly analyzed before it runs.
	// Facts must be pointers.
	//
	// FactTypes establishes a "vertical" dependency between
	// analysis passes (same analyzer, different packages).
	FactTypes []Fact
}

func (a *Analyzer) String() string { return a.Name }

// A Pass provides information to the Run function that
// applies a specific analyzer to a single Go package.
//
// It forms the interface between the analysis logic and the driver
// program, and has both input and an output components.
//
// As in a compiler, one pass may depend on the result computed by another.
//
// The Run function should not call any of the Pass functions concurrently.
type Pass struct {
	Analyzer *Analyzer // the identity of the current analyzer

	// syntax and type information
	Fset         *token.FileSet // file position information
	Files        []*ast.File    // the abstract syntax tree of each file
	OtherFiles   []string       // names of non-Go files of this package
	IgnoredFiles []string       // names of ignored source files in this package
	Pkg          *types.Package // type information about the package
	TypesInfo    *types.Info    // type information about the syntax trees
	TypesSizes   types.Sizes    // function for computing sizes of types
	TypeErrors   []types.Error  // type errors (only if Analyzer.RunDespiteErrors)

	// Report reports a Diagnostic, a finding about a specific location
	// in the analyzed source code such as a potential mistake.
	// It may be called by the Run function.
	Report func(Diagnostic)

	// ResultOf provides the inputs to this analysis pass, which are
	// the corresponding results of its prerequisite analyzers.
	// The map keys are the elements of Analysis.Required,
	// and the type of each corresponding value is the required
	// analysis's ResultType.
	ResultOf map[*Analyzer]interface{}

	// ReadFile returns the contents of the named file.
	//
	// The only valid file names are the elements of OtherFiles
	// and IgnoredFiles, and names returned by
	// Fset.File(f.FileStart).Name() for each f in Files.
	//
	// Analyzers must use this function (if provided) instead of
	// accessing the file system directly. This allows a driver to
	// provide a virtualized file tree (including, for example,
	// unsaved editor buffers) and to track dependencies precisely
	// to avoid unnecessary recomputation.
	ReadFile func(filename string) ([]byte, error)

	// -- facts --

	// ImportObjectFact retrieves a fact associated with obj.
	// Given a value ptr of type *T, where *T satisfies Fact,
	// ImportObjectFact copies the value to *ptr.
	//
	// ImportObjectFact panics if called after the pass is complete.
	// ImportObjectFact is not concurrency-safe.
	ImportObjectFact func(obj types.Object, fact Fact) bool

	// ImportPackageFact retrieves a fact associated with package pkg,
	// which must be this package or one of its dependencies.
	// See comments for ImportObjectFact.
	ImportPackageFact func(pkg *types.Package, fact Fact) bool

	// ExportObjectFact associates a fact of type *T with the obj,
	// replacing any previous fact of that type.
	//
	// ExportObjectFact panics if it is called after the pass is
	// complete, or if obj does not belong to the package being analyzed.
	// ExportObjectFact is not concurrency-safe.
	ExportObjectFact func(obj types.Object, fact Fact)

	// ExportPackageFact associates a fact with the current package.
	// See comments for ExportObjectFact.
	ExportPackageFact func(fact Fact)

	// AllPackageFacts returns a new slice containing all package
	// facts of the analysis's FactTypes in unspecified order.
	AllPackageFacts func() []PackageFact

	// AllObjectFacts returns a new slice containing all object
	// facts of the analysis's FactTypes in unspecified order.
	AllObjectFacts func() []ObjectFact

	/* Further fields may be added in future. */
}

// PackageFact is a package together with an associated fact.
type PackageFact struct {
	Package *types.Package
	Fact    Fact
}

// ObjectFact is an object together with an associated fact.
type ObjectFact struct {
	Object types.Object
	Fact   Fact
}

// Reportf is a helper function that reports a Diagnostic using the
// specified position and formatted error message.
func (pass *Pass) Reportf(pos token.Pos, format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	pass.Report(Diagnostic{Pos: pos, Message: msg})
}

// The Range interface provides a range. It's equivalent to and satisfied by
// ast.Node.
type Range interface {
	Pos() token.Pos // position of first character belonging to the node
	End() token.Pos // position of first character immediately after the node
}

// ReportRangef is a helper function that reports a Diagnostic using the
// range provided. ast.Node values can be passed in as the range because
// they satisfy the Range interface.
func (pass *Pass) ReportRangef(rng Range, format string, args ...interface{}) {
	msg := fmt.Sprintf(format, args...)
	pass.Report(Diagnostic{Pos: rng.Pos(), End: rng.End(), Message: msg})
}

func (pass *Pass) String() string {
	return fmt.Sprintf("%s@%s", pass.Analyzer.Name, pass.Pkg.Path())
}

// A Fact is an intermediate fact produced during analysis.
//
// Each fact is associated with a named declaration (a types.Object) or
// with a package as a whole. A single object or package may have
// multiple associated facts, but only one of any particular fact type.
//
// A Fact represents a predicate such as "never returns", but does not
// represent the subject of the predicate such as "function F" or "package P".
//
// Facts may be produced in one analysis pass and consumed by another
// analysis pass even if these are in different address spaces.
// If package P imports Q, all facts about Q produced during
// analysis of that package will be available during later analysis of P.
// Facts are analogous to type export data in a build system:
// just as export data enables separate compilation of several passes,
// facts enable "separate analysis".
//
// Each pass (a, p) starts with the set of facts produced by the
// same analyzer a applied to the packages directly imported by p.
// The analysis may add facts to the set, and they may be exported in turn.
// An analysis's Run function may retrieve facts by calling
// Pass.Import{Object,Package}Fact and update them using
// Pass.Export{Object,Package}Fact.
//
// A fact is logically private to its Analysis. To pass values
// between different analyzers, use the results mechanism;
// see Analyzer.Requires, Analyzer.ResultType, and Pass.ResultOf.
//
// A Fact type must be a pointer.
// Facts are encoded and decoded using encoding/gob.
// A Fact may implement the GobEncoder/GobDecoder interfaces
// to customize its encoding. Fact encoding should not fail.
//
// A Fact should not be modified once exported.
type Fact interface {
	AFact() // dummy method to avoid type errors
}
