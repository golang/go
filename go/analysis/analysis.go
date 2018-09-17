// The analysis package defines a uniform interface for static checkers
// of Go source code. By implementing a common interface, checkers from
// a variety of sources can be easily selected, incorporated, and reused
// in a wide range of programs including command-line tools, text
// editors and IDEs, build systems, test frameworks, code review tools,
// and batch pipelines for large code bases. For the design, see
// https://docs.google.com/document/d/1-azPLXaLgTCKeKDNg0HVMq2ovMlD-e7n1ZHzZVzOlJk
//
// Each analysis is invoked once per Go package, and is provided the
// abstract syntax trees (ASTs) and type information for that package.
//
// The principal data types of this package are structs, not interfaces,
// to permit later addition of optional fields as the API evolves.
package analysis

import (
	"flag"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"reflect"
)

// An Analysis describes an analysis function and its options.
type Analysis struct {
	// The Name of the analysis must be a valid Go identifier
	// as it may appear in command-line flags, URLs, and so on.
	Name string

	// Doc is the documentation for the analysis.
	Doc string

	// Flags defines any flags accepted by the analysis.
	// The manner in which these flags are exposed to the user
	// depends on the driver which runs the analysis.
	Flags flag.FlagSet

	// Run applies the analysis to a package.
	// It returns an error if the analysis failed.
	Run func(*Unit) error

	// RunDespiteErrors allows the driver to invoke
	// the Run method of this analysis even on a
	// package that contains parse or type errors.
	RunDespiteErrors bool

	// Requires is a set of analyses that must run successfully
	// before this one on a given package. This analysis may inspect
	// the outputs produced by each analysis in Requires.
	// The graph over analyses implied by Requires edges must be acyclic.
	//
	// Requires establishes a "horizontal" dependency between
	// analysis units (different analyses, same package).
	Requires []*Analysis

	// OutputType is the type of the optional Output value
	// computed by this analysis and stored in Unit.Output.
	// (The Output is provided as an Input to
	// each analysis that Requires this one.)
	OutputType reflect.Type

	// LemmaTypes is the set of types of lemmas produced and
	// consumed by this analysis. An analysis that uses lemmas
	// may assume that its import dependencies have been
	// similarly analyzed before it runs. Lemmas are pointers.
	//
	// LemmaTypes establishes a "vertical" dependency between
	// analysis units (same analysis, different packages).
	LemmaTypes []reflect.Type
}

func (a *Analysis) String() string { return a.Name }

// A Unit provides information to the Run function that
// applies a specific analysis to a single Go package.
//
// It forms the interface between the analysis logic and the driver
// program, and has both input and an output components.
type Unit struct {
	// -- inputs --

	Analysis *Analysis // the identity of the current analysis

	// syntax and type information
	Fset   *token.FileSet // file position information
	Syntax []*ast.File    // the abstract syntax tree of each file
	Pkg    *types.Package // type information about the package
	Info   *types.Info    // type information about the syntax trees

	// Inputs provides the inputs to this analysis unit, which are
	// the corresponding outputs of its prerequisite analysis.
	// The map keys are the elements of Analysis.Required,
	// and the type of each corresponding value is the required
	// analysis's OutputType.
	Inputs map[*Analysis]interface{}

	// ObjectLemma retrieves a lemma associated with obj.
	// Given a value ptr of type *T, where *T satisfies Lemma,
	// ObjectLemma copies the value to *ptr.
	//
	// ObjectLemma may panic if applied to a lemma type that
	// the analysis did not declare among its LemmaTypes,
	// or if called after analysis of the unit is complete.
	//
	// ObjectLemma is not concurrency-safe.
	ObjectLemma func(obj types.Object, lemma Lemma) bool

	// PackageLemma retrives a lemma associated with package pkg,
	// which must be this package or one if its dependencies.
	// See comments for ObjectLemma.
	PackageLemma func(pkg *types.Package, lemma Lemma) bool

	// -- outputs --

	// Findings is a list of findings about specific locations
	// in the analyzed source code, such as potential mistakes.
	// It is populated by the Run function.
	Findings []*Finding

	// SetObjectLemma associates a lemma of type *T with the obj,
	// replacing any previous lemma of that type.
	//
	// SetObjectLemma panics if the lemma's type is not among
	// Analysis.LemmaTypes, or if obj does not belong to the package
	// being analyzed, or if it is called after analysis of the unit
	// is complete.
	//
	// SetObjectLemma is not concurrency-safe.
	SetObjectLemma func(obj types.Object, lemma Lemma)

	// SetPackageLemma associates a lemma with the current package.
	// See comments for SetObjectLemma.
	SetPackageLemma func(lemma Lemma)

	// Output is an immutable result computed by this analysis unit
	// and set by the Run function.
	// It will be made available as an input to any analysis that
	// depends directly on this one; see Analysis.Requires.
	// Its type must match Analysis.OutputType.
	//
	// Outputs are available as Inputs to later analyses of the
	// same package. To pass analysis results between packages (and
	// thus potentially between address spaces), use Lemmas, which
	// are serializable.
	Output interface{}

	/* Further fields may be added in future. */
	// For example, suggested or applied refactorings.
}

// Findingf is a helper function that creates a new Finding using the
// specified position and formatted error message, appends it to
// unit.Findings, and returns it.
func (unit *Unit) Findingf(pos token.Pos, format string, args ...interface{}) *Finding {
	msg := fmt.Sprintf(format, args...)
	f := &Finding{Pos: pos, Message: msg}
	unit.Findings = append(unit.Findings, f)
	return f
}

func (unit *Unit) String() string {
	return fmt.Sprintf("%s@%s", unit.Analysis.Name, unit.Pkg.Path())
}

// A Lemma is an intermediate fact produced during analysis.
//
// Each lemma is associated with a named declaration (a types.Object).
// A single object may have multiple associated lemmas, but only one of
// any particular lemma type.
//
// A Lemma represents a predicate such as "never returns", but does not
// represent the subject of the predicate such as "function F".
//
// Lemmas may be produced in one analysis unit and consumed by another
// analysis unit even if these are in different address spaces.
// If package P imports Q, all lemmas about objects of Q produced during
// analysis of that package will be available during later analysis of P.
// Lemmas are analogous to type export data in a build system:
// just as export data enables separate compilation of several units,
// lemmas enable "separate analysis".
//
// Each unit of analysis starts with the set of lemmas produced by the
// same analysis applied to the packages directly imported by the
// current one. The analysis may add additional lemmas to the set, and
// they may be exported in turn. An analysis's Run function may retrieve
// lemmas by calling Unit.Lemma and set them using Unit.SetLemma.
//
// Each type of Lemma may be produced by at most one Analysis.
// Lemmas are logically private to their Analysis; to pass values
// between different analysis, use the Input/Output mechanism.
//
// A Lemma type must be a pointer. (Unit.GetLemma relies on it.)
// Lemmas are encoded and decoded using encoding/gob.
// A Lemma may implement the GobEncoder/GobDecoder interfaces
// to customize its encoding; Lemma encoding should not fail.
//
// A Lemma should not be modified once passed to SetLemma.
type Lemma interface {
	IsLemma() // dummy method to avoid type errors
}

// A Finding is a message associated with a source location.
//
// An Analysis may return a variety of findings; the optional Category,
// which should be a constant, may be used to classify them.
// It is primarily intended to make it easy to look up documentation.
type Finding struct {
	Pos      token.Pos
	Category string // optional
	Message  string
}
