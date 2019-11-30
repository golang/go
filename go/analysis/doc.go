/*

The analysis package defines the interface between a modular static
analysis and an analysis driver program.


Background

A static analysis is a function that inspects a package of Go code and
reports a set of diagnostics (typically mistakes in the code), and
perhaps produces other results as well, such as suggested refactorings
or other facts. An analysis that reports mistakes is informally called a
"checker". For example, the printf checker reports mistakes in
fmt.Printf format strings.

A "modular" analysis is one that inspects one package at a time but can
save information from a lower-level package and use it when inspecting a
higher-level package, analogous to separate compilation in a toolchain.
The printf checker is modular: when it discovers that a function such as
log.Fatalf delegates to fmt.Printf, it records this fact, and checks
calls to that function too, including calls made from another package.

By implementing a common interface, checkers from a variety of sources
can be easily selected, incorporated, and reused in a wide range of
driver programs including command-line tools (such as vet), text editors and
IDEs, build and test systems (such as go build, Bazel, or Buck), test
frameworks, code review tools, code-base indexers (such as SourceGraph),
documentation viewers (such as godoc), batch pipelines for large code
bases, and so on.


Analyzer

The primary type in the API is Analyzer. An Analyzer statically
describes an analysis function: its name, documentation, flags,
relationship to other analyzers, and of course, its logic.

To define an analysis, a user declares a (logically constant) variable
of type Analyzer. Here is a typical example from one of the analyzers in
the go/analysis/passes/ subdirectory:

	package unusedresult

	var Analyzer = &analysis.Analyzer{
		Name: "unusedresult",
		Doc:  "check for unused results of calls to some functions",
		Run:  run,
		...
	}

	func run(pass *analysis.Pass) (interface{}, error) {
		...
	}

An analysis driver is a program such as vet that runs a set of
analyses and prints the diagnostics that they report.
The driver program must import the list of Analyzers it needs.
Typically each Analyzer resides in a separate package.
To add a new Analyzer to an existing driver, add another item to the list:

	import ( "unusedresult"; "nilness"; "printf" )

	var analyses = []*analysis.Analyzer{
		unusedresult.Analyzer,
		nilness.Analyzer,
		printf.Analyzer,
	}

A driver may use the name, flags, and documentation to provide on-line
help that describes the analyses it performs.
The doc comment contains a brief one-line summary,
optionally followed by paragraphs of explanation.
The vet command, shown below, is an example of a driver that runs
multiple analyzers. It is based on the multichecker package
(see the "Standalone commands" section for details).

	$ go build golang.org/x/tools/go/analysis/cmd/vet
	$ ./vet help
	vet is a tool for static analysis of Go programs.

	Usage: vet [-flag] [package]

	Registered analyzers:

	    asmdecl      report mismatches between assembly files and Go declarations
	    assign       check for useless assignments
	    atomic       check for common mistakes using the sync/atomic package
	    ...
	    unusedresult check for unused results of calls to some functions

	$ ./vet help unusedresult
	unusedresult: check for unused results of calls to some functions

	Analyzer flags:

	  -unusedresult.funcs value
	        comma-separated list of functions whose results must be used (default Error,String)
	  -unusedresult.stringmethods value
	        comma-separated list of names of methods of type func() string whose results must be used

	Some functions like fmt.Errorf return a result and have no side effects,
	so it is always a mistake to discard the result. This analyzer reports
	calls to certain functions in which the result of the call is ignored.

	The set of functions may be controlled using flags.

The Analyzer type has more fields besides those shown above:

	type Analyzer struct {
		Name             string
		Doc              string
		Flags            flag.FlagSet
		Run              func(*Pass) (interface{}, error)
		RunDespiteErrors bool
		ResultType       reflect.Type
		Requires         []*Analyzer
		FactTypes        []Fact
	}

The Flags field declares a set of named (global) flag variables that
control analysis behavior. Unlike vet, analysis flags are not declared
directly in the command line FlagSet; it is up to the driver to set the
flag variables. A driver for a single analysis, a, might expose its flag
f directly on the command line as -f, whereas a driver for multiple
analyses might prefix the flag name by the analysis name (-a.f) to avoid
ambiguity. An IDE might expose the flags through a graphical interface,
and a batch pipeline might configure them from a config file.
See the "findcall" analyzer for an example of flags in action.

The RunDespiteErrors flag indicates whether the analysis is equipped to
handle ill-typed code. If not, the driver will skip the analysis if
there were parse or type errors.
The optional ResultType field specifies the type of the result value
computed by this analysis and made available to other analyses.
The Requires field specifies a list of analyses upon which
this one depends and whose results it may access, and it constrains the
order in which a driver may run analyses.
The FactTypes field is discussed in the section on Modularity.
The analysis package provides a Validate function to perform basic
sanity checks on an Analyzer, such as that its Requires graph is
acyclic, its fact and result types are unique, and so on.

Finally, the Run field contains a function to be called by the driver to
execute the analysis on a single package. The driver passes it an
instance of the Pass type.


Pass

A Pass describes a single unit of work: the application of a particular
Analyzer to a particular package of Go code.
The Pass provides information to the Analyzer's Run function about the
package being analyzed, and provides operations to the Run function for
reporting diagnostics and other information back to the driver.

	type Pass struct {
		Fset       *token.FileSet
		Files      []*ast.File
		OtherFiles []string
		Pkg        *types.Package
		TypesInfo  *types.Info
		ResultOf   map[*Analyzer]interface{}
		Report     func(Diagnostic)
		...
	}

The Fset, Files, Pkg, and TypesInfo fields provide the syntax trees,
type information, and source positions for a single package of Go code.

The OtherFiles field provides the names, but not the contents, of non-Go
files such as assembly that are part of this package. See the "asmdecl"
or "buildtags" analyzers for examples of loading non-Go files and reporting
diagnostics against them.

The ResultOf field provides the results computed by the analyzers
required by this one, as expressed in its Analyzer.Requires field. The
driver runs the required analyzers first and makes their results
available in this map. Each Analyzer must return a value of the type
described in its Analyzer.ResultType field.
For example, the "ctrlflow" analyzer returns a *ctrlflow.CFGs, which
provides a control-flow graph for each function in the package (see
golang.org/x/tools/go/cfg); the "inspect" analyzer returns a value that
enables other Analyzers to traverse the syntax trees of the package more
efficiently; and the "buildssa" analyzer constructs an SSA-form
intermediate representation.
Each of these Analyzers extends the capabilities of later Analyzers
without adding a dependency to the core API, so an analysis tool pays
only for the extensions it needs.

The Report function emits a diagnostic, a message associated with a
source position. For most analyses, diagnostics are their primary
result.
For convenience, Pass provides a helper method, Reportf, to report a new
diagnostic by formatting a string.
Diagnostic is defined as:

	type Diagnostic struct {
		Pos      token.Pos
		Category string // optional
		Message  string
	}

The optional Category field is a short identifier that classifies the
kind of message when an analysis produces several kinds of diagnostic.

Most Analyzers inspect typed Go syntax trees, but a few, such as asmdecl
and buildtag, inspect the raw text of Go source files or even non-Go
files such as assembly. To report a diagnostic against a line of a
raw text file, use the following sequence:

	content, err := ioutil.ReadFile(filename)
	if err != nil { ... }
	tf := fset.AddFile(filename, -1, len(content))
	tf.SetLinesForContent(content)
	...
	pass.Reportf(tf.LineStart(line), "oops")


Modular analysis with Facts

To improve efficiency and scalability, large programs are routinely
built using separate compilation: units of the program are compiled
separately, and recompiled only when one of their dependencies changes;
independent modules may be compiled in parallel. The same technique may
be applied to static analyses, for the same benefits. Such analyses are
described as "modular".

A compiler’s type checker is an example of a modular static analysis.
Many other checkers we would like to apply to Go programs can be
understood as alternative or non-standard type systems. For example,
vet's printf checker infers whether a function has the "printf wrapper"
type, and it applies stricter checks to calls of such functions. In
addition, it records which functions are printf wrappers for use by
later analysis passes to identify other printf wrappers by induction.
A result such as “f is a printf wrapper” that is not interesting by
itself but serves as a stepping stone to an interesting result (such as
a diagnostic) is called a "fact".

The analysis API allows an analysis to define new types of facts, to
associate facts of these types with objects (named entities) declared
within the current package, or with the package as a whole, and to query
for an existing fact of a given type associated with an object or
package.

An Analyzer that uses facts must declare their types:

	var Analyzer = &analysis.Analyzer{
		Name:      "printf",
		FactTypes: []analysis.Fact{new(isWrapper)},
		...
	}

	type isWrapper struct{} // => *types.Func f “is a printf wrapper”

The driver program ensures that facts for a pass’s dependencies are
generated before analyzing the package and is responsible for propagating
facts from one package to another, possibly across address spaces.
Consequently, Facts must be serializable. The API requires that drivers
use the gob encoding, an efficient, robust, self-describing binary
protocol. A fact type may implement the GobEncoder/GobDecoder interfaces
if the default encoding is unsuitable. Facts should be stateless.

The Pass type has functions to import and export facts,
associated either with an object or with a package:

	type Pass struct {
		...
		ExportObjectFact func(types.Object, Fact)
		ImportObjectFact func(types.Object, Fact) bool

		ExportPackageFact func(fact Fact)
		ImportPackageFact func(*types.Package, Fact) bool
	}

An Analyzer may only export facts associated with the current package or
its objects, though it may import facts from any package or object that
is an import dependency of the current package.

Conceptually, ExportObjectFact(obj, fact) inserts fact into a hidden map keyed by
the pair (obj, TypeOf(fact)), and the ImportObjectFact function
retrieves the entry from this map and copies its value into the variable
pointed to by fact. This scheme assumes that the concrete type of fact
is a pointer; this assumption is checked by the Validate function.
See the "printf" analyzer for an example of object facts in action.

Some driver implementations (such as those based on Bazel and Blaze) do
not currently apply analyzers to packages of the standard library.
Therefore, for best results, analyzer authors should not rely on
analysis facts being available for standard packages.
For example, although the printf checker is capable of deducing during
analysis of the log package that log.Printf is a printf wrapper,
this fact is built in to the analyzer so that it correctly checks
calls to log.Printf even when run in a driver that does not apply
it to standard packages. We would like to remove this limitation in future.


Testing an Analyzer

The analysistest subpackage provides utilities for testing an Analyzer.
In a few lines of code, it is possible to run an analyzer on a package
of testdata files and check that it reported all the expected
diagnostics and facts (and no more). Expectations are expressed using
"// want ..." comments in the input code.


Standalone commands

Analyzers are provided in the form of packages that a driver program is
expected to import. The vet command imports a set of several analyzers,
but users may wish to define their own analysis commands that perform
additional checks. To simplify the task of creating an analysis command,
either for a single analyzer or for a whole suite, we provide the
singlechecker and multichecker subpackages.

The singlechecker package provides the main function for a command that
runs one analyzer. By convention, each analyzer such as
go/passes/findcall should be accompanied by a singlechecker-based
command such as go/analysis/passes/findcall/cmd/findcall, defined in its
entirety as:

	package main

	import (
		"golang.org/x/tools/go/analysis/passes/findcall"
		"golang.org/x/tools/go/analysis/singlechecker"
	)

	func main() { singlechecker.Main(findcall.Analyzer) }

A tool that provides multiple analyzers can use multichecker in a
similar way, giving it the list of Analyzers.

*/
package analysis
