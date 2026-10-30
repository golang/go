// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package analysis defines the interface between a modular static
analysis and an analysis driver program.

# Background

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

# Analyzer

The primary type in the API is [Analyzer]. An Analyzer statically
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

The [Analyzer] type has more fields besides those shown above:

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

# Pass

A [Pass] describes a single unit of work: the application of a particular
Analyzer to a particular package of Go code.
The Pass provides information to the Analyzer's Run function about the
package being analyzed, and provides operations to the Run function for
reporting diagnostics and other information back to the driver.

	type Pass struct {
		Fset         *token.FileSet
		Files        []*ast.File
		OtherFiles   []string
		IgnoredFiles []string
		Pkg          *types.Package
		TypesInfo    *types.Info
		ResultOf     map[*Analyzer]interface{}
		Report       func(Diagnostic)
		...
	}

The Fset, Files, Pkg, and TypesInfo fields provide the syntax trees,
type information, and source positions for a single package of Go code.

The OtherFiles field provides the names of non-Go
files such as assembly that are part of this package.
Similarly, the IgnoredFiles field provides the names of Go and non-Go
source files that are not part of this package with the current build
configuration but may be part of other build configurations.
The contents of these files may be read using Pass.ReadFile;
see the "asmdecl" or "buildtags" analyzers for examples of loading
non-Go files and reporting diagnostics against them.

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

The [Diagnostic] struct does not have a field to indicate its severity
because opinions about the relative importance of Analyzers and their
diagnostics vary widely among users. The design of this framework does
not hold each Analyzer responsible for identifying the severity of its
diagnostics. Instead, we expect that drivers will allow the user to
customize the filtering and prioritization of diagnostics based on the
producing Analyzer and optional Category, according to the user's
preferences.

Most Analyzers inspect typed Go syntax trees, but a few, such as asmdecl
and buildtag, inspect the raw text of Go source files or even non-Go
files such as assembly. To report a diagnostic against a line of a
raw text file, use the following sequence:

	content, err := pass.ReadFile(filename)
	if err != nil { ... }
	tf := fset.AddFile(filename, -1, len(content))
	tf.SetLinesForContent(content)
	...
	pass.Reportf(tf.LineStart(line), "oops")

# Modular analysis with Facts

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
a diagnostic) is called a [Fact].

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
Because serialized facts may appear within build outputs, the gob encoding
of a fact must be deterministic, to avoid spurious cache misses in
build systems that use content-addressable caches.
The driver makes a single call to the gob encoder for all facts
exported by a given analysis pass, so that the topology of
shared data structures referenced by multiple facts is preserved.

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

A driver should apply the analyzers to every package in the transitive
closure of the packages of interest, including those of the standard
library, since the facts and types for a package exist only if that
package was analyzed.
(Historically some drivers skipped the standard library.
Blaze's analysis driver has since been fixed, and Bazel's driver is
expected to follow; see https://github.com/bazel-contrib/rules_go/issues/4702.)

# Testing an Analyzer

The [golang.org/x/tools/go/analysis/analysistest] subpackage provides
utilities for testing an Analyzer.
In a few lines of code, it is possible to run an analyzer on a package
of testdata files and check that it reported all the expected
diagnostics and facts (and no more). Expectations are expressed using
"// want ..." comments in the input code.

# Drivers

A driver is a program that applies a set of analyzers to a set of Go
packages and reports the diagnostics they produce.
It chooses the packages to analyze, provides the type-annotated syntax
for each one, runs the analyzers in an order consistent with their
Requires graphs, conveys the facts exported by each package to the
packages that import it, and finally presents the resulting diagnostics
and optionally applies any fixes.
The analysis API is the interface that allows one analyzer to work
with any driver.

A number of drivers have been successfully deployed.
Their designs vary along several axes, including:

 1. How is the import graph of packages obtained? Some drivers execute
    the build system's metadata query command, such as "go list", or
    for other build systems, a GOPACKAGESDRIVER program wrapping a
    query such as "bazel query". Other drivers, conversely, are
    executed by the build system itself, similar to how "go build"
    executes cmd/compile once per package.

 2. How are intermediate results propagated between analysis actions?
    Some drivers compute the whole operation in one process, which can
    be efficient, but means that analysis of large code bases is limited
    by available memory and intermediate results are not cached and
    reused. Other drivers save all the intermediate results in the file
    system, like an incremental build system; they pay the costs of
    reading and writing files, but recompute only what is necessary.

 3. How do requests for analysis arrive? Some drivers are batch
    commands that analyze a set of packages, then exit. Others are
    long-lived servers that continuously detect and respond to changes
    in the source files by re-executing analysis as needed, like an
    IDE.

We'll describe several specific drivers in terms of these axes.

Unitchecker: the [golang.org/x/tools/go/analysis/unitchecker] driver
is analogous to a compiler.
It reads the source files of a single package, along with summary
information for each of its direct dependency packages, then writes a
summary of the current package for use in turn downstream.
It is invoked by an orchestrating build system (such as "go vet")
analogous to the way the Go compiler cmd/compile is invoked once per
package by "go build", in dependencies-first order.
Like a compiler, it is (ideally) a pure function from input files to
output files, so the build system can treat it as an ordinary build
action, running it in parallel with others, caching its result, and
skipping it entirely if none of its inputs have changed.
This allows "separate analysis" to scale in the same manner as
separate compilation.

Unitchecker is the basis of the vet tool (cmd/vet) executed by "go
vet" and "go test", and the fix tool (cmd/fix) executed by "go fix".
These tools may also be executed by Bazel's nogo
(https://github.com/bazel-contrib/rules_go/blob/master/go/nogo.rst),
and by analogous mechanisms in other build systems.

Each build system provides a flag to allow you to specify an
alternative unitchecker-based tool with its own analyzer suite, for
example:

	$ go vet -vettool=my.exe [packages...]

Several aspects of the analysis API follow from the demands of
separate ("package at a time") analysis.
Facts must be serializable, and their encoding deterministic,
because they travel between processes in files that may be recorded
in a content-addressable build cache.
An analyzer must not assume that the whole program is available to it,
nor communicate between packages by any means other than facts.
And the unitchecker package does not use golang.org/x/tools/go/packages
since it runs within a build action and should not start a metadata
query of its own.

Multichecker, singlechecker, and analysistest:
the [golang.org/x/tools/go/analysis/multichecker] driver,
by contrast, executes the build system's metadata query tool
(such as "go list") to obtain the import graph over packages.
It then computes the whole analysis, in one address space,
passing facts from one package to another in memory.

It is named multichecker because it runs multiple analyzers; the
singlechecker variant is identical except that it runs only a single
analyzer and its command-line interface is correspondingly simpler.
There is also a variant for analyzer testing, named analysistest, that
works in a similar way.
All three drivers are based on a common implementation in the
[golang.org/x/tools/go/analysis/checker] package.

A driver of this form is easy to deploy, being a single self-contained
command, but it repeats all of its work on each run, and may not
scale to very large code bases since it is limited by available
memory.

Gopls: Go's language server (https://go.dev/gopls) has an integrated
analysis driver that combines aspects of both approaches.
Like multichecker, it executes the build system's metadata query
tool to obtain the import graph; and like unitchecker, it analyzes
one package at a time, recording the diagnostics, types, and facts
of each package in a summary file that is cached in the file system.
It incorporates an incremental build system that detects when files
have changed and automatically re-executes analysis actions
or metadata queries as needed, delivering up-to-date diagnostics
even at interactive editing latency.

The third-party driver tools staticcheck (https://staticcheck.dev) and
golangci-lint (https://golangci-lint.run) both work similarly to gopls,
but as batch programs instead of a long-lived server.
Both also provide suites of their own analyzers.

# Standalone commands

Analyzers are provided in the form of packages that a driver program is
expected to import. The vet command imports a suite of several analyzers
(see [golang.org/x/tools/go/analysis/suite/vet]),
but users may wish to define their own analysis commands that perform
additional checks. To simplify the task of creating an analysis command,
either for a single analyzer or for a whole suite, we provide the
singlechecker and multichecker subpackages.

The singlechecker package provides the main function for a command that
runs one analyzer. By convention, each analyzer such as
go/analysis/passes/findcall should be accompanied by a singlechecker-based
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

For convenience, the singlechecker and multichecker packages also detect
when they are being used in a package-at-a-time context such as
"go vet -vettool=my.exe", and automatically switch to unitchecker
behavior in that case.
This saves building and distributing two executables, but
it does mean that a single command has two rather different modes,
with different ways of choosing packages and different performance.
A tool intended only for use within a build system should call
unitchecker.Main directly, and thus avoid the dependency on
go/packages.
*/
package analysis
