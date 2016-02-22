// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package loader loads a complete Go program from source code, parsing
// and type-checking the initial packages plus their transitive closure
// of dependencies.  The ASTs and the derived facts are retained for
// later use.
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
//      // Parse the specified files and create an ad hoc package with path "foo".
//      // All files must have the same 'package' declaration.
//      conf.CreateFromFilenames("foo", "foo.go", "bar.go")
//
//      // Create an ad hoc package with path "foo" from
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
// See examples_test.go for examples of API usage.
//
//
// CONCEPTS AND TERMINOLOGY
//
// The WORKSPACE is the set of packages accessible to the loader.  The
// workspace is defined by Config.Build, a *build.Context.  The
// default context treats subdirectories of $GOROOT and $GOPATH as
// packages, but this behavior may be overridden.
//
// An AD HOC package is one specified as a set of source files on the
// command line.  In the simplest case, it may consist of a single file
// such as $GOROOT/src/net/http/triv.go.
//
// EXTERNAL TEST packages are those comprised of a set of *_test.go
// files all with the same 'package foo_test' declaration, all in the
// same directory.  (go/build.Package calls these files XTestFiles.)
//
// An IMPORTABLE package is one that can be referred to by some import
// spec.  Every importable package is uniquely identified by its
// PACKAGE PATH or just PATH, a string such as "fmt", "encoding/json",
// or "cmd/vendor/golang.org/x/arch/x86/x86asm".  A package path
// typically denotes a subdirectory of the workspace.
//
// An import declaration uses an IMPORT PATH to refer to a package.
// Most import declarations use the package path as the import path.
//
// Due to VENDORING (https://golang.org/s/go15vendor), the
// interpretation of an import path may depend on the directory in which
// it appears.  To resolve an import path to a package path, go/build
// must search the enclosing directories for a subdirectory named
// "vendor".
//
// ad hoc packages and external test packages are NON-IMPORTABLE.  The
// path of an ad hoc package is inferred from the package
// declarations of its files and is therefore not a unique package key.
// For example, Config.CreatePkgs may specify two initial ad hoc
// packages, both with path "main".
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

// IMPLEMENTATION NOTES
//
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
// are labelled by the package's path.  The remaining nodes are
// ad hoc packages and lists of in-package *_test.go files that augment
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
// We use a concurrency-safe non-blocking cache (importer.imported) to
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

// TODO(adonovan): overhaul the package documentation.
