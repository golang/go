// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package oracle contains the implementation of the oracle tool whose
// command-line is provided by code.google.com/p/go.tools/cmd/oracle.
//
// http://golang.org/s/oracle-design
// http://golang.org/s/oracle-user-manual
//
package oracle

// This file defines oracle.Query, the entry point for the oracle tool.
// The actual executable is defined in cmd/oracle.

// TODO(adonovan): new queries
// - show all statements that may update the selected lvalue
//   (local, global, field, etc).
// - show all places where an object of type T is created
//   (&T{}, var t T, new(T), new(struct{array [3]T}), etc.

// ORACLE CONTROL FLOW
//
// The Oracle is somewhat convoluted due to the need to support two
// very different use-cases, "one-shot" and "long running", and to do
// so quickly.
//
// The cmd/oracle tool issues "one-shot" queries via the exported
// Query function, which creates an Oracle to answer a single query.
// newOracle consults the 'needs' flags of the query mode and the
// package containing the query to avoid doing more work than it needs
// (loading, parsing, type checking, SSA construction).
//
// The Pythia tool (github.com/fzipp/pythia‎) is an example of a "long
// running" tool.  It calls New() and then loops, calling
// ParseQueryPos and (*Oracle).Query to handle each incoming HTTP
// query.  Since New cannot see which queries will follow, it must
// load, parse, type-check and SSA-build the entire transitive closure
// of the analysis scope, retaining full debug information and all
// typed ASTs.
//
// TODO(adonovan): experiment with inverting the control flow by
// making each mode consist of two functions: a "one-shot setup"
// function and the existing "impl" function.  The one-shot setup
// function would do all of the work of Query and newOracle,
// specialized to each mode, calling library utilities for the common
// things.  This would give it more control over "scope reduction".
// Long running tools would not call the one-shot setup function but
// would have their own setup function equivalent to the existing
// 'needsAll' flow path.

import (
	"fmt"
	"go/ast"
	"go/build"
	"go/token"
	"io"

	"code.google.com/p/go.tools/astutil"
	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/importer"
	"code.google.com/p/go.tools/oracle/serial"
	"code.google.com/p/go.tools/pointer"
	"code.google.com/p/go.tools/ssa"
)

// An Oracle holds the program state required for one or more queries.
type Oracle struct {
	fset      *token.FileSet                           // file set [all queries]
	prog      *ssa.Program                             // the SSA program [needSSA]
	ptaConfig pointer.Config                           // pointer analysis configuration [needPTA]
	typeInfo  map[*types.Package]*importer.PackageInfo // type info for all ASTs in the program [needRetainTypeInfo]
}

// A set of bits indicating the analytical requirements of each mode.
//
// Typed ASTs for the whole program are always constructed
// transiently; they are retained only for the queried package unless
// needRetainTypeInfo is set.
const (
	needPos            = 1 << iota // needs a position
	needExactPos                   // needs an exact AST selection; implies needPos
	needRetainTypeInfo             // needs to retain type info for all ASTs in the program
	needSSA                        // needs ssa.Packages for whole program
	needSSADebug                   // needs debug info for ssa.Packages
	needPTA            = needSSA   // needs pointer analysis
	needAll            = -1        // needs everything (e.g. a sequence of queries)
)

type modeInfo struct {
	name  string
	needs int
	impl  func(*Oracle, *QueryPos) (queryResult, error)
}

var modes = []*modeInfo{
	// Pointer analyses, whole program:
	{"callees", needPTA | needExactPos, callees},
	{"callers", needPTA | needPos, callers},
	{"callgraph", needPTA, callgraph},
	{"callstack", needPTA | needPos, callstack},
	{"peers", needPTA | needSSADebug | needPos, peers},
	{"pointsto", needPTA | needSSADebug | needExactPos, pointsto},

	// Type-based, modular analyses:
	{"definition", needPos, definition},
	{"describe", needExactPos, describe},
	{"freevars", needPos, freevars},

	// Type-based, whole-program analyses:
	{"implements", needRetainTypeInfo | needPos, implements},
	{"referrers", needRetainTypeInfo | needPos, referrers},
}

func findMode(mode string) *modeInfo {
	for _, m := range modes {
		if m.name == mode {
			return m
		}
	}
	return nil
}

type printfFunc func(pos interface{}, format string, args ...interface{})

// queryResult is the interface of each query-specific result type.
type queryResult interface {
	toSerial(res *serial.Result, fset *token.FileSet)
	display(printf printfFunc)
}

// A QueryPos represents the position provided as input to a query:
// a textual extent in the program's source code, the AST node it
// corresponds to, and the package to which it belongs.
// Instances are created by ParseQueryPos.
//
type QueryPos struct {
	fset       *token.FileSet
	start, end token.Pos             // source extent of query
	path       []ast.Node            // AST path from query node to root of ast.File
	exact      bool                  // 2nd result of PathEnclosingInterval
	info       *importer.PackageInfo // type info for the queried package (nil for fastQueryPos)
}

// TypeString prints type T relative to the query position.
func (qpos *QueryPos) TypeString(T types.Type) string {
	return types.TypeString(qpos.info.Pkg, T)
}

// ObjectString prints object obj relative to the query position.
func (qpos *QueryPos) ObjectString(obj types.Object) string {
	return types.ObjectString(qpos.info.Pkg, obj)
}

// SelectionString prints selection sel relative to the query position.
func (qpos *QueryPos) SelectionString(sel *types.Selection) string {
	return types.SelectionString(qpos.info.Pkg, sel)
}

// A Result encapsulates the result of an oracle.Query.
type Result struct {
	fset     *token.FileSet
	q        queryResult       // the query-specific result
	mode     string            // query mode
	warnings []pointer.Warning // pointer analysis warnings
}

// Serial returns an instance of serial.Result, which implements the
// {xml,json}.Marshaler interfaces so that query results can be
// serialized as JSON or XML.
//
func (res *Result) Serial() *serial.Result {
	resj := &serial.Result{Mode: res.mode}
	res.q.toSerial(resj, res.fset)
	for _, w := range res.warnings {
		resj.Warnings = append(resj.Warnings, serial.PTAWarning{
			Pos:     res.fset.Position(w.Pos).String(),
			Message: w.Message,
		})
	}
	return resj
}

// Query runs a single oracle query.
//
// args specify the main package in importer.LoadInitialPackages syntax.
// mode is the query mode ("callers", etc).
// ptalog is the (optional) pointer-analysis log file.
// buildContext is the go/build configuration for locating packages.
// reflection determines whether to model reflection soundly (currently slow).
//
// Clients that intend to perform multiple queries against the same
// analysis scope should use this pattern instead:
//
//	imp := importer.New(&importer.Config{Build: buildContext})
// 	o, err := oracle.New(imp, args, nil)
//	if err != nil { ... }
//	for ... {
//		qpos, err := oracle.ParseQueryPos(imp, pos, needExact)
//		if err != nil { ... }
//
//		res, err := o.Query(mode, qpos)
//		if err != nil { ... }
//
//		// use res
//	}
//
// TODO(adonovan): the ideal 'needsExact' parameter for ParseQueryPos
// depends on the query mode; how should we expose this?
//
func Query(args []string, mode, pos string, ptalog io.Writer, buildContext *build.Context, reflection bool) (*Result, error) {
	if mode == "what" {
		// Bypass package loading, type checking, SSA construction.
		return what(pos, buildContext)
	}

	minfo := findMode(mode)
	if minfo == nil {
		return nil, fmt.Errorf("invalid mode type: %q", mode)
	}

	impcfg := importer.Config{Build: buildContext}

	// For queries needing only a single typed package,
	// reduce the analysis scope to that package.
	if minfo.needs&(needSSA|needRetainTypeInfo) == 0 {
		reduceScope(pos, &impcfg, &args)
	}

	// TODO(adonovan): report type errors to the user via Serial
	// types, not stderr?
	// impcfg.TypeChecker.Error = func(err error) {
	// 	E := err.(types.Error)
	// 	fmt.Fprintf(os.Stderr, "%s: %s\n", E.Fset.Position(E.Pos), E.Msg)
	// }
	imp := importer.New(&impcfg)
	o, err := newOracle(imp, args, ptalog, minfo.needs, reflection)
	if err != nil {
		return nil, err
	}

	var qpos *QueryPos
	if minfo.needs&(needPos|needExactPos) != 0 {
		qpos, err = ParseQueryPos(imp, pos, minfo.needs&needExactPos != 0)
		if err != nil {
			return nil, err
		}
	}

	// SSA is built and we have the QueryPos.
	// Release the other ASTs and type info to the GC.
	imp = nil

	return o.query(minfo, qpos)
}

// reduceScope is called for one-shot queries that need only a single
// typed package.  It attempts to guess the query package from pos and
// reduce the analysis scope (set of loaded packages) to just that one
// plus (the exported parts of) its dependencies.  It leaves its
// arguments unchanged on failure.
//
// TODO(adonovan): this is a real mess... but it's fast.
//
func reduceScope(pos string, impcfg *importer.Config, args *[]string) {
	// TODO(adonovan): make the 'args' argument of
	// (*Importer).LoadInitialPackages part of the
	// importer.Config, and inline LoadInitialPackages into
	// NewImporter.  Then we won't need the 'args' argument.

	fqpos, err := fastQueryPos(pos)
	if err != nil {
		return // bad query
	}

	// TODO(adonovan): fix: this gives the wrong results for files
	// in non-importable packages such as tests and ad-hoc packages
	// specified as a list of files (incl. the oracle's tests).
	_, importPath, err := guessImportPath(fqpos.fset.File(fqpos.start).Name(), impcfg.Build)
	if err != nil {
		return // can't find GOPATH dir
	}
	if importPath == "" {
		return
	}

	// Check that it's possible to load the queried package.
	// (e.g. oracle tests contain different 'package' decls in same dir.)
	// Keep consistent with logic in importer/util.go!
	ctxt2 := *impcfg.Build
	ctxt2.CgoEnabled = false
	bp, err := ctxt2.Import(importPath, "", 0)
	if err != nil {
		return // no files for package
	}
	_ = bp

	// TODO(adonovan): fix: also check that the queried file appears in the package.
	//  for _, f := range bp.GoFiles, bp.TestGoFiles, bp.XTestGoFiles {
	//  	if sameFile(f, fqpos.filename) { goto found }
	//  }
	//  return // not found
	// found:

	impcfg.TypeCheckFuncBodies = func(p string) bool { return p == importPath }
	*args = []string{importPath}
}

// New constructs a new Oracle that can be used for a sequence of queries.
//
// imp will be used to load source code for imported packages.
// It must not yet have loaded any packages.
//
// args specify the main package in importer.LoadInitialPackages syntax.
//
// ptalog is the (optional) pointer-analysis log file.
// reflection determines whether to model reflection soundly (currently slow).
//
func New(imp *importer.Importer, args []string, ptalog io.Writer, reflection bool) (*Oracle, error) {
	return newOracle(imp, args, ptalog, needAll, reflection)
}

func newOracle(imp *importer.Importer, args []string, ptalog io.Writer, needs int, reflection bool) (*Oracle, error) {
	o := &Oracle{fset: imp.Fset}

	// Load/parse/type-check program from args.
	initialPkgInfos, args, err := imp.LoadInitialPackages(args)
	if err != nil {
		return nil, err // I/O or parser error
	}
	if len(args) > 0 {
		return nil, fmt.Errorf("surplus arguments: %q", args)
	}

	// Retain type info for all ASTs in the program.
	if needs&needRetainTypeInfo != 0 {
		m := make(map[*types.Package]*importer.PackageInfo)
		for _, p := range imp.AllPackages() {
			m[p.Pkg] = p
		}
		o.typeInfo = m
	}

	// Create SSA package for the initial packages and their dependencies.
	if needs&needSSA != 0 {
		prog := ssa.NewProgram(o.fset, 0)

		// Create SSA packages.
		if err := prog.CreatePackages(imp); err != nil {
			return nil, err
		}

		// For each initial package (specified on the command line),
		// if it has a main function, analyze that,
		// otherwise analyze its tests, if any.
		var testPkgs, mains []*ssa.Package
		for _, info := range initialPkgInfos {
			initialPkg := prog.Package(info.Pkg)

			// Add package to the pointer analysis scope.
			if initialPkg.Func("main") != nil {
				mains = append(mains, initialPkg)
			} else {
				testPkgs = append(testPkgs, initialPkg)
			}
		}
		if testPkgs != nil {
			if p := prog.CreateTestMainPackage(testPkgs...); p != nil {
				mains = append(mains, p)
			}
		}
		if mains == nil {
			return nil, fmt.Errorf("analysis scope has no main and no tests")
		}
		o.ptaConfig.Log = ptalog
		o.ptaConfig.Reflection = reflection
		o.ptaConfig.Mains = mains

		if needs&needSSADebug != 0 {
			for _, pkg := range prog.AllPackages() {
				pkg.SetDebugMode(true)
			}
		}

		o.prog = prog
	}

	return o, nil
}

// Query runs the query of the specified mode and selection.
//
// TODO(adonovan): fix: this function does not currently support the
// "what" query, which needs to access the go/build.Context.
//
func (o *Oracle) Query(mode string, qpos *QueryPos) (*Result, error) {
	minfo := findMode(mode)
	if minfo == nil {
		return nil, fmt.Errorf("invalid mode type: %q", mode)
	}
	return o.query(minfo, qpos)
}

func (o *Oracle) query(minfo *modeInfo, qpos *QueryPos) (*Result, error) {
	// Clear out residue of previous query (for long-running clients).
	o.ptaConfig.Queries = nil
	o.ptaConfig.IndirectQueries = nil

	res := &Result{
		mode: minfo.name,
		fset: o.fset,
	}
	var err error
	res.q, err = minfo.impl(o, qpos)
	if err != nil {
		return nil, err
	}
	return res, nil
}

// ParseQueryPos parses the source query position pos.
// If needExact, it must identify a single AST subtree;
// this is appropriate for queries that allow fairly arbitrary syntax,
// e.g. "describe".
//
func ParseQueryPos(imp *importer.Importer, posFlag string, needExact bool) (*QueryPos, error) {
	filename, startOffset, endOffset, err := parsePosFlag(posFlag)
	if err != nil {
		return nil, err
	}
	start, end, err := findQueryPos(imp.Fset, filename, startOffset, endOffset)
	if err != nil {
		return nil, err
	}
	info, path, exact := imp.PathEnclosingInterval(start, end)
	if path == nil {
		return nil, fmt.Errorf("no syntax here")
	}
	if needExact && !exact {
		return nil, fmt.Errorf("ambiguous selection within %s", astutil.NodeDescription(path[0]))
	}
	return &QueryPos{imp.Fset, start, end, path, exact, info}, nil
}

// WriteTo writes the oracle query result res to out in a compiler diagnostic format.
func (res *Result) WriteTo(out io.Writer) {
	printf := func(pos interface{}, format string, args ...interface{}) {
		fprintf(out, res.fset, pos, format, args...)
	}
	res.q.display(printf)

	// Print warnings after the main output.
	if res.warnings != nil {
		fmt.Fprintln(out, "\nPointer analysis warnings:")
		for _, w := range res.warnings {
			printf(w.Pos, "warning: "+w.Message)
		}
	}
}

// ---------- Utilities ----------

// buildSSA constructs the SSA representation of Go-source function bodies.
// Not needed in simpler modes, e.g. freevars.
//
func buildSSA(o *Oracle) {
	o.prog.BuildAll()
}

// ptrAnalysis runs the pointer analysis and returns its result.
func ptrAnalysis(o *Oracle) *pointer.Result {
	return pointer.Analyze(&o.ptaConfig)
}

// unparen returns e with any enclosing parentheses stripped.
func unparen(e ast.Expr) ast.Expr {
	for {
		p, ok := e.(*ast.ParenExpr)
		if !ok {
			break
		}
		e = p.X
	}
	return e
}

// deref returns a pointer's element type; otherwise it returns typ.
func deref(typ types.Type) types.Type {
	if p, ok := typ.Underlying().(*types.Pointer); ok {
		return p.Elem()
	}
	return typ
}

// fprintf prints to w a message of the form "location: message\n"
// where location is derived from pos.
//
// pos must be one of:
//    - a token.Pos, denoting a position
//    - an ast.Node, denoting an interval
//    - anything with a Pos() method:
//         ssa.Member, ssa.Value, ssa.Instruction, types.Object, pointer.Label, etc.
//    - a QueryPos, denoting the extent of the user's query.
//    - nil, meaning no position at all.
//
// The output format is is compatible with the 'gnu'
// compilation-error-regexp in Emacs' compilation mode.
// TODO(adonovan): support other editors.
//
func fprintf(w io.Writer, fset *token.FileSet, pos interface{}, format string, args ...interface{}) {
	var start, end token.Pos
	switch pos := pos.(type) {
	case ast.Node:
		start = pos.Pos()
		end = pos.End()
	case token.Pos:
		start = pos
		end = start
	case interface {
		Pos() token.Pos
	}:
		start = pos.Pos()
		end = start
	case *QueryPos:
		start = pos.start
		end = pos.end
	case nil:
		// no-op
	default:
		panic(fmt.Sprintf("invalid pos: %T", pos))
	}

	if sp := fset.Position(start); start == end {
		// (prints "-: " for token.NoPos)
		fmt.Fprintf(w, "%s: ", sp)
	} else {
		ep := fset.Position(end)
		// The -1 below is a concession to Emacs's broken use of
		// inclusive (not half-open) intervals.
		// Other editors may not want it.
		// TODO(adonovan): add an -editor=vim|emacs|acme|auto
		// flag; auto uses EMACS=t / VIM=... / etc env vars.
		fmt.Fprintf(w, "%s:%d.%d-%d.%d: ",
			sp.Filename, sp.Line, sp.Column, ep.Line, ep.Column-1)
	}
	fmt.Fprintf(w, format, args...)
	io.WriteString(w, "\n")
}
