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

// TODO(adonovan): new query: show all statements that may update the
// selected lvalue (local, global, field, etc).

import (
	"bytes"
	encjson "encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/build"
	"go/printer"
	"go/token"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"code.google.com/p/go.tools/go/types"
	"code.google.com/p/go.tools/importer"
	"code.google.com/p/go.tools/oracle/json"
	"code.google.com/p/go.tools/pointer"
	"code.google.com/p/go.tools/ssa"
)

// An Oracle holds the program state required for one or more queries.
type Oracle struct {
	out    io.Writer      // standard output
	prog   *ssa.Program   // the SSA program [only populated if need&SSA]
	config pointer.Config // pointer analysis configuration [TODO rename ptaConfig]

	// need&AllTypeInfo
	typeInfo map[*types.Package]*importer.PackageInfo // type info for all ASTs in the program

	timers map[string]time.Duration // phase timing information
}

// A set of bits indicating the analytical requirements of each mode.
//
// Typed ASTs for the whole program are always constructed
// transiently; they are retained only for the queried package unless
// needAllTypeInfo is set.
const (
	needPos         = 1 << iota // needs a position
	needExactPos                // needs an exact AST selection; implies needPos
	needAllTypeInfo             // needs to retain type info for all ASTs in the program
	needSSA                     // needs ssa.Packages for whole program
	needSSADebug                // needs debug info for ssa.Packages
	needPTA         = needSSA   // needs pointer analysis
	needAll         = -1        // needs everything (e.g. a sequence of queries)
)

type modeInfo struct {
	name  string
	needs int
	impl  func(*Oracle, *QueryPos) (queryResult, error)
}

var modes = []*modeInfo{
	{"callees", needPTA | needExactPos, callees},
	{"callers", needPTA | needPos, callers},
	{"callgraph", needPTA, callgraph},
	{"callstack", needPTA | needPos, callstack},
	{"describe", needPTA | needSSADebug | needExactPos, describe},
	{"freevars", needPos, freevars},
	{"implements", needPos, implements},
	{"peers", needPTA | needSSADebug | needPos, peers},
	{"referrers", needAllTypeInfo | needPos, referrers},
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
	toJSON(res *json.Result, fset *token.FileSet)
	display(printf printfFunc)
}

type warning struct {
	pos    token.Pos
	format string
	args   []interface{}
}

// A QueryPos represents the position provided as input to a query:
// a textual extent in the program's source code, the AST node it
// corresponds to, and the package to which it belongs.
// Instances are created by ParseQueryPos.
//
type QueryPos struct {
	start, end token.Pos             // source extent of query
	info       *importer.PackageInfo // type info for the queried package
	path       []ast.Node            // AST path from query node to root of ast.File
}

// A Result encapsulates the result of an oracle.Query.
//
// Result instances implement the json.Marshaler interface, i.e. they
// can be JSON-serialized.
type Result struct {
	fset *token.FileSet
	// fprintf is a closure over the oracle's fileset and start/end position.
	fprintf  func(w io.Writer, pos interface{}, format string, args ...interface{})
	q        queryResult // the query-specific result
	mode     string      // query mode
	warnings []warning   // pointer analysis warnings
}

func (res *Result) MarshalJSON() ([]byte, error) {
	resj := &json.Result{Mode: res.mode}
	res.q.toJSON(resj, res.fset)
	for _, w := range res.warnings {
		resj.Warnings = append(resj.Warnings, json.PTAWarning{
			Pos:     res.fset.Position(w.pos).String(),
			Message: fmt.Sprintf(w.format, w.args...),
		})
	}
	return encjson.Marshal(resj)
}

// Query runs a single oracle query.
//
// args specify the main package in importer.CreatePackageFromArgs syntax.
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
	minfo := findMode(mode)
	if minfo == nil {
		return nil, fmt.Errorf("invalid mode type: %q", mode)
	}

	imp := importer.New(&importer.Config{Build: buildContext})
	o, err := New(imp, args, ptalog, reflection)
	if err != nil {
		return nil, err
	}

	// Phase timing diagnostics.
	// TODO(adonovan): needs more work.
	// if false {
	// 	defer func() {
	// 		fmt.Println()
	// 		for name, duration := range o.timers {
	// 			fmt.Printf("# %-30s %s\n", name, duration)
	// 		}
	// 	}()
	// }

	var qpos *QueryPos
	if minfo.needs&(needPos|needExactPos) != 0 {
		var err error
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

// New constructs a new Oracle that can be used for a sequence of queries.
//
// imp will be used to load source code for imported packages.
// It must not yet have loaded any packages.
//
// args specify the main package in importer.CreatePackageFromArgs syntax.
//
// ptalog is the (optional) pointer-analysis log file.
// reflection determines whether to model reflection soundly (currently slow).
//
func New(imp *importer.Importer, args []string, ptalog io.Writer, reflection bool) (*Oracle, error) {
	return newOracle(imp, args, ptalog, needAll, reflection)
}

func newOracle(imp *importer.Importer, args []string, ptalog io.Writer, needs int, reflection bool) (*Oracle, error) {
	o := &Oracle{
		prog:   ssa.NewProgram(imp.Fset, 0),
		timers: make(map[string]time.Duration),
	}
	o.config.Log = ptalog
	o.config.Reflection = reflection

	// Load/parse/type-check program from args.
	start := time.Now()
	initialPkgInfos, args, err := imp.LoadInitialPackages(args)
	if err != nil {
		return nil, err // I/O or parser error
	}
	if len(args) > 0 {
		return nil, fmt.Errorf("surplus arguments: %q", args)
	}
	o.timers["load/parse/type"] = time.Since(start)

	// Retain type info for all ASTs in the program.
	if needs&needAllTypeInfo != 0 {
		m := make(map[*types.Package]*importer.PackageInfo)
		for _, p := range imp.AllPackages() {
			m[p.Pkg] = p
		}
		o.typeInfo = m
	}

	// Create SSA package for the initial package and its dependencies.
	if needs&needSSA != 0 {
		start = time.Now()

		// Create SSA packages.
		if err := o.prog.CreatePackages(imp); err != nil {
			return nil, o.errorf(nil, "%s", err)
		}

		// Initial packages (specified on command line)
		for _, info := range initialPkgInfos {
			initialPkg := o.prog.Package(info.Pkg)

			// Add package to the pointer analysis scope.
			if initialPkg.Func("main") == nil {
				// TODO(adonovan): to simulate 'go test' more faithfully, we
				// should build a single synthetic testmain package,
				// not synthetic main functions to many packages.
				if initialPkg.CreateTestMainFunction() == nil {
					return nil, o.errorf(nil, "analysis scope has no main() entry points")
				}
			}
			o.config.Mains = append(o.config.Mains, initialPkg)
		}

		if needs&needSSADebug != 0 {
			for _, pkg := range o.prog.AllPackages() {
				pkg.SetDebugMode(true)
			}
		}

		o.timers["SSA-create"] = time.Since(start)
	}

	return o, nil
}

// Query runs the query of the specified mode and selection.
func (o *Oracle) Query(mode string, qpos *QueryPos) (*Result, error) {
	minfo := findMode(mode)
	if minfo == nil {
		return nil, fmt.Errorf("invalid mode type: %q", mode)
	}
	return o.query(minfo, qpos)
}

func (o *Oracle) query(minfo *modeInfo, qpos *QueryPos) (*Result, error) {
	res := &Result{
		mode:    minfo.name,
		fset:    o.prog.Fset,
		fprintf: o.fprintf, // captures o.prog, o.{start,end}Pos for later printing
	}
	o.config.Warn = func(pos token.Pos, format string, args ...interface{}) {
		res.warnings = append(res.warnings, warning{pos, format, args})
	}
	var err error
	res.q, err = minfo.impl(o, qpos)
	if err != nil {
		return nil, err
	}
	return res, nil
}

// ParseQueryPos parses the source query position pos.
// If needExact, it must identify a single AST subtree.
//
func ParseQueryPos(imp *importer.Importer, pos string, needExact bool) (*QueryPos, error) {
	start, end, err := parseQueryPos(imp.Fset, pos)
	if err != nil {
		return nil, err
	}
	info, path, exact := imp.PathEnclosingInterval(start, end)
	if path == nil {
		return nil, errors.New("no syntax here")
	}
	if needExact && !exact {
		return nil, fmt.Errorf("ambiguous selection within %s", importer.NodeDescription(path[0]))
	}
	return &QueryPos{start, end, info, path}, nil
}

// WriteTo writes the oracle query result res to out in a compiler diagnostic format.
func (res *Result) WriteTo(out io.Writer) {
	printf := func(pos interface{}, format string, args ...interface{}) {
		res.fprintf(out, pos, format, args...)
	}
	res.q.display(printf)

	// Print warnings after the main output.
	if res.warnings != nil {
		fmt.Fprintln(out, "\nPointer analysis warnings:")
		for _, w := range res.warnings {
			printf(w.pos, "warning: "+w.format, w.args...)
		}
	}
}

// ---------- Utilities ----------

// buildSSA constructs the SSA representation of Go-source function bodies.
// Not needed in simpler modes, e.g. freevars.
//
func buildSSA(o *Oracle) {
	start := time.Now()
	o.prog.BuildAll()
	o.timers["SSA-build"] = time.Since(start)
}

// ptrAnalysis runs the pointer analysis and returns the synthetic
// root of the callgraph.
//
func ptrAnalysis(o *Oracle) pointer.CallGraphNode {
	start := time.Now()
	root := pointer.Analyze(&o.config)
	o.timers["pointer analysis"] = time.Since(start)
	return root
}

// parseOctothorpDecimal returns the numeric value if s matches "#%d",
// otherwise -1.
func parseOctothorpDecimal(s string) int {
	if s != "" && s[0] == '#' {
		if s, err := strconv.ParseInt(s[1:], 10, 32); err == nil {
			return int(s)
		}
	}
	return -1
}

// parseQueryPos parses a string of the form "file:pos" or
// file:start,end" where pos, start, end match #%d and represent byte
// offsets, and returns the extent to which it refers.
//
// (Numbers without a '#' prefix are reserved for future use,
// e.g. to indicate line/column positions.)
//
func parseQueryPos(fset *token.FileSet, queryPos string) (start, end token.Pos, err error) {
	if queryPos == "" {
		err = fmt.Errorf("no source position specified (-pos flag)")
		return
	}

	colon := strings.LastIndex(queryPos, ":")
	if colon < 0 {
		err = fmt.Errorf("invalid source position -pos=%q", queryPos)
		return
	}
	filename, offset := queryPos[:colon], queryPos[colon+1:]
	startOffset := -1
	endOffset := -1
	if hyphen := strings.Index(offset, ","); hyphen < 0 {
		// e.g. "foo.go:#123"
		startOffset = parseOctothorpDecimal(offset)
		endOffset = startOffset
	} else {
		// e.g. "foo.go:#123,#456"
		startOffset = parseOctothorpDecimal(offset[:hyphen])
		endOffset = parseOctothorpDecimal(offset[hyphen+1:])
	}
	if startOffset < 0 || endOffset < 0 {
		err = fmt.Errorf("invalid -pos offset %q", offset)
		return
	}

	var file *token.File
	fset.Iterate(func(f *token.File) bool {
		if sameFile(filename, f.Name()) {
			// (f.Name() is absolute)
			file = f
			return false // done
		}
		return true // continue
	})
	if file == nil {
		err = fmt.Errorf("couldn't find file containing position -pos=%q", queryPos)
		return
	}

	// Range check [start..end], inclusive of both end-points.

	if 0 <= startOffset && startOffset <= file.Size() {
		start = file.Pos(int(startOffset))
	} else {
		err = fmt.Errorf("start position is beyond end of file -pos=%q", queryPos)
		return
	}

	if 0 <= endOffset && endOffset <= file.Size() {
		end = file.Pos(int(endOffset))
	} else {
		err = fmt.Errorf("end position is beyond end of file -pos=%q", queryPos)
		return
	}

	return
}

// sameFile returns true if x and y have the same basename and denote
// the same file.
//
func sameFile(x, y string) bool {
	if filepath.Base(x) == filepath.Base(y) { // (optimisation)
		if xi, err := os.Stat(x); err == nil {
			if yi, err := os.Stat(y); err == nil {
				return os.SameFile(xi, yi)
			}
		}
	}
	return false
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
func (o *Oracle) fprintf(w io.Writer, pos interface{}, format string, args ...interface{}) {
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

	if sp := o.prog.Fset.Position(start); start == end {
		// (prints "-: " for token.NoPos)
		fmt.Fprintf(w, "%s: ", sp)
	} else {
		ep := o.prog.Fset.Position(end)
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

// errorf is like fprintf, but returns a formatted error string.
func (o *Oracle) errorf(pos interface{}, format string, args ...interface{}) error {
	var buf bytes.Buffer
	o.fprintf(&buf, pos, format, args...)
	return errors.New(buf.String())
}

// printNode returns the pretty-printed syntax of n.
func (o *Oracle) printNode(n ast.Node) string {
	var buf bytes.Buffer
	printer.Fprint(&buf, o.prog.Fset, n)
	return buf.String()
}
