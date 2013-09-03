// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

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

type oracle struct {
	out    io.Writer      // standard output
	prog   *ssa.Program   // the SSA program [need&SSA]
	config pointer.Config // pointer analysis configuration

	// need&(Pos|ExactPos):
	startPos, endPos token.Pos             // source extent of query
	queryPkgInfo     *importer.PackageInfo // type info for the queried package
	queryPath        []ast.Node            // AST path from query node to root of ast.File

	timers map[string]time.Duration // phase timing information
}

// A set of bits indicating the analytical requirements of each mode.
const (
	Pos         = 1 << iota // needs a position
	ExactPos                // needs an exact AST selection; implies Pos
	SSA                     // needs SSA intermediate form
	WholeSource             // needs ASTs/SSA (not just types) for whole program

	// TODO(adonovan): implement more efficiently than WholeSource|SSA.
	TypedAST = WholeSource | SSA // needs typed AST for the queried package; implies Pos
)

type modeInfo struct {
	needs int
	impl  func(*oracle) (queryResult, error)
}

var modes = map[string]modeInfo{
	"callees":    modeInfo{WholeSource | SSA | ExactPos, callees},
	"callers":    modeInfo{WholeSource | SSA | Pos, callers},
	"callgraph":  modeInfo{WholeSource | SSA, callgraph},
	"callstack":  modeInfo{WholeSource | SSA | Pos, callstack},
	"describe":   modeInfo{WholeSource | SSA | ExactPos, describe},
	"freevars":   modeInfo{TypedAST | Pos, freevars},
	"implements": modeInfo{TypedAST | Pos, implements},
	"peers":      modeInfo{WholeSource | SSA | Pos, peers},
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

// Query runs the oracle.
// args specify the main package in importer.CreatePackageFromArgs syntax.
// mode is the query mode ("callers", etc).
// pos is the selection in parseQueryPos() syntax.
// ptalog is the (optional) pointer-analysis log file.
// buildContext is the optional configuration for locating packages.
//
func Query(args []string, mode, pos string, ptalog io.Writer, buildContext *build.Context) (*Result, error) {
	minfo, ok := modes[mode]
	if !ok {
		if mode == "" {
			return nil, errors.New("you must specify a -mode to perform")
		}
		return nil, fmt.Errorf("invalid mode type: %q", mode)
	}

	var loader importer.SourceLoader
	if minfo.needs&WholeSource != 0 {
		loader = importer.MakeGoBuildLoader(buildContext)
	}
	imp := importer.New(&importer.Config{Loader: loader})
	o := &oracle{
		prog:   ssa.NewProgram(imp.Fset, 0),
		timers: make(map[string]time.Duration),
	}
	o.config.Log = ptalog

	var res Result
	o.config.Warn = func(pos token.Pos, format string, args ...interface{}) {
		res.warnings = append(res.warnings, warning{pos, format, args})
	}

	// Phase timing diagnostics.
	if false {
		defer func() {
			fmt.Println()
			for name, duration := range o.timers {
				fmt.Printf("# %-30s %s\n", name, duration)
			}
		}()
	}

	// Load/parse/type-check program from args.
	start := time.Now()
	initialPkgInfo, _, err := importer.CreatePackageFromArgs(imp, args)
	if err != nil {
		return nil, err // I/O, parser or type error
	}
	o.timers["load/parse/type"] = time.Since(start)

	// Parse the source query position.
	if minfo.needs&(Pos|ExactPos) != 0 {
		var err error
		o.startPos, o.endPos, err = parseQueryPos(o.prog.Fset, pos)
		if err != nil {
			return nil, err
		}

		var exact bool
		o.queryPkgInfo, o.queryPath, exact = imp.PathEnclosingInterval(o.startPos, o.endPos)
		if o.queryPath == nil {
			return nil, o.errorf(false, "no syntax here")
		}
		if minfo.needs&ExactPos != 0 && !exact {
			return nil, o.errorf(o.queryPath[0], "ambiguous selection within %s",
				importer.NodeDescription(o.queryPath[0]))
		}
	}

	// Create SSA package for the initial package and its dependencies.
	if minfo.needs&SSA != 0 {
		start = time.Now()

		// All packages.
		for _, info := range imp.Packages {
			o.prog.CreatePackage(info) // create ssa.Package
		}

		// Initial package (specified on command line)
		initialPkg := o.prog.Package(initialPkgInfo.Pkg)

		// Add package to the pointer analysis scope.
		if initialPkg.Func("main") == nil {
			if initialPkg.CreateTestMainFunction() == nil {
				return nil, o.errorf(false, "analysis scope has no main() entry points")
			}
		}
		o.config.Mains = append(o.config.Mains, initialPkg)

		// Query package.
		if o.queryPkgInfo != nil {
			pkg := o.prog.Package(o.queryPkgInfo.Pkg)
			pkg.SetDebugMode(true)
			pkg.Build()
		}

		o.timers["SSA-create"] = time.Since(start)
	}

	// SSA is built and we have query{Path,PkgInfo}.
	// Release the other ASTs and type info to the GC.
	imp = nil

	res.q, err = minfo.impl(o)
	if err != nil {
		return nil, err
	}
	res.mode = mode
	res.fset = o.prog.Fset
	res.fprintf = o.fprintf // captures o.prog, o.{start,end}Pos for later printing
	return &res, nil
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
func buildSSA(o *oracle) {
	start := time.Now()
	o.prog.BuildAll()
	o.timers["SSA-build"] = time.Since(start)
}

// ptrAnalysis runs the pointer analysis and returns the synthetic
// root of the callgraph.
//
func ptrAnalysis(o *oracle) pointer.CallGraphNode {
	start := time.Now()
	root := pointer.Analyze(&o.config)
	o.timers["pointer analysis"] = time.Since(start)
	return root
}

func parseDecimal(s string) int {
	if s, err := strconv.ParseInt(s, 10, 32); err == nil {
		return int(s)
	}
	return -1
}

// parseQueryPos parses a string of the form "file:pos" or
// file:start-end" where pos, start, end are decimal integers, and
// returns the extent to which it refers.
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
	if hyphen := strings.Index(offset, "-"); hyphen < 0 {
		// e.g. "foo.go:123"
		startOffset = parseDecimal(offset)
		endOffset = startOffset
	} else {
		// e.g. "foo.go:123-456"
		startOffset = parseDecimal(offset[:hyphen])
		endOffset = parseDecimal(offset[hyphen+1:])
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
//    - a bool, meaning the extent [o.startPos, o.endPos) of the user's query.
//      (the value is ignored)
//    - nil, meaning no position at all.
//
// The output format is is compatible with the 'gnu'
// compilation-error-regexp in Emacs' compilation mode.
// TODO(adonovan): support other editors.
//
func (o *oracle) fprintf(w io.Writer, pos interface{}, format string, args ...interface{}) {
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
	case bool:
		start = o.startPos
		end = o.endPos
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
func (o *oracle) errorf(pos interface{}, format string, args ...interface{}) error {
	var buf bytes.Buffer
	o.fprintf(&buf, pos, format, args...)
	return errors.New(buf.String())
}

// printNode returns the pretty-printed syntax of n.
func (o *oracle) printNode(n ast.Node) string {
	var buf bytes.Buffer
	printer.Fprint(&buf, o.prog.Fset, n)
	return buf.String()
}
