// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build go1.5

// Package oracle contains the implementation of the oracle tool whose
// command-line is provided by golang.org/x/tools/cmd/oracle.
//
// http://golang.org/s/oracle-design
// http://golang.org/s/oracle-user-manual
//
package oracle // import "golang.org/x/tools/oracle"

// This file defines oracle.Query, the entry point for the oracle tool.
// The actual executable is defined in cmd/oracle.

// TODO(adonovan): new queries
// - show all statements that may update the selected lvalue
//   (local, global, field, etc).
// - show all places where an object of type T is created
//   (&T{}, var t T, new(T), new(struct{array [3]T}), etc.

import (
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"go/types"
	"io"
	"path/filepath"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/pointer"
	"golang.org/x/tools/go/ssa"
	"golang.org/x/tools/oracle/serial"
)

type printfFunc func(pos interface{}, format string, args ...interface{})

// queryResult is the interface of each query-specific result type.
type queryResult interface {
	toSerial(res *serial.Result, fset *token.FileSet)
	display(printf printfFunc)
}

// A QueryPos represents the position provided as input to a query:
// a textual extent in the program's source code, the AST node it
// corresponds to, and the package to which it belongs.
// Instances are created by parseQueryPos.
type queryPos struct {
	fset       *token.FileSet
	start, end token.Pos           // source extent of query
	path       []ast.Node          // AST path from query node to root of ast.File
	exact      bool                // 2nd result of PathEnclosingInterval
	info       *loader.PackageInfo // type info for the queried package (nil for fastQueryPos)
}

// TypeString prints type T relative to the query position.
func (qpos *queryPos) typeString(T types.Type) string {
	return types.TypeString(T, types.RelativeTo(qpos.info.Pkg))
}

// ObjectString prints object obj relative to the query position.
func (qpos *queryPos) objectString(obj types.Object) string {
	return types.ObjectString(obj, types.RelativeTo(qpos.info.Pkg))
}

// SelectionString prints selection sel relative to the query position.
func (qpos *queryPos) selectionString(sel *types.Selection) string {
	return types.SelectionString(sel, types.RelativeTo(qpos.info.Pkg))
}

// A Query specifies a single oracle query.
type Query struct {
	Mode  string         // query mode ("callers", etc)
	Pos   string         // query position
	Build *build.Context // package loading configuration

	// pointer analysis options
	Scope      []string  // main packages in (*loader.Config).FromArgs syntax
	PTALog     io.Writer // (optional) pointer-analysis log file
	Reflection bool      // model reflection soundly (currently slow).

	// Populated during Run()
	Fset   *token.FileSet
	result queryResult
}

// Serial returns an instance of serial.Result, which implements the
// {xml,json}.Marshaler interfaces so that query results can be
// serialized as JSON or XML.
//
func (q *Query) Serial() *serial.Result {
	resj := &serial.Result{Mode: q.Mode}
	q.result.toSerial(resj, q.Fset)
	return resj
}

// WriteTo writes the oracle query result res to out in a compiler diagnostic format.
func (q *Query) WriteTo(out io.Writer) {
	printf := func(pos interface{}, format string, args ...interface{}) {
		fprintf(out, q.Fset, pos, format, args...)
	}
	q.result.display(printf)
}

// Run runs an oracle query and populates its Fset and Result.
func Run(q *Query) error {
	switch q.Mode {
	case "callees":
		return callees(q)
	case "callers":
		return callers(q)
	case "callstack":
		return callstack(q)
	case "peers":
		return peers(q)
	case "pointsto":
		return pointsto(q)
	case "whicherrs":
		return whicherrs(q)
	case "definition":
		return definition(q)
	case "describe":
		return describe(q)
	case "freevars":
		return freevars(q)
	case "implements":
		return implements(q)
	case "referrers":
		return referrers(q)
	case "what":
		return what(q)
	default:
		return fmt.Errorf("invalid mode: %q", q.Mode)
	}
}

func setPTAScope(lconf *loader.Config, scope []string) error {
	if len(scope) == 0 {
		return fmt.Errorf("no packages specified for pointer analysis scope")
	}

	// Determine initial packages for PTA.
	args, err := lconf.FromArgs(scope, true)
	if err != nil {
		return err
	}
	if len(args) > 0 {
		return fmt.Errorf("surplus arguments: %q", args)
	}
	return nil
}

// Create a pointer.Config whose scope is the initial packages of lprog
// and their dependencies.
func setupPTA(prog *ssa.Program, lprog *loader.Program, ptaLog io.Writer, reflection bool) (*pointer.Config, error) {
	// TODO(adonovan): the body of this function is essentially
	// duplicated in all go/pointer clients.  Refactor.

	// For each initial package (specified on the command line),
	// if it has a main function, analyze that,
	// otherwise analyze its tests, if any.
	var testPkgs, mains []*ssa.Package
	for _, info := range lprog.InitialPackages() {
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
	return &pointer.Config{
		Log:        ptaLog,
		Reflection: reflection,
		Mains:      mains,
	}, nil
}

// importQueryPackage finds the package P containing the
// query position and tells conf to import it.
// It returns the package's path.
func importQueryPackage(pos string, conf *loader.Config) (string, error) {
	fqpos, err := fastQueryPos(pos)
	if err != nil {
		return "", err // bad query
	}
	filename := fqpos.fset.File(fqpos.start).Name()

	// This will not work for ad-hoc packages
	// such as $GOROOT/src/net/http/triv.go.
	// TODO(adonovan): ensure we report a clear error.
	_, importPath, err := guessImportPath(filename, conf.Build)
	if err != nil {
		return "", err // can't find GOPATH dir
	}
	if importPath == "" {
		return "", fmt.Errorf("can't guess import path from %s", filename)
	}

	// Check that it's possible to load the queried package.
	// (e.g. oracle tests contain different 'package' decls in same dir.)
	// Keep consistent with logic in loader/util.go!
	cfg2 := *conf.Build
	cfg2.CgoEnabled = false
	bp, err := cfg2.Import(importPath, "", 0)
	if err != nil {
		return "", err // no files for package
	}

	switch pkgContainsFile(bp, filename) {
	case 'T':
		conf.ImportWithTests(importPath)
	case 'X':
		conf.ImportWithTests(importPath)
		importPath += "_test" // for TypeCheckFuncBodies
	case 'G':
		conf.Import(importPath)
	default:
		return "", fmt.Errorf("package %q doesn't contain file %s",
			importPath, filename)
	}

	conf.TypeCheckFuncBodies = func(p string) bool { return p == importPath }

	return importPath, nil
}

// pkgContainsFile reports whether file was among the packages Go
// files, Test files, eXternal test files, or not found.
func pkgContainsFile(bp *build.Package, filename string) byte {
	for i, files := range [][]string{bp.GoFiles, bp.TestGoFiles, bp.XTestGoFiles} {
		for _, file := range files {
			if sameFile(filepath.Join(bp.Dir, file), filename) {
				return "GTX"[i]
			}
		}
	}
	return 0 // not found
}

// ParseQueryPos parses the source query position pos and returns the
// AST node of the loaded program lprog that it identifies.
// If needExact, it must identify a single AST subtree;
// this is appropriate for queries that allow fairly arbitrary syntax,
// e.g. "describe".
//
func parseQueryPos(lprog *loader.Program, posFlag string, needExact bool) (*queryPos, error) {
	filename, startOffset, endOffset, err := parsePosFlag(posFlag)
	if err != nil {
		return nil, err
	}
	start, end, err := findQueryPos(lprog.Fset, filename, startOffset, endOffset)
	if err != nil {
		return nil, err
	}
	info, path, exact := lprog.PathEnclosingInterval(start, end)
	if path == nil {
		return nil, fmt.Errorf("no syntax here")
	}
	if needExact && !exact {
		return nil, fmt.Errorf("ambiguous selection within %s", astutil.NodeDescription(path[0]))
	}
	return &queryPos{lprog.Fset, start, end, path, exact, info}, nil
}

// ---------- Utilities ----------

// allowErrors causes type errors to be silently ignored.
// (Not suitable if SSA construction follows.)
func allowErrors(lconf *loader.Config) {
	ctxt := *lconf.Build // copy
	ctxt.CgoEnabled = false
	lconf.Build = &ctxt
	lconf.AllowErrors = true
	// AllErrors makes the parser always return an AST instead of
	// bailing out after 10 errors and returning an empty ast.File.
	lconf.ParserMode = parser.AllErrors
	lconf.TypeChecker.Error = func(err error) {}
}

// ptrAnalysis runs the pointer analysis and returns its result.
func ptrAnalysis(conf *pointer.Config) *pointer.Result {
	result, err := pointer.Analyze(conf)
	if err != nil {
		panic(err) // pointer analysis internal error
	}
	return result
}

func unparen(e ast.Expr) ast.Expr { return astutil.Unparen(e) }

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
	case *queryPos:
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
