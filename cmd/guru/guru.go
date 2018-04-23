// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// TODO(adonovan): new queries
// - show all statements that may update the selected lvalue
//   (local, global, field, etc).
// - show all places where an object of type T is created
//   (&T{}, var t T, new(T), new(struct{array [3]T}), etc.

import (
	"encoding/json"
	"fmt"
	"go/ast"
	"go/build"
	"go/parser"
	"go/token"
	"go/types"
	"io"
	"log"
	"path/filepath"
	"strings"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/go/buildutil"
	"golang.org/x/tools/go/loader"
	"golang.org/x/tools/go/pointer"
	"golang.org/x/tools/go/ssa"
)

type printfFunc func(pos interface{}, format string, args ...interface{})

// A QueryResult is an item of output.  Each query produces a stream of
// query results, calling Query.Output for each one.
type QueryResult interface {
	// JSON returns the QueryResult in JSON form.
	JSON(fset *token.FileSet) []byte

	// PrintPlain prints the QueryResult in plain text form.
	// The implementation calls printfFunc to print each line of output.
	PrintPlain(printf printfFunc)
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

// A Query specifies a single guru query.
type Query struct {
	Pos   string         // query position
	Build *build.Context // package loading configuration

	// pointer analysis options
	Scope      []string  // main packages in (*loader.Config).FromArgs syntax
	PTALog     io.Writer // (optional) pointer-analysis log file
	Reflection bool      // model reflection soundly (currently slow).

	// result-printing function, safe for concurrent use
	Output func(*token.FileSet, QueryResult)
}

// Run runs an guru query and populates its Fset and Result.
func Run(mode string, q *Query) error {
	switch mode {
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
		return fmt.Errorf("invalid mode: %q", mode)
	}
}

func setPTAScope(lconf *loader.Config, scope []string) error {
	pkgs := buildutil.ExpandPatterns(lconf.Build, scope)
	if len(pkgs) == 0 {
		return fmt.Errorf("no packages specified for pointer analysis scope")
	}
	// The value of each entry in pkgs is true,
	// giving ImportWithTests (not Import) semantics.
	lconf.ImportPkgs = pkgs
	return nil
}

// Create a pointer.Config whose scope is the initial packages of lprog
// and their dependencies.
func setupPTA(prog *ssa.Program, lprog *loader.Program, ptaLog io.Writer, reflection bool) (*pointer.Config, error) {
	// For each initial package (specified on the command line),
	// if it has a main function, analyze that,
	// otherwise analyze its tests, if any.
	var mains []*ssa.Package
	for _, info := range lprog.InitialPackages() {
		p := prog.Package(info.Pkg)

		// Add package to the pointer analysis scope.
		if p.Pkg.Name() == "main" && p.Func("main") != nil {
			mains = append(mains, p)
		} else if main := prog.CreateTestMainPackage(p); main != nil {
			mains = append(mains, main)
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
	fqpos, err := fastQueryPos(conf.Build, pos)
	if err != nil {
		return "", err // bad query
	}
	filename := fqpos.fset.File(fqpos.start).Name()

	_, importPath, err := guessImportPath(filename, conf.Build)
	if err != nil {
		// Can't find GOPATH dir.
		// Treat the query file as its own package.
		importPath = "command-line-arguments"
		conf.CreateFromFilenames(importPath, filename)
	} else {
		// Check that it's possible to load the queried package.
		// (e.g. guru tests contain different 'package' decls in same dir.)
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
			// This happens for ad-hoc packages like
			// $GOROOT/src/net/http/triv.go.
			return "", fmt.Errorf("package %q doesn't contain file %s",
				importPath, filename)
		}
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
func parseQueryPos(lprog *loader.Program, pos string, needExact bool) (*queryPos, error) {
	filename, startOffset, endOffset, err := parsePos(pos)
	if err != nil {
		return nil, err
	}

	// Find the named file among those in the loaded program.
	var file *token.File
	lprog.Fset.Iterate(func(f *token.File) bool {
		if sameFile(filename, f.Name()) {
			file = f
			return false // done
		}
		return true // continue
	})
	if file == nil {
		return nil, fmt.Errorf("file %s not found in loaded program", filename)
	}

	start, end, err := fileOffsetToPos(file, startOffset, endOffset)
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

// loadWithSoftErrors calls lconf.Load, suppressing "soft" errors.  (See Go issue 16530.)
// TODO(adonovan): Once the loader has an option to allow soft errors,
// replace calls to loadWithSoftErrors with loader calls with that parameter.
func loadWithSoftErrors(lconf *loader.Config) (*loader.Program, error) {
	lconf.AllowErrors = true

	// Ideally we would just return conf.Load() here, but go/types
	// reports certain "soft" errors that gc does not (Go issue 14596).
	// As a workaround, we set AllowErrors=true and then duplicate
	// the loader's error checking but allow soft errors.
	// It would be nice if the loader API permitted "AllowErrors: soft".
	prog, err := lconf.Load()
	if err != nil {
		return nil, err
	}
	var errpkgs []string
	// Report hard errors in indirectly imported packages.
	for _, info := range prog.AllPackages {
		if containsHardErrors(info.Errors) {
			errpkgs = append(errpkgs, info.Pkg.Path())
		} else {
			// Enable SSA construction for packages containing only soft errors.
			info.TransitivelyErrorFree = true
		}
	}
	if errpkgs != nil {
		var more string
		if len(errpkgs) > 3 {
			more = fmt.Sprintf(" and %d more", len(errpkgs)-3)
			errpkgs = errpkgs[:3]
		}
		return nil, fmt.Errorf("couldn't load packages due to errors: %s%s",
			strings.Join(errpkgs, ", "), more)
	}
	return prog, err
}

func containsHardErrors(errors []error) bool {
	for _, err := range errors {
		if err, ok := err.(types.Error); ok && err.Soft {
			continue
		}
		return true
	}
	return false
}

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
	case *types.PkgName:
		// The Pos of most PkgName objects does not coincide with an identifier,
		// so we suppress the usual start+len(name) heuristic for types.Objects.
		start = pos.Pos()
		end = start
	case types.Object:
		start = pos.Pos()
		end = start + token.Pos(len(pos.Name())) // heuristic
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

func toJSON(x interface{}) []byte {
	b, err := json.MarshalIndent(x, "", "\t")
	if err != nil {
		log.Fatalf("JSON error: %v", err)
	}
	return b
}
