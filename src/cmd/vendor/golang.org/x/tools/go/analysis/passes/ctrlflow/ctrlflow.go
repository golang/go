// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ctrlflow is an analysis that provides a syntactic
// control-flow graph (CFG) for the body of a function.
// It records whether a function cannot return.
// By itself, it does not report any diagnostics.
package ctrlflow

import (
	"go/ast"
	"go/types"
	"log"
	"reflect"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/cfg"
	"golang.org/x/tools/go/types/typeutil"
	"golang.org/x/tools/internal/typesinternal"
)

var Analyzer = &analysis.Analyzer{
	Name:       "ctrlflow",
	Doc:        "build a control-flow graph",
	URL:        "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/ctrlflow",
	Run:        run,
	ResultType: reflect.TypeFor[*CFGs](),
	FactTypes:  []analysis.Fact{new(noReturn)},
	Requires:   []*analysis.Analyzer{inspect.Analyzer},
}

// noReturn is a fact indicating that a function does not return.
type noReturn struct{}

func (*noReturn) AFact() {}

func (*noReturn) String() string { return "noReturn" }

// A CFGs holds the control-flow graphs
// for all the functions of the current package.
type CFGs struct {
	defs      map[*ast.Ident]types.Object // from Pass.TypesInfo.Defs
	funcDecls map[*types.Func]*declInfo
	funcLits  map[*ast.FuncLit]*litInfo
	noReturn  map[*types.Func]bool // functions lacking a reachable return statement
	pass      *analysis.Pass       // transient; nil after construction
}

// NoReturn reports whether the specified control-flow graph cannot return normally.
//
// It is defined for at least all function symbols that appear as the static callee of a
// CallExpr in the current package, even if the callee was imported from a dependency.
//
// The result may incorporate interprocedural information based on induction of
// the "no return" property over the static call graph within the package.
// For example, if f simply calls g and g always calls os.Exit, then both f and g may
// be deemed never to return.
func (c *CFGs) NoReturn(fn *types.Func) bool {
	return c.noReturn[fn]
}

// CFGs has two maps: funcDecls for named functions and funcLits for
// unnamed ones. Unlike funcLits, the funcDecls map is not keyed by its
// syntax node, *ast.FuncDecl, because callMayReturn needs to do a
// look-up by *types.Func, and you can get from an *ast.FuncDecl to a
// *types.Func but not the other way.

type declInfo struct {
	decl    *ast.FuncDecl
	cfg     *cfg.CFG // iff decl.Body != nil
	started bool     // to break cycles
}

type litInfo struct {
	cfg      *cfg.CFG
	noReturn bool // (currently unused)
}

// FuncDecl returns the control-flow graph for a named function.
// It returns nil if decl.Body==nil.
func (c *CFGs) FuncDecl(decl *ast.FuncDecl) *cfg.CFG {
	if decl.Body == nil {
		return nil
	}
	fn := c.defs[decl.Name].(*types.Func)
	return c.funcDecls[fn].cfg
}

// FuncLit returns the control-flow graph for a literal function.
func (c *CFGs) FuncLit(lit *ast.FuncLit) *cfg.CFG {
	return c.funcLits[lit].cfg
}

func run(pass *analysis.Pass) (any, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	// Because CFG construction consumes and produces noReturn
	// facts, CFGs for exported FuncDecls must be built before 'run'
	// returns; we cannot construct them lazily.
	// (We could build CFGs for FuncLits lazily,
	// but the benefit is marginal.)

	// Pass 1. Map types.Funcs to ast.FuncDecls in this package.
	funcDecls := make(map[*types.Func]*declInfo) // functions and methods
	funcLits := make(map[*ast.FuncLit]*litInfo)

	var decls []*types.Func // keys(funcDecls), in order
	var lits []*ast.FuncLit // keys(funcLits), in order

	nodeFilter := []ast.Node{
		(*ast.FuncDecl)(nil),
		(*ast.FuncLit)(nil),
	}
	inspect.Preorder(nodeFilter, func(n ast.Node) {
		switch n := n.(type) {
		case *ast.FuncDecl:
			// Type information may be incomplete.
			if fn, ok := pass.TypesInfo.Defs[n.Name].(*types.Func); ok {
				funcDecls[fn] = &declInfo{decl: n}
				decls = append(decls, fn)
			}
		case *ast.FuncLit:
			funcLits[n] = new(litInfo)
			lits = append(lits, n)
		}
	})

	c := &CFGs{
		defs:      pass.TypesInfo.Defs,
		funcDecls: funcDecls,
		funcLits:  funcLits,
		noReturn:  make(map[*types.Func]bool),
		pass:      pass,
	}

	// Pass 2. Build CFGs.

	// Build CFGs for named functions.
	// Cycles in the static call graph are broken
	// arbitrarily but deterministically.
	// We create noReturn facts as discovered.
	for _, fn := range decls {
		c.buildDecl(fn, funcDecls[fn])
	}

	// Build CFGs for literal functions.
	// These aren't relevant to facts (since they aren't named)
	// but are required for the CFGs.FuncLit API.
	for _, lit := range lits {
		li := funcLits[lit]
		if li.cfg == nil {
			li.cfg = cfg.New(lit.Body, c.callMayReturn)
			if li.cfg.NoReturn() {
				li.noReturn = true
			}
		}
	}

	// All CFGs are now built.
	c.pass = nil

	return c, nil
}

// di.cfg may be nil on return.
func (c *CFGs) buildDecl(fn *types.Func, di *declInfo) {
	// buildDecl may call itself recursively for the same function,
	// because cfg.New is passed the callMayReturn method, which
	// builds the CFG of the callee, leading to recursion.
	// The buildDecl call tree thus resembles the static call graph.
	// We mark each node when we start working on it to break cycles.

	if di.started {
		return // break cycle
	}
	di.started = true

	noreturn, known := knownIntrinsic(fn)
	if !known {
		if di.decl.Body != nil {
			di.cfg = cfg.New(di.decl.Body, c.callMayReturn)
			if di.cfg.NoReturn() {
				noreturn = true
			}
		}
	}
	if noreturn {
		c.pass.ExportObjectFact(fn, new(noReturn))
		c.noReturn[fn] = true
	}

	// debugging
	if false {
		log.Printf("CFG for %s:\n%s (noreturn=%t)\n", fn, di.cfg.Format(c.pass.Fset), noreturn)
	}
}

// callMayReturn reports whether the called function may return.
// It is passed to the CFG builder.
func (c *CFGs) callMayReturn(call *ast.CallExpr) (r bool) {
	if id, ok := call.Fun.(*ast.Ident); ok && c.pass.TypesInfo.Uses[id] == panicBuiltin {
		return false // panic never returns
	}

	// Is this a static call? Also includes static functions
	// parameterized by a type. Such functions may or may not
	// return depending on the parameter type, but in some
	// cases the answer is definite. We let ctrlflow figure
	// that out.
	fn := typeutil.StaticCallee(c.pass.TypesInfo, call)
	if fn == nil {
		return true // callee not statically known; be conservative
	}

	// Function or method declared in this package?
	if di, ok := c.funcDecls[fn]; ok {
		c.buildDecl(fn, di)
		return !c.noReturn[fn]
	}

	// Not declared in this package.
	// Is there a fact from another package?
	if c.pass.ImportObjectFact(fn, new(noReturn)) {
		c.noReturn[fn] = true
		return false
	}

	return true
}

var panicBuiltin = types.Universe.Lookup("panic").(*types.Builtin)

// knownIntrinsic reports whether a function intrinsically never
// returns because it stops execution of the calling thread, or does
// in fact return, contrary to its apparent body, because it is
// handled specially by the compiler.
//
// It is the base case in the recursion.
func knownIntrinsic(fn *types.Func) (noreturn, known bool) {
	// Add functions here as the need arises, but don't allocate memory.

	// Functions known intrinsically never to return.
	if typesinternal.IsFunctionNamed(fn, "syscall", "Exit", "ExitProcess", "ExitThread") ||
		typesinternal.IsFunctionNamed(fn, "runtime", "Goexit", "fatalthrow", "fatalpanic", "exit") ||
		// Following staticcheck (see go/ir/exits.go) we include functions
		// in several popular logging packages whose no-return status is
		// beyond the analysis to infer.
		// TODO(adonovan): make this list extensible.
		typesinternal.IsMethodNamed(fn, "go.uber.org/zap", "Logger", "Fatal", "Panic") ||
		typesinternal.IsMethodNamed(fn, "go.uber.org/zap", "SugaredLogger", "Fatal", "Fatalw", "Fatalf", "Panic", "Panicw", "Panicf") ||
		typesinternal.IsMethodNamed(fn, "github.com/sirupsen/logrus", "Logger", "Exit", "Panic", "Panicf", "Panicln") ||
		typesinternal.IsMethodNamed(fn, "github.com/sirupsen/logrus", "Entry", "Panicf", "Panicln") ||
		typesinternal.IsFunctionNamed(fn, "k8s.io/klog", "Exit", "ExitDepth", "Exitf", "Exitln", "Fatal", "FatalDepth", "Fatalf", "Fatalln") ||
		typesinternal.IsFunctionNamed(fn, "k8s.io/klog/v2", "Exit", "ExitDepth", "Exitf", "Exitln", "Fatal", "FatalDepth", "Fatalf", "Fatalln") {
		return true, true
	}

	// Compiler intrinsics known to return, contrary to
	// what analysis of the function body would conclude.
	//
	// Not all such intrinsics must be listed here: ctrlflow
	// considers any function called for its value--such as
	// crypto/internal/constanttime.bool2Uint8--to potentially
	// return; only functions called as a statement, for effects,
	// are no-return candidates.
	//
	// Unfortunately this does sometimes mean peering into internals.
	// Where possible, use the nearest enclosing public API function.
	if typesinternal.IsFunctionNamed(fn, "internal/abi", "EscapeNonString") ||
		typesinternal.IsFunctionNamed(fn, "hash/maphash", "Comparable") {
		return false, true
	}

	return // unknown
}
