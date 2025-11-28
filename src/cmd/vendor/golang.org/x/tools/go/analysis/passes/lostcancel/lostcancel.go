// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lostcancel

import (
	_ "embed"
	"fmt"
	"go/ast"
	"go/types"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/ctrlflow"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/go/cfg"
	"golang.org/x/tools/internal/analysis/analyzerutil"
	"golang.org/x/tools/internal/astutil"
	"golang.org/x/tools/internal/typesinternal"
)

//go:embed doc.go
var doc string

var Analyzer = &analysis.Analyzer{
	Name: "lostcancel",
	Doc:  analyzerutil.MustExtractDoc(doc, "lostcancel"),
	URL:  "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/lostcancel",
	Run:  run,
	Requires: []*analysis.Analyzer{
		inspect.Analyzer,
		ctrlflow.Analyzer,
	},
}

const debug = false

var contextPackage = "context"

// checkLostCancel reports a failure to the call the cancel function
// returned by context.WithCancel, either because the variable was
// assigned to the blank identifier, or because there exists a
// control-flow path from the call to a return statement and that path
// does not "use" the cancel function.  Any reference to the variable
// counts as a use, even within a nested function literal.
// If the variable's scope is larger than the function
// containing the assignment, we assume that other uses exist.
//
// checkLostCancel analyzes a single named or literal function.
func run(pass *analysis.Pass) (any, error) {
	// Fast path: bypass check if file doesn't use context.WithCancel.
	if !typesinternal.Imports(pass.Pkg, contextPackage) {
		return nil, nil
	}

	// Call runFunc for each Func{Decl,Lit}.
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)
	nodeTypes := []ast.Node{
		(*ast.FuncLit)(nil),
		(*ast.FuncDecl)(nil),
	}
	inspect.Preorder(nodeTypes, func(n ast.Node) {
		runFunc(pass, n)
	})
	return nil, nil
}

func runFunc(pass *analysis.Pass, node ast.Node) {
	// Find scope of function node
	var funcScope *types.Scope
	switch v := node.(type) {
	case *ast.FuncLit:
		funcScope = pass.TypesInfo.Scopes[v.Type]
	case *ast.FuncDecl:
		funcScope = pass.TypesInfo.Scopes[v.Type]
	}

	// Maps each cancel variable to its defining ValueSpec/AssignStmt.
	cancelvars := make(map[*types.Var]ast.Node)

	// TODO(adonovan): opt: refactor to make a single pass
	// over the AST using inspect.WithStack and node types
	// {FuncDecl,FuncLit,CallExpr,SelectorExpr}.

	// Find the set of cancel vars to analyze.
	astutil.PreorderStack(node, nil, func(n ast.Node, stack []ast.Node) bool {
		if _, ok := n.(*ast.FuncLit); ok && len(stack) > 0 {
			return false // don't stray into nested functions
		}

		// Look for n=SelectorExpr beneath stack=[{AssignStmt,ValueSpec} CallExpr]:
		//
		//   ctx, cancel    := context.WithCancel(...)
		//   ctx, cancel     = context.WithCancel(...)
		//   var ctx, cancel = context.WithCancel(...)
		//
		if !isContextWithCancel(pass.TypesInfo, n) || !isCall(stack[len(stack)-1]) {
			return true
		}
		var id *ast.Ident // id of cancel var
		stmt := stack[len(stack)-2]
		switch stmt := stmt.(type) {
		case *ast.ValueSpec:
			if len(stmt.Names) > 1 {
				id = stmt.Names[1]
			}
		case *ast.AssignStmt:
			if len(stmt.Lhs) > 1 {
				id, _ = stmt.Lhs[1].(*ast.Ident)
			}
		}
		if id != nil {
			if id.Name == "_" {
				pass.ReportRangef(id,
					"the cancel function returned by context.%s should be called, not discarded, to avoid a context leak",
					n.(*ast.SelectorExpr).Sel.Name)
			} else if v, ok := pass.TypesInfo.Uses[id].(*types.Var); ok {
				// If the cancel variable is defined outside function scope,
				// do not analyze it.
				if funcScope.Contains(v.Pos()) {
					cancelvars[v] = stmt
				}
			} else if v, ok := pass.TypesInfo.Defs[id].(*types.Var); ok {
				cancelvars[v] = stmt
			}
		}
		return true
	})

	if len(cancelvars) == 0 {
		return // no need to inspect CFG
	}

	// Obtain the CFG.
	cfgs := pass.ResultOf[ctrlflow.Analyzer].(*ctrlflow.CFGs)
	var g *cfg.CFG
	var sig *types.Signature
	switch node := node.(type) {
	case *ast.FuncDecl:
		sig, _ = pass.TypesInfo.Defs[node.Name].Type().(*types.Signature)
		if node.Name.Name == "main" && sig.Recv() == nil && pass.Pkg.Name() == "main" {
			// Returning from main.main terminates the process,
			// so there's no need to cancel contexts.
			return
		}
		g = cfgs.FuncDecl(node)

	case *ast.FuncLit:
		sig, _ = pass.TypesInfo.Types[node.Type].Type.(*types.Signature)
		g = cfgs.FuncLit(node)
	}
	if sig == nil {
		return // missing type information
	}

	// Print CFG.
	if debug {
		fmt.Println(g.Format(pass.Fset))
	}

	// Examine the CFG for each variable in turn.
	// (It would be more efficient to analyze all cancelvars in a
	// single pass over the AST, but seldom is there more than one.)
	for v, stmt := range cancelvars {
		if ret := lostCancelPath(pass, g, v, stmt, sig); ret != nil {
			lineno := pass.Fset.Position(stmt.Pos()).Line
			pass.ReportRangef(stmt, "the %s function is not used on all paths (possible context leak)", v.Name())

			pos, end := ret.Pos(), ret.End()
			// golang/go#64547: cfg.Block.Return may return a synthetic
			// ReturnStmt that overflows the file.
			if pass.Fset.File(pos) != pass.Fset.File(end) {
				end = pos
			}
			pass.Report(analysis.Diagnostic{
				Pos:     pos,
				End:     end,
				Message: fmt.Sprintf("this return statement may be reached without using the %s var defined on line %d", v.Name(), lineno),
			})
		}
	}
}

func isCall(n ast.Node) bool { _, ok := n.(*ast.CallExpr); return ok }

// isContextWithCancel reports whether n is one of the qualified identifiers
// context.With{Cancel,Timeout,Deadline}.
func isContextWithCancel(info *types.Info, n ast.Node) bool {
	sel, ok := n.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	switch sel.Sel.Name {
	case "WithCancel", "WithCancelCause",
		"WithTimeout", "WithTimeoutCause",
		"WithDeadline", "WithDeadlineCause":
	default:
		return false
	}
	if x, ok := sel.X.(*ast.Ident); ok {
		if pkgname, ok := info.Uses[x].(*types.PkgName); ok {
			return pkgname.Imported().Path() == contextPackage
		}
		// Import failed, so we can't check package path.
		// Just check the local package name (heuristic).
		return x.Name == "context"
	}
	return false
}

// lostCancelPath finds a path through the CFG, from stmt (which defines
// the 'cancel' variable v) to a return statement, that doesn't "use" v.
// If it finds one, it returns the return statement (which may be synthetic).
// sig is the function's type, if known.
func lostCancelPath(pass *analysis.Pass, g *cfg.CFG, v *types.Var, stmt ast.Node, sig *types.Signature) *ast.ReturnStmt {
	vIsNamedResult := sig != nil && tupleContains(sig.Results(), v)

	// uses reports whether stmts contain a "use" of variable v.
	uses := func(pass *analysis.Pass, v *types.Var, stmts []ast.Node) bool {
		found := false
		for _, stmt := range stmts {
			ast.Inspect(stmt, func(n ast.Node) bool {
				switch n := n.(type) {
				case *ast.Ident:
					if pass.TypesInfo.Uses[n] == v {
						found = true
					}
				case *ast.ReturnStmt:
					// A naked return statement counts as a use
					// of the named result variables.
					if n.Results == nil && vIsNamedResult {
						found = true
					}
				}
				return !found
			})
		}
		return found
	}

	// blockUses computes "uses" for each block, caching the result.
	memo := make(map[*cfg.Block]bool)
	blockUses := func(pass *analysis.Pass, v *types.Var, b *cfg.Block) bool {
		res, ok := memo[b]
		if !ok {
			res = uses(pass, v, b.Nodes)
			memo[b] = res
		}
		return res
	}

	// Find the var's defining block in the CFG,
	// plus the rest of the statements of that block.
	var defblock *cfg.Block
	var rest []ast.Node
outer:
	for _, b := range g.Blocks {
		for i, n := range b.Nodes {
			if n == stmt {
				defblock = b
				rest = b.Nodes[i+1:]
				break outer
			}
		}
	}
	if defblock == nil {
		panic("internal error: can't find defining block for cancel var")
	}

	// Is v "used" in the remainder of its defining block?
	if uses(pass, v, rest) {
		return nil
	}

	// Does the defining block return without using v?
	if ret := defblock.Return(); ret != nil {
		return ret
	}

	// Search the CFG depth-first for a path, from defblock to a
	// return block, in which v is never "used".
	seen := make(map[*cfg.Block]bool)
	var search func(blocks []*cfg.Block) *ast.ReturnStmt
	search = func(blocks []*cfg.Block) *ast.ReturnStmt {
		for _, b := range blocks {
			if seen[b] {
				continue
			}
			seen[b] = true

			// Prune the search if the block uses v.
			if blockUses(pass, v, b) {
				continue
			}

			// Found path to return statement?
			if ret := b.Return(); ret != nil {
				if debug {
					fmt.Printf("found path to return in block %s\n", b)
				}
				return ret // found
			}

			// Recur
			if ret := search(b.Succs); ret != nil {
				if debug {
					fmt.Printf(" from block %s\n", b)
				}
				return ret
			}
		}
		return nil
	}
	return search(defblock.Succs)
}

func tupleContains(tuple *types.Tuple, v *types.Var) bool {
	for v0 := range tuple.Variables() {
		if v0 == v {
			return true
		}
	}
	return false
}
