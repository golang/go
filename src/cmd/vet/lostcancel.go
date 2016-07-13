// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmd/vet/internal/cfg"
	"fmt"
	"go/ast"
	"go/types"
	"strconv"
)

func init() {
	register("lostcancel",
		"check for failure to call cancelation function returned by context.WithCancel",
		checkLostCancel,
		funcDecl, funcLit)
}

const debugLostCancel = false

var contextPackage = "context"

// checkLostCancel reports a failure to the call the cancel function
// returned by context.WithCancel, either because the variable was
// assigned to the blank identifier, or because there exists a
// control-flow path from the call to a return statement and that path
// does not "use" the cancel function.  Any reference to the variable
// counts as a use, even within a nested function literal.
//
// checkLostCancel analyzes a single named or literal function.
func checkLostCancel(f *File, node ast.Node) {
	// Fast path: bypass check if file doesn't use context.WithCancel.
	if !hasImport(f.file, contextPackage) {
		return
	}

	// Maps each cancel variable to its defining ValueSpec/AssignStmt.
	cancelvars := make(map[*types.Var]ast.Node)

	// Find the set of cancel vars to analyze.
	stack := make([]ast.Node, 0, 32)
	ast.Inspect(node, func(n ast.Node) bool {
		switch n.(type) {
		case *ast.FuncLit:
			if len(stack) > 0 {
				return false // don't stray into nested functions
			}
		case nil:
			stack = stack[:len(stack)-1] // pop
			return true
		}
		stack = append(stack, n) // push

		// Look for [{AssignStmt,ValueSpec} CallExpr SelectorExpr]:
		//
		//   ctx, cancel    := context.WithCancel(...)
		//   ctx, cancel     = context.WithCancel(...)
		//   var ctx, cancel = context.WithCancel(...)
		//
		if isContextWithCancel(f, n) && isCall(stack[len(stack)-2]) {
			var id *ast.Ident // id of cancel var
			stmt := stack[len(stack)-3]
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
					f.Badf(id.Pos(), "the cancel function returned by context.%s should be called, not discarded, to avoid a context leak",
						n.(*ast.SelectorExpr).Sel.Name)
				} else if v, ok := f.pkg.uses[id].(*types.Var); ok {
					cancelvars[v] = stmt
				} else if v, ok := f.pkg.defs[id].(*types.Var); ok {
					cancelvars[v] = stmt
				}
			}
		}

		return true
	})

	if len(cancelvars) == 0 {
		return // no need to build CFG
	}

	// Tell the CFG builder which functions never return.
	info := &types.Info{Uses: f.pkg.uses, Selections: f.pkg.selectors}
	mayReturn := func(call *ast.CallExpr) bool {
		name := callName(info, call)
		return !noReturnFuncs[name]
	}

	// Build the CFG.
	var g *cfg.CFG
	var sig *types.Signature
	switch node := node.(type) {
	case *ast.FuncDecl:
		sig, _ = f.pkg.defs[node.Name].Type().(*types.Signature)
		g = cfg.New(node.Body, mayReturn)
	case *ast.FuncLit:
		sig, _ = f.pkg.types[node.Type].Type.(*types.Signature)
		g = cfg.New(node.Body, mayReturn)
	}

	// Print CFG.
	if debugLostCancel {
		fmt.Println(g.Format(f.fset))
	}

	// Examine the CFG for each variable in turn.
	// (It would be more efficient to analyze all cancelvars in a
	// single pass over the AST, but seldom is there more than one.)
	for v, stmt := range cancelvars {
		if ret := lostCancelPath(f, g, v, stmt, sig); ret != nil {
			lineno := f.fset.Position(stmt.Pos()).Line
			f.Badf(stmt.Pos(), "the %s function is not used on all paths (possible context leak)", v.Name())
			f.Badf(ret.Pos(), "this return statement may be reached without using the %s var defined on line %d", v.Name(), lineno)
		}
	}
}

func isCall(n ast.Node) bool { _, ok := n.(*ast.CallExpr); return ok }

func hasImport(f *ast.File, path string) bool {
	for _, imp := range f.Imports {
		v, _ := strconv.Unquote(imp.Path.Value)
		if v == path {
			return true
		}
	}
	return false
}

// isContextWithCancel reports whether n is one of the qualified identifiers
// context.With{Cancel,Timeout,Deadline}.
func isContextWithCancel(f *File, n ast.Node) bool {
	if sel, ok := n.(*ast.SelectorExpr); ok {
		switch sel.Sel.Name {
		case "WithCancel", "WithTimeout", "WithDeadline":
			if x, ok := sel.X.(*ast.Ident); ok {
				if pkgname, ok := f.pkg.uses[x].(*types.PkgName); ok {
					return pkgname.Imported().Path() == contextPackage
				}
				// Import failed, so we can't check package path.
				// Just check the local package name (heuristic).
				return x.Name == "context"
			}
		}
	}
	return false
}

// lostCancelPath finds a path through the CFG, from stmt (which defines
// the 'cancel' variable v) to a return statement, that doesn't "use" v.
// If it finds one, it returns the return statement (which may be synthetic).
// sig is the function's type, if known.
func lostCancelPath(f *File, g *cfg.CFG, v *types.Var, stmt ast.Node, sig *types.Signature) *ast.ReturnStmt {
	vIsNamedResult := sig != nil && tupleContains(sig.Results(), v)

	// uses reports whether stmts contain a "use" of variable v.
	uses := func(f *File, v *types.Var, stmts []ast.Node) bool {
		found := false
		for _, stmt := range stmts {
			ast.Inspect(stmt, func(n ast.Node) bool {
				switch n := n.(type) {
				case *ast.Ident:
					if f.pkg.uses[n] == v {
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
	blockUses := func(f *File, v *types.Var, b *cfg.Block) bool {
		res, ok := memo[b]
		if !ok {
			res = uses(f, v, b.Nodes)
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
	if uses(f, v, rest) {
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
			if !seen[b] {
				seen[b] = true

				// Prune the search if the block uses v.
				if blockUses(f, v, b) {
					continue
				}

				// Found path to return statement?
				if ret := b.Return(); ret != nil {
					if debugLostCancel {
						fmt.Printf("found path to return in block %s\n", b)
					}
					return ret // found
				}

				// Recur
				if ret := search(b.Succs); ret != nil {
					if debugLostCancel {
						fmt.Printf(" from block %s\n", b)
					}
					return ret
				}
			}
		}
		return nil
	}
	return search(defblock.Succs)
}

func tupleContains(tuple *types.Tuple, v *types.Var) bool {
	for i := 0; i < tuple.Len(); i++ {
		if tuple.At(i) == v {
			return true
		}
	}
	return false
}

var noReturnFuncs = map[string]bool{
	"(*testing.common).FailNow": true,
	"(*testing.common).Fatal":   true,
	"(*testing.common).Fatalf":  true,
	"(*testing.common).Skip":    true,
	"(*testing.common).SkipNow": true,
	"(*testing.common).Skipf":   true,
	"log.Fatal":                 true,
	"log.Fatalf":                true,
	"log.Fatalln":               true,
	"os.Exit":                   true,
	"panic":                     true,
	"runtime.Goexit":            true,
}

// callName returns the canonical name of the builtin, method, or
// function called by call, if known.
func callName(info *types.Info, call *ast.CallExpr) string {
	switch fun := call.Fun.(type) {
	case *ast.Ident:
		// builtin, e.g. "panic"
		if obj, ok := info.Uses[fun].(*types.Builtin); ok {
			return obj.Name()
		}
	case *ast.SelectorExpr:
		if sel, ok := info.Selections[fun]; ok && sel.Kind() == types.MethodVal {
			// method call, e.g. "(*testing.common).Fatal"
			meth := sel.Obj()
			return fmt.Sprintf("(%s).%s",
				meth.Type().(*types.Signature).Recv().Type(),
				meth.Name())
		}
		if obj, ok := info.Uses[fun.Sel]; ok {
			// qualified identifier, e.g. "os.Exit"
			return fmt.Sprintf("%s.%s",
				obj.Pkg().Path(),
				obj.Name())
		}
	}

	// function with no name, or defined in missing imported package
	return ""
}
