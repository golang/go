// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inline

// This file defines the analysis of callee effects.

import (
	"go/ast"
	"go/token"
	"go/types"
)

const (
	rinf = -1 //  R∞: arbitrary read from memory
	winf = -2 //  W∞: arbitrary write to memory (or unknown control)
)

// calleefx returns a list of parameter indices indicating the order
// in which parameters are first referenced during evaluation of the
// callee, relative both to each other and to other effects of the
// callee (if any), such as arbitrary reads (rinf) and arbitrary
// effects (winf), including unknown control flow. Each parameter
// that is referenced appears once in the list.
//
// For example, the effects list of this function:
//
//	func f(x, y, z int) int {
//	    return y + x + g() + z
//	}
//
// is [1 0 -2 2], indicating reads of y and x, followed by the unknown
// effects of the g() call. and finally the read of parameter z. This
// information is used during inlining to ascertain when it is safe
// for parameter references to be replaced by their corresponding
// argument expressions. Such substitutions are permitted only when
// they do not cause "write" operations (those with effects) to
// commute with "read" operations (those that have no effect but are
// not pure). Impure operations may be reordered with other impure
// operations, and pure operations may be reordered arbitrarily.
//
// The analysis ignores the effects of runtime panics, on the
// assumption that well-behaved programs shouldn't encounter them.
func calleefx(info *types.Info, body *ast.BlockStmt, paramInfos map[*types.Var]*paramInfo) []int {
	// This traversal analyzes the callee's statements (in syntax
	// form, though one could do better with SSA) to compute the
	// sequence of events of the following kinds:
	//
	// 1  read of a parameter variable.
	// 2. reads from other memory.
	// 3. writes to memory

	var effects []int // indices of parameters, or rinf/winf (-ve)
	seen := make(map[int]bool)
	effect := func(i int) {
		if !seen[i] {
			seen[i] = true
			effects = append(effects, i)
		}
	}

	// unknown is called for statements of unknown effects (or control).
	unknown := func() {
		effect(winf)

		// Ensure that all remaining parameters are "seen"
		// after we go into the unknown (unless they are
		// unreferenced by the function body). This lets us
		// not bother implementing the complete traversal into
		// control structures.
		//
		// TODO(adonovan): add them in a deterministic order.
		// (This is not a bug but determinism is good.)
		for _, pinfo := range paramInfos {
			if !pinfo.IsResult && len(pinfo.Refs) > 0 {
				effect(pinfo.Index)
			}
		}
	}

	var visitExpr func(n ast.Expr)
	var visitStmt func(n ast.Stmt) bool
	visitExpr = func(n ast.Expr) {
		switch n := n.(type) {
		case *ast.Ident:
			if v, ok := info.Uses[n].(*types.Var); ok && !v.IsField() {
				// Use of global?
				if v.Parent() == v.Pkg().Scope() {
					effect(rinf) // read global var
				}

				// Use of parameter?
				if pinfo, ok := paramInfos[v]; ok && !pinfo.IsResult {
					effect(pinfo.Index) // read parameter var
				}

				// Use of local variables is ok.
			}

		case *ast.BasicLit:
			// no effect

		case *ast.FuncLit:
			// A func literal has no read or write effect
			// until called, and (most) function calls are
			// considered to have arbitrary effects.
			// So, no effect.

		case *ast.CompositeLit:
			for _, elt := range n.Elts {
				visitExpr(elt) // note: visits KeyValueExpr
			}

		case *ast.ParenExpr:
			visitExpr(n.X)

		case *ast.SelectorExpr:
			if sel, ok := info.Selections[n]; ok {
				visitExpr(n.X)
				if sel.Indirect() {
					effect(rinf) // indirect read x.f of heap variable
				}
			} else {
				// qualified identifier: treat like unqualified
				visitExpr(n.Sel)
			}

		case *ast.IndexExpr:
			if tv := info.Types[n.Index]; tv.IsType() {
				// no effect (G[T] instantiation)
			} else {
				visitExpr(n.X)
				visitExpr(n.Index)
				switch tv.Type.Underlying().(type) {
				case *types.Slice, *types.Pointer: // []T, *[n]T (not string, [n]T)
					effect(rinf) // indirect read of slice/array element
				}
			}

		case *ast.IndexListExpr:
			// no effect (M[K,V] instantiation)

		case *ast.SliceExpr:
			visitExpr(n.X)
			visitExpr(n.Low)
			visitExpr(n.High)
			visitExpr(n.Max)

		case *ast.TypeAssertExpr:
			visitExpr(n.X)

		case *ast.CallExpr:
			if info.Types[n.Fun].IsType() {
				// conversion T(x)
				visitExpr(n.Args[0])
			} else {
				// call f(args)
				visitExpr(n.Fun)
				for i, arg := range n.Args {
					if i == 0 && info.Types[arg].IsType() {
						continue // new(T), make(T, n)
					}
					visitExpr(arg)
				}

				// The pure built-ins have no effects beyond
				// those of their operands (not even memory reads).
				// All other calls have unknown effects.
				if !callsPureBuiltin(info, n) {
					unknown() // arbitrary effects
				}
			}

		case *ast.StarExpr:
			visitExpr(n.X)
			effect(rinf) // *ptr load or store depends on state of heap

		case *ast.UnaryExpr: // + - ! ^ & ~ <-
			visitExpr(n.X)
			if n.Op == token.ARROW {
				unknown() // effect: channel receive
			}

		case *ast.BinaryExpr:
			visitExpr(n.X)
			visitExpr(n.Y)

		case *ast.KeyValueExpr:
			visitExpr(n.Key) // may be a struct field
			visitExpr(n.Value)

		case *ast.BadExpr:
			// no effect

		case nil:
			// optional subtree

		default:
			// type syntax: unreachable given traversal
			panic(n)
		}
	}

	// visitStmt's result indicates the continuation:
	// false for return, true for the next statement.
	//
	// We could treat return as an unknown, but this way
	// yields definite effects for simple sequences like
	// {S1; S2; return}, so unreferenced parameters are
	// not spuriously added to the effects list, and thus
	// not spuriously disqualified from elimination.
	visitStmt = func(n ast.Stmt) bool {
		switch n := n.(type) {
		case *ast.DeclStmt:
			decl := n.Decl.(*ast.GenDecl)
			for _, spec := range decl.Specs {
				switch spec := spec.(type) {
				case *ast.ValueSpec:
					for _, v := range spec.Values {
						visitExpr(v)
					}

				case *ast.TypeSpec:
					// no effect
				}
			}

		case *ast.LabeledStmt:
			return visitStmt(n.Stmt)

		case *ast.ExprStmt:
			visitExpr(n.X)

		case *ast.SendStmt:
			visitExpr(n.Chan)
			visitExpr(n.Value)
			unknown() // effect: channel send

		case *ast.IncDecStmt:
			visitExpr(n.X)
			unknown() // effect: variable increment

		case *ast.AssignStmt:
			for _, lhs := range n.Lhs {
				visitExpr(lhs)
			}
			for _, rhs := range n.Rhs {
				visitExpr(rhs)
			}
			for _, lhs := range n.Lhs {
				id, _ := lhs.(*ast.Ident)
				if id != nil && id.Name == "_" {
					continue // blank assign has no effect
				}
				if n.Tok == token.DEFINE && id != nil && info.Defs[id] != nil {
					continue // new var declared by := has no effect
				}
				unknown() // assignment to existing var
				break
			}

		case *ast.GoStmt:
			visitExpr(n.Call.Fun)
			for _, arg := range n.Call.Args {
				visitExpr(arg)
			}
			unknown() // effect: create goroutine

		case *ast.DeferStmt:
			visitExpr(n.Call.Fun)
			for _, arg := range n.Call.Args {
				visitExpr(arg)
			}
			unknown() // effect: push defer

		case *ast.ReturnStmt:
			for _, res := range n.Results {
				visitExpr(res)
			}
			return false

		case *ast.BlockStmt:
			for _, stmt := range n.List {
				if !visitStmt(stmt) {
					return false
				}
			}

		case *ast.BranchStmt:
			unknown() // control flow

		case *ast.IfStmt:
			visitStmt(n.Init)
			visitExpr(n.Cond)
			unknown() // control flow

		case *ast.SwitchStmt:
			visitStmt(n.Init)
			visitExpr(n.Tag)
			unknown() // control flow

		case *ast.TypeSwitchStmt:
			visitStmt(n.Init)
			visitStmt(n.Assign)
			unknown() // control flow

		case *ast.SelectStmt:
			unknown() // control flow

		case *ast.ForStmt:
			visitStmt(n.Init)
			visitExpr(n.Cond)
			unknown() // control flow

		case *ast.RangeStmt:
			visitExpr(n.X)
			unknown() // control flow

		case *ast.EmptyStmt, *ast.BadStmt:
			// no effect

		case nil:
			// optional subtree

		default:
			panic(n)
		}
		return true
	}
	visitStmt(body)

	return effects
}
