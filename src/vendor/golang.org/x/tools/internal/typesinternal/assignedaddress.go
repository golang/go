// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typesinternal

import (
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/go/ast/edge"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/typeparams"
)

// IsAssignedOrAddressTaken reports whether the expression cur denotes a
// variable and appears in a context that assigns it or that takes its address,
// potentially leading to indirect assignment.
//
// These examples cause IsAssignedOrAddressTaken on the identifier for x to
// return true:
//
//	x = 1
//	x++
//	x[i] = 1	 (assume x is an array)
//	x.a[i] = 1	 (assume x.a is a non-pointer struct field)
//	use(&x)
//	use(x[i:j])	 (assume x is an array; the slice aliases x)
//
// whereas these cause it to return false:
//
//	y = x
//	f(x)
//	use(x.a[i])
//	use(*x)
//
// The expression may itself be a compound, for example:
//
//	use(&(*ptr))  => IsAssignedOrAddressTaken("*ptr") = true
//	x.a[i] = 1    => IsAssignedOrAddressTaken("x.a")  = true
//	_ = x.a[i]    => IsAssignedOrAddressTaken("x.a")  = false
//
// A variable's declaration is not considered to be an assignment:
//
//	var x int     => IsAssignedOrAddressTaken(x) = false
//	x := 1        => IsAssignedOrAddressTaken(x) = false
//
// For some ill-formed code, such as a non-identifier on the LHS of :=, or when
// the type of x in x[i] or x[i:j] is unknown, we conservatively return true.
//
// TODO(adonovan): revisit the surprising behavior for declarations.
func IsAssignedOrAddressTaken(info *types.Info, cur inspector.Cursor) bool {
	// Unfortunately we can't simply use info.Types[e].Assignable()
	// as it is always true for a variable even when that variable is
	// used only as an r-value. So we must inspect enclosing syntax.
outer:
	// Ascend to the outermost expression that denotes the
	// same variable as the original cur, or part of it:
	//    x -> (x) | x.f | x[i]
	// where x.f does not go through a pointer and x[i] indexes an array.
	// Other forms (array slices x[i:j], method selections, pointer
	// indirections) are decided immediately within the loop.
	for cur = range cur.Enclosing() {
		switch cur.ParentEdgeKind() {
		case edge.ParenExpr_X:
			// If x is an lvalue, then (x) is an lvalue.
		case edge.SelectorExpr_X:
			// If x is an lvalue, then x.f is an lvalue iff
			// the selection does not traverse a pointer.
			sel := cur.Parent().Node().(*ast.SelectorExpr)
			if seln, ok := info.Selections[sel]; ok {
				// Note: there is a bug in Indirect() where it spuriously returns true
				// when both the selection receiver and parameter are pointers. However,
				// it's okay in this case because there is no address taken when a
				// pointer receiver method is called on a pointer type.
				if seln.Indirect() {
					return false
				}
				if seln.Kind() == types.MethodVal {
					sig := seln.Obj().Type().(*types.Signature)
					if is[*types.Pointer](sig.Recv().Type().Underlying()) {
						t := seln.Recv()
						// The receiver may be an embedded field, so we need
						// to get the inner-most type (right before the method
						// call in seln.Index())
						for field := range ImplicitFieldSelections(seln) {
							t = field.Type()
						}
						if !is[*types.Pointer](t.Underlying()) {
							return true // takes address of receiver
						}
					}
					return false
				}
			}
		case edge.IndexExpr_X, edge.SliceExpr_X:
			// x[i] and x[i:j] refer to x's own memory only if x is an
			// array; for slices, maps, strings, and pointers to arrays,
			// they go through a reference and do not involve x itself.
			// If x is a type parameter, it may be instantiated with an
			// array if any type in its type set is an array.
			t := info.TypeOf(cur.Node().(ast.Expr))
			if t == nil || t == types.Typ[types.Invalid] {
				return true // no type information (ill-typed code); be conservative
			}
			if !mayBeArray(t) {
				return false
			}
			if cur.ParentEdgeKind() == edge.SliceExpr_X {
				return true // x[i:j] on an array implicitly takes &x
			}
		default:
			break outer
		}
	}
	switch cur.ParentEdgeKind() {
	case edge.AssignStmt_Lhs:
		assign := cur.Parent().Node().(*ast.AssignStmt)
		if assign.Tok != token.DEFINE {
			return true // x = j or x += j
		}
		id, ok := cur.Node().(*ast.Ident)
		if !ok {
			return true // ill-formed, e.g. (x) := 1; be conservative
		}
		// Re-assigned identifiers are recorded in the Uses map.
		if _, ok := info.Uses[id]; ok {
			return true // reassignment of x (x, y := 1, 2)
		}
	case edge.RangeStmt_Key, edge.RangeStmt_Value:
		rng := cur.Parent().Node().(*ast.RangeStmt)
		if rng.Tok == token.ASSIGN {
			return true // "for k, v = range x" is like an AssignStmt to k, v
		}
	case edge.IncDecStmt_X:
		return true // x++, x--
	case edge.UnaryExpr_X:
		if cur.Parent().Node().(*ast.UnaryExpr).Op == token.AND {
			return true // &x
		}
	}
	return false
}

// mayBeArray reports whether t is an array type, or a type parameter
// (or other interface) whose type set includes an array type.
//
// It conservatively returns true if the type set has no specific
// types, since such an operand cannot be indexed or sliced in
// well-typed code.
func mayBeArray(t types.Type) bool {
	if !types.IsInterface(t) { // common case: not a type parameter
		return is[*types.Array](t.Underlying())
	}
	tset := typeparams.TypeSetOf(t)
	if tset.Unrestricted() {
		return true // ill-typed code; be conservative
	}
	for u := range tset.UnderlyingTypes() {
		if is[*types.Array](u) {
			return true
		}
	}
	return false
}

func is[T any](x any) bool {
	_, ok := x.(T)
	return ok
}
