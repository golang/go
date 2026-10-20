// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// deferErasesErrdeferState is correctness::defer_erases_errdefer_state (SPEC
// Sec 1.2, #77): in a function with errdefer, a plain deferred function
// literal assigns nil to the error result. The deferred literal runs before
// the errdefer cleanup, clears the result, and the errdefer sees a success
// and skips its rollback. The rule reads the function that owns the defer: an
// errdefer statement and a deferred literal count for the function that
// contains them directly, never for an enclosing or nested function literal,
// and the assigned variable must be the last, named result of that function,
// matched by object so a shadowing variable does not count. A plain deferred
// literal is `defer func() {...}()`: defer f()?, scope defer, and a deferred
// call of a named function are not. An assignment inside a function literal
// nested in the deferred literal is not searched, because that literal may
// never run. The order of the defer and the errdefer does not matter. The
// rule is forbid and applies to loGo/s packages only.
var deferErasesErrdeferState = Rule{
	ID:    "correctness::defer_erases_errdefer_state",
	Group: Correctness,
	Check: checkDeferErases,
}

// checkDeferErases is the Check of deferErasesErrdeferState.
func checkDeferErases(pass *Pass) {
	for _, f := range pass.Files {
		syntax.Inspect(f, func(n syntax.Node) bool {
			switch n := n.(type) {
			case *syntax.FuncDecl:
				if n.Body != nil {
					checkErasures(pass, n.Type, n.Body)
				}
			case *syntax.FuncLit:
				checkErasures(pass, n.Type, n.Body)
			}
			return true
		})
	}
}

// checkErasures reports the plain deferred literals of one function body that
// assign nil to its error result.
func checkErasures(pass *Pass, sig *syntax.FuncType, body *syntax.BlockStmt) {
	result := errorResult(pass, sig)
	if result == nil || !hasErrdefer(body) {
		return
	}
	inspectOwn(body, func(n syntax.Node) {
		if lit := plainDeferredLiteral(n); lit != nil {
			reportErasures(pass, result, lit)
		}
	})
}

// errorResult returns the object of the last result of sig when it is named
// and of type error, or nil.
func errorResult(pass *Pass, sig *syntax.FuncType) types2.Object {
	if len(sig.ResultList) == 0 {
		return nil
	}
	name := sig.ResultList[len(sig.ResultList)-1].Name
	if name == nil || name.Value == "_" {
		return nil
	}
	obj := pass.Info.Defs[name]
	if obj == nil || !isErrorType(obj.Type()) {
		return nil
	}
	return obj
}

// hasErrdefer reports whether body holds an errdefer statement of its own.
func hasErrdefer(body *syntax.BlockStmt) bool {
	found := false
	inspectOwn(body, func(n syntax.Node) {
		_, ok := n.(*syntax.ErrdeferStmt)
		found = found || ok
	})
	return found
}

// inspectOwn calls visit for every node under root that belongs to the
// function of root: it does not enter a function literal.
func inspectOwn(root syntax.Node, visit func(syntax.Node)) {
	syntax.Inspect(root, func(n syntax.Node) bool {
		if n == nil {
			return false
		}
		if _, ok := n.(*syntax.FuncLit); ok {
			return false
		}
		visit(n)
		return true
	})
}

// plainDeferredLiteral returns the function literal of a statement of the
// form defer func() {...}(), or nil for any other node.
func plainDeferredLiteral(n syntax.Node) *syntax.FuncLit {
	stmt, ok := n.(*syntax.CallStmt)
	if !ok || stmt.Tok != syntax.Defer {
		return nil
	}
	call, ok := stmt.Call.(*syntax.CallExpr)
	if !ok {
		return nil
	}
	lit, _ := syntax.Unparen(call.Fun).(*syntax.FuncLit)
	return lit
}

// reportErasures reports each assignment of nil to result in the body of lit,
// outside any function literal nested in it.
func reportErasures(pass *Pass, result types2.Object, lit *syntax.FuncLit) {
	inspectOwn(lit.Body, func(n syntax.Node) {
		as, ok := n.(*syntax.AssignStmt)
		if !ok || as.Op != 0 || as.Rhs == nil || !assignsNilTo(pass, result, as) {
			return
		}
		pass.Reportf(as, "deferred function literal assigns nil to result %s, erasing the error state that errdefer reads", result.Name())
	})
}

// assignsNilTo reports whether the assignment as stores the predeclared nil
// into result.
func assignsNilTo(pass *Pass, result types2.Object, as *syntax.AssignStmt) bool {
	lhs, rhs := exprList(as.Lhs), exprList(as.Rhs)
	if len(lhs) != len(rhs) {
		return false
	}
	for i := range lhs {
		if denotes(pass, lhs[i], result) && isNilLiteral(pass, rhs[i]) {
			return true
		}
	}
	return false
}

// denotes reports whether e, looking through parentheses, is an identifier
// that refers to obj.
func denotes(pass *Pass, e syntax.Expr, obj types2.Object) bool {
	n, ok := syntax.Unparen(e).(*syntax.Name)
	return ok && pass.Info.Uses[n] == obj
}
