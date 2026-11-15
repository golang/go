// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

type logosGroupFrame struct {
	asig  *Signature
	child bool
	csig  *Signature
}

const logosGroupReturnInChild = "[correctness::return_in_group_child] return cannot leave a group child block"
const logosGroupChildOutside = "[correctness::child_outside_group] go child block outside a go group body"
const logosGroupUnsupervised = "[concurrency::unsupervised_goroutine] go statement outside a go group body"
const logosGroupNeedsError = "[correctness::group_needs_error_result] go group requires the enclosing function to return error"

func (check *Checker) logosGroupTop() (logosGroupFrame, bool) {
	if n := len(check.logosGroup); n > 0 {
		if top := check.logosGroup[n-1]; top.asig == check.sig {
			return top, true
		}
	}
	return logosGroupFrame{}, false
}

func (check *Checker) logosGroupStmt(ctxt stmtContext, s *syntax.GroupStmt) {
	if !check.conf.Logos {
		check.error(s, InvalidSyntaxTree, "go group requires logos mode")
		return
	}
	check.logosGroupContext(s)
	check.logosGroupLimit(s)
	if !check.logosEnclosingError() {
		check.error(s.Ctx, WrongResultCount, logosGroupNeedsError)
	}
	check.logosGroup = append(check.logosGroup, logosGroupFrame{asig: check.sig})
	check.openScope(s.Body, "group")
	check.stmtList(ctxt, s.Body.List)
	check.closeScope()
	check.logosGroup = check.logosGroup[:len(check.logosGroup)-1]
}

func (check *Checker) logosGroupContext(s *syntax.GroupStmt) {
	check.logosContextArg(s.Ctx, "go group context")
}

func (check *Checker) logosContextArg(e syntax.Expr, what string) bool {
	var x operand
	check.expr(nil, &x, e)
	if !x.isValid() {
		return false
	}
	if named, ok := Unalias(x.typ()).(*Named); ok {
		if obj := named.Obj(); obj != nil && obj.Pkg() != nil &&
			obj.Pkg().Path() == "context" && obj.Name() == "Context" {
			return true
		}
	}
	check.errorf(e, MismatchedTypes, "%s must be context.Context, have %s", what, x.typ())
	return false
}

func (check *Checker) logosGroupLimit(s *syntax.GroupStmt) {
	if s.Limit == nil {
		return
	}
	var x operand
	check.expr(nil, &x, s.Limit)
	if !x.isValid() {
		return
	}
	if !isInteger(x.typ()) {
		check.errorf(s.Limit, MismatchedTypes, "go group limit must be an integer, have %s", x.typ())
	}
}

func (check *Checker) logosGroupChild(s *syntax.GroupChild) {
	if !check.conf.Logos {
		check.error(s, InvalidSyntaxTree, "go child block requires logos mode")
		return
	}
	if _, ok := check.logosGroupTop(); !ok {
		check.error(s, InvalidGo, logosGroupChildOutside)
	}
	scope := NewScope(check.scope, s.Pos(), endPos(s.Body), "group child")
	scope.isFunc = true
	results := NewTuple(newVar(ResultVar, s.Pos(), check.pkg, "", universeError))
	csig := NewSignatureType(nil, nil, nil, NewTuple(), results, false)
	csig.scope = scope
	check.logosGroup = append(check.logosGroup, logosGroupFrame{asig: csig, child: true, csig: csig})
	saved := check.environment
	check.environment = environment{
		decl:    check.decl,
		scope:   scope,
		version: check.version,
		iota:    check.iota,
		sig:     csig,
	}
	check.logosEnterLit(s.Body)
	check.stmtList(0, s.Body.List)
	check.logosExitFunc()
	if check.hasLabel && !check.conf.IgnoreBranchErrors {
		check.labels(s.Body)
	}
	check.usage(scope)
	check.environment = saved
	check.logosGroup = check.logosGroup[:len(check.logosGroup)-1]
}

func (check *Checker) logosGroupGo(s *syntax.CallStmt) bool {
	if !check.conf.Logos || s.Tok != syntax.Go {
		return false
	}
	if _, ok := check.logosGroupTop(); !ok {
		check.error(s, InvalidGo, logosGroupUnsupervised)
		check.use(s.Call)
		check.logosRegionSuspended("go", s.Call)
		return true
	}
	check.suspendedCall("go", s.Call)
	return true
}

func (check *Checker) logosGroupReturn(s *syntax.ReturnStmt) bool {
	if !check.conf.Logos {
		return false
	}
	top, ok := check.logosGroupTop()
	if !ok || !top.child || top.csig != check.sig {
		return false
	}
	check.error(s, InvalidSyntaxTree, logosGroupReturnInChild)
	for _, r := range syntax.UnpackListExpr(s.Results) {
		check.use(r)
	}
	return true
}

func LogosErrorResults(t Type) (int, bool) {
	if tup, ok := t.(*Tuple); ok {
		if n := tup.Len(); n > 0 && Identical(tup.At(n-1).Type(), universeError) {
			return n - 1, true
		}
		return 0, false
	}
	if t != nil && Identical(t, universeError) {
		return 0, true
	}
	return 0, false
}

func (check *Checker) logosGroupBranches(all *Scope, parent *block, lstmt *syntax.LabeledStmt, s *syntax.GroupStmt) []*syntax.BranchStmt {
	return check.blockBranches(all, parent, lstmt, s.Body.List)
}
