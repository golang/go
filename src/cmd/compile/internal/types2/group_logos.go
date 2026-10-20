// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

// LoGo/s M8-S2 type rules for `go group` and `go { }` children (SPEC
// Sec 2.6, T-206). stmt.go and labels.go call in through hook lines
// only; the lowering lives in logos/group.go.
//
// One frame stack tracks lexical group nesting per function. Each frame
// carries the signature its body checks under: group bodies keep the
// enclosing signature, child bodies check under a synthesized `func()
// error`. A frame belongs to the current function only while its active
// signature matches, so eager lambda bodies (which install their own
// signature) and deferred literal bodies (checked after the stack pops)
// never inherit group membership.

// logosGroupFrame is one group-body or child-body scope: asig is the
// signature the body checks under, child tells a child body from a
// group body, and csig is the child signature for return checks.
type logosGroupFrame struct {
	asig  *Signature
	child bool
	csig  *Signature
}

const logosGroupReturnInChild = "[correctness::return_in_group_child] return cannot leave a group child block"
const logosGroupChildOutside = "[correctness::child_outside_group] go child block outside a go group body"
const logosGroupUnsupervised = "[concurrency::unsupervised_goroutine] go statement outside a go group body"
const logosGroupNeedsError = "[correctness::group_needs_error_result] go group requires the enclosing function to return error"

// logosGroupTop reports the innermost group frame of the current
// function: the top frame whose active signature matches. Frames of
// enclosing functions never match, which keeps eager lambda bodies out.
func (check *Checker) logosGroupTop() (logosGroupFrame, bool) {
	if n := len(check.logosGroup); n > 0 {
		if top := check.logosGroup[n-1]; top.asig == check.sig {
			return top, true
		}
	}
	return logosGroupFrame{}, false
}

// logosGroupStmt checks a group statement: the context must be
// context.Context, the limit an integer, and the enclosing function
// must return error last. The body checks inline (no new scope, so the
// lowering can splice it in place) under a group frame, keeping the
// caller's branch context for breaks and continues.
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
	check.stmtList(ctxt, s.Body.List)
	check.logosGroup = check.logosGroup[:len(check.logosGroup)-1]
}

// logosGroupContext requires the group context to name context.Context.
func (check *Checker) logosGroupContext(s *syntax.GroupStmt) {
	var x operand
	check.expr(nil, &x, s.Ctx)
	if !x.isValid() {
		return
	}
	if named, ok := Unalias(x.typ()).(*Named); ok {
		if obj := named.Obj(); obj != nil && obj.Pkg() != nil &&
			obj.Pkg().Path() == "context" && obj.Name() == "Context" {
			return
		}
	}
	check.errorf(s.Ctx, MismatchedTypes, "go group context must be context.Context, have %s", x.typ())
}

// logosGroupLimit requires the limit to be an integer. A missing limit
// leaves concurrency unbounded.
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

// logosGroupChild checks a child block: outside a group body of the
// current function it fails, otherwise the body checks as a `func()
// error` closure (fresh scope, no branch targets outside, labels
// resolved inside), so `?` returns to the group and `return` is picketed
// by logosGroupReturn.
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
	check.logosEnterFunc(s.Body)
	check.stmtList(0, s.Body.List)
	check.logosExitFunc()
	if check.hasLabel && !check.conf.IgnoreBranchErrors {
		check.labels(s.Body)
	}
	check.usage(scope)
	check.environment = saved
	check.logosGroup = check.logosGroup[:len(check.logosGroup)-1]
}

// logosGroupGo checks a `go` call statement in LogosMode: inside a group
// body of the current function it is a child (the call checks through
// the shared suspended-call rule, and error-last results flow to the
// group at lowering), anywhere else it is an unsupervised goroutine. It
// reports true when it handled the statement.
func (check *Checker) logosGroupGo(s *syntax.CallStmt) bool {
	if !check.conf.Logos || s.Tok != syntax.Go {
		return false
	}
	if _, ok := check.logosGroupTop(); !ok {
		check.error(s, InvalidGo, logosGroupUnsupervised)
		check.use(s.Call)
		return true
	}
	check.suspendedCall("go", s.Call)
	return true
}

// logosGroupReturn rejects a return that would leave a child block: the
// innermost frame of the current function is a child. It reports true
// when it handled the statement, marking the results used so no
// follow-on errors surface.
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

// LogosErrorResults splits the result type t of a checked call into its
// non-error value count and whether an error follows: a tuple ending in
// error, or a lone error. Anything else carries no error for the group.
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

// logosGroupBranches checks the labels of a group body inline: the body
// runs in the enclosing scope at lowering, so its labels unify with the
// function like a plain block. Child bodies keep their own label pass in
// logosGroupChild and never reach here.
func (check *Checker) logosGroupBranches(all *Scope, parent *block, lstmt *syntax.LabeledStmt, s *syntax.GroupStmt) []*syntax.BranchStmt {
	return check.blockBranches(all, parent, lstmt, s.Body.List)
}
