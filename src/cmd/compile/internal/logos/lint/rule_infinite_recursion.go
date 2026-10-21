// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"slices"
	"strings"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// infiniteRecursion is correctness::infinite_recursion (SPEC Sec 1.2,
// #157): a function whose every path from entry reaches a call to itself,
// with arguments identical to its parameters, before any return. The rule is
// forbid in a loGo/s package, where nothing can silence it, so it fails
// safe: a path that can leave the function another way (a return, a branch,
// a panic, a process exit, a loGo/s ? or yield) or never reach the call (an
// endless loop) clears the function, and so does any assignment to or
// address-of a parameter, because the arguments then carry other values than
// the parameters had at entry. Arguments match by object, so a shadowing
// variable of the same name does not count. The analysis reads the syntax
// tree and the checker's objects only, so it holds for Go and loGo/s
// packages alike.
var infiniteRecursion = Rule{
	ID:    "correctness::infinite_recursion",
	Group: Correctness,
	Go:    true,
	Check: checkInfiniteRecursion,
}

// recursion is the analysis of one function declaration.
type recursion struct {
	pass     *Pass
	fn       *types2.Func
	recv     types2.Object
	params   []types2.Object
	variadic bool
}

// checkInfiniteRecursion is the Check of infiniteRecursion.
func checkInfiniteRecursion(pass *Pass) {
	for _, f := range pass.Files {
		for _, d := range f.DeclList {
			fd, ok := d.(*syntax.FuncDecl)
			if !ok || fd.Body == nil {
				continue
			}
			if r := newRecursion(pass, fd); r != nil && r.always(fd.Body.List) {
				pass.Reportf(fd.Name, "function %s calls itself with its own arguments on every path", fd.Name.Value)
			}
		}
	}
}

// newRecursion prepares the analysis of fd. It returns nil when no call can
// pass the entry values of the parameters: a receiver or parameter is
// unnamed or blank, or the body assigns one.
func newRecursion(pass *Pass, fd *syntax.FuncDecl) *recursion {
	fn, _ := pass.Info.Defs[fd.Name].(*types2.Func)
	if fn == nil {
		return nil
	}
	r := &recursion{pass: pass, fn: fn}
	if fd.Recv != nil {
		if r.recv = r.definedBy(fd.Recv); r.recv == nil {
			return nil
		}
	}
	for _, p := range fd.Type.ParamList {
		obj := r.definedBy(p)
		if obj == nil {
			return nil
		}
		r.params = append(r.params, obj)
		_, r.variadic = p.Type.(*syntax.DotsType)
	}
	if r.assignsParam(fd.Body) {
		return nil
	}
	return r
}

// definedBy returns the object a named receiver or parameter declares, or
// nil for an unnamed or blank one.
func (r *recursion) definedBy(f *syntax.Field) types2.Object {
	if f.Name == nil || f.Name.Value == "_" {
		return nil
	}
	return r.pass.Info.Defs[f.Name]
}

// assignsParam reports whether body assigns, increments, ranges into, or
// takes the address of the receiver or a parameter.
func (r *recursion) assignsParam(body *syntax.BlockStmt) bool {
	found := false
	syntax.Inspect(body, func(n syntax.Node) bool {
		found = found || r.mutates(n)
		return !found
	})
	return found
}

// mutates reports whether n assigns, ranges into, or takes the address of
// an expression rooted at the receiver or a parameter.
func (r *recursion) mutates(n syntax.Node) bool {
	switch n := n.(type) {
	case *syntax.AssignStmt:
		return r.rooted(n.Lhs)
	case *syntax.RangeClause:
		return n.Lhs != nil && r.rooted(n.Lhs)
	case *syntax.Operation:
		return n.Op == syntax.And && n.Y == nil && r.rooted(n.X)
	}
	return false
}

// rooted reports whether e, or an element of the tuple e, is selected or
// indexed from the receiver or a parameter.
func (r *recursion) rooted(e syntax.Expr) bool {
	for _, x := range exprList(e) {
		name := rootName(x)
		if name == nil {
			continue
		}
		obj := r.pass.Info.Uses[name]
		if obj != nil && (obj == r.recv || slices.Contains(r.params, obj)) {
			return true
		}
	}
	return false
}

// rootName returns the identifier an expression is selected, indexed, or
// dereferenced from, or nil.
func rootName(e syntax.Expr) *syntax.Name {
	for {
		switch x := e.(type) {
		case *syntax.Name:
			return x
		case *syntax.SelectorExpr:
			e = x.X
		case *syntax.IndexExpr:
			e = x.X
		case *syntax.ParenExpr:
			e = x.X
		case *syntax.Operation:
			if x.Y != nil {
				return nil
			}
			e = x.X
		default:
			return nil
		}
	}
}

// always reports whether every path through list reaches a self-call
// before leaving the function. A branch whose leaving paths all call
// first covers those paths, and the paths that fall through continue to
// the statements below.
func (r *recursion) always(list []syntax.Stmt) bool {
	for _, s := range list {
		if r.stmtCalls(s) {
			return true
		}
		if r.mayLeave(s) && !r.exitsCovered(s) {
			return false
		}
	}
	return false
}

// stmtCalls reports whether every path through s reaches a self-call.
func (r *recursion) stmtCalls(s syntax.Stmt) bool {
	switch s := s.(type) {
	case *syntax.ExprStmt:
		return r.exprCalls(s.X)
	case *syntax.ReturnStmt:
		return s.Results != nil && r.exprCalls(s.Results)
	case *syntax.AssignStmt:
		return s.Rhs != nil && r.exprCalls(s.Rhs)
	case *syntax.BlockStmt:
		return r.always(s.List)
	case *syntax.IfStmt:
		return r.ifCalls(s)
	case *syntax.SwitchStmt:
		return r.switchCalls(s)
	case *syntax.ConstSwitchStmt:
		return r.constSwitchCalls(s)
	}
	return false
}

// ifCalls reports whether an if statement reaches a self-call on every
// path: the condition calls, or both branches do.
func (r *recursion) ifCalls(s *syntax.IfStmt) bool {
	if s.Init != nil {
		return false
	}
	if r.exprCalls(s.Cond) {
		return true
	}
	if s.Else == nil || !r.always(s.Then.List) {
		return false
	}
	return r.stmtCalls(s.Else)
}

// switchCalls reports whether a switch statement reaches a self-call on
// every path: the tag calls, or a default arm exists and every arm does.
func (r *recursion) switchCalls(s *syntax.SwitchStmt) bool {
	if s.Init != nil {
		return false
	}
	if s.Tag != nil && r.exprCalls(s.Tag) {
		return true
	}
	var def bool
	for _, c := range s.Body {
		if c.Cases == nil {
			def = true
		}
		if !r.always(c.Body) {
			return false
		}
	}
	return def
}

// constSwitchCalls reports whether a `const switch` statement reaches a
// self-call on every path: a concrete tag folds to its first matching case
// at compile time, so only the selected body runs, while a type-parameter
// tag dispatches at run time and the arm it does not take calls nothing. A
// concrete tag no case matches folds the statement away, so control falls
// through to the next statement. Selection is the shared checker operation
// (types2 LogosConstSelectedCase), so lint and lowering agree on every case.
func (r *recursion) constSwitchCalls(s *syntax.ConstSwitchStmt) bool {
	sel, static := types2.LogosConstSelectedCase(r.pass.Info, s)
	return static && sel != nil && r.always(sel.Body)
}

// exitsCovered reports whether every path through s that leaves the
// enclosing statement list reaches a self-call first: a taken arm that
// self-calls before returning covers its path, while a path that falls
// through continues to the statements below. Any other statement covers
// no leaving path.
func (r *recursion) exitsCovered(s syntax.Stmt) bool {
	switch s := s.(type) {
	case *syntax.IfStmt:
		if s.Init != nil {
			return false
		}
		if s.Else == nil {
			return r.branchCovered(s.Then.List)
		}
		return r.branchCovered(s.Then.List) && r.branchCovered([]syntax.Stmt{s.Else})
	case *syntax.SwitchStmt:
		if s.Init != nil {
			return false
		}
		for _, c := range s.Body {
			if !r.branchCovered(c.Body) {
				return false
			}
		}
		return true
	case *syntax.ConstSwitchStmt:
		return r.constSwitchCovered(s)
	}
	return false
}

// constSwitchCovered reports whether every path through s that leaves the
// enclosing statement list reaches a self-call first. Only the selected
// body can leave: a dead branch never runs, so its returns must not cover
// or clear the function. A type-parameter tag dispatches at run time, so
// every arm's leaving paths must call first, as for a Go switch. A
// statement no case matches folds away, so no path leaves it.
func (r *recursion) constSwitchCovered(s *syntax.ConstSwitchStmt) bool {
	sel, static := types2.LogosConstSelectedCase(r.pass.Info, s)
	if !static {
		for _, c := range s.Cases {
			if !r.branchCovered(c.Body) {
				return false
			}
		}
		return true
	}
	if sel == nil {
		return true
	}
	return r.branchCovered(sel.Body)
}

// branchCovered reports whether every path through list either reaches a
// self-call or falls out of list: a path that leaves the enclosing list
// from inside list must call first, while a path that falls through
// continues to the statements below.
func (r *recursion) branchCovered(list []syntax.Stmt) bool {
	for _, s := range list {
		if r.stmtCalls(s) {
			return true
		}
		if r.mayLeave(s) && !r.exitsCovered(s) {
			return false
		}
	}
	return true
}

// exprCalls reports whether evaluating e always reaches a self-call. The
// right operand of && and || is conditional and is not searched.
func (r *recursion) exprCalls(e syntax.Expr) bool {
	switch e := e.(type) {
	case *syntax.ParenExpr:
		return r.exprCalls(e.X)
	case *syntax.ListExpr:
		return r.anyCalls(e.ElemList)
	case *syntax.CallExpr:
		return r.isSelfCall(e) || r.anyCalls(e.ArgList)
	case *syntax.Operation:
		return r.operationCalls(e)
	}
	return false
}

// operationCalls reports whether evaluating the operation e always reaches
// a self-call: through its left operand, or through its right operand when
// that operand is not conditional.
func (r *recursion) operationCalls(e *syntax.Operation) bool {
	if r.exprCalls(e.X) {
		return true
	}
	conditional := e.Op == syntax.AndAnd || e.Op == syntax.OrOr
	return e.Y != nil && !conditional && r.exprCalls(e.Y)
}

// anyCalls reports whether any of list always reaches a self-call.
func (r *recursion) anyCalls(list []syntax.Expr) bool {
	for _, e := range list {
		if r.exprCalls(e) {
			return true
		}
	}
	return false
}

// isSelfCall reports whether call invokes the function under analysis with
// exactly its parameters, in order.
func (r *recursion) isSelfCall(call *syntax.CallExpr) bool {
	if len(call.ArgList) != len(r.params) || call.HasDots != r.variadic || !r.callsFn(call.Fun) {
		return false
	}
	for i, a := range call.ArgList {
		if n, ok := a.(*syntax.Name); !ok || r.pass.Info.Uses[n] != r.params[i] {
			return false
		}
	}
	return true
}

// callsFn reports whether fun names the function under analysis: a plain
// identifier, the receiver with the method name, or an explicit
// instantiation of either with its own type parameters in order.
func (r *recursion) callsFn(fun syntax.Expr) bool {
	switch f := fun.(type) {
	case *syntax.ParenExpr:
		return r.callsFn(f.X)
	case *syntax.Name:
		return r.recv == nil && r.pass.Info.Uses[f] == r.fn
	case *syntax.SelectorExpr:
		x, ok := f.X.(*syntax.Name)
		return ok && r.recv != nil && r.pass.Info.Uses[x] == r.recv && r.pass.Info.Uses[f.Sel] == r.fn
	case *syntax.IndexExpr:
		return r.callsGeneric(f)
	}
	return false
}

// callsGeneric reports whether idx instantiates the function under
// analysis with its own type parameters (f[T], (f[T]), f[T, U]): the base
// must name it as in callsFn, and the recorded type arguments must be
// identical to its type parameters, position by position.
func (r *recursion) callsGeneric(idx *syntax.IndexExpr) bool {
	sig, ok := r.fn.Type().(*types2.Signature)
	if !ok {
		return false
	}
	tparams := sig.TypeParams()
	if tparams == nil || tparams.Len() == 0 {
		return false
	}
	var id *syntax.Name
	switch b := syntax.Unparen(idx.X).(type) {
	case *syntax.Name:
		if r.recv != nil || r.pass.Info.Uses[b] != r.fn {
			return false
		}
		id = b
	case *syntax.SelectorExpr:
		x, ok := b.X.(*syntax.Name)
		if !ok || r.recv == nil || r.pass.Info.Uses[x] != r.recv || !r.methodIsFn(b) {
			return false
		}
		id = b.Sel
	default:
		return false
	}
	inst, ok := r.pass.Info.Instances[id]
	if !ok || inst.TypeArgs == nil || inst.TypeArgs.Len() != tparams.Len() {
		return false
	}
	for i := range tparams.Len() {
		if !types2.Identical(inst.TypeArgs.At(i), tparams.At(i)) {
			return false
		}
	}
	return true
}

// methodIsFn reports whether sel selects the function under analysis: the
// checker's recorded selection when present, otherwise the denoted object.
// Methods on generic receivers resolve to expanded clones, so identity is
// by origin, not pointer.
func (r *recursion) methodIsFn(sel *syntax.SelectorExpr) bool {
	if s, ok := r.pass.Info.Selections[sel]; ok && s != nil {
		fn, ok := s.Obj().(*types2.Func)
		return ok && fn.Origin() == r.fn.Origin()
	}
	fn, ok := r.pass.Info.Uses[sel.Sel].(*types2.Func)
	return ok && fn.Origin() == r.fn.Origin()
}

// mayLeave reports whether s contains a node after which control may not
// reach the next statement. Function literals and lambdas are separate
// functions and are not searched.
func (r *recursion) mayLeave(s syntax.Stmt) bool {
	found := false
	syntax.Inspect(s, func(n syntax.Node) bool {
		found = found || r.leavesSequence(n)
		return !found && !isFunctionBody(n)
	})
	return found
}

// isFunctionBody reports whether n starts a function of its own.
func isFunctionBody(n syntax.Node) bool {
	switch n.(type) {
	case *syntax.FuncLit, *syntax.LambdaExpr:
		return true
	}
	return false
}

// leavesSequence reports whether n itself is a return, a branch, a loGo/s
// early exit (? and yield) or concurrency block, a call that panics or ends
// the process, or a for statement with no condition and no range.
func (r *recursion) leavesSequence(n syntax.Node) bool {
	switch n := n.(type) {
	case *syntax.ReturnStmt, *syntax.BranchStmt, *syntax.TryExpr, *syntax.YieldStmt, *syntax.GroupStmt, *syntax.GroupChild:
		return true
	case *syntax.CallExpr:
		return r.endsProcess(n)
	case *syntax.ForStmt:
		_, ranged := n.Init.(*syntax.RangeClause)
		return n.Cond == nil && !ranged
	}
	return false
}

// terminators are the prefixes of the full names of functions that do not
// return.
var terminators = []string{"os.Exit", "syscall.Exit", "runtime.Goexit", "log.Fatal", "log.Panic", "(*log.Logger).Fatal", "(*log.Logger).Panic"}

// endsProcess reports whether call is a call of panic or of a function that
// never returns normally.
func (r *recursion) endsProcess(call *syntax.CallExpr) bool {
	if name, ok := call.Fun.(*syntax.Name); ok && name.Value == "panic" {
		return true
	}
	fn := calleeFunc(r.pass, call)
	if fn == nil {
		return false
	}
	full := fn.FullName()
	return slices.ContainsFunc(terminators, func(t string) bool { return strings.HasPrefix(full, t) })
}
