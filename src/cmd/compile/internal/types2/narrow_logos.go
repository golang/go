// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
	"maps"
	"slices"
)

// LoGo/s M3-S4 nil narrowing and the error-first guard (SPEC Sec 2.2,
// T-202). Upstream files call in through hook lines only: stmt.go
// (funcBody pre-pass, stmtList early-exit stack, IfStmt branch scopes),
// typexpr.go (ident resolution), expr.go (`&&` chains), assignments.go
// (LHS suppression, kill on reassign, tuple tracking), and
// option_logos.go (`?.` and `??` on narrowed receivers). Every rule below
// applies only to a package of .lgo files (Config.Logos), so plain Go
// keeps its types exactly as upstream. All state lives on the Checker,
// alongside logosConforms and logosPropSel.
//
// An `if x != nil` test narrows an option-typed local or parameter x to
// its element inside the guarded block, through `&&` chains, and after
// an early exit on `x == nil` (return, panic, break, or continue). A
// later assignment to x ends narrowing. Narrowing never applies to
// fields (only plain names narrow), to variables whose address is taken,
// or to variables assigned inside a closure or a loop body; a syntactic
// pre-pass per function body collects those disqualified names.
//
// The error-first guard tracks `n, err := f()` for callees declared in
// .lgo whose second result is error: reading n before the `err == nil`
// proof fails with result_before_error_check. The proof is the
// fall-through of `if err != nil { exit }` or the inside of
// `if err == nil {}`. Callees from other packages (the standard library
// and every other Go package) keep their own contracts.

const logosResultBeforeErr = "[correctness::result_before_error_check]"

// logosEnterFunc runs the M3-S4 pre-pass for one function body: it walks
// body for address-taken names and for names assigned inside closures or
// loop bodies, pushes the disqualified set, and ensures the per-function
// maps exist. Hooked from funcBody; logosExitFunc pops the set.
func (check *Checker) logosEnterFunc(body *syntax.BlockStmt) {
	if !check.conf.Logos {
		return
	}
	dq := &logosDisqualWalk{
		addr:   make(map[string]bool),
		inLoop: make(map[string]bool),
		inClos: make(map[string]bool),
	}
	if body != nil {
		syntax.Walk(body, dq)
	}
	set := make(map[string]bool, len(dq.addr)+len(dq.inLoop)+len(dq.inClos))
	for name := range dq.addr {
		set[name] = true
	}
	for name := range dq.inLoop {
		set[name] = true
	}
	for name := range dq.inClos {
		set[name] = true
	}
	check.logosDisqual = append(check.logosDisqual, set)
	if body != nil {
		if check.Info.LogosDisqual == nil {
			check.Info.LogosDisqual = make(map[*syntax.BlockStmt][]map[string]bool)
		}
		check.Info.LogosDisqual[body] = slices.Clone(check.logosDisqual)
	}
}

// logosExitFunc pops the disqualified set pushed by logosEnterFunc.
func (check *Checker) logosExitFunc() {
	if !check.conf.Logos {
		return
	}
	if n := len(check.logosDisqual); n > 0 {
		check.logosDisqual = check.logosDisqual[:n-1]
	}
}

// logosDisqualWalk is a syntax.Visitor collecting M3-S4 disqualified
// names in one walk: every `&name` (any depth), and every plain-name
// assignment target under a loop or closure. Short declarations with
// `:=` declare fresh names (shadowing is not an assignment to the outer
// variable), so only `=` forms disqualify. Walk calls Visit(nil) after
// each node's children, which pops the node stack.
type logosDisqualWalk struct {
	stack  []syntax.Node
	loop   int
	clos   int
	addr   map[string]bool
	inLoop map[string]bool
	inClos map[string]bool
}

func (w *logosDisqualWalk) Visit(node syntax.Node) syntax.Visitor {
	if node == nil {
		top := w.stack[len(w.stack)-1]
		w.stack = w.stack[:len(w.stack)-1]
		switch top.(type) {
		case *syntax.FuncLit:
			w.clos--
		case *syntax.ForStmt:
			w.loop--
		}
		return w
	}
	switch t := node.(type) {
	case *syntax.FuncLit:
		w.clos++
	case *syntax.ForStmt:
		w.loop++
	case *syntax.Operation:
		if t.Op == syntax.And && t.Y == nil {
			if id, ok := syntax.Unparen(t.X).(*syntax.Name); ok && id.Value != "_" {
				w.addr[id.Value] = true
			}
		}
	case *syntax.AssignStmt:
		if t.Op != syntax.Def && t.Lhs != nil {
			for _, name := range logosAssignNames(t.Lhs) {
				if w.loop > 0 {
					w.inLoop[name] = true
				}
				if w.clos > 0 {
					w.inClos[name] = true
				}
			}
		}
	case *syntax.RangeClause:
		if !t.Def && t.Lhs != nil {
			for _, name := range logosAssignNames(t.Lhs) {
				w.inLoop[name] = true
				if w.clos > 0 {
					w.inClos[name] = true
				}
			}
		}
	}
	w.stack = append(w.stack, node)
	return w
}

// logosAssignNames returns the plain names written by an assignment lhs.
func logosAssignNames(lhs syntax.Expr) []string {
	var names []string
	for _, e := range syntax.UnpackListExpr(lhs) {
		if id, ok := syntax.Unparen(e).(*syntax.Name); ok && id.Value != "_" {
			names = append(names, id.Value)
		}
	}
	return names
}

// logosDisqualified reports whether name may never narrow in the current
// function: its address was taken or it is assigned in a loop or closure.
func (check *Checker) logosDisqualified(name string) bool {
	for _, set := range check.logosDisqual {
		if set[name] {
			return true
		}
	}
	return false
}

// logosNarrowable reports whether v can carry narrowing: a same-package
// option-typed variable of local or parameter kind (local, receiver,
// parameter, or result) that is not disqualified. Package-level
// variables are excluded.
func (check *Checker) logosNarrowable(v *Var) bool {
	if v == nil || v.pkg != check.pkg || v.typ == nil {
		return false
	}
	switch v.Kind() {
	case LocalVar, RecvVar, ParamVar, ResultVar:
	default:
		return false
	}
	if logosAsOption(v.typ) == nil {
		return false
	}
	return !check.logosDisqualified(v.name)
}

// logosNilTestOf matches `name != nil` or `name == nil` (either operand
// order, through parentheses) for a plain name. It returns the resolved
// variable and whether the operator is `!=`. Selectors and other
// expressions never match: fields do not narrow.
func (check *Checker) logosNilTestOf(cond syntax.Expr) (v *Var, isNeq bool, ok bool) {
	op, ok := syntax.Unparen(cond).(*syntax.Operation)
	if !ok || (op.Op != syntax.Eql && op.Op != syntax.Neq) || op.Y == nil {
		return nil, false, false
	}
	var name *syntax.Name
	if check.isNil(op.X) {
		name, _ = syntax.Unparen(op.Y).(*syntax.Name)
	} else if check.isNil(op.Y) {
		name, _ = syntax.Unparen(op.X).(*syntax.Name)
	} else {
		return nil, false, false
	}
	if name == nil || name.Value == "_" {
		return nil, false, false
	}
	obj := check.lookup(name.Value)
	w, _ := obj.(*Var)
	if w == nil {
		return nil, false, false
	}
	return w, op.Op == syntax.Neq, true
}

// logosCollectNeNil gathers every variable tested `!= nil` along the
// `&&` spine of cond: `x != nil && rest` narrows x for rest and, through
// the whole chain, for the guarded block.
func (check *Checker) logosCollectNeNil(cond syntax.Expr) []*Var {
	op, ok := syntax.Unparen(cond).(*syntax.Operation)
	if !ok || op.Y == nil {
		return nil
	}
	if op.Op == syntax.AndAnd {
		left := check.logosCollectNeNil(op.X)
		right := check.logosCollectNeNil(op.Y)
		return append(left, right...)
	}
	if op.Op != syntax.Neq {
		return nil
	}
	v, isNeq, ok := check.logosNilTestOf(cond)
	if !ok || !isNeq {
		return nil
	}
	return []*Var{v}
}

// logosPushNarrow pushes one narrowing frame for the narrowable vars in
// vars and reports whether a frame was pushed. Pair every true return
// with logosPopNarrow.
func (check *Checker) logosPushNarrow(vars []*Var) bool {
	if !check.conf.Logos {
		return false
	}
	var frame []*Var
	for _, v := range vars {
		if check.logosNarrowable(v) {
			frame = append(frame, v)
		}
	}
	if len(frame) == 0 {
		return false
	}
	if check.logosNarrow == nil {
		check.logosNarrow = make(map[*Var]int)
	}
	for _, v := range frame {
		check.logosNarrow[v]++
	}
	check.logosNarrowStack = append(check.logosNarrowStack, frame)
	return true
}

// logosPopNarrow pops one frame pushed by logosPushNarrow.
func (check *Checker) logosPopNarrow() {
	frames := check.logosNarrowStack
	frame := frames[len(frames)-1]
	check.logosNarrowStack = frames[:len(frames)-1]
	for _, v := range frame {
		if c := check.logosNarrow[v]; c <= 1 {
			delete(check.logosNarrow, v)
		} else {
			check.logosNarrow[v] = c - 1
		}
	}
}

// logosNarrowBase records the narrowing stack height for stmtList; the
// matching logosPopToNarrow drops every frame pushed for later
// statements in the list (early-exit narrowing).
func (check *Checker) logosNarrowBase() int {
	return len(check.logosNarrowStack)
}

// logosPopToNarrow drops narrowing frames down to base.
func (check *Checker) logosPopToNarrow(base int) {
	for len(check.logosNarrowStack) > base {
		check.logosPopNarrow()
	}
}

// logosPushIfThen pushes narrowing for the `!= nil` spine of an if
// condition, for the Then block.
func (check *Checker) logosPushIfThen(cond syntax.Expr) bool {
	return check.logosPushNarrow(check.logosCollectNeNil(cond))
}

// logosPushAndCond pushes narrowing for the `!= nil` tests in the left
// side of an `&&` before the right side is checked.
func (check *Checker) logosPushAndCond(op syntax.Operator, lhs syntax.Expr) bool {
	if !check.conf.Logos || op != syntax.AndAnd {
		return false
	}
	return check.logosPushNarrow(check.logosCollectNeNil(lhs))
}

// logosAfterStmt applies early-exit rules for the rest of a statement
// list: `if x == nil { return|panic|break|continue }` (with no
// fall-through else) narrows the option x afterwards, and any exiting
// nil test on err discharges the error-first guard for results bound to
// err. Hooked from stmtList after each statement.
func (check *Checker) logosAfterStmt(s syntax.Stmt) {
	if !check.conf.Logos {
		return
	}
	ifStmt, ok := s.(*syntax.IfStmt)
	if !ok || ifStmt.Then == nil {
		return
	}
	v, isNeq, ok := check.logosNilTestOf(ifStmt.Cond)
	if !ok {
		return
	}
	if !check.logosExits(ifStmt.Then) {
		return
	}
	if ifStmt.Else != nil && !check.logosExits(ifStmt.Else) && (isNeq || check.logosElseAssigns(ifStmt.Else, v)) {
		return
	}
	if !isNeq {
		check.logosPushNarrow([]*Var{v})
	}
	check.logosDischargeErr(v)
}

// logosExits reports whether s never falls through: a return, a panic
// call, a break, continue, or goto, a block or if whose every path
// exits, or an infinite loop. Break and continue count as exits (unlike
// isTerminating) because the fall-through after the if still proves the
// nil test.
func (check *Checker) logosExits(s syntax.Stmt) bool {
	switch s := s.(type) {
	case *syntax.ReturnStmt:
		return true
	case *syntax.BranchStmt:
		return s.Tok != syntax.Fallthrough
	case *syntax.BlockStmt:
		list := s.List
		for len(list) > 0 {
			if _, ok := list[len(list)-1].(*syntax.EmptyStmt); !ok {
				break
			}
			list = list[:len(list)-1]
		}
		if len(list) == 0 {
			return false
		}
		return check.logosExits(list[len(list)-1])
	case *syntax.IfStmt:
		return s.Else != nil && check.logosExits(s.Then) && check.logosExits(s.Else)
	case *syntax.LabeledStmt:
		return check.logosExits(s.Stmt)
	case *syntax.ExprStmt:
		if call, ok := syntax.Unparen(s.X).(*syntax.CallExpr); ok {
			if check.isPanic[call] {
				return true
			}
			if name, ok := syntax.Unparen(call.Fun).(*syntax.Name); ok && name.Value == "panic" {
				return true
			}
		}
		return false
	case *syntax.ForStmt:
		if s.Cond != nil {
			return false
		}
		if _, ok := s.Init.(*syntax.RangeClause); ok {
			return false
		}
		hasBreak := false
		syntax.Inspect(s.Body, func(n syntax.Node) bool {
			if b, ok := n.(*syntax.BranchStmt); ok && b.Tok == syntax.Break {
				hasBreak = true
				return false
			}
			return !hasBreak
		})
		return !hasBreak
	default:
		return false
	}
}

// logosElseAssigns reports whether the else branch may assign v, voiding
// the early-exit proof after the if. The scan is syntactic (by name).
func (check *Checker) logosElseAssigns(els syntax.Stmt, v *Var) bool {
	assigns := false
	syntax.Inspect(els, func(n syntax.Node) bool {
		if assigns {
			return false
		}
		switch t := n.(type) {
		case *syntax.AssignStmt:
			if t.Op != syntax.Def {
				for _, name := range logosAssignNames(t.Lhs) {
					if name == v.name {
						assigns = true
						return false
					}
				}
			}
		case *syntax.RangeClause:
			if !t.Def {
				for _, name := range logosAssignNames(t.Lhs) {
					if name == v.name {
						assigns = true
						return false
					}
				}
			}
		}
		return true
	})
	return assigns
}

// logosPushIfElse pushes narrowing for the else block of an `x == nil`
// condition: the else branch proves `x != nil`.
func (check *Checker) logosPushIfElse(cond syntax.Expr) bool {
	v, isNeq, ok := check.logosNilTestOf(cond)
	if !ok || isNeq {
		return false
	}
	return check.logosPushNarrow([]*Var{v})
}

// logosNarrowIdent is the ident hook: a value read of a narrowed option
// variable takes its element type, so field and method selection,
// indirection, and stores into non-nil slots check against the narrowed
// type. A read of an unchecked error-first result reports
// result_before_error_check. Assignment targets resolve through
// logosLHSPush and skip both rules. Hooked from ident.
func (check *Checker) logosNarrowIdent(x *operand, obj Object) {
	if !check.conf.Logos {
		return
	}
	v, ok := obj.(*Var)
	if !ok || v.pkg != check.pkg {
		return
	}
	if n := len(check.logosLHS); n > 0 && check.logosLHS[n-1] == v {
		return
	}
	if errV, ok := check.logosErrUncheck[v]; ok {
		if !check.logosStmtReadsErr(errV.name) {
			check.errorf(x.expr, IncompatibleAssign, "%s read of %s before %s is checked (read %s only after %s is checked)", logosResultBeforeErr, v.name, errV.name, v.name, errV.name)
		}
		return
	}
	if check.logosNarrow[v] > 0 {
		if opt := logosAsOption(v.typ); opt != nil {
			x.typ_ = opt.Elem
		}
	}
}

// logosStmtEnter records the innermost enclosing statement for the
// joint-use scan; logosStmtExit restores the outer one. Hooked from
// stmtList around each statement.
func (check *Checker) logosStmtEnter(s syntax.Stmt) syntax.Stmt {
	prev := check.logosCurStmt
	check.logosCurStmt = s
	return prev
}

// logosStmtExit restores the statement saved by logosStmtEnter.
func (check *Checker) logosStmtExit(prev syntax.Stmt) {
	check.logosCurStmt = prev
}

// logosStmtReadsErr reports whether the innermost enclosing statement
// also reads the error name: `fmt.Println(v, err)` and `return v, err`
// observe both together, so the result use is error-aware and exempt.
// The scan is syntactic (by name); shadowing across scopes is ignored.
func (check *Checker) logosStmtReadsErr(name string) bool {
	if name == "" || name == "_" || check.logosCurStmt == nil {
		return false
	}
	found := false
	syntax.Walk(check.logosCurStmt, logosNameFinder{name: name, found: &found})
	return found
}

// logosNameFinder is a syntax.Visitor marking found when a plain name
// with the target value occurs.
type logosNameFinder struct {
	name  string
	found *bool
}

func (w logosNameFinder) Visit(node syntax.Node) syntax.Visitor {
	if node == nil || *w.found {
		return nil
	}
	if id, ok := node.(*syntax.Name); ok && id.Value == w.name {
		*w.found = true
		return nil
	}
	return w
}

// logosLHSPush marks the assignment target of lhs so ident skips the
// read rules for the write itself (sub-expression reads still check).
// Hooked from lhsVar around the lhs evaluation; logosLHSPop clears it.
func (check *Checker) logosLHSPush(lhs syntax.Expr) {
	if !check.conf.Logos {
		return
	}
	var v *Var
	if id, ok := syntax.Unparen(lhs).(*syntax.Name); ok && id.Value != "_" {
		v, _ = check.lookup(id.Value).(*Var)
	}
	check.logosLHS = append(check.logosLHS, v)
}

// logosLHSPop clears the mark pushed by logosLHSPush.
func (check *Checker) logosLHSPop() {
	if !check.conf.Logos {
		return
	}
	if n := len(check.logosLHS); n > 0 {
		check.logosLHS = check.logosLHS[:n-1]
	}
}

// logosKillNarrow ends narrowing and error-first tracking for the plain
// variable written by lhs: `x = other` kills narrowing until the next
// re-narrow. Reassigning a result kills its guard; reassigning an error
// voids every result bound to it. Hooked from assignVar.
func (check *Checker) logosKillNarrow(lhs syntax.Expr) {
	if !check.conf.Logos {
		return
	}
	id, ok := syntax.Unparen(lhs).(*syntax.Name)
	if !ok || id.Value == "_" {
		return
	}
	obj := check.lookup(id.Value)
	v, ok := obj.(*Var)
	if !ok || v.pkg != check.pkg {
		return
	}
	check.logosKillVar(v)
}

// logosShortDecl kills narrowing for redeclared `:=` targets and tracks
// `n, err := f()` tuples for the error-first guard. Hooked from
// shortVarDecl after the initializers run.
func (check *Checker) logosShortDecl(lhs, rhs []syntax.Expr, lhsVars []*Var) {
	if !check.conf.Logos {
		return
	}
	for _, e := range lhs {
		id, ok := syntax.Unparen(e).(*syntax.Name)
		if !ok || id.Value == "_" {
			continue
		}
		if alt, _ := check.scope.Lookup(id.Value).(*Var); alt != nil {
			check.logosKillVar(alt)
		}
	}
	if len(lhs) != 2 || len(rhs) != 1 || len(lhsVars) != 2 {
		return
	}
	call, ok := syntax.Unparen(rhs[0]).(*syntax.CallExpr)
	if !ok {
		return
	}
	n, errV := lhsVars[0], lhsVars[1]
	if n == nil || errV == nil || n.name == "_" || errV.name == "_" || n == errV {
		return
	}
	check.logosTrackTuple(n, errV, call)
}

// logosTrackCallAssign tracks `n, err = f()` tuples for plain
// assignments. Hooked from assignVars on the single-call path, after the
// per-variable kills in assignVar.
func (check *Checker) logosTrackCallAssign(lhs []syntax.Expr, rhs syntax.Expr) {
	if !check.conf.Logos || len(lhs) != 2 {
		return
	}
	call, ok := syntax.Unparen(rhs).(*syntax.CallExpr)
	if !ok {
		return
	}
	n, ok := check.logosLhsVar(lhs[0])
	if !ok {
		return
	}
	errV, ok := check.logosLhsVar(lhs[1])
	if !ok || n == errV {
		return
	}
	check.logosTrackTuple(n, errV, call)
}

// logosLhsVar resolves a plain assignment lhs to its variable.
func (check *Checker) logosLhsVar(e syntax.Expr) (*Var, bool) {
	id, ok := syntax.Unparen(e).(*syntax.Name)
	if !ok || id.Value == "_" {
		return nil, false
	}
	v, ok := check.lookup(id.Value).(*Var)
	if !ok || v.pkg != check.pkg {
		return nil, false
	}
	return v, true
}

// logosKillVar ends narrowing and guard tracking for one variable.
func (check *Checker) logosKillVar(v *Var) {
	if v == nil || v.pkg != check.pkg {
		return
	}
	delete(check.logosNarrow, v)
	if _, ok := check.logosErrUncheck[v]; ok {
		delete(check.logosErrUncheck, v)
		check.logosNoteDead(v)
	}
	maps.DeleteFunc(check.logosErrUncheck, func(m *Var, e *Var) bool {
		if e != v {
			return false
		}
		check.logosNoteDead(m)
		return true
	})
}

// logosNoteDead marks a result skipped by err-frame restore after a kill.
func (check *Checker) logosNoteDead(v *Var) {
	if check.logosErrDead == nil {
		check.logosErrDead = make(map[*Var]bool)
	}
	check.logosErrDead[v] = true
}

// logosTrackTuple starts error-first tracking for one `n, err := f()`
// binding: n must be used only after err is checked. Only callees
// declared in .lgo (same package) arm the guard, and only when the
// second result is error. Rebinding err voids previous guards.
func (check *Checker) logosTrackTuple(n, errV *Var, call *syntax.CallExpr) {
	if n.pkg != check.pkg || errV.pkg != check.pkg {
		return
	}
	if errV.typ == nil || !Identical(errV.typ, universeError) {
		return
	}
	if !check.logosIsLgoCall(call) {
		return
	}
	maps.DeleteFunc(check.logosErrUncheck, func(m *Var, e *Var) bool {
		if e != errV {
			return false
		}
		check.logosNoteDead(m)
		return true
	})
	delete(check.logosErrDead, n)
	if check.logosErrUncheck == nil {
		check.logosErrUncheck = make(map[*Var]*Var)
	}
	check.logosErrUncheck[n] = errV
}

// logosIsLgoCall reports whether call targets a function declared in
// .lgo (the current package). Package-qualified calls into other
// packages and methods declared outside keep their own contracts.
func (check *Checker) logosIsLgoCall(call *syntax.CallExpr) bool {
	switch fun := syntax.Unparen(call.Fun).(type) {
	case *syntax.Name:
		if fn, ok := check.lookup(fun.Value).(*Func); ok {
			return fn.pkg == check.pkg
		}
		return true
	case *syntax.SelectorExpr:
		if xn, ok := syntax.Unparen(fun.X).(*syntax.Name); ok {
			if obj := check.lookup(xn.Value); obj != nil {
				if pn, ok := obj.(*PkgName); ok {
					return pn.imported == check.pkg
				}
			}
		}
		if sel := check.Selections[fun]; sel != nil {
			if fn, ok := sel.Obj().(*Func); ok && fn.pkg != nil {
				return fn.pkg == check.pkg
			}
		}
		return true
	default:
		return true
	}
}

// logosDischargeErr permanently discharges every result bound to err:
// the fall-through of `if err != nil { exit }` proves err is nil.
func (check *Checker) logosDischargeErr(errV *Var) {
	if errV == nil {
		return
	}
	maps.DeleteFunc(check.logosErrUncheck, func(_ *Var, e *Var) bool {
		return e == errV
	})
}

// logosPushChecked temporarily discharges the results bound to the `err`
// of an `if err == nil` or `if err != nil` condition for the Then block:
// both branches read err under scrutiny, so the error-branch use in
// try_chain.lgo stays legal. Pair every true return with logosPopChecked.
func (check *Checker) logosPushChecked(cond syntax.Expr) bool {
	if !check.conf.Logos {
		return false
	}
	v, _, ok := check.logosNilTestOf(cond)
	if !ok {
		return false
	}
	var frame map[*Var]*Var
	for m, e := range check.logosErrUncheck {
		if e == v {
			if frame == nil {
				frame = make(map[*Var]*Var)
			}
			frame[m] = e
		}
	}
	if len(frame) == 0 {
		return false
	}
	for m := range frame {
		delete(check.logosErrUncheck, m)
	}
	check.logosErrStack = append(check.logosErrStack, frame)
	return true
}

// logosPopChecked restores one frame pushed by logosPushChecked, except
// results killed inside the block.
func (check *Checker) logosPopChecked() {
	frames := check.logosErrStack
	frame := frames[len(frames)-1]
	check.logosErrStack = frames[:len(frames)-1]
	for m, e := range frame {
		if check.logosErrDead[m] {
			continue
		}
		if _, ok := check.logosErrUncheck[m]; !ok {
			check.logosErrUncheck[m] = e
		}
	}
}

// logosErrBase records the err-frame height for stmtList; the matching
// logosPopToErr restores every frame pushed for later statements.
func (check *Checker) logosErrBase() int {
	return len(check.logosErrStack)
}

// logosPopToErr drops err frames down to base.
func (check *Checker) logosPopToErr(base int) {
	for len(check.logosErrStack) > base {
		check.logosPopChecked()
	}
}

// logosUnnarrowOperand restores the declared option type of a narrowed
// receiver for `?.` and `??`, which require an option operand. Hooked
// from logosOptSelect and logosCoalesce.
func (check *Checker) logosUnnarrowOperand(x *operand) {
	if !check.conf.Logos || !x.isValid() {
		return
	}
	id, ok := syntax.Unparen(x.expr).(*syntax.Name)
	if !ok {
		return
	}
	obj := check.lookup(id.Value)
	v, ok := obj.(*Var)
	if !ok || v.pkg != check.pkg || check.logosNarrow[v] == 0 {
		return
	}
	if logosAsOption(v.typ) == nil {
		return
	}
	x.typ_ = v.typ
	check.updateExprType(x.expr, v.typ, false)
}
