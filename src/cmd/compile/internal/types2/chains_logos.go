// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"slices"

	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

const logosChainMutation = "[correctness::const_view_mutation]"

const logosSideEffectBeforeSort = "[suspicious::side_effect_before_sort]"

const logosChainJump = "[correctness::chain_jump] jump must not cross a chain lambda boundary"

type logosChainLabel struct {
	name string
	loop bool
	brk  bool
}

type logosChainDepth struct {
	loop int
	brk  int
	sw   int
}

func (d logosChainDepth) inLoop() logosChainDepth {
	d.loop++
	d.brk++
	return d
}
func (d logosChainDepth) inSwitch() logosChainDepth {
	d.brk++
	d.sw++
	return d
}
func (d logosChainDepth) inTypeSwitch() logosChainDepth {
	d.brk++
	return d
}
func (d logosChainDepth) inSelect() logosChainDepth {
	d.brk++
	return d
}

type logosChainJumpScan struct {
	check  *Checker
	inside map[string]bool
	bad    bool
}

func (check *Checker) logosChainJumps(block *syntax.BlockStmt) bool {
	v := &logosChainJumpScan{check: check, inside: make(map[string]bool)}
	syntax.Inspect(block, func(n syntax.Node) bool {
		switch n.(type) {
		case *syntax.FuncLit, *syntax.LambdaExpr:
			return false
		}
		if l, ok := n.(*syntax.LabeledStmt); ok && l.Label != nil {
			v.inside[l.Label.Value] = true
		}
		return true
	})
	v.seq(block.List, logosChainDepth{}, nil)
	return !v.bad
}

func (v *logosChainJumpScan) seq(list []syntax.Stmt, d logosChainDepth, st []logosChainLabel) {
	for _, s := range list {
		v.one(s, d, st)
	}
}

func (v *logosChainJumpScan) one(s syntax.Stmt, d logosChainDepth, st []logosChainLabel) {
	switch s := s.(type) {
	case *syntax.BlockStmt:
		v.seq(s.List, d, st)
	case *syntax.IfStmt:
		if s.Then != nil {
			v.one(s.Then, d, st)
		}
		if s.Else != nil {
			v.one(s.Else, d, st)
		}
	case *syntax.ForStmt:
		v.one(s.Body, d.inLoop(), st)
	case *syntax.SwitchStmt:
		dd := d.inSwitch()
		if _, ok := s.Tag.(*syntax.TypeSwitchGuard); ok {
			dd = d.inTypeSwitch()
		}
		for _, c := range s.Body {
			v.seq(c.Body, dd, st)
		}
	case *syntax.SelectStmt:
		for _, c := range s.Body {
			v.seq(c.Body, d.inSelect(), st)
		}
	case *syntax.LabeledStmt:
		v.one(s.Stmt, d, append(st, logosChainLabelOf(s)))
	case *syntax.BranchStmt:
		v.branch(s, d, st)
	}
}
func logosChainLabelOf(s *syntax.LabeledStmt) logosChainLabel {
	e := logosChainLabel{name: s.Label.Value}
	inner := s.Stmt
	for {
		l, ok := inner.(*syntax.LabeledStmt)
		if !ok {
			break
		}
		inner = l.Stmt
	}
	switch inner.(type) {
	case *syntax.ForStmt:
		e.loop = true
		e.brk = true
	case *syntax.SwitchStmt, *syntax.SelectStmt:
		e.brk = true
	}
	return e
}

func logosChainLookup(st []logosChainLabel, name string) (logosChainLabel, bool) {
	for i := len(st) - 1; i >= 0; i-- {
		if st[i].name == name {
			return st[i], true
		}
	}
	return logosChainLabel{}, false
}

func logosChainBreaks(st []logosChainLabel, name string) bool {
	e, ok := logosChainLookup(st, name)
	return ok && e.brk
}

func logosChainContinues(st []logosChainLabel, name string) bool {
	e, ok := logosChainLookup(st, name)
	return ok && e.loop
}

func (v *logosChainJumpScan) branch(s *syntax.BranchStmt, d logosChainDepth, st []logosChainLabel) {
	switch s.Tok {
	case syntax.Break:
		if s.Label != nil {
			if !logosChainBreaks(st, s.Label.Value) {
				v.fail(s.Pos(), MisplacedLabel)
			}
			return
		}
		if d.brk == 0 {
			v.fail(s.Pos(), MisplacedBreak)
		}
	case syntax.Continue:
		if s.Label != nil {
			if !logosChainContinues(st, s.Label.Value) {
				v.fail(s.Pos(), MisplacedLabel)
			}
			return
		}
		if d.loop == 0 {
			v.fail(s.Pos(), MisplacedContinue)
		}
	case syntax.Fallthrough:
		if d.sw == 0 {
			v.fail(s.Pos(), MisplacedFallthrough)
		}
	case syntax.Goto:
		if s.Label == nil {
			return
		}
		if !v.inside[s.Label.Value] {
			v.fail(s.Pos(), MisplacedLabel)
		}
	}
}

func (v *logosChainJumpScan) fail(pos syntax.Pos, code Code) {
	v.check.error(pos, code, logosChainJump)
	v.bad = true
}

var logosChainMethods = map[string]bool{
	"filter": true,
	"map":    true,
	"find":   true,
	"any":    true,
	"all":    true,
	"sortBy": true,
	"first":  true,
	"last":   true,
	"get":    true,
	"take":   true,
	"drop":   true,
	"sum":    true,
	"count":  true,
}

// LogosChainMethod reports whether name is a method of a loGo/s chain.
func LogosChainMethod(name string) bool { return logosChainMethods[name] }

func (check *Checker) logosChainCall(x *operand, call *syntax.CallExpr) bool {
	if !check.conf.Logos {
		return false
	}
	if check.logosStageChain(x, call, logosStageValue) || check.logosNamedArgCall(x, call) {
		return true
	}
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if !ok {
		return false
	}
	if !logosChainMethods[sel.Sel.Value] {
		return false
	}
	var rx operand
	check.exprOrType(&rx, sel.X, false)
	if !rx.isValid() {
		x.mode_ = invalid
		x.expr = call
		x.typ_ = Typ[Invalid]
		return true
	}
	if rx.mode() == typexpr {
		return false
	}
	st, ok := check.logosChainSlice(&rx, sel.Sel.Value)
	if !ok {
		return false
	}
	check.logosChainMethod(x, call, sel, st.Elem())
	return true
}

func (check *Checker) logosChainRoots(e syntax.Expr) []logosWritePath {
	return []logosWritePath{check.logosWritePath(e)}
}

func (check *Checker) logosChainSlice(rx *operand, method string) (*Slice, bool) {
	rt, descendants, readonly := logosReadonlyBase(rx.typ())
	if obj, _, _ := lookupFieldOrMethod(rt, rx.mode() == variable, check.pkg, method, false); obj != nil {
		return nil, false
	}
	st, ok := Unalias(rt).(*Slice)
	if ok && (descendants || readonly) {
		st = NewSlice(logosWithReadonlyDescendants(st.Elem()))
	}
	return st, ok
}

func (check *Checker) logosChainMethod(x *operand, call *syntax.CallExpr, sel *syntax.SelectorExpr, elem Type) {
	method := sel.Sel.Value
	roots := check.logosChainRoots(sel.X)
	fail := func() {
		x.mode_ = invalid
		x.expr = call
		x.typ_ = Typ[Invalid]
	}
	done := func(t Type) {
		x.mode_ = value
		x.expr = call
		x.typ_ = t
		check.hasCallOrRecv = true
	}
	args := call.ArgList
	switch method {
	case "filter", "find", "any", "all":
		if !check.logosChainArity(call, args, 1) {
			fail()
			return
		}
		u, _, ok := check.logosChainPred(args[0], elem, method, roots)
		if !ok {
			fail()
			return
		}
		_ = u
		switch method {
		case "filter":
			done(NewSlice(elem))
		case "find":
			done(logosWrapOption(elem))
		default:
			done(Typ[Bool])
		}
	case "map":
		if !check.logosChainArity(call, args, 1) {
			fail()
			return
		}
		u, _, ok := check.logosChainMapArg(args[0], elem, roots)
		if !ok {
			fail()
			return
		}
		done(NewSlice(u))
	case "sortBy":
		if !check.logosChainArity(call, args, 1) {
			fail()
			return
		}
		_, _, ok := check.logosChainKeyArg(args[0], elem, roots)
		if !ok {
			fail()
			return
		}
		check.logosWarnSideEffectBeforeSort(sel, sel.X)
		done(NewSlice(elem))
	case "first", "last":
		if !check.logosChainArity(call, args, 0) {
			fail()
			return
		}
		done(logosWrapOption(elem))
	case "get":
		if !check.logosChainArity(call, args, 1) {
			fail()
			return
		}
		if !check.logosChainIntArg(args[0], method) {
			fail()
			return
		}
		done(logosWrapOption(elem))
	case "take", "drop":
		if !check.logosChainArity(call, args, 1) {
			fail()
			return
		}
		if !check.logosChainIntArg(args[0], method) {
			fail()
			return
		}
		result := Type(NewSlice(elem))
		if _, descendants, readonly := logosReadonlyBase(check.Info.TypeOf(sel.X)); descendants || readonly {
			result = &ConstView{Elem: result}
		}
		done(result)
	case "sum":
		if !check.logosChainArity(call, args, 0) {
			fail()
			return
		}
		if b, ok := Unalias(elem).(*Basic); !ok || b.Info()&IsNumeric == 0 {
			check.errorf(call, IncompatibleAssign, "cannot sum non-numeric element type %s", elem)
			fail()
			return
		}
		done(elem)
	case "count":
		if len(args) == 0 {
			done(Typ[Int])
			return
		}
		if !check.logosChainArity(call, args, 1) {
			fail()
			return
		}
		if _, _, ok := check.logosChainPred(args[0], elem, method, roots); !ok {
			fail()
			return
		}
		done(Typ[Int])
	}
}

func (check *Checker) logosChainArity(call *syntax.CallExpr, args []syntax.Expr, want int) bool {
	if hasDots(call) {
		check.errorf(call, InvalidDotDotDot, "invalid use of ... in chain call to %s", call.Fun)
		return false
	}
	if len(args) != want {
		check.errorf(call, WrongArgCount, "wrong number of arguments in chain call to %s: have %d, want %d", call.Fun, len(args), want)
		return false
	}
	return true
}

func (check *Checker) logosChainIntArg(arg syntax.Expr, method string) bool {
	var op operand
	check.expr(nil, &op, arg)
	if !op.isValid() {
		return false
	}
	if b, ok := Unalias(op.typ()).(*Basic); !ok || b.Info()&IsInteger == 0 {
		check.errorf(&op, IncompatibleAssign, "chain %s index must be an integer, have %s", method, op.typ())
		return false
	}
	return true
}

func (check *Checker) logosChainPred(arg syntax.Expr, elem Type, method string, roots []logosWritePath) (Type, bool, bool) {
	if lam, ok := syntax.Unparen(arg).(*syntax.LambdaExpr); ok {
		return check.logosChainLambda(lam, elem, logosPredKind, roots)
	}
	return check.logosChainFuncArg(arg, elem, Typ[Bool], method)
}

func (check *Checker) logosChainMapArg(arg syntax.Expr, elem Type, roots []logosWritePath) (Type, bool, bool) {
	if lam, ok := syntax.Unparen(arg).(*syntax.LambdaExpr); ok {
		return check.logosChainLambda(lam, elem, logosMapKind, roots)
	}
	return check.logosChainFuncInfer(arg, elem)
}

func (check *Checker) logosChainKeyArg(arg syntax.Expr, elem Type, roots []logosWritePath) (Type, bool, bool) {
	if lam, ok := syntax.Unparen(arg).(*syntax.LambdaExpr); ok {
		return check.logosChainLambda(lam, elem, logosKeyKind, roots)
	}
	u, isTry, ok := check.logosChainFuncInfer(arg, elem)
	if !ok {
		return nil, false, false
	}
	if !logosOrderedKey(u) {
		check.errorf(arg, IncompatibleAssign, "chain sortBy key must be ordered, have %s", u)
		return nil, false, false
	}
	return u, isTry, true
}

type logosChainKind int

const (
	logosPredKind logosChainKind = iota
	logosMapKind
	logosKeyKind
)

func logosIsTryChain(call *syntax.CallExpr) *syntax.TryExpr {
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if !ok || sel.Sel == nil || !logosChainMethods[sel.Sel.Value] {
		return nil
	}
	for _, arg := range call.ArgList {
		if lam, ok := syntax.Unparen(arg).(*syntax.LambdaExpr); ok {
			if tb := LogosLambdaTry(lam); tb != nil {
				return tb
			}
		}
	}
	return nil
}

func LogosLambdaTry(lam *syntax.LambdaExpr) *syntax.TryExpr {
	if block, ok := lam.Body.(*syntax.BlockStmt); ok {
		return logosBlockTry(block)
	}
	if body, ok := lam.Body.(syntax.Expr); ok {
		tb, _ := syntax.Unparen(body).(*syntax.TryExpr)
		return tb
	}
	return nil
}

func logosBlockTry(block *syntax.BlockStmt) *syntax.TryExpr {
	var found *syntax.TryExpr
	syntax.Inspect(block, func(n syntax.Node) bool {
		if found != nil {
			return false
		}
		switch n := n.(type) {
		case *syntax.FuncLit, *syntax.LambdaExpr:
			return false
		case *syntax.TryExpr:
			found = n
			return false
		}
		return true
	})
	return found
}

func (check *Checker) logosChainNestedTry(body syntax.Node) {
	syntax.Inspect(body, func(n syntax.Node) bool {
		if _, ok := n.(*syntax.FuncLit); ok {
			return false
		}
		if call, ok := n.(*syntax.CallExpr); ok {
			if tb := logosIsTryChain(call); tb != nil {
				check.error(tb.Pos(), IncompatibleAssign, "[correctness::nested_try_capture] ? cannot propagate from a nested chain")
				return false
			}
		}
		return true
	})
}

func (check *Checker) logosChainLambda(lam *syntax.LambdaExpr, elem Type, kind logosChainKind, roots []logosWritePath) (Type, bool, bool) {
	if check.logosInPlace == nil {
		check.logosInPlace = make(map[*syntax.LambdaExpr]bool)
	}
	check.logosInPlace[lam] = true
	if len(lam.Params) != 1 {
		check.errorf(lam.Arrow, IncompatibleAssign, "chain lambda takes exactly one parameter, have %d", len(lam.Params))
		return nil, false, false
	}
	check.logosChainNestedTry(lam.Body)
	name := lam.Params[0]
	scope := NewScope(check.scope, lam.Pos(), endPos(lam.Body), "chain lambda")
	scope.isFunc = true
	p := NewParam(name.Pos(), check.pkg, name.Value, elem)
	check.declare(scope, name, p, name.Pos())
	merged := append(slices.Clone(roots), logosWritePath{root: p})
	saved := check.environment
	check.environment = environment{decl: check.decl, scope: scope, version: check.version, iota: check.iota, sig: check.sig}
	defer func() { check.environment = saved }()
	defer check.logosChainNoMutate(merged, lam.Body)
	if block, ok := lam.Body.(*syntax.BlockStmt); ok {
		u, isTry, ok := check.logosChainBlock(lam, scope, elem, kind, block)
		check.usage(scope)
		return u, isTry, ok
	}
	body, ok := lam.Body.(syntax.Expr)
	if !ok {
		check.error(lam.Body, InvalidSyntaxTree, "invalid chain lambda body")
		return nil, false, false
	}
	if t, ok := syntax.Unparen(body).(*syntax.TryExpr); ok {
		results, ok := check.logosTryResults(t)
		if !ok {
			return nil, false, false
		}
		if len(results) != 1 {
			check.errorf(t.Pos(), WrongResultCount, "chain lambda ? provides %d values, want 1", len(results))
			return nil, false, false
		}
		u := results[0].typ()
		if kind == logosPredKind {
			check.assignment(results[0], Typ[Bool], "chain predicate")
			check.usage(check.scope)
		}
		if kind == logosKeyKind && !logosOrderedKey(u) {
			check.errorf(body, IncompatibleAssign, "chain sortBy key must be ordered, have %s", u)
			return nil, false, false
		}
		if isUntyped(u) {
			u = Default(u)
		}
		check.usage(check.scope)
		return u, true, true
	}
	var op operand
	check.expr(nil, &op, body)
	if !op.isValid() {
		return nil, false, false
	}
	u := op.typ()
	if kind == logosPredKind {
		check.assignment(&op, Typ[Bool], "chain predicate")
		check.usage(scope)
		return Typ[Bool], false, true
	}
	if isUntyped(u) {
		if u == Typ[UntypedNil] {
			check.error(&op, UntypedNilUse, "cannot infer chain element type from nil")
			return nil, false, false
		}
		u = Default(u)
	}
	if kind == logosKeyKind && !logosOrderedKey(u) {
		check.errorf(&op, IncompatibleAssign, "chain sortBy key must be ordered, have %s", u)
		return nil, false, false
	}
	check.usage(scope)
	return u, false, true
}

func logosOrderedKey(u Type) bool {
	b, ok := Unalias(u).(*Basic)
	return ok && b.Info()&IsOrdered != 0
}

func (check *Checker) logosChainBlock(lam *syntax.LambdaExpr, scope *Scope, elem Type, kind logosChainKind, block *syntax.BlockStmt) (Type, bool, bool) {
	check.logosChainBlockTryForms(block)
	saved := check.environment.chainRets
	st := &chainReturnState{}
	check.environment.chainRets = st
	failed := !check.logosChainJumps(block)
	check.stmtList(0, block.List)
	check.environment.chainRets = saved
	if st.bad || failed {
		return nil, false, false
	}
	collected := st.ops
	if kind == logosPredKind {
		for _, op := range collected {
			check.assignment(op, Typ[Bool], "return statement")
		}
		if !check.isTerminating(block, "") {
			check.error(block.Rbrace, MissingReturn, "missing return")
			return nil, false, false
		}
		return Typ[Bool], false, true
	}
	var types []Type
	for _, op := range collected {
		t := op.typ()
		if isUntyped(t) {
			if t == Typ[UntypedNil] {
				check.error(op.expr, UntypedNilUse, "cannot infer chain element type from nil")
				return nil, false, false
			}
			t = Default(t)
		}
		types = append(types, t)
	}
	if len(types) == 0 {
		check.error(block, IncompatibleAssign, "cannot infer chain element type: no return")
		return nil, false, false
	}
	mismatch := false
	for _, op := range collected {
		if ok, _ := op.assignableTo(check, types[0], nil); !ok {
			check.assignment(op, types[0], "return statement")
			mismatch = true
		}
	}
	if mismatch {
		return nil, false, false
	}
	if !check.isTerminating(block, "") {
		check.error(block.Rbrace, MissingReturn, "missing return")
		return nil, false, false
	}
	if kind == logosKeyKind && !logosOrderedKey(types[0]) {
		check.errorf(block, IncompatibleAssign, "chain sortBy key must be ordered, have %s", types[0])
		return nil, false, false
	}
	return types[0], false, true
}

type chainReturnState struct {
	ops []*operand
	bad bool
}

var logosChainMutatingBuiltins = map[string]string{
	"append": "append to",
	"copy":   "copy into",
	"clear":  "clear",
	"delete": "delete from",
}

func (check *Checker) logosChainReturn(s *syntax.ReturnStmt) bool {
	st := check.environment.chainRets
	if st == nil {
		return false
	}
	results := syntax.UnpackListExpr(s.Results)
	if len(results) != 1 {
		check.errorf(s, WrongResultCount, "chain lambda return provides %d values, want 1", len(results))
		st.bad = true
		return true
	}
	var op operand
	check.expr(nil, &op, results[0])
	if !op.isValid() {
		st.bad = true
		return true
	}
	if _, ok := op.typ().(*Tuple); ok {
		check.error(results[0], WrongResultCount, "cannot infer chain element type from multi-value return")
		st.bad = true
		return true
	}
	st.ops = append(st.ops, &op)
	return true
}

func (check *Checker) logosChainBlockTryForms(block *syntax.BlockStmt) {
	syntax.Inspect(block, func(n syntax.Node) bool {
		switch n := n.(type) {
		case *syntax.FuncLit, *syntax.LambdaExpr:
			return false
		case *syntax.CallStmt:
			if n.Tok == syntax.Defer {
				if t, ok := syntax.Unparen(n.Call).(*syntax.TryExpr); ok {
					check.logosTryMisplaced(t)
				}
			}
		case *syntax.ErrdeferStmt:
			check.error(n, InvalidDefer, logosErrdeferNeedsError)
			check.logosErrdeferUse(n.Call)
		}
		return true
	})
}

func (check *Checker) logosChainNoMutate(roots []logosWritePath, body syntax.Node) {
	syntax.Inspect(body, func(n syntax.Node) bool {
		var targets []syntax.Expr
		switch n := n.(type) {
		case *syntax.AssignStmt:
			if n.Op != syntax.Def {
				targets = syntax.UnpackListExpr(n.Lhs)
			}
		case *syntax.RangeClause:
			if !n.Def {
				targets = syntax.UnpackListExpr(n.Lhs)
			}
		case *syntax.CallExpr:
			check.logosChainMutatingCall(n, roots)
		}
		for _, lhs := range targets {
			if logosChainHits(check.logosWritePath(lhs), roots) {
				check.errorf(lhs, IncompatibleAssign, "%s cannot assign to chain receiver or parameter", logosChainMutation)
			}
		}
		return true
	})
}

func (check *Checker) logosChainMutatingCall(call *syntax.CallExpr, roots []logosWritePath) {
	name, ok := syntax.Unparen(call.Fun).(*syntax.Name)
	if !ok || len(call.ArgList) == 0 {
		return
	}
	verb, mutating := logosChainMutatingBuiltins[name.Value]
	_, builtin := check.Info.Uses[name].(*Builtin)
	if mutating && builtin && logosChainHits(check.logosContentsPath(call.ArgList[0]), roots) {
		check.errorf(call.ArgList[0], IncompatibleAssign, "%s cannot %s chain receiver or parameter", logosChainMutation, verb)
	}
}

func logosChainHits(target logosWritePath, roots []logosWritePath) bool {
	return slices.ContainsFunc(roots, func(root logosWritePath) bool { return logosWriteHits(root, target) })
}

func (check *Checker) logosChainFuncArg(arg syntax.Expr, elem, want Type, method string) (Type, bool, bool) {
	var op operand
	check.expr(nil, &op, arg)
	if !op.isValid() {
		return nil, false, false
	}
	sig, ok := Unalias(op.typ()).(*Signature)
	if !ok || sig.Params().Len() != 1 {
		check.errorf(arg, IncompatibleAssign, "chain %s needs func with one parameter, have %s", method, op.typ())
		return nil, false, false
	}
	fn := NewSignatureType(nil, nil, nil, NewTuple(NewParam(nopos, check.pkg, "", elem)), NewTuple(NewParam(nopos, check.pkg, "", want)), false)
	check.assignment(&op, fn, "chain "+method)
	return want, false, op.isValid()
}

func (check *Checker) logosWarnSideEffectBeforeSort(sel *syntax.SelectorExpr, recv syntax.Expr) {
	for _, lam := range logosPrefixLambdas(recv) {
		if logosLambdaHasImpureCall(check, lam) {
			check.softErrorf(sel.Sel, IncompatibleAssign, "%s chain lambda with a call of unknown purity before sortBy", logosSideEffectBeforeSort)
			return
		}
	}
}

func logosPrefixLambdas(recv syntax.Expr) []*syntax.LambdaExpr {
	var lams []*syntax.LambdaExpr
	for {
		call, ok := syntax.Unparen(recv).(*syntax.CallExpr)
		if !ok {
			return lams
		}
		sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
		if !ok || sel.Sel == nil || !logosChainMethods[sel.Sel.Value] {
			return lams
		}
		for _, arg := range call.ArgList {
			if lam, ok := syntax.Unparen(arg).(*syntax.LambdaExpr); ok {
				lams = append(lams, lam)
			}
		}
		recv = sel.X
	}
}

func logosLambdaHasImpureCall(check *Checker, lam *syntax.LambdaExpr) bool {
	found := false
	syntax.Inspect(lam.Body, func(n syntax.Node) bool {
		if found {
			return false
		}
		if _, ok := n.(*syntax.FuncLit); ok {
			return false
		}
		call, ok := n.(*syntax.CallExpr)
		if !ok {
			return true
		}
		if !logosKnownPureCall(check, call) {
			found = true
			return false
		}
		return true
	})
	return found
}

func logosKnownPureCall(check *Checker, call *syntax.CallExpr) bool {
	name, ok := syntax.Unparen(call.Fun).(*syntax.Name)
	if !ok {
		return false
	}
	obj := check.lookup(name.Value)
	if obj == nil {
		return false
	}
	if _, ok := obj.(*TypeName); ok {
		return true
	}
	if b, ok := obj.(*Builtin); ok {
		switch b.Name() {
		case "copy", "clear", "delete", "append", "panic", "recover", "print", "println":
			return false
		}
		return true
	}
	return false
}

func (check *Checker) logosChainFuncInfer(arg syntax.Expr, elem Type) (Type, bool, bool) {
	var op operand
	check.expr(nil, &op, arg)
	if !op.isValid() {
		return nil, false, false
	}
	sig, ok := Unalias(op.typ()).(*Signature)
	if !ok || sig.Params().Len() != 1 || sig.Results().Len() != 1 {
		check.errorf(arg, IncompatibleAssign, "chain map needs func with one parameter and one result, have %s", op.typ())
		return nil, false, false
	}
	u := sig.Results().At(0).Type()
	fn := NewSignatureType(nil, nil, nil, NewTuple(NewParam(nopos, check.pkg, "", elem)), NewTuple(NewParam(nopos, check.pkg, "", u)), false)
	check.assignment(&op, fn, "chain map")
	return u, false, op.isValid()
}
