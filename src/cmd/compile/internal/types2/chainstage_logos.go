// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
	"slices"
)

const logosNamedArgument = "[correctness::named_argument] a named argument is allowed only as limit = N in a group stage call"

func (check *Checker) logosNamedArgs(args []syntax.Expr) bool {
	found := false
	for _, arg := range args {
		if kv, ok := arg.(*syntax.KeyValueExpr); ok {
			check.error(kv, InvalidCall, logosNamedArgument)
			check.use(kv.Value)
			found = true
		}
	}
	return found
}

func (check *Checker) logosNamedArgCall(x *operand, call *syntax.CallExpr) bool {
	if !slices.ContainsFunc(call.ArgList, func(arg syntax.Expr) bool {
		_, named := arg.(*syntax.KeyValueExpr)
		return named
	}) {
		return false
	}
	if check.logosUseCallee(call.Fun) {
		check.logosNamedArgs(call.ArgList)
	} else {
		for _, arg := range call.ArgList {
			if kv, ok := arg.(*syntax.KeyValueExpr); ok {
				check.use(kv.Value)
			}
		}
	}
	check.logosUsePositional(call.ArgList)
	logosStageInvalid(x, call)
	return true
}

func (check *Checker) logosUseCallee(fun syntax.Expr) bool {
	sel, ok := syntax.Unparen(fun).(*syntax.SelectorExpr)
	if !ok || !logosChainMethods[sel.Sel.Value] {
		return check.use(fun)
	}
	var rx operand
	check.exprOrType(&rx, sel.X, false)
	if !rx.isValid() {
		return false
	}
	if _, ok := check.logosChainSlice(&rx, sel.Sel.Value); ok {
		return true
	}
	return check.use(fun)
}

type logosStageForm int

const (
	logosStageValue logosStageForm = iota
	logosStageRange
)

// LogosGroupStage reports whether call is a group stage call, src.group(ctx,
// limit = N) or the .unordered() directly after it, and returns the element
// type of src. src may be a slice, an array, an iter.Seq[T], a receive
// channel, or a named type of those, and must have no field or method named
// group. The receiver types are read from info, so call must come from a
// package that has been checked.
func LogosGroupStage(pkg *Package, info *Info, call *syntax.CallExpr) (Type, bool) {
	sel := logosStageSelector(call)
	if sel == nil {
		return nil, false
	}
	switch sel.Sel.Value {
	case "group":
		recv := info.TypeOf(sel.X)
		if recv == nil {
			return nil, false
		}
		return logosStageSource(pkg, recv)
	case "unordered":
		prev, ok := syntax.Unparen(sel.X).(*syntax.CallExpr)
		if !ok {
			return nil, false
		}
		if psel := logosStageSelector(prev); psel == nil || psel.Sel.Value != "group" {
			return nil, false
		}
		return LogosGroupStage(pkg, info, prev)
	}
	return nil, false
}

func logosStageSource(pkg *Package, recv Type) (Type, bool) {
	if c := logosAsConstView(recv); c != nil {
		recv = c.Elem
	}
	if obj, _, _ := lookupFieldOrMethod(recv, true, pkg, "group", false); obj != nil {
		return nil, false
	}
	switch u := recv.Underlying().(type) {
	case *Slice:
		return u.elem, true
	case *Array:
		return u.elem, true
	case *Chan:
		return u.elem, u.dir != SendOnly
	case *Signature:
		return logosSeqElem(u)
	}
	return nil, false
}

func logosSeqElem(seq *Signature) (Type, bool) {
	if seq.params.Len() != 1 || seq.results.Len() != 0 || seq.variadic {
		return nil, false
	}
	yield, ok := seq.params.At(0).typ.Underlying().(*Signature)
	if !ok || yield.params.Len() != 1 || yield.results.Len() != 1 || yield.variadic {
		return nil, false
	}
	if !Identical(yield.results.At(0).typ, Typ[Bool]) {
		return nil, false
	}
	return yield.params.At(0).typ, true
}

func logosStageSelector(call *syntax.CallExpr) *syntax.SelectorExpr {
	sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr)
	if !ok || sel.Sel == nil {
		return nil
	}
	return sel
}

func logosStageCalls(top *syntax.CallExpr) []*syntax.CallExpr {
	var calls []*syntax.CallExpr
	for cur := top; ; {
		sel := logosStageSelector(cur)
		if sel == nil {
			return nil
		}
		name := sel.Sel.Value
		if name != "group" && name != "unordered" && !logosChainMethods[name] {
			return nil
		}
		calls = append(calls, cur)
		if name == "group" {
			slices.Reverse(calls)
			return calls
		}
		next, ok := syntax.Unparen(sel.X).(*syntax.CallExpr)
		if !ok {
			return nil
		}
		cur = next
	}
}

func logosStageInvalid(x *operand, top *syntax.CallExpr) {
	x.mode_ = invalid
	x.expr = top
	x.typ_ = Typ[Invalid]
}

func (check *Checker) logosStageChain(x *operand, top *syntax.CallExpr, form logosStageForm) bool {
	calls := logosStageCalls(top)
	if calls == nil {
		return false
	}
	var recv operand
	check.exprOrType(&recv, logosStageSelector(calls[0]).X, false)
	if !recv.isValid() {
		logosStageInvalid(x, top)
		return true
	}
	elem, ok := logosStageSource(check.pkg, recv.typ())
	if !ok || recv.mode() == typexpr {
		return false
	}
	if !check.logosStageRun(x, calls, elem, form) {
		logosStageInvalid(x, top)
	}
	return true
}

func (check *Checker) logosStageRun(x *operand, calls []*syntax.CallExpr, elem Type, form logosStageForm) bool {
	head, top := calls[0], calls[len(calls)-1]
	if !check.logosStageHead(head) {
		return false
	}
	var cur Type = NewSlice(elem)
	check.record(&operand{mode_: value, expr: head, typ_: cur})
	rest := calls[1:]
	if len(rest) > 0 && logosStageMethod(rest[0]) == "unordered" {
		if check.logosNamedArgs(rest[0].ArgList) || !check.logosChainArity(rest[0], rest[0].ArgList, 0) {
			return false
		}
		check.record(&operand{mode_: value, expr: rest[0], typ_: cur})
		rest = rest[1:]
	}
	if i := slices.IndexFunc(rest, func(call *syntax.CallExpr) bool { return logosStageMethod(call) == "unordered" }); i >= 0 {
		check.error(logosStageSelector(rest[i]).Sel, InvalidCall, "unordered must directly follow group")
		return false
	}
	n := logosSegmentLen(rest)
	if n == 0 {
		check.error(logosStageSelector(head).Sel, InvalidCall, "group stage needs map or filter after it")
		return false
	}
	saved := check.sig
	check.sig = check.logosStageSignature()
	cur, ok := check.logosStageSteps(x, rest[:n], top, cur)
	check.sig = saved
	if !ok {
		return false
	}
	if form == logosStageRange {
		return check.logosStageRangeOperand(x, head, rest[n:], cur)
	}
	if cur, ok = check.logosStageSteps(x, rest[n:], top, cur); !ok {
		return false
	}
	x.typ_ = NewTuple(newVar(ResultVar, top.Pos(), check.pkg, "", cur), newVar(ResultVar, top.Pos(), check.pkg, "", universeError))
	return true
}

func logosStageMethod(call *syntax.CallExpr) string {
	return logosStageSelector(call).Sel.Value
}

func logosSegmentLen(calls []*syntax.CallExpr) int {
	n := 0
	for n < len(calls) && (logosStageMethod(calls[n]) == "map" || logosStageMethod(calls[n]) == "filter") {
		n++
	}
	return n
}

func (check *Checker) logosStageSignature() *Signature {
	result := newVar(ResultVar, nopos, check.pkg, "", universeError)
	return NewSignatureType(nil, nil, nil, nil, NewTuple(result), false)
}

func (check *Checker) logosStageSteps(x *operand, steps []*syntax.CallExpr, top *syntax.CallExpr, cur Type) (Type, bool) {
	for _, call := range steps {
		*x = operand{}
		sel := logosStageSelector(call)
		if check.logosNamedArgs(call.ArgList) {
			return nil, false
		}
		slice, ok := cur.(*Slice)
		if !ok {
			check.errorf(sel.Sel, InvalidCall, "chain method %s cannot follow a terminal", sel.Sel.Value)
			return nil, false
		}
		check.logosChainMethod(x, call, sel, slice.elem)
		if !x.isValid() {
			return nil, false
		}
		cur = x.typ()
		if call != top {
			check.record(x)
		}
	}
	return cur, true
}

func (check *Checker) logosStageRangeOperand(x *operand, head *syntax.CallExpr, rest []*syntax.CallExpr, cur Type) bool {
	if len(rest) > 0 {
		check.error(logosStageSelector(rest[0]).Sel, InvalidRangeExpr, "only map and filter may follow group in a range clause")
		return false
	}
	if !check.logosEnclosingError() {
		check.error(logosStageSelector(head).Sel, WrongResultCount, logosGroupNeedsError)
	}
	yield := NewSignatureType(nil, nil, nil, NewTuple(NewParam(nopos, check.pkg, "", cur.(*Slice).elem)), NewTuple(NewParam(nopos, check.pkg, "", Typ[Bool])), false)
	x.typ_ = NewSignatureType(nil, nil, nil, NewTuple(NewParam(nopos, check.pkg, "yield", yield)), nil, false)
	return true
}

func (check *Checker) logosRangeExpr(x *operand, e syntax.Expr) bool {
	if !check.conf.Logos {
		return false
	}
	call, ok := syntax.Unparen(e).(*syntax.CallExpr)
	if !ok || !check.logosStageChain(x, call, logosStageRange) {
		return false
	}
	check.record(x)
	return true
}

func (check *Checker) logosStageHead(head *syntax.CallExpr) bool {
	sel := logosStageSelector(head)
	args := head.ArgList
	if hasDots(head) {
		check.error(sel.Sel, InvalidDotDotDot, "invalid use of ... in group stage call")
		return false
	}
	if len(args) == 0 {
		check.error(sel.Sel, WrongArgCount, "group stage requires a context and limit = N")
		return false
	}
	ok := !check.logosNamedArgs(args[:1]) && check.logosContextArg(args[0], "group stage context")
	if len(args) == 1 {
		check.error(sel.Sel, WrongArgCount, "group stage requires limit = N")
		return false
	}
	ok = check.logosStageLimit(args[1]) && ok
	if len(args) > 2 {
		if !check.logosNamedArgs(args[2:]) {
			check.error(args[2], WrongArgCount, "too many arguments in group stage call")
		}
		check.logosUsePositional(args[2:])
		return false
	}
	return ok
}

func (check *Checker) logosUsePositional(args []syntax.Expr) {
	for _, arg := range args {
		switch arg.(type) {
		case *syntax.KeyValueExpr, *syntax.LambdaExpr:
		default:
			check.use(arg)
		}
	}
}

func (check *Checker) logosStageLimit(arg syntax.Expr) bool {
	kv, ok := arg.(*syntax.KeyValueExpr)
	if !ok {
		check.use(arg)
		check.error(arg, InvalidCall, "group stage limit must be written limit = N")
		return false
	}
	if key, _ := kv.Key.(*syntax.Name); key == nil || key.Value != "limit" {
		check.logosNamedArgs([]syntax.Expr{kv})
		return false
	}
	var limit operand
	check.expr(nil, &limit, kv.Value)
	if !limit.isValid() {
		return false
	}
	check.assignment(&limit, Typ[Int], "group limit")
	return limit.isValid()
}
