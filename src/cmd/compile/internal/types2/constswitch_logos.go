// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

// LoGo/s M7-S4 `const switch` checking (SPEC Sec 2.5, T-205). Each case
// body checks with the switched type replaced by the case type for exact
// and tilde cases, or with the interface methods gained for interface
// cases. Shadowing follows source order: a case whose match set is
// covered by earlier cases reports unreachable_case.

const (
	logosUnreachableCase = "[comptime::unreachable_case]"
	logosSwitchNotType   = "[comptime::switch_not_type]"
	logosCaseNotType     = "[comptime::case_not_type]"
	logosTildeNotBasic   = "[comptime::tilde_not_basic]"
)

// logosCaseKind classifies one const-switch case.
type logosCaseKind int

const (
	logosCaseExact logosCaseKind = iota
	logosCaseTilde
	logosCaseIface
)

// logosCaseInfo is one checked const-switch case: the resolved case type
// (the base type for tilde cases), its kind, and the switched type
// parameter, or nil for a concrete tag.
type logosCaseInfo struct {
	syn  *syntax.ConstSwitchCase
	typ  Type
	kind logosCaseKind
	tpar *TypeParam
}

// LogosConstCase is the checked case-scope substitution of one `const
// switch` case (SPEC Sec 2.5, T-205). Target is the case-scope name of the
// switched type, and Vars maps each case-scope variable whose type names
// the switched type to its substituted type. Body names resolve to these
// shadow objects, so lowering matches them by identity. Only an exact case
// of a type parameter records anything: tilde and interface cases keep the
// switched type in their body (SPEC Sec 2.5 rebinds exact cases only).
type LogosConstCase struct {
	Target *TypeName
	Vars   map[*Var]Type
}

// logosIfaceCase is one active interface-case context for the selector
// fallback in call.go: receivers of the switched type gain the methods
// of iface inside the case body.
type logosIfaceCase struct {
	recv  Type
	iface *Interface
	syn   *syntax.ConstSwitchCase
}

// logosConstSwitch checks a `const switch` statement. The tag must denote
// a type; each case resolves to exact, tilde, or interface, reports
// shadowing, and checks its body under the case substitution.
func (check *Checker) logosConstSwitch(ctxt stmtContext, s *syntax.ConstSwitchStmt) {
	tag := check.logosConstTag(s.Targ)
	if tag == nil {
		return
	}
	inner := ctxt &^ (fallthroughOk | finalSwitchCase | inTypeSwitch)
	var seen []logosCaseInfo
	for _, c := range s.Cases {
		info := check.logosConstCase(c, tag)
		if info == nil {
			continue
		}
		if check.logosCaseShadowed(info, seen, tag) {
			continue
		}
		seen = append(seen, *info)
		check.logosConstCaseBody(inner, c, tag, info)
	}
}

// logosConstTag resolves the switched type. It reports switch_not_type
// when the tag is not a type and returns nil. The tag type is recorded
// for lowering, which reads it back from info.
func (check *Checker) logosConstTag(e syntax.Expr) Type {
	var x operand
	check.exprOrType(&x, e, false)
	if !x.isValid() {
		return nil
	}
	if x.mode() != typexpr {
		check.errorf(e, InvalidSyntaxTree, "%s const switch needs a type, not a value", logosSwitchNotType)
		return nil
	}
	check.record(&x)
	return x.typ()
}

// logosConstCase resolves one case to its kind. It returns nil for an
// invalid case, after reporting.
func (check *Checker) logosConstCase(c *syntax.ConstSwitchCase, tag Type) *logosCaseInfo {
	var x operand
	check.exprOrType(&x, c.Type, true)
	if !x.isValid() {
		return nil
	}
	if x.mode() != typexpr {
		check.errorf(c.Type, InvalidSyntaxTree, "%s const switch case needs a type, not a value", logosCaseNotType)
		return nil
	}
	check.record(&x)
	info := &logosCaseInfo{syn: c, typ: x.typ()}
	if tpar, ok := tag.(*TypeParam); ok {
		info.tpar = tpar
	}
	info.kind = logosConstCaseKind(x.typ(), c.Tilde)
	if info.kind == logosCaseTilde {
		base, ok := underBasic(x.typ())
		if !ok {
			check.errorf(c.TildePos, InvalidSyntaxTree, "%s const switch tilde case needs a basic type, not %s", logosTildeNotBasic, x.typ())
			return nil
		}
		info.typ = base
	}
	return info
}

// logosConstCaseKind classifies one const-switch case from its resolved
// type: a tilde case, an interface case, or an exact case. Checking and
// compile-time selection share it, so both agree on every case.
func logosConstCaseKind(typ Type, tilde bool) logosCaseKind {
	switch {
	case tilde:
		return logosCaseTilde
	case isIfaceCase(typ):
		return logosCaseIface
	default:
		return logosCaseExact
	}
}

// LogosConstSelectedCase reports the case a `const switch` statement selects
// at compile time: the first case matching the tag, with exact cases by
// identity, tilde cases by the underlying basic type, and interface cases by
// implementation. It returns static false when selection is not static: the
// tag is a type parameter, so the switch dispatches at run time, or a type
// the checker did not record. A nil case with static true means no case
// matches and the statement folds away.
func LogosConstSelectedCase(info *Info, s *syntax.ConstSwitchStmt) (sel *syntax.ConstSwitchCase, static bool) {
	tag := logosConstTypeOf(info, s.Targ)
	if tag == nil {
		return nil, false
	}
	if _, ok := tag.(*TypeParam); ok {
		return nil, false
	}
	for _, c := range s.Cases {
		typ := logosConstTypeOf(info, c.Type)
		if typ == nil {
			return nil, false
		}
		if logosConstMatches(tag, typ, c) {
			return c, true
		}
	}
	return nil, true
}

// logosConstTypeOf returns the checked type denoted by e, resolving a type
// name through its declared object, or nil when the checker recorded none.
func logosConstTypeOf(info *Info, e syntax.Expr) Type {
	var name *syntax.Name
	switch e := e.(type) {
	case *syntax.Name:
		name = e
	case *syntax.SelectorExpr:
		name = e.Sel
	}
	if name != nil {
		if tn, ok := info.Uses[name].(*TypeName); ok {
			return tn.Type()
		}
	}
	return info.TypeOf(e)
}

// logosConstMatches reports whether the concrete tag selects the case of
// resolved type typ.
func logosConstMatches(tag Type, typ Type, c *syntax.ConstSwitchCase) bool {
	switch logosConstCaseKind(typ, c.Tilde) {
	case logosCaseTilde:
		ub, ok := Unalias(tag).Underlying().(*Basic)
		return ok && Identical(ub, typ)
	case logosCaseIface:
		iface, _ := Unalias(typ).Underlying().(*Interface)
		return iface != nil && Implements(tag, iface)
	default:
		return Identical(typ, tag)
	}
}

// isIfaceCase reports whether a case type is an interface case: any
// interface type, including an empty one.
func isIfaceCase(t Type) bool {
	_, ok := Unalias(t).Underlying().(*Interface)
	return ok
}

// underBasic returns the basic type underlying t, reporting whether t
// denotes one. Only basic bases admit tilde cases: their kind decides
// the run-time match exactly.
func underBasic(t Type) (Type, bool) {
	if b, ok := Unalias(t).Underlying().(*Basic); ok {
		return b, true
	}
	return nil, false
}

// logosCaseShadowed reports unreachable_case when info matches only types
// already covered by earlier cases, or nothing at all.
func (check *Checker) logosCaseShadowed(info *logosCaseInfo, seen []logosCaseInfo, tag Type) bool {
	if !check.logosCaseCovered(info, seen, tag) {
		return false
	}
	check.errorf(info.syn.Type, InvalidSyntaxTree, "%s const switch case is shadowed by an earlier case", logosUnreachableCase)
	return true
}

// logosCaseCovered reports whether no type can reach info: an exact case
// that cannot satisfy the switched constraint is dead on arrival, and any
// case is dead once one earlier case covers it. Coverage follows the
// lowering precedence: exact cases first in source order, then tilde
// cases, then interface cases.
func (check *Checker) logosCaseCovered(info *logosCaseInfo, seen []logosCaseInfo, tag Type) bool {
	if info.kind == logosCaseExact && !check.logosCaseSatisfiable(info.typ, tag) {
		return true
	}
	for i := range seen {
		if check.logosCaseCoveredBy(&seen[i], info) {
			return true
		}
	}
	return false
}

// logosCaseCoveredBy reports whether every type that reaches info already
// matches the earlier case prev.
func (check *Checker) logosCaseCoveredBy(prev, info *logosCaseInfo) bool {
	switch info.kind {
	case logosCaseExact:
		return check.logosCoversExact(prev, info.typ)
	case logosCaseTilde:
		return logosCoversTilde(prev, info.typ)
	}
	return prev.kind == logosCaseIface && logosMethodSetCovers(prev.iface(), info.iface())
}

// logosCoversExact reports whether prev matches the exact type typ.
func (check *Checker) logosCoversExact(prev *logosCaseInfo, typ Type) bool {
	switch prev.kind {
	case logosCaseExact:
		return Identical(prev.typ, typ)
	case logosCaseTilde:
		return logosUnderlyingIs(typ, prev.typ)
	}
	return check.implements(typ, prev.iface(), false, nil)
}

// logosCoversTilde reports whether prev matches every type of the tilde
// case on the basic type typ.
func logosCoversTilde(prev *logosCaseInfo, typ Type) bool {
	switch prev.kind {
	case logosCaseTilde:
		return Identical(prev.typ, typ)
	case logosCaseIface:
		return prev.iface().Empty()
	}
	return false
}

// iface returns the case type known to be an interface.
func (info *logosCaseInfo) iface() *Interface {
	iface, _ := Unalias(info.typ).Underlying().(*Interface)
	return iface
}

// logosCaseSatisfiable reports whether the exact case type could be the
// switched type: for a type parameter tag, the case must satisfy its
// constraint; for a concrete tag, it must be identical to it.
func (check *Checker) logosCaseSatisfiable(c, tag Type) bool {
	if tpar, ok := tag.(*TypeParam); ok {
		return check.implements(c, tpar.iface(), false, nil)
	}
	return Identical(c, tag)
}

// logosUnderlyingIs reports whether the underlying type of t is the basic
// type b.
func logosUnderlyingIs(t, b Type) bool {
	ub, ok := Unalias(t).Underlying().(*Basic)
	return ok && Identical(ub, b)
}

// logosMethodSetCovers reports whether every implementer of sub also
// implements super: the methods of super are all present in sub.
func logosMethodSetCovers(super, sub *Interface) bool {
	if super == nil || sub == nil {
		return false
	}
	for i := range super.NumMethods() {
		m := super.Method(i)
		obj, _, _ := lookupFieldOrMethod(sub, false, nil, m.Name(), false)
		fn, ok := obj.(*Func)
		if !ok || !Identical(fn.Type(), m.Type()) {
			return false
		}
	}
	return true
}

// logosConstCaseBody checks one case body in a scope of its own. An exact
// case of a type parameter runs under the substitution of the parameter by
// the case type, an interface case runs with the case methods gained on the
// switched type, and a tilde case keeps the switched type, so a value of
// it needs a conversion to reach the basic type.
// Unlabeled break and continue stay outside the case: lowering rewrites
// the construct around the body, so only labeled jumps and jumps local
// to nested statements keep their meaning.
func (check *Checker) logosConstCaseBody(ctxt stmtContext, c *syntax.ConstSwitchCase, tag Type, info *logosCaseInfo) {
	ctxt &= ^(breakOk | continueOk)
	if info.kind == logosCaseIface {
		check.logosCSwitch = append(check.logosCSwitch, logosIfaceCase{recv: tag, iface: info.iface(), syn: c})
		defer func() { check.logosCSwitch = check.logosCSwitch[:len(check.logosCSwitch)-1] }()
	}
	check.openScope(c.Type, "const switch case")
	defer check.closeScope()
	if tpar, ok := tag.(*TypeParam); ok && info.kind == logosCaseExact {
		check.logosShadowConstTarget(tpar, info)
	}
	check.stmtList(ctxt, c.Body)
}

// logosTerminatingConstSwitch reports whether a `const switch`
// terminates: only a concrete tag with a terminating match, since any
// other tag may miss every case and fall through.
func (check *Checker) logosTerminatingConstSwitch(s *syntax.ConstSwitchStmt) bool {
	tag := check.Info.TypeOf(s.Targ)
	if tag == nil || isTypeParam(tag) {
		return false
	}
	for _, c := range s.Cases {
		if !check.logosConcreteCaseMatches(c, tag) {
			continue
		}
		return check.isTerminatingList(c.Body, "")
	}
	return false
}

// logosConcreteCaseMatches reports whether a concrete tag matches one
// source case: identical exact types, tilde bases equal to the tag
// underlying, or implemented interfaces.
func (check *Checker) logosConcreteCaseMatches(c *syntax.ConstSwitchCase, tag Type) bool {
	ctv := check.Info.TypeOf(c.Type)
	if ctv == nil {
		return false
	}
	if c.Tilde {
		ub, ok := Unalias(tag).Underlying().(*Basic)
		return ok && Identical(ub, ctv)
	}
	if isIfaceCase(ctv) {
		iface, _ := Unalias(ctv).Underlying().(*Interface)
		return iface != nil && Implements(tag, iface)
	}
	return Identical(ctv, tag)
}

// logosHasBreakConstSwitch reports whether any case body holds a break
// matching the label search. No case consumes a break.
func logosHasBreakConstSwitch(s *syntax.ConstSwitchStmt, label string, implicit bool) bool {
	for _, c := range s.Cases {
		if hasBreakList(c.Body, label, implicit) {
			return true
		}
	}
	return false
}

// logosShadowConstTarget declares the case substitution of an exact case in
// the current scope: the name of the switched type parameter resolves to the
// case type, and every visible variable whose type mentions the parameter
// gains a shadow with the substituted type. Lowering rebinds the same set,
// so checking and lowering agree.
func (check *Checker) logosShadowConstTarget(tpar *TypeParam, info *logosCaseInfo) {
	pos := info.syn.Type.Pos()
	shadow := NewTypeName(pos, check.pkg, tpar.Obj().Name(), info.typ)
	check.scope.Insert(shadow)
	rec := LogosConstCase{Target: shadow, Vars: make(map[*Var]Type)}
	for sc := check.scope.Parent(); sc != nil; sc = sc.Parent() {
		for _, objName := range sc.Names() {
			v, ok := sc.Lookup(objName).(*Var)
			if !ok {
				continue
			}
			sub := logosSubstOne(check, pos, v.Type(), tpar, info.typ)
			if sub == v.Type() {
				continue
			}
			shadowVar := NewVar(v.Pos(), v.Pkg(), v.Name(), sub)
			if alt := check.scope.Insert(shadowVar); alt != nil {
				continue
			}
			check.usedVars[shadowVar] = true
			rec.Vars[shadowVar] = sub
		}
	}
	if check.Info.LogosConstCases == nil {
		check.Info.LogosConstCases = make(map[*syntax.ConstSwitchCase]LogosConstCase)
	}
	check.Info.LogosConstCases[info.syn] = rec
}

// logosConstIfaceSelect resolves a selector on the switched type against
// the active interface case. It reports true when it handled the
// selection, in which case the caller skips the upstream error. The
// resolved interface type is recorded for lowering, which rewrites the
// receiver to an assertion.
func (check *Checker) logosConstIfaceSelect(x *operand, e *syntax.SelectorExpr, sel string) bool {
	if !check.conf.Logos || len(check.logosCSwitch) == 0 {
		return false
	}
	top := check.logosCSwitch[len(check.logosCSwitch)-1]
	if x.typ() != top.recv {
		return false
	}
	obj, _, _ := lookupFieldOrMethod(top.iface, false, check.pkg, sel, false)
	fn, ok := obj.(*Func)
	if !ok {
		return false
	}
	check.objDecl(fn)
	sig := *fn.Type().(*Signature)
	sig.recvold = sig.recv
	sig.recv = nil
	x.mode_ = value
	x.typ_ = &sig
	x.expr = e
	check.recordUse(e.Sel, fn)
	if check.Info.LogosConstSel == nil {
		check.Info.LogosConstSel = make(map[*syntax.SelectorExpr]*syntax.ConstSwitchCase)
	}
	check.Info.LogosConstSel[e] = top.syn
	return true
}
