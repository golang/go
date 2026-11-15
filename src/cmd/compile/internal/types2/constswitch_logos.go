// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

const (
	logosUnreachableCase = "[comptime::unreachable_case]"
	logosSwitchNotType   = "[comptime::switch_not_type]"
	logosCaseNotType     = "[comptime::case_not_type]"
	logosTildeNotBasic   = "[comptime::tilde_not_basic]"
)

type logosCaseKind int

const (
	logosCaseExact logosCaseKind = iota
	logosCaseTilde
	logosCaseIface
)

type logosCaseInfo struct {
	syn  *syntax.ConstSwitchCase
	typ  Type
	kind logosCaseKind
	tpar *TypeParam
}

type LogosConstCase struct {
	Target *TypeName
	Vars   map[*Var]Type
}

type logosIfaceCase struct {
	recv  Type
	iface *Interface
	syn   *syntax.ConstSwitchCase
}

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
		if !check.logosCopy && check.logosCaseShadowed(info, seen, tag) {
			continue
		}
		seen = append(seen, *info)
		if check.logosDeadCopyCase(c, tag, info) {
			continue
		}
		check.logosConstCaseBody(inner, c, tag, info)
	}
}

func (check *Checker) logosDeadCopyCase(c *syntax.ConstSwitchCase, tag Type, info *logosCaseInfo) bool {
	if !check.logosCopy {
		return false
	}
	if _, ok := tag.(*TypeParam); ok {
		return false
	}
	return !logosConstMatches(tag, info.typ, c)
}

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

func isIfaceCase(t Type) bool {
	_, ok := Unalias(t).Underlying().(*Interface)
	return ok
}

func underBasic(t Type) (Type, bool) {
	if b, ok := Unalias(t).Underlying().(*Basic); ok {
		return b, true
	}
	return nil, false
}

func (check *Checker) logosCaseShadowed(info *logosCaseInfo, seen []logosCaseInfo, tag Type) bool {
	if !check.logosCaseCovered(info, seen, tag) {
		return false
	}
	check.errorf(info.syn.Type, InvalidSyntaxTree, "%s const switch case is shadowed by an earlier case", logosUnreachableCase)
	return true
}

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

func (check *Checker) logosCaseCoveredBy(prev, info *logosCaseInfo) bool {
	switch info.kind {
	case logosCaseExact:
		return check.logosCoversExact(prev, info.typ)
	case logosCaseTilde:
		return logosCoversTilde(prev, info.typ)
	}
	return prev.kind == logosCaseIface && logosMethodSetCovers(prev.iface(), info.iface())
}

func (check *Checker) logosCoversExact(prev *logosCaseInfo, typ Type) bool {
	switch prev.kind {
	case logosCaseExact:
		return Identical(prev.typ, typ)
	case logosCaseTilde:
		return logosUnderlyingIs(typ, prev.typ)
	}
	return check.implements(typ, prev.iface(), false, nil)
}

func logosCoversTilde(prev *logosCaseInfo, typ Type) bool {
	switch prev.kind {
	case logosCaseTilde:
		return Identical(prev.typ, typ)
	case logosCaseIface:
		return prev.iface().Empty()
	}
	return false
}

func (info *logosCaseInfo) iface() *Interface {
	iface, _ := Unalias(info.typ).Underlying().(*Interface)
	return iface
}

func (check *Checker) logosCaseSatisfiable(c, tag Type) bool {
	if tpar, ok := tag.(*TypeParam); ok {
		return check.implements(c, tpar.iface(), false, nil)
	}
	return Identical(c, tag)
}

func logosUnderlyingIs(t, b Type) bool {
	ub, ok := Unalias(t).Underlying().(*Basic)
	return ok && Identical(ub, b)
}

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

func (check *Checker) logosConstCaseBody(ctxt stmtContext, c *syntax.ConstSwitchCase, tag Type, info *logosCaseInfo) {
	if ctxt&continueOk != 0 {
		ctxt |= logosConstSwitchContinue
	}
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

func logosHasBreakConstSwitch(s *syntax.ConstSwitchStmt, label string, implicit bool) bool {
	for _, c := range s.Cases {
		if hasBreakList(c.Body, label, implicit) {
			return true
		}
	}
	return false
}

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
			sub := check.subst(pos, v.Type(), makeSubstMap([]*TypeParam{tpar}, []Type{info.typ}), nil, check.context())
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
