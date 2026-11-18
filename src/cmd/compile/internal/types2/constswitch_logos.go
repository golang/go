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
	pend bool // case type is an unbound f.Type: unknown until instantiation
	vary bool // case type is an outer loop's bound f.Type: concrete per iteration but varies across them
}

type LogosConstCase struct {
	Target *TypeName
	Vars   map[*Var]Type
}

type logosIfaceCase struct {
	recv  Type
	iface *Interface
	typ   Type
	syn   *syntax.ConstSwitchCase
}

func (check *Checker) logosConstSwitch(ctxt stmtContext, s *syntax.ConstSwitchStmt) {
	tag := check.logosConstTag(s.Targ)
	if tag == nil {
		return
	}
	// A tag resolved through a const-for field binding is concrete for
	// this iteration only: cases that cannot match it are dead for the
	// iteration, like copy-evaluation cases, not authoring errors.
	fieldTag := check.logosFieldBind != nil && check.logosFieldBound[s.Targ]
	// A tag resolved through an unbound field type, or through a field
	// of an enclosing outer loop, is the comptime any or varies per
	// outer iteration: reachability defers to the instantiations that
	// bind it.
	owner := check.logosFieldTypeOwner[syntax.Unparen(s.Targ)]
	tagPend := check.logosFieldTypePend(owner)
	narrow := owner
	inner := ctxt &^ (fallthroughOk | finalSwitchCase | inTypeSwitch)
	var seen []logosCaseInfo
	for _, c := range s.Cases {
		info := check.logosConstCase(c, tag)
		if info == nil {
			continue
		}
		// Field-bound tags vary per iteration, so the tag-satisfiable
		// clause does not apply; coverage among the cases themselves
		// still makes a case dead for every iteration.
		if !check.logosCopy && check.logosCaseShadowed(info, seen, tag, fieldTag, tagPend) {
			continue
		}
		seen = append(seen, *info)
		if check.logosDeadCopyCase(c, tag, info, fieldTag, seen[:len(seen)-1]) {
			continue
		}
		check.logosConstCaseBody(inner, c, tag, info, narrow)
	}
}

// logosDeadCopyCase answers whether a case body is dead for the current
// copy or field iteration: when the tag's type is known, only the first
// matching case ever runs it.
func (check *Checker) logosDeadCopyCase(c *syntax.ConstSwitchCase, tag Type, info *logosCaseInfo, fieldTag bool, seen []logosCaseInfo) bool {
	if !check.logosCopy && !fieldTag {
		return false
	}
	if _, ok := tag.(*TypeParam); ok {
		return false
	}
	for i := range seen {
		// A pending case resolves per iteration; it neither kills a
		// later case nor is killed by an earlier one statically.
		if seen[i].pend {
			continue
		}
		if logosConstMatches(tag, seen[i].typ, seen[i].syn) {
			return true
		}
	}
	if info.pend {
		return false
	}
	return !logosConstMatches(tag, info.typ, c)
}

// logosFieldTypeSel admits e as a type when it is f.Type of a const-for
// field variable, the same admission type positions get.
func (check *Checker) logosFieldTypeSel(e syntax.Expr, x *operand) bool {
	sel, ok := syntax.Unparen(e).(*syntax.SelectorExpr)
	return ok && check.logosFieldTypeType(sel, x)
}

func (check *Checker) logosConstTag(e syntax.Expr) Type {
	var x operand
	if !check.logosFieldTypeSel(e, &x) {
		check.exprOrType(&x, e, false)
	}
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
	if !check.logosFieldTypeSel(c.Type, &x) {
		check.exprOrType(&x, c.Type, true)
	}
	if !x.isValid() {
		return nil
	}
	if x.mode() != typexpr {
		check.errorf(c.Type, InvalidSyntaxTree, "%s const switch case needs a type, not a value", logosCaseNotType)
		return nil
	}
	check.record(&x)
	owner := check.logosFieldTypeOwner[syntax.Unparen(c.Type)]
	info := &logosCaseInfo{syn: c, typ: x.typ(), pend: check.logosFieldTypePend(owner), vary: check.logosFieldTypeVary(owner)}
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

// LogosConstMatch answers whether tag selects case c the way a const
// switch does (exact, ~basic, or interface match); the lowering uses it
// to pick the one case an unrolled field type runs.
func LogosConstMatch(tag, typ Type, c *syntax.ConstSwitchCase) bool {
	return logosConstMatches(tag, typ, c)
}

func logosConstMatches(tag Type, typ Type, c *syntax.ConstSwitchCase) bool {
	switch logosConstCaseKind(typ, c.Tilde) {
	case logosCaseTilde:
		ub, ok := Unalias(tag).Underlying().(*Basic)
		if !ok {
			return false
		}
		cb, ok := underBasic(typ)
		return ok && Identical(ub, cb)
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

func (check *Checker) logosCaseShadowed(info *logosCaseInfo, seen []logosCaseInfo, tag Type, fieldTag, tagPend bool) bool {
	if !check.logosCaseCovered(info, seen, tag, fieldTag, tagPend) {
		return false
	}
	check.errorf(info.syn.Type, InvalidSyntaxTree, "%s const switch case is shadowed by an earlier case", logosUnreachableCase)
	return true
}

func (check *Checker) logosCaseCovered(info *logosCaseInfo, seen []logosCaseInfo, tag Type, fieldTag, tagPend bool) bool {
	if info.pend || info.vary {
		// A pending or varying case resolves only per iteration; whether
		// earlier cases cover it varies with them, so it is never
		// statically covered.
		return false
	}
	if !fieldTag && info.kind == logosCaseExact && !check.logosCaseSatisfiable(info.typ, tag, tagPend) {
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
	if prev.pend || prev.vary {
		// An unbound or outer-bound f.Type case resolves only at
		// instantiation and covers nothing statically.
		return false
	}
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

func (check *Checker) logosCaseSatisfiable(c, tag Type, tagPend bool) bool {
	if tpar, ok := tag.(*TypeParam); ok {
		return check.implements(c, tpar.iface(), false, nil)
	}
	// An unbound field type is the comptime any: every case stays
	// reachable until an iteration binds it to a field's type. A plain
	// any tag does not defer: cases that cannot match it are dead.
	if tagPend && Identical(tag, logosComptimeAny) {
		return true
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

func (check *Checker) logosConstCaseBody(ctxt stmtContext, c *syntax.ConstSwitchCase, tag Type, info *logosCaseInfo, narrow *LogosConstFor) {
	if ctxt&continueOk != 0 {
		ctxt |= logosConstSwitchContinue
	}
	ctxt &= ^(breakOk | continueOk)
	if narrow != nil && !info.pend {
		// Inside `case T:` over an unbound f.Type, the field projects
		// as T: the case runs only for fields of that type. A bound
		// field keeps its own type under ~T; an unbound one narrows to
		// a parameter constrained by ~T so conversions to T check while
		// assignments to T still fail until the field binds.
		var ntyp Type
		switch {
		case info.kind != logosCaseTilde:
			ntyp = info.typ
		case Identical(tag, logosComptimeAny):
			tname := NewTypeName(info.syn.Type.Pos(), check.pkg, "T", nil)
			ntyp = NewTypeParam(tname, NewInterfaceType(nil, []Type{NewUnion([]*Term{NewTerm(true, info.typ)})}))
		}
		if ntyp != nil {
			old, had := check.logosFieldNarrow[narrow]
			if check.logosFieldNarrow == nil {
				check.logosFieldNarrow = make(map[*LogosConstFor]Type)
			}
			check.logosFieldNarrow[narrow] = ntyp
			defer func() {
				if had {
					check.logosFieldNarrow[narrow] = old
				} else {
					delete(check.logosFieldNarrow, narrow)
				}
			}()
		}
	}
	if info.kind == logosCaseIface {
		check.logosCSwitch = append(check.logosCSwitch, logosIfaceCase{recv: tag, iface: info.iface(), typ: info.typ, syn: c})
		defer func() { check.logosCSwitch = check.logosCSwitch[:len(check.logosCSwitch)-1] }()
	}
	check.openScope(c.Type, "const switch case")
	defer check.closeScope()
	if tpar, ok := tag.(*TypeParam); ok && info.kind != logosCaseIface {
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
	bind := info.typ
	var shadow *TypeName
	if info.kind == logosCaseTilde {
		// Inside a ~B case the switched type keeps its identity — a nested
		// const switch still dispatches on the actual type argument — but
		// gains the constraint ~B, so a value of a numeric type can be
		// converted to it and variables naming it keep their spelling.
		shadow = NewTypeName(pos, check.pkg, tpar.Obj().Name(), nil)
		bind = NewTypeParam(shadow, NewInterfaceType(nil, []Type{NewUnion([]*Term{NewTerm(true, info.typ)})}))
	} else {
		shadow = NewTypeName(pos, check.pkg, tpar.Obj().Name(), bind)
	}
	check.scope.Insert(shadow)
	rec := LogosConstCase{Target: shadow, Vars: make(map[*Var]Type)}
	seen := make(map[string]bool)
	for sc := check.scope.Parent(); sc != nil; sc = sc.Parent() {
		for _, objName := range sc.Names() {
			if seen[objName] {
				// A nearer scope already binds this name, so uses inside the
				// case resolve to that object, not to an outer variable.
				continue
			}
			seen[objName] = true
			v, ok := sc.Lookup(objName).(*Var)
			if !ok {
				continue
			}
			sub := check.subst(pos, v.Type(), makeSubstMap([]*TypeParam{tpar}, []Type{bind}), nil, check.context())
			if sub == v.Type() {
				continue
			}
			emit := sub
			if info.kind == logosCaseTilde {
				// The lowered code spells the rebound variable with the
				// basic case type (int, not the narrowed parameter), so the
				// rebinding puns the storage; the tilde match guarantees
				// identical layouts.
				emit = check.subst(pos, v.Type(), makeSubstMap([]*TypeParam{tpar}, []Type{info.typ}), nil, check.context())
			}
			shadowVar := NewVar(v.Pos(), v.Pkg(), v.Name(), sub)
			if alt := check.scope.Insert(shadowVar); alt != nil {
				continue
			}
			check.usedVars[shadowVar] = true
			rec.Vars[shadowVar] = emit
		}
	}
	if check.Info.LogosConstCases == nil {
		check.Info.LogosConstCases = make(map[*syntax.ConstSwitchCase]LogosConstCase)
	}
	check.Info.LogosConstCases[info.syn] = rec
}

// logosConstIfaceAssign admits a value of the switched type where the active
// interface case's type is required: inside `case I:` a value of type T keeps
// its concrete layout and gains I's method set, so it can be stored or passed
// as I (SPEC Sec 2.5). The operand is recorded so the lowerer boxes it as
// rt.Box[I](x) at the use site; the type name T itself is not rebound.
func (check *Checker) logosConstIfaceAssign(x *operand, T Type) bool {
	if !check.conf.Logos || len(check.logosCSwitch) == 0 || x.expr == nil {
		return false
	}
	top := check.logosCSwitch[len(check.logosCSwitch)-1]
	if x.typ() != top.recv || !Identical(T, top.typ) {
		return false
	}
	if check.Info.LogosConstConv == nil {
		check.Info.LogosConstConv = make(map[syntax.Expr]*syntax.ConstSwitchCase)
	}
	check.Info.LogosConstConv[x.expr] = top.syn
	return true
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
