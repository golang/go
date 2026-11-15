// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"strings"

	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

const (
	logosOpenUnion   = "[correctness::open_union_value]"
	logosNonMember   = "[correctness::union_nonmember]"
	logosMemberLacks = "[correctness::union_member_method]"
	logosNonExhaust  = "[correctness::nonexhaustive_switch]"
	logosArmMismatch = "[correctness::switch_arm_mismatch]"
	logosNeedDefault = "[correctness::switch_expr_needs_default]"
	logosAssertFail  = "[correctness::assertion_failed]"
)

func logosIsClosed(tset *_TypeSet) bool {
	if !tset.hasTerms() {
		return false
	}
	for _, t := range tset.terms {
		if t.tilde {
			return false
		}
	}
	return true
}

func logosClosedUnion(t Type) *_TypeSet {
	if t == nil || isTypeParam(t) {
		return nil
	}
	iface, _ := t.Underlying().(*Interface)
	if iface == nil {
		return nil
	}
	tset := iface.typeSet()
	if !logosIsClosed(tset) {
		return nil
	}
	return tset
}

func logosMemberOf(t Type, tset *_TypeSet) bool {
	for _, term := range tset.terms {
		if Identical(term.typ, t) {
			return true
		}
	}
	return false
}

func (check *Checker) logosUnionsOK() bool {
	return check.conf.Logos || check.conf.LogosUnions
}

func (check *Checker) logosUnionMethods(ityp *Interface, iface *syntax.InterfaceType) {
	if !check.logosUnionsOK() || ityp.typeSet().NumMethods() == 0 {
		return
	}
	next := 0
	for _, f := range iface.MethodList {
		if f.Name != nil {
			continue
		}
		u, _ := ityp.embeddeds[next].(*Union)
		next++
		if u == nil {
			continue
		}
		_, tlist := flattenUnion(nil, f.Type)
		for i, t := range u.terms {
			if i < len(tlist) {
				check.logosTermMethods(ityp, t, tlist[i])
			}
		}
	}
}

func (check *Checker) logosTermMethods(ityp *Interface, t *Term, at syntax.Expr) {
	if t.tilde || !isValid(t.typ) {
		return
	}
	members := []Type{t.typ}
	if nested, _ := t.typ.Underlying().(*Interface); nested != nil {
		members = members[:0]
		for _, term := range nested.typeSet().terms {
			if !term.tilde {
				members = append(members, term.typ)
			}
		}
	}
	for _, m := range members {
		var cause string
		if !check.hasAllMethods(m, ityp, true, Identical, &cause) {
			check.errorf(at, InvalidUnion, "%s union member %s does not implement the methods of the union %s", logosMemberLacks, m, cause)
		}
	}
}

func (check *Checker) logosValueType(pos syntax.Pos, typ Type, tset *_TypeSet) bool {
	if !check.logosUnionsOK() {
		return false
	}
	switch {
	case logosIsClosed(tset):
		return true
	case tset.hasTerms() || tset.comparable:
		check.softErrorf(pos, MisplacedConstraintIface, "%s %s has ~ terms or an unbounded type set", logosOpenUnion, typ)
		return true
	}
	return false
}

func (check *Checker) logosConversionUnion(T Type) bool {
	return check.logosUnionsOK() && logosClosedUnion(T) != nil
}

func (check *Checker) logosUnionDefault(x *operand, u *Interface) (Type, bool) {
	if !check.logosUnionsOK() || !logosIsClosed(u.typeSet()) {
		return nil, false
	}
	if check.logosReadonlyBoundary(x, u, "union default") {
		return nil, false
	}
	d := Default(x.typ())
	if !check.implements(d, u, false, nil) {
		return nil, false
	}
	return d, true
}

func (check *Checker) logosUnionConversion(x *operand, T Type) bool {
	if !check.logosConversionUnion(T) {
		return false
	}
	if check.logosReadonlyBoundary(x, T, "union conversion") {
		return true
	}
	if ok, _ := x.assignableTo(check, T, nil); ok {
		return false
	}
	from := x.typ()
	if isUntyped(from) {
		from = Default(from)
	}
	check.errorf(x, InvalidConversion, "%s %s is not a member of %s", logosNonMember, from, T)
	x.invalidate()
	return true
}

func (check *Checker) logosNonMemberAssert(e syntax.Expr, x *operand, T Type) bool {
	if !check.conf.Logos {
		return false
	}
	if logosContainsReadonlyDescendants(x.typ()) {
		probe := *x
		if check.logosReadonlyBoundary(&probe, T, "type assertion") && !probe.isValid() {
			x.invalidate()
			return true
		}
	}
	tset := logosClosedUnion(x.typ())
	if tset == nil || isTypeParam(T) || logosMemberOf(T, tset) {
		return false
	}
	if _, ok := e.(*syntax.AssertExpr); ok {
		return true
	}
	if IsInterface(T) {
		if iface, ok := T.Underlying().(*Interface); ok {
			if ttset := iface.typeSet(); ttset != nil && ttset.hasTerms() && !ttset.subsetOf(tset) {
				check.errorf(e, ImpossibleAssert, "%s %s is not a member of %s", logosNonMember, T, x.typ())
				return true
			}
		}
		return false
	}
	check.errorf(e, ImpossibleAssert, "%s %s is not a member of %s", logosNonMember, T, x.typ())
	return true
}
func (check *Checker) logosUnionAssign(x *operand, T Type, context string) bool {
	if !check.conf.Logos {
		return false
	}
	tset := logosClosedUnion(T)
	if tset == nil {
		return false
	}
	if check.logosReadonlyBoundary(x, T, context) {
		return true
	}
	from := x.typ()
	if from == nil {
		return false
	}
	if isUntyped(from) {
		from = Default(from)
	}
	if logosMemberOf(from, tset) {
		return false
	}
	check.errorf(x, IncompatibleAssign, "%s cannot use %s as %s value in %s: %s is not a member of %s", logosNonMember, x, T, context, from, T)
	x.invalidate()
	return true
}

func logosUnionShape(t Type) ([]Type, bool, bool) {
	if t == nil {
		return nil, false, false
	}
	iface, ok := Unalias(t).Underlying().(*Interface)
	if !ok {
		return nil, false, false
	}
	var out []Type
	for i := range iface.NumEmbeddeds() {
		u, ok := iface.EmbeddedType(i).(*Union)
		if !ok {
			continue
		}
		for j := range u.Len() {
			term := u.Term(j)
			if term.Tilde() {
				return nil, true, false
			}
			out = append(out, term.Type())
		}
	}
	if len(out) == 0 {
		return nil, false, false
	}
	return out, true, true
}

func LogosUnionTerms(t Type) ([]Type, bool) {
	direct, isUnion, closed := logosUnionShape(t)
	if !isUnion || !closed {
		return nil, false
	}
	var out []Type
	var expanding []Type
	var expand func(m Type) bool
	expand = func(m Type) bool {
		sub, isSub, closedSub := logosUnionShape(m)
		if !isSub {
			for _, e := range out {
				if Identical(e, m) {
					return true
				}
			}
			out = append(out, m)
			return true
		}
		if !closedSub {
			return false
		}
		for _, e := range expanding {
			if Identical(e, m) {
				return true
			}
		}
		expanding = append(expanding, m)
		for _, s := range sub {
			if !expand(s) {
				return false
			}
		}
		return true
	}
	for _, d := range direct {
		if !expand(d) {
			return nil, false
		}
	}
	if len(out) == 0 {
		return nil, false
	}
	iface, _ := Unalias(t).Underlying().(*Interface)
	if iface != nil {
		var kept []Type
		for _, m := range out {
			if _, imported := logosForeignUnionName(m); imported || Implements(m, iface) {
				kept = append(kept, m)
			}
		}
		out = kept
	}
	return out, true
}

type LogosUnionMember struct {
	Type Type
	Pkg  string
	Name string
	Ptr  bool
}

func (m LogosUnionMember) Text() string {
	text := m.Pkg + "." + m.Name
	if m.Ptr {
		return "*" + text
	}
	return text
}

func LogosUnionMembers(t Type) ([]LogosUnionMember, bool) {
	t = Unalias(t)
	if cv, ok := t.(*ConstView); ok {
		t = Unalias(cv.Elem)
	}
	if rd := logosAsReadonlyDescendants(t); rd != nil {
		t = Unalias(rd.Elem)
	}
	var out []LogosUnionMember
	seen := make(map[string]bool)
	visited := make(map[string]bool)
	if terms, ok := LogosUnionTerms(t); ok {
		for _, term := range terms {
			if name, imported := logosForeignUnionName(term); imported {
				logosAppendImportedMembers(name, &out, seen, visited)
				continue
			}
			logosAppendTypeMember(term, &out, seen)
		}
	} else if name, imported := logosForeignUnionName(t); imported {
		logosAppendImportedMembers(name, &out, seen, visited)
	}
	return out, len(out) > 0
}

func logosAppendTypeMember(term Type, out *[]LogosUnionMember, seen map[string]bool) {
	key := TypeString(term, nil)
	if seen[key] {
		return
	}
	seen[key] = true
	m := LogosUnionMember{Type: term}
	elem := Unalias(term)
	if rd := logosAsReadonlyDescendants(elem); rd != nil {
		elem = Unalias(rd.Elem)
	}
	if p, ok := elem.(*Pointer); ok {
		m.Ptr = true
		elem = Unalias(p.Elem())
		if rd := logosAsReadonlyDescendants(elem); rd != nil {
			elem = Unalias(rd.Elem)
		}
	}
	if named, ok := elem.(*Named); ok && named.Obj() != nil && named.Obj().Pkg() != nil {
		m.Pkg = named.Obj().Pkg().Path()
		m.Name = named.Obj().Name()
	}
	*out = append(*out, m)
}

func logosAppendImportedMembers(name logosForeignName, out *[]LogosUnionMember, seen, visited map[string]bool) {
	key := name.pkgPath + "." + name.typeName
	if visited[key] {
		return
	}
	visited[key] = true
	for _, member := range logosLookupImportedUnionMembers(name.pkgPath, name.typeName) {
		trimmed, ptr := strings.CutPrefix(member.Member, "*")
		pkgPath, typeName, ok := strings.CutLast(trimmed, ".")
		if !ok || pkgPath == "" || typeName == "" {
			continue
		}
		nested := logosForeignName{pkgPath: pkgPath, typeName: typeName}
		if !ptr && len(logosLookupImportedUnionMembers(pkgPath, typeName)) > 0 {
			logosAppendImportedMembers(nested, out, seen, visited)
			continue
		}
		if seen[member.Member] {
			continue
		}
		seen[member.Member] = true
		*out = append(*out, LogosUnionMember{Pkg: pkgPath, Name: typeName, Ptr: ptr})
	}
}

func (check *Checker) logosUnionLiteralMiss(x, y *operand) (lit *operand, union Type, from Type) {
	if !check.conf.Logos {
		return nil, nil, nil
	}
	for _, pair := range [2][2]*operand{{x, y}, {y, x}} {
		un, c := pair[0], pair[1]
		tset := logosClosedUnion(un.typ())
		if tset == nil || c.mode() != constant_ || !isUntyped(c.typ()) {
			continue
		}
		if d := Default(c.typ()); !logosMemberOf(d, tset) {
			return c, un.typ(), d
		}
	}
	return nil, nil, nil
}

func (check *Checker) logosUnionCompareMiss(x, y *operand, op syntax.Operator) bool {
	lit, union, from := check.logosUnionLiteralMiss(x, y)
	if lit == nil {
		return false
	}
	check.errorf(y, MismatchedTypes, "%s invalid operation: %s %s %s (%s is not a member of %s)", logosNonMember, x.expr, op, y.expr, from, union)
	x.invalidate()
	return true
}

func (check *Checker) logosUnionConstMiss(x, y *operand) bool {
	lit, union, from := check.logosUnionLiteralMiss(x, y)
	if lit == nil {
		return false
	}
	check.errorf(lit, InvalidUntypedConversion, "%s %s is not a member of %s", logosNonMember, from, union)
	return true
}
