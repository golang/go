// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

// LoGo/s M2 type rules for closed unions (SPEC Sec 2.3, T-203.1). A
// closed union is an interface whose type set is a finite union of
// non-`~` terms; its methods, if any, are implemented by every member.
// Upstream files call in through hook lines only: typexpr.go (value
// positions), call.go and conversions.go (conversions), expr.go
// (untyped constants and assertions). Every rule below applies only to a
// package of .lgo files (Config.Logos for the diagnostics of the first
// check, Config.LogosUnions for value positions in every check), so plain
// Go keeps rejecting such interfaces outside type constraints.
//
// The run-time representation of a union is the upstream interface, so a
// membership rule only decides what type checks; nothing here changes
// what the backend sees. The recheck of lowered code runs with Logos off
// and LogosUnions on: lowering leaves unions in value positions, and the
// unified IR reader folds the type set away.

const (
	logosOpenUnion   = "[correctness::open_union_value]"
	logosNonMember   = "[correctness::union_nonmember]"
	logosNonExhaust  = "[correctness::nonexhaustive_switch]"
	logosArmMismatch = "[correctness::switch_arm_mismatch]"
	logosNeedDefault = "[correctness::switch_expr_needs_default]"
	logosAssertFail  = "[correctness::assertion_failed]"
)

// logosIsClosed reports whether tset describes a closed union: specific
// terms, none of them `~` terms.
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

// logosClosedUnion returns the type set of t when t is a closed union
// type and nil otherwise. A type parameter is never a union.
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

// logosMemberOf reports whether t is a member of the closed union whose
// type set is tset: some term of the union denotes the identical type.
func logosMemberOf(t Type, tset *_TypeSet) bool {
	for _, term := range tset.terms {
		if Identical(term.typ, t) {
			return true
		}
	}
	return false
}

// logosUnionsOK reports whether the package under check is a package of
// .lgo files, in which closed unions are legal.
func (check *Checker) logosUnionsOK() bool {
	return check.conf.Logos || check.conf.LogosUnions
}

// logosValueType decides whether the interface type typ, whose type set
// tset is not a method set, may occupy a value position. A closed union
// may; a union with `~` terms and an unbounded set such as comparable may
// not and report open_union_value. It reports whether it handled typ, in
// which case the caller skips the upstream error. It runs from the
// delayed check of validVarType, so tset is complete.
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

// logosConversionUnion reports whether T is a closed union that a
// conversion T(x) may target; logosUnionConversion decides membership.
func (check *Checker) logosConversionUnion(T Type) bool {
	return check.logosUnionsOK() && logosClosedUnion(T) != nil
}

// logosUnionDefault gives an untyped constant or untyped boolean x paired
// with the closed union u its default type when that type is a member of
// u (SPEC Sec 2.3: the constant converts to its default type and
// membership is required). It reports false otherwise, leaving the
// upstream refusal in place.
func (check *Checker) logosUnionDefault(x *operand, u *Interface) (Type, bool) {
	if !check.logosUnionsOK() || !logosIsClosed(u.typeSet()) {
		return nil, false
	}
	d := Default(x.typ())
	if !check.implements(d, u, false, nil) {
		return nil, false
	}
	return d, true
}

// logosUnionConversion checks the conversion of x to the closed union T.
// Converting to an interface type is assignment, so x must be assignable:
// a member, or a closed union whose members are all members of T.
// Otherwise it reports union_nonmember, invalidates x, and reports true.
// It reports false, leaving the conversion to the upstream rules, for
// every other target and for a member x.
func (check *Checker) logosUnionConversion(x *operand, T Type) bool {
	if !check.logosConversionUnion(T) {
		return false
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

// logosNonMemberAssert reports union_nonmember for an assertion or case
// type T outside the member set of the closed union operand x, and
// reports true. A non-interface T is a member iff it denotes a member.
// An interface T is a member iff its type set is a subset of the
// operand's: a sub-union covering only members, `any`, or a method-only
// interface with no specific terms. A T with no type set (a type
// parameter) is never reported here. Both callers flow through
// typeAssertion (assertion and switch-case contexts alike), so no flag
// distinguishes them. e is the type expression, for the position.
func (check *Checker) logosNonMemberAssert(e syntax.Expr, x *operand, T Type) bool {
	if !check.conf.Logos {
		return false
	}
	tset := logosClosedUnion(x.typ())
	if tset == nil || isTypeParam(T) || logosMemberOf(T, tset) {
		return false
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
