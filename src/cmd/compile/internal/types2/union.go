// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import "cmd/compile/internal/syntax"

// ----------------------------------------------------------------------------
// API

// A Union represents a union of terms embedded in an interface.
type Union struct {
	terms []*term  // list of syntactical terms (not a canonicalized termlist)
	tset  *TypeSet // type set described by this union, computed lazily
}

// NewUnion returns a new Union type with the given terms (types[i], tilde[i]).
// The lengths of both arguments must match. It is an error to create an empty
// union; they are syntactically not possible.
func NewUnion(types []Type, tilde []bool) *Union { return newUnion(types, tilde) }

func (u *Union) NumTerms() int           { return len(u.terms) }
func (u *Union) Term(i int) (Type, bool) { t := u.terms[i]; return t.typ, t.tilde }

func (u *Union) Underlying() Type { return u }
func (u *Union) String() string   { return TypeString(u, nil) }

// ----------------------------------------------------------------------------
// Implementation

func newUnion(types []Type, tilde []bool) *Union {
	assert(len(types) == len(tilde))
	if len(types) == 0 {
		panic("empty union")
	}
	t := new(Union)
	t.terms = make([]*term, len(types))
	for i, typ := range types {
		t.terms[i] = &term{tilde[i], typ}
	}
	return t
}

func parseUnion(check *Checker, tlist []syntax.Expr) Type {
	var terms []*term
	for _, x := range tlist {
		tilde, typ := parseTilde(check, x)
		if len(tlist) == 1 && !tilde {
			return typ // single type
		}
		terms = append(terms, &term{tilde, typ})
	}

	// Check validity of terms.
	// Do this check later because it requires types to be set up.
	// Note: This is a quadratic algorithm, but unions tend to be short.
	check.later(func() {
		for i, t := range terms {
			typ := expand(t.typ)
			if typ == Typ[Invalid] {
				continue
			}

			x := tlist[i]
			pos := syntax.StartPos(x)
			// We may not know the position of x if it was a typechecker-
			// introduced ~T term for a type list entry T. Use the position
			// of T instead.
			// TODO(gri) remove this test once we don't support type lists anymore
			if !pos.IsKnown() {
				if op, _ := x.(*syntax.Operation); op != nil {
					pos = syntax.StartPos(op.X)
				}
			}

			u := under(typ)
			f, _ := u.(*Interface)
			if t.tilde {
				if f != nil {
					check.errorf(x, "invalid use of ~ (%s is an interface)", typ)
					continue // don't report another error for t
				}

				if !Identical(u, typ) {
					check.errorf(x, "invalid use of ~ (underlying type of %s is %s)", typ, u)
					continue // don't report another error for t
				}
			}

			// Stand-alone embedded interfaces are ok and are handled by the single-type case
			// in the beginning. Embedded interfaces with tilde are excluded above. If we reach
			// here, we must have at least two terms in the union.
			if f != nil && !f.typeSet().IsTypeSet() {
				check.errorf(pos, "cannot use %s in union (interface contains methods)", t)
				continue // don't report another error for t
			}

			// Report overlapping (non-disjoint) terms such as
			// a|a, a|~a, ~a|~a, and ~a|A (where under(A) == a).
			if j := overlappingTerm(terms[:i], t); j >= 0 {
				check.softErrorf(pos, "overlapping terms %s and %s", t, terms[j])
			}
		}
	})

	return &Union{terms, nil}
}

func parseTilde(check *Checker, x syntax.Expr) (tilde bool, typ Type) {
	if op, _ := x.(*syntax.Operation); op != nil && op.Op == syntax.Tilde {
		x = op.X
		tilde = true
	}
	typ = check.anyType(x)
	// embedding stand-alone type parameters is not permitted (issue #47127).
	if _, ok := under(typ).(*TypeParam); ok {
		check.error(x, "cannot embed a type parameter")
		typ = Typ[Invalid]
	}
	return
}

// overlappingTerm reports the index of the term x in terms which is
// overlapping (not disjoint) from y. The result is < 0 if there is no
// such term.
func overlappingTerm(terms []*term, y *term) int {
	for i, x := range terms {
		// disjoint requires non-nil, non-top arguments
		if debug {
			if x == nil || x.typ == nil || y == nil || y.typ == nil {
				panic("internal error: empty or top union term")
			}
		}
		if !x.disjoint(y) {
			return i
		}
	}
	return -1
}
