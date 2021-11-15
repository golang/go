// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"go/ast"
	"go/token"
)

// ----------------------------------------------------------------------------
// API

// A Union represents a union of terms embedded in an interface.
type Union struct {
	terms []*Term   // list of syntactical terms (not a canonicalized termlist)
	tset  *_TypeSet // type set described by this union, computed lazily
}

// NewUnion returns a new Union type with the given terms.
// It is an error to create an empty union; they are syntactically not possible.
func NewUnion(terms []*Term) *Union {
	if len(terms) == 0 {
		panic("empty union")
	}
	return &Union{terms, nil}
}

func (u *Union) Len() int         { return len(u.terms) }
func (u *Union) Term(i int) *Term { return u.terms[i] }

func (u *Union) Underlying() Type { return u }
func (u *Union) String() string   { return TypeString(u, nil) }

// A Term represents a term in a Union.
type Term term

// NewTerm returns a new union term.
func NewTerm(tilde bool, typ Type) *Term { return &Term{tilde, typ} }

func (t *Term) Tilde() bool    { return t.tilde }
func (t *Term) Type() Type     { return t.typ }
func (t *Term) String() string { return (*term)(t).String() }

// ----------------------------------------------------------------------------
// Implementation

// Avoid excessive type-checking times due to quadratic termlist operations.
const maxTermCount = 100

// parseUnion parses the given list of type expressions tlist as a union of
// those expressions. The result is a Union type, or Typ[Invalid] for some
// errors.
func parseUnion(check *Checker, tlist []ast.Expr) Type {
	var terms []*Term
	for _, x := range tlist {
		tilde, typ := parseTilde(check, x)
		if len(tlist) == 1 && !tilde {
			// Single type. Ok to return early because all relevant
			// checks have been performed in parseTilde (no need to
			// run through term validity check below).
			return typ
		}
		if len(terms) >= maxTermCount {
			check.errorf(x, _InvalidUnion, "cannot handle more than %d union terms (implementation limitation)", maxTermCount)
			return Typ[Invalid]
		}
		terms = append(terms, NewTerm(tilde, typ))
	}

	// Check validity of terms.
	// Do this check later because it requires types to be set up.
	// Note: This is a quadratic algorithm, but unions tend to be short.
	check.later(func() {
		for i, t := range terms {
			if t.typ == Typ[Invalid] {
				continue
			}

			u := under(t.typ)
			f, _ := u.(*Interface)
			if t.tilde {
				if f != nil {
					check.errorf(tlist[i], _InvalidUnion, "invalid use of ~ (%s is an interface)", t.typ)
					continue // don't report another error for t
				}

				if !Identical(u, t.typ) {
					check.errorf(tlist[i], _InvalidUnion, "invalid use of ~ (underlying type of %s is %s)", t.typ, u)
					continue // don't report another error for t
				}
			}

			// Stand-alone embedded interfaces are ok and are handled by the single-type case
			// in the beginning. Embedded interfaces with tilde are excluded above. If we reach
			// here, we must have at least two terms in the union.
			if f != nil && !f.typeSet().IsTypeSet() {
				check.errorf(tlist[i], _InvalidUnion, "cannot use %s in union (interface contains methods)", t)
				continue // don't report another error for t
			}

			// Report overlapping (non-disjoint) terms such as
			// a|a, a|~a, ~a|~a, and ~a|A (where under(A) == a).
			if j := overlappingTerm(terms[:i], t); j >= 0 {
				check.softErrorf(tlist[i], _InvalidUnion, "overlapping terms %s and %s", t, terms[j])
			}
		}
	})

	return &Union{terms, nil}
}

func parseTilde(check *Checker, x ast.Expr) (tilde bool, typ Type) {
	if op, _ := x.(*ast.UnaryExpr); op != nil && op.Op == token.TILDE {
		x = op.X
		tilde = true
	}
	typ = check.typ(x)
	// Embedding stand-alone type parameters is not permitted (issue #47127).
	// Do this check later because it requires computation of the underlying type (see also issue #46461).
	// Note: If an underlying type cannot be a type parameter, the call to
	//       under() will not be needed and then we don't need to delay this
	//       check to later and could return Typ[Invalid] instead.
	check.later(func() {
		if _, ok := under(typ).(*TypeParam); ok {
			check.error(x, _MisplacedTypeParam, "cannot embed a type parameter")
		}
	})
	return
}

// overlappingTerm reports the index of the term x in terms which is
// overlapping (not disjoint) from y. The result is < 0 if there is no
// such term.
func overlappingTerm(terms []*Term, y *Term) int {
	for i, x := range terms {
		// disjoint requires non-nil, non-top arguments
		if debug {
			if x == nil || x.typ == nil || y == nil || y.typ == nil {
				panic("empty or top union term")
			}
		}
		if !(*term)(x).disjoint((*term)(y)) {
			return i
		}
	}
	return -1
}
