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

// A Union represents a union of terms.
type Union struct {
	terms []*term
}

// NewUnion returns a new Union type with the given terms (types[i], tilde[i]).
// The lengths of both arguments must match. An empty union represents the set
// of no types.
func NewUnion(types []Type, tilde []bool) *Union { return newUnion(types, tilde) }

func (u *Union) IsEmpty() bool           { return len(u.terms) == 0 }
func (u *Union) NumTerms() int           { return len(u.terms) }
func (u *Union) Term(i int) (Type, bool) { t := u.terms[i]; return t.typ, t.tilde }

func (u *Union) Underlying() Type { return u }
func (u *Union) String() string   { return TypeString(u, nil) }

// ----------------------------------------------------------------------------
// Implementation

var emptyUnion = new(Union)

func newUnion(types []Type, tilde []bool) *Union {
	assert(len(types) == len(tilde))
	if len(types) == 0 {
		return emptyUnion
	}
	t := new(Union)
	t.terms = make([]*term, len(types))
	for i, typ := range types {
		t.terms[i] = &term{tilde[i], typ}
	}
	return t
}

// is reports whether f returns true for all terms of u.
func (u *Union) is(f func(*term) bool) bool {
	if u.IsEmpty() {
		return false
	}
	for _, t := range u.terms {
		if !f(t) {
			return false
		}
	}
	return true
}

// underIs reports whether f returned true for the underlying types of all terms of u.
func (u *Union) underIs(f func(Type) bool) bool {
	if u.IsEmpty() {
		return false
	}
	for _, t := range u.terms {
		if !f(under(t.typ)) {
			return false
		}
	}
	return true
}

func parseUnion(check *Checker, tlist []ast.Expr) Type {
	var types []Type
	var tilde []bool
	for _, x := range tlist {
		t, d := parseTilde(check, x)
		if len(tlist) == 1 && !d {
			return t // single type
		}
		types = append(types, t)
		tilde = append(tilde, d)
	}

	// Ensure that each type is only present once in the type list.
	// It's ok to do this check later because it's not a requirement
	// for correctness of the code.
	// Note: This is a quadratic algorithm, but unions tend to be short.
	check.later(func() {
		for i, t := range types {
			t := expand(t)
			if t == Typ[Invalid] {
				continue
			}

			x := tlist[i]
			pos := x.Pos()
			// We may not know the position of x if it was a typechecker-
			// introduced ~T term for a type list entry T. Use the position
			// of T instead.
			// TODO(rfindley) remove this test once we don't support type lists anymore
			if !pos.IsValid() {
				if op, _ := x.(*ast.UnaryExpr); op != nil {
					pos = op.X.Pos()
				}
			}

			u := under(t)
			f, _ := u.(*Interface)
			if tilde[i] {
				if f != nil {
					check.errorf(x, _Todo, "invalid use of ~ (%s is an interface)", t)
					continue // don't report another error for t
				}

				if !Identical(u, t) {
					check.errorf(x, _Todo, "invalid use of ~ (underlying type of %s is %s)", t, u)
					continue // don't report another error for t
				}
			}

			// Stand-alone embedded interfaces are ok and are handled by the single-type case
			// in the beginning. Embedded interfaces with tilde are excluded above. If we reach
			// here, we must have at least two terms in the union.
			if f != nil && !f.typeSet().IsTypeSet() {
				check.errorf(atPos(pos), _Todo, "cannot use %s in union (interface contains methods)", t)
				continue // don't report another error for t
			}

			// Complain about duplicate entries a|a, but also a|~a, and ~a|~a.
			// TODO(gri) We should also exclude myint|~int since myint is included in ~int.
			if includes(types[:i], t) {
				// TODO(rfindley) this currently doesn't print the ~ if present
				check.softErrorf(atPos(pos), _Todo, "duplicate term %s in union element", t)
			}
		}
	})

	return newUnion(types, tilde)
}

func parseTilde(check *Checker, x ast.Expr) (typ Type, tilde bool) {
	if op, _ := x.(*ast.UnaryExpr); op != nil && op.Op == token.TILDE {
		x = op.X
		tilde = true
	}
	typ = check.anyType(x)
	// embedding stand-alone type parameters is not permitted (issue #47127).
	if _, ok := under(typ).(*TypeParam); ok {
		check.error(x, _Todo, "cannot embed a type parameter")
		typ = Typ[Invalid]
	}
	return
}

// intersect computes the intersection of the types x and y,
// A nil type stands for the set of all types; an empty union
// stands for the set of no types.
func intersect(x, y Type) (r Type) {
	// If one of the types is nil (no restrictions)
	// the result is the other type.
	switch {
	case x == nil:
		return y
	case y == nil:
		return x
	}

	// Compute the terms which are in both x and y.
	// TODO(gri) This is not correct as it may not always compute
	//           the "largest" intersection. For instance, for
	//           x = myInt|~int, y = ~int
	//           we get the result myInt but we should get ~int.
	xu, _ := x.(*Union)
	yu, _ := y.(*Union)
	switch {
	case xu != nil && yu != nil:
		return &Union{intersectTerms(xu.terms, yu.terms)}

	case xu != nil:
		if r, _ := xu.intersect(y, false); r != nil {
			return y
		}

	case yu != nil:
		if r, _ := yu.intersect(x, false); r != nil {
			return x
		}

	default: // xu == nil && yu == nil
		if Identical(x, y) {
			return x
		}
	}

	return emptyUnion
}

// includes reports whether typ is in list.
func includes(list []Type, typ Type) bool {
	for _, e := range list {
		if Identical(typ, e) {
			return true
		}
	}
	return false
}

// intersect computes the intersection of the union u and term (y, yt)
// and returns the intersection term, if any. Otherwise the result is
// (nil, false).
// TODO(gri) this needs to cleaned up/removed once we switch to lazy
//           union type set computation.
func (u *Union) intersect(y Type, yt bool) (Type, bool) {
	under_y := under(y)
	for _, x := range u.terms {
		xt := x.tilde
		// determine which types xx, yy to compare
		xx := x.typ
		if yt {
			xx = under(xx)
		}
		yy := y
		if xt {
			yy = under_y
		}
		if Identical(xx, yy) {
			//  T ∩  T =  T
			//  T ∩ ~t =  T
			// ~t ∩  T =  T
			// ~t ∩ ~t = ~t
			return xx, xt && yt
		}
	}
	return nil, false
}

func identicalTerms(list1, list2 []*term) bool {
	if len(list1) != len(list2) {
		return false
	}
	// Every term in list1 must be in list2.
	// Quadratic algorithm, but probably good enough for now.
	// TODO(gri) we need a fast quick type ID/hash for all types.
L:
	for _, x := range list1 {
		for _, y := range list2 {
			if x.equal(y) {
				continue L // x is in list2
			}
		}
		return false
	}
	return true
}

func intersectTerms(list1, list2 []*term) (list []*term) {
	// Quadratic algorithm, but good enough for now.
	// TODO(gri) fix asymptotic performance
	for _, x := range list1 {
		for _, y := range list2 {
			if r := x.intersect(y); r != nil {
				list = append(list, r)
			}
		}
	}
	return
}
