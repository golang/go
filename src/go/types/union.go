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
// A term is a type, possibly with a ~ (tilde) flag.
type Union struct {
	types []Type // types are unique
	tilde []bool // if tilde[i] is set, terms[i] is of the form ~T
}

func NewUnion(types []Type, tilde []bool) Type { return newUnion(types, tilde) }

func (u *Union) NumTerms() int           { return len(u.types) }
func (u *Union) Term(i int) (Type, bool) { return u.types[i], u.tilde[i] }

func (u *Union) Underlying() Type { return u }
func (u *Union) String() string   { return TypeString(u, nil) }

// ----------------------------------------------------------------------------
// Implementation

func newUnion(types []Type, tilde []bool) Type {
	assert(len(types) == len(tilde))
	if types == nil {
		return nil
	}
	t := new(Union)
	t.types = types
	t.tilde = tilde
	return t
}

// is reports whether f returned true for all terms (type, tilde) of u.
func (u *Union) is(f func(Type, bool) bool) bool {
	if u == nil {
		return false
	}
	for i, t := range u.types {
		if !f(t, u.tilde[i]) {
			return false
		}
	}
	return true
}

// is reports whether f returned true for the underlying types of all terms of u.
func (u *Union) underIs(f func(Type) bool) bool {
	if u == nil {
		return false
	}
	for _, t := range u.types {
		if !f(under(t)) {
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
	// It's ok to do this check at the end because it's not a requirement
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
			// introduced ~T type of a type list entry T. Use the position
			// of T instead.
			// TODO(rfindley) remove this test once we don't support type lists anymore
			if !pos.IsValid() {
				if op, _ := x.(*ast.UnaryExpr); op != nil {
					pos = op.X.Pos()
				}
			}

			u := under(t)
			if tilde[i] {
				// TODO(rfindley) enable this check once we have converted tests
				// if !Identical(u, t) {
				// 	check.errorf(x, "invalid use of ~ (underlying type of %s is %s)", t, u)
				// }
			}
			if _, ok := u.(*Interface); ok {
				check.errorf(atPos(pos), _Todo, "cannot use interface %s with ~ or inside a union (implementation restriction)", t)
			}

			// Complain about duplicate entries a|a, but also a|~a, and ~a|~a.
			if includes(types[:i], t) {
				// TODO(rfindley) this currently doesn't print the ~ if present
				check.softErrorf(atPos(pos), _Todo, "duplicate term %s in union element", t)
			}
		}
	})

	return newUnion(types, tilde)
}

func parseTilde(check *Checker, x ast.Expr) (Type, bool) {
	tilde := false
	if op, _ := x.(*ast.UnaryExpr); op != nil && op.Op == token.TILDE {
		x = op.X
		tilde = true
	}
	return check.anyType(x), tilde
}

// intersect computes the intersection of the types x and y.
// Note: An incomming nil type stands for the top type. A top
// type result is returned as nil.
func intersect(x, y Type) (r Type) {
	defer func() {
		if r == theTop {
			r = nil
		}
	}()

	switch {
	case x == theBottom || y == theBottom:
		return theBottom
	case x == nil || x == theTop:
		return y
	case y == nil || x == theTop:
		return x
	}

	// Compute the terms which are in both x and y.
	xu, _ := x.(*Union)
	yu, _ := y.(*Union)
	switch {
	case xu != nil && yu != nil:
		// Quadratic algorithm, but good enough for now.
		// TODO(gri) fix asymptotic performance
		var types []Type
		var tilde []bool
		for _, y := range yu.types {
			if includes(xu.types, y) {
				types = append(types, y)
				tilde = append(tilde, true) // TODO(gri) fix this
			}
		}
		if types != nil {
			return newUnion(types, tilde)
		}

	case xu != nil:
		if includes(xu.types, y) {
			return y
		}

	case yu != nil:
		if includes(yu.types, x) {
			return x
		}

	default: // xu == nil && yu == nil
		if Identical(x, y) {
			return x
		}
	}

	return theBottom
}
