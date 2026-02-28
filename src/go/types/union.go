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
	terms []*Term // list of syntactical terms (not a canonicalized termlist)
}

// NewUnion returns a new Union type with the given terms.
// It is an error to create an empty union; they are syntactically not possible.
func NewUnion(terms []*Term) *Union {
	if len(terms) == 0 {
		panic("empty union")
	}
	return &Union{terms}
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

// parseUnion parses uexpr as a union of expressions.
// The result is a Union type, or Typ[Invalid] for some errors.
func parseUnion(check *Checker, uexpr ast.Expr) Type {
	blist, tlist := flattenUnion(nil, uexpr)
	assert(len(blist) == len(tlist)-1)

	var terms []*Term

	var u Type
	for i, x := range tlist {
		term := parseTilde(check, x)
		if len(tlist) == 1 && !term.tilde {
			// Single type. Ok to return early because all relevant
			// checks have been performed in parseTilde (no need to
			// run through term validity check below).
			return term.typ // typ already recorded through check.typ in parseTilde
		}
		if len(terms) >= maxTermCount {
			if u != Typ[Invalid] {
				check.errorf(x, _InvalidUnion, "cannot handle more than %d union terms (implementation limitation)", maxTermCount)
				u = Typ[Invalid]
			}
		} else {
			terms = append(terms, term)
			u = &Union{terms}
		}

		if i > 0 {
			check.recordTypeAndValue(blist[i-1], typexpr, u, nil)
		}
	}

	if u == Typ[Invalid] {
		return u
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
					continue
				}
			}

			// Stand-alone embedded interfaces are ok and are handled by the single-type case
			// in the beginning. Embedded interfaces with tilde are excluded above. If we reach
			// here, we must have at least two terms in the syntactic term list (but not necessarily
			// in the term list of the union's type set).
			if f != nil {
				tset := f.typeSet()
				switch {
				case tset.NumMethods() != 0:
					check.errorf(tlist[i], _InvalidUnion, "cannot use %s in union (%s contains methods)", t, t)
				case t.typ == universeComparable.Type():
					check.error(tlist[i], _InvalidUnion, "cannot use comparable in union")
				case tset.comparable:
					check.errorf(tlist[i], _InvalidUnion, "cannot use %s in union (%s embeds comparable)", t, t)
				}
				continue // terms with interface types are not subject to the no-overlap rule
			}

			// Report overlapping (non-disjoint) terms such as
			// a|a, a|~a, ~a|~a, and ~a|A (where under(A) == a).
			if j := overlappingTerm(terms[:i], t); j >= 0 {
				check.softErrorf(tlist[i], _InvalidUnion, "overlapping terms %s and %s", t, terms[j])
			}
		}
	}).describef(uexpr, "check term validity %s", uexpr)

	return u
}

func parseTilde(check *Checker, tx ast.Expr) *Term {
	x := tx
	var tilde bool
	if op, _ := x.(*ast.UnaryExpr); op != nil && op.Op == token.TILDE {
		x = op.X
		tilde = true
	}
	typ := check.typ(x)
	// Embedding stand-alone type parameters is not permitted (issue #47127).
	// We don't need this restriction anymore if we make the underlying type of a type
	// parameter its constraint interface: if we embed a lone type parameter, we will
	// simply use its underlying type (like we do for other named, embedded interfaces),
	// and since the underlying type is an interface the embedding is well defined.
	if isTypeParam(typ) {
		if tilde {
			check.errorf(x, _MisplacedTypeParam, "type in term %s cannot be a type parameter", tx)
		} else {
			check.error(x, _MisplacedTypeParam, "term cannot be a type parameter")
		}
		typ = Typ[Invalid]
	}
	term := NewTerm(tilde, typ)
	if tilde {
		check.recordTypeAndValue(tx, typexpr, &Union{[]*Term{term}}, nil)
	}
	return term
}

// overlappingTerm reports the index of the term x in terms which is
// overlapping (not disjoint) from y. The result is < 0 if there is no
// such term. The type of term y must not be an interface, and terms
// with an interface type are ignored in the terms list.
func overlappingTerm(terms []*Term, y *Term) int {
	assert(!IsInterface(y.typ))
	for i, x := range terms {
		if IsInterface(x.typ) {
			continue
		}
		// disjoint requires non-nil, non-top arguments,
		// and non-interface types as term types.
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

// flattenUnion walks a union type expression of the form A | B | C | ...,
// extracting both the binary exprs (blist) and leaf types (tlist).
func flattenUnion(list []ast.Expr, x ast.Expr) (blist, tlist []ast.Expr) {
	if o, _ := x.(*ast.BinaryExpr); o != nil && o.Op == token.OR {
		blist, tlist = flattenUnion(list, o.X)
		blist = append(blist, o)
		x = o.Y
	}
	return blist, append(tlist, x)
}
