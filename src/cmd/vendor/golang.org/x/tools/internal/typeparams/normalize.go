// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeparams

import (
	"errors"
	"fmt"
	"go/types"
	"os"
	"strings"
)

//go:generate go run copytermlist.go

const debug = false

var ErrEmptyTypeSet = errors.New("empty type set")

// StructuralTerms returns a slice of terms representing the normalized
// structural type restrictions of a type parameter, if any.
//
// Structural type restrictions of a type parameter are created via
// non-interface types embedded in its constraint interface (directly, or via a
// chain of interface embeddings). For example, in the declaration
//
//	type T[P interface{~int; m()}] int
//
// the structural restriction of the type parameter P is ~int.
//
// With interface embedding and unions, the specification of structural type
// restrictions may be arbitrarily complex. For example, consider the
// following:
//
//	type A interface{ ~string|~[]byte }
//
//	type B interface{ int|string }
//
//	type C interface { ~string|~int }
//
//	type T[P interface{ A|B; C }] int
//
// In this example, the structural type restriction of P is ~string|int: A|B
// expands to ~string|~[]byte|int|string, which reduces to ~string|~[]byte|int,
// which when intersected with C (~string|~int) yields ~string|int.
//
// StructuralTerms computes these expansions and reductions, producing a
// "normalized" form of the embeddings. A structural restriction is normalized
// if it is a single union containing no interface terms, and is minimal in the
// sense that removing any term changes the set of types satisfying the
// constraint. It is left as a proof for the reader that, modulo sorting, there
// is exactly one such normalized form.
//
// Because the minimal representation always takes this form, StructuralTerms
// returns a slice of tilde terms corresponding to the terms of the union in
// the normalized structural restriction. An error is returned if the
// constraint interface is invalid, exceeds complexity bounds, or has an empty
// type set. In the latter case, StructuralTerms returns ErrEmptyTypeSet.
//
// StructuralTerms makes no guarantees about the order of terms, except that it
// is deterministic.
func StructuralTerms(tparam *types.TypeParam) ([]*types.Term, error) {
	constraint := tparam.Constraint()
	if constraint == nil {
		return nil, fmt.Errorf("%s has nil constraint", tparam)
	}
	iface, _ := constraint.Underlying().(*types.Interface)
	if iface == nil {
		return nil, fmt.Errorf("constraint is %T, not *types.Interface", constraint.Underlying())
	}
	return InterfaceTermSet(iface)
}

// InterfaceTermSet computes the normalized terms for a constraint interface,
// returning an error if the term set cannot be computed or is empty. In the
// latter case, the error will be ErrEmptyTypeSet.
//
// See the documentation of StructuralTerms for more information on
// normalization.
func InterfaceTermSet(iface *types.Interface) ([]*types.Term, error) {
	return computeTermSet(iface)
}

// UnionTermSet computes the normalized terms for a union, returning an error
// if the term set cannot be computed or is empty. In the latter case, the
// error will be ErrEmptyTypeSet.
//
// See the documentation of StructuralTerms for more information on
// normalization.
func UnionTermSet(union *types.Union) ([]*types.Term, error) {
	return computeTermSet(union)
}

func computeTermSet(typ types.Type) ([]*types.Term, error) {
	tset, err := computeTermSetInternal(typ, make(map[types.Type]*termSet), 0)
	if err != nil {
		return nil, err
	}
	if tset.terms.isEmpty() {
		return nil, ErrEmptyTypeSet
	}
	if tset.terms.isAll() {
		return nil, nil
	}
	var terms []*types.Term
	for _, term := range tset.terms {
		terms = append(terms, types.NewTerm(term.tilde, term.typ))
	}
	return terms, nil
}

// A termSet holds the normalized set of terms for a given type.
//
// The name termSet is intentionally distinct from 'type set': a type set is
// all types that implement a type (and includes method restrictions), whereas
// a term set just represents the structural restrictions on a type.
type termSet struct {
	complete bool
	terms    termlist
}

func indentf(depth int, format string, args ...interface{}) {
	fmt.Fprintf(os.Stderr, strings.Repeat(".", depth)+format+"\n", args...)
}

func computeTermSetInternal(t types.Type, seen map[types.Type]*termSet, depth int) (res *termSet, err error) {
	if t == nil {
		panic("nil type")
	}

	if debug {
		indentf(depth, "%s", t.String())
		defer func() {
			if err != nil {
				indentf(depth, "=> %s", err)
			} else {
				indentf(depth, "=> %s", res.terms.String())
			}
		}()
	}

	const maxTermCount = 100
	if tset, ok := seen[t]; ok {
		if !tset.complete {
			return nil, fmt.Errorf("cycle detected in the declaration of %s", t)
		}
		return tset, nil
	}

	// Mark the current type as seen to avoid infinite recursion.
	tset := new(termSet)
	defer func() {
		tset.complete = true
	}()
	seen[t] = tset

	switch u := t.Underlying().(type) {
	case *types.Interface:
		// The term set of an interface is the intersection of the term sets of its
		// embedded types.
		tset.terms = allTermlist
		for i := 0; i < u.NumEmbeddeds(); i++ {
			embedded := u.EmbeddedType(i)
			if _, ok := embedded.Underlying().(*types.TypeParam); ok {
				return nil, fmt.Errorf("invalid embedded type %T", embedded)
			}
			tset2, err := computeTermSetInternal(embedded, seen, depth+1)
			if err != nil {
				return nil, err
			}
			tset.terms = tset.terms.intersect(tset2.terms)
		}
	case *types.Union:
		// The term set of a union is the union of term sets of its terms.
		tset.terms = nil
		for i := 0; i < u.Len(); i++ {
			t := u.Term(i)
			var terms termlist
			switch t.Type().Underlying().(type) {
			case *types.Interface:
				tset2, err := computeTermSetInternal(t.Type(), seen, depth+1)
				if err != nil {
					return nil, err
				}
				terms = tset2.terms
			case *types.TypeParam, *types.Union:
				// A stand-alone type parameter or union is not permitted as union
				// term.
				return nil, fmt.Errorf("invalid union term %T", t)
			default:
				if t.Type() == types.Typ[types.Invalid] {
					continue
				}
				terms = termlist{{t.Tilde(), t.Type()}}
			}
			tset.terms = tset.terms.union(terms)
			if len(tset.terms) > maxTermCount {
				return nil, fmt.Errorf("exceeded max term count %d", maxTermCount)
			}
		}
	case *types.TypeParam:
		panic("unreachable")
	default:
		// For all other types, the term set is just a single non-tilde term
		// holding the type itself.
		if u != types.Typ[types.Invalid] {
			tset.terms = termlist{{false, t}}
		}
	}
	return tset, nil
}

// under is a facade for the go/types internal function of the same name. It is
// used by typeterm.go.
func under(t types.Type) types.Type {
	return t.Underlying()
}
