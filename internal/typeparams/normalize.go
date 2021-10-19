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

// NormalizeInterface returns the normal form of the interface iface, or nil if iface
// has an empty type set (i.e. there are no types that satisfy iface). If the
// resulting interface is non-nil, it will be identical to iface.
//
// An error is returned if the interface type is invalid, or too complicated to
// reasonably normalize (for example, contains unions with more than a hundred
// terms).
//
// An interface is in normal form if and only if:
//   - it has 0 or 1 embedded types.
//   - its embedded type is either a types.Union or has a concrete
//     (non-interface) underlying type
//   - if the embedded type is a union, each term of the union has a concrete
//     underlying type, and no terms may be removed without changing the type set
//     of the interface
func NormalizeInterface(iface *types.Interface) (*types.Interface, error) {
	var methods []*types.Func
	for i := 0; i < iface.NumMethods(); i++ {
		methods = append(methods, iface.Method(i))
	}
	var embeddeds []types.Type
	tset, err := computeTermSet(iface, make(map[types.Type]*termSet), 0)
	if err != nil {
		return nil, err
	}
	switch {
	case tset.terms.isEmpty():
		// Special case: as documented
		return nil, nil

	case tset.terms.isAll():
		// No embeddeds.

	case len(tset.terms) == 1:
		if !tset.terms[0].tilde {
			embeddeds = append(embeddeds, tset.terms[0].typ)
			break
		}
		fallthrough
	default:
		var terms []*Term
		for _, term := range tset.terms {
			terms = append(terms, NewTerm(term.tilde, term.typ))
		}
		embeddeds = append(embeddeds, NewUnion(terms))
	}

	return types.NewInterfaceType(methods, embeddeds), nil
}

var ErrEmptyTypeSet = errors.New("empty type set")

// StructuralTerms returns the normalized structural type restrictions of a
// type, if any. For types that are not type parameters, it returns term slice
// containing a single non-tilde term holding the given type. For type
// parameters, it returns the normalized term list of the type parameter's
// constraint. See NormalizeInterface for more information on the normal form
// of a constraint interface.
//
// StructuralTerms returns an error if the structural term list cannot be
// computed. If the type set of typ is empty, it returns ErrEmptyTypeSet.
func StructuralTerms(typ types.Type) ([]*Term, error) {
	switch typ := typ.(type) {
	case *TypeParam:
		iface, _ := typ.Constraint().(*types.Interface)
		if iface == nil {
			return nil, fmt.Errorf("constraint is %T, not *types.Interface", typ)
		}
		tset, err := computeTermSet(iface, make(map[types.Type]*termSet), 0)
		if err != nil {
			return nil, err
		}
		if tset.terms.isEmpty() {
			return nil, ErrEmptyTypeSet
		}
		if tset.terms.isAll() {
			return nil, nil
		}
		var terms []*Term
		for _, term := range tset.terms {
			terms = append(terms, NewTerm(term.tilde, term.typ))
		}
		return terms, nil
	default:
		return []*Term{NewTerm(false, typ)}, nil
	}
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

func computeTermSet(t types.Type, seen map[types.Type]*termSet, depth int) (res *termSet, err error) {
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
			if _, ok := embedded.Underlying().(*TypeParam); ok {
				return nil, fmt.Errorf("invalid embedded type %T", embedded)
			}
			tset2, err := computeTermSet(embedded, seen, depth+1)
			if err != nil {
				return nil, err
			}
			tset.terms = tset.terms.intersect(tset2.terms)
		}
	case *Union:
		// The term set of a union is the union of term sets of its terms.
		tset.terms = nil
		for i := 0; i < u.Len(); i++ {
			t := u.Term(i)
			var terms termlist
			switch t.Type().Underlying().(type) {
			case *types.Interface:
				tset2, err := computeTermSet(t.Type(), seen, depth+1)
				if err != nil {
					return nil, err
				}
				terms = tset2.terms
			case *TypeParam, *Union:
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
	case *TypeParam:
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
