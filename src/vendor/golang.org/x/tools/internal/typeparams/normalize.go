// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeparams

import (
	"fmt"
	"go/types"
	"os"
	"strings"
)

//go:generate go run copytermlist.go

const debug = false

// A termSet holds the normalized set of terms for a given type.
//
// The name termSet is intentionally distinct from 'type set': a type set is
// all types that implement a type (and includes method restrictions), whereas
// a term set just represents the structural restrictions on a type.
type termSet struct {
	complete bool
	terms    termlist
}

func indentf(depth int, format string, args ...any) {
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
		for embedded := range u.EmbeddedTypes() {
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
		for t := range u.Terms() {
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
