// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typeparams

import (
	"go/types"

	"golang.org/x/tools/internal/aliases"
)

// Free is a memoization of the set of free type parameters within a
// type. It makes a sequence of calls to [Free.Has] for overlapping
// types more efficient. The zero value is ready for use.
//
// NOTE: Adapted from go/types/infer.go. If it is later exported, factor.
type Free struct {
	seen map[types.Type]bool
}

// Has reports whether the specified type has a free type parameter.
func (w *Free) Has(typ types.Type) (res bool) {

	// detect cycles
	if x, ok := w.seen[typ]; ok {
		return x
	}
	if w.seen == nil {
		w.seen = make(map[types.Type]bool)
	}
	w.seen[typ] = false
	defer func() {
		w.seen[typ] = res
	}()

	switch t := typ.(type) {
	case nil, *types.Basic: // TODO(gri) should nil be handled here?
		break

	case *aliases.Alias:
		return w.Has(aliases.Unalias(t))

	case *types.Array:
		return w.Has(t.Elem())

	case *types.Slice:
		return w.Has(t.Elem())

	case *types.Struct:
		for i, n := 0, t.NumFields(); i < n; i++ {
			if w.Has(t.Field(i).Type()) {
				return true
			}
		}

	case *types.Pointer:
		return w.Has(t.Elem())

	case *types.Tuple:
		n := t.Len()
		for i := 0; i < n; i++ {
			if w.Has(t.At(i).Type()) {
				return true
			}
		}

	case *types.Signature:
		// t.tparams may not be nil if we are looking at a signature
		// of a generic function type (or an interface method) that is
		// part of the type we're testing. We don't care about these type
		// parameters.
		// Similarly, the receiver of a method may declare (rather than
		// use) type parameters, we don't care about those either.
		// Thus, we only need to look at the input and result parameters.
		return w.Has(t.Params()) || w.Has(t.Results())

	case *types.Interface:
		for i, n := 0, t.NumMethods(); i < n; i++ {
			if w.Has(t.Method(i).Type()) {
				return true
			}
		}
		terms, err := InterfaceTermSet(t)
		if err != nil {
			panic(err)
		}
		for _, term := range terms {
			if w.Has(term.Type()) {
				return true
			}
		}

	case *types.Map:
		return w.Has(t.Key()) || w.Has(t.Elem())

	case *types.Chan:
		return w.Has(t.Elem())

	case *types.Named:
		args := t.TypeArgs()
		// TODO(taking): this does not match go/types/infer.go. Check with rfindley.
		if params := t.TypeParams(); params.Len() > args.Len() {
			return true
		}
		for i, n := 0, args.Len(); i < n; i++ {
			if w.Has(args.At(i)) {
				return true
			}
		}
		return w.Has(t.Underlying()) // recurse for types local to parameterized functions

	case *types.TypeParam:
		return true

	default:
		panic(t) // unreachable
	}

	return false
}
