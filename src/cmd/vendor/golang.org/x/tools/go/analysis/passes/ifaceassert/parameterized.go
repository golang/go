// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package ifaceassert

import (
	"go/types"

	"golang.org/x/tools/internal/typeparams"
)

// isParameterized reports whether typ contains any of the type parameters of tparams.
//
// NOTE: Adapted from go/types/infer.go. If that is exported in a future release remove this copy.
func isParameterized(typ types.Type) bool {
	w := tpWalker{
		seen: make(map[types.Type]bool),
	}
	return w.isParameterized(typ)
}

type tpWalker struct {
	seen map[types.Type]bool
}

func (w *tpWalker) isParameterized(typ types.Type) (res bool) {
	// detect cycles
	if x, ok := w.seen[typ]; ok {
		return x
	}
	w.seen[typ] = false
	defer func() {
		w.seen[typ] = res
	}()

	switch t := typ.(type) {
	case nil, *types.Basic: // TODO(gri) should nil be handled here?
		break

	case *types.Array:
		return w.isParameterized(t.Elem())

	case *types.Slice:
		return w.isParameterized(t.Elem())

	case *types.Struct:
		for i, n := 0, t.NumFields(); i < n; i++ {
			if w.isParameterized(t.Field(i).Type()) {
				return true
			}
		}

	case *types.Pointer:
		return w.isParameterized(t.Elem())

	case *types.Tuple:
		n := t.Len()
		for i := 0; i < n; i++ {
			if w.isParameterized(t.At(i).Type()) {
				return true
			}
		}

	case *types.Signature:
		// t.tparams may not be nil if we are looking at a signature
		// of a generic function type (or an interface method) that is
		// part of the type we're testing. We don't care about these type
		// parameters.
		// Similarly, the receiver of a method may declare (rather then
		// use) type parameters, we don't care about those either.
		// Thus, we only need to look at the input and result parameters.
		return w.isParameterized(t.Params()) || w.isParameterized(t.Results())

	case *types.Interface:
		for i, n := 0, t.NumMethods(); i < n; i++ {
			if w.isParameterized(t.Method(i).Type()) {
				return true
			}
		}
		terms, err := typeparams.InterfaceTermSet(t)
		if err != nil {
			panic(err)
		}
		for _, term := range terms {
			if w.isParameterized(term.Type()) {
				return true
			}
		}

	case *types.Map:
		return w.isParameterized(t.Key()) || w.isParameterized(t.Elem())

	case *types.Chan:
		return w.isParameterized(t.Elem())

	case *types.Named:
		list := typeparams.NamedTypeArgs(t)
		for i, n := 0, list.Len(); i < n; i++ {
			if w.isParameterized(list.At(i)) {
				return true
			}
		}

	case *typeparams.TypeParam:
		return true

	default:
		panic(t) // unreachable
	}

	return false
}
