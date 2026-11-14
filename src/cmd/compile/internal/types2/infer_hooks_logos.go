// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import "cmd/compile/internal/syntax"

func (check *Checker) logosInferProperty(pos syntax.Pos, tpar *TypeParam, tx Type, constraint *Interface, err *error_) bool {
	var propCause string
	if !check.logosImplementsProperty(tx, constraint, constraint, &propCause, pos) {
		err.addf(pos, "%s (type %s) does not satisfy %s %s", tpar, tx, tpar.Constraint(), propCause)
		return true
	}
	return false
}

func (w *tpWalker) logosInferParameterized(typ Type) bool {
	switch t := typ.(type) {
	case *ReadonlyDescendants:
		return w.isParameterized(t.Elem)
	case *ConstView:
		return w.isParameterized(t.Elem)
	case *Option:
		return w.isParameterized(t.Elem)
	case *LogosTuple:
		for _, e := range t.types {
			if w.isParameterized(e) {
				return true
			}
		}
	}
	return false
}

func (w *cycleFinder) logosInferType(typ Type) {
	switch t := typ.(type) {
	case *ReadonlyDescendants:
		w.typ(t.Elem)
	case *ConstView:
		w.typ(t.Elem)
	case *Option:
		w.typ(t.Elem)
	case *LogosTuple:
		for _, e := range t.types {
			w.typ(e)
		}
	}
}
