// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

type typeInfo uint

// validType verifies that the given type does not "expand" infinitely
// producing a cycle in the type graph. Cycles are detected by marking
// defined types.
// (Cycles involving alias types, as in "type A = [10]A" are detected
// earlier, via the objDecl cycle detection mechanism.)
func (check *Checker) validType(typ Type, path []Object) typeInfo {
	const (
		unknown typeInfo = iota
		marked
		valid
		invalid
	)

	switch t := typ.(type) {
	case *Array:
		return check.validType(t.elem, path)

	case *Struct:
		for _, f := range t.fields {
			if check.validType(f.typ, path) == invalid {
				return invalid
			}
		}

	case *Union:
		for _, t := range t.terms {
			if check.validType(t.typ, path) == invalid {
				return invalid
			}
		}

	case *Interface:
		for _, etyp := range t.embeddeds {
			if check.validType(etyp, path) == invalid {
				return invalid
			}
		}

	case *Named:
		// If t is parameterized, we should be considering the instantiated (expanded)
		// form of t, but in general we can't with this algorithm: if t is an invalid
		// type it may be so because it infinitely expands through a type parameter.
		// Instantiating such a type would lead to an infinite sequence of instantiations.
		// In general, we need "type flow analysis" to recognize those cases.
		// Example: type A[T any] struct{ x A[*T] } (issue #48951)
		// In this algorithm we always only consider the original, uninstantiated type.
		// This won't recognize some invalid cases with parameterized types, but it
		// will terminate.
		t = t.orig

		// don't touch the type if it is from a different package or the Universe scope
		// (doing so would lead to a race condition - was issue #35049)
		if t.obj.pkg != check.pkg {
			return valid
		}

		// don't report a 2nd error if we already know the type is invalid
		// (e.g., if a cycle was detected earlier, via under).
		if t.underlying == Typ[Invalid] {
			t.info = invalid
			return invalid
		}

		switch t.info {
		case unknown:
			t.info = marked
			t.info = check.validType(t.fromRHS, append(path, t.obj)) // only types of current package added to path
		case marked:
			// cycle detected
			for i, tn := range path {
				if t.obj.pkg != check.pkg {
					panic("type cycle via package-external type")
				}
				if tn == t.obj {
					check.cycleError(path[i:])
					t.info = invalid
					t.underlying = Typ[Invalid]
					return invalid
				}
			}
			panic("cycle start not found")
		}
		return t.info
	}

	return valid
}
