// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

// validType verifies that the given type does not "expand" indefinitely
// producing a cycle in the type graph. Cycles are detected by marking
// defined types.
// (Cycles involving alias types, as in "type A = [10]A" are detected
// earlier, via the objDecl cycle detection mechanism.)
func (check *Checker) validType(typ *Named) {
	check.validType0(typ, nil)
}

type typeInfo uint

func (check *Checker) validType0(typ Type, path []Object) typeInfo {
	const (
		unknown typeInfo = iota
		marked
		valid
		invalid
	)

	switch t := typ.(type) {
	case *Array:
		return check.validType0(t.elem, path)

	case *Struct:
		for _, f := range t.fields {
			if check.validType0(f.typ, path) == invalid {
				return invalid
			}
		}

	case *Union:
		for _, t := range t.terms {
			if check.validType0(t.typ, path) == invalid {
				return invalid
			}
		}

	case *Interface:
		for _, etyp := range t.embeddeds {
			if check.validType0(etyp, path) == invalid {
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

		// don't report a 2nd error if we already know the type is invalid
		// (e.g., if a cycle was detected earlier, via under).
		if t.underlying == Typ[Invalid] {
			check.infoMap[t] = invalid
			return invalid
		}

		switch check.infoMap[t] {
		case unknown:
			check.infoMap[t] = marked
			check.infoMap[t] = check.validType0(t.fromRHS, append(path, t.obj))
		case marked:
			// cycle detected
			for i, tn := range path {
				// Even though validType now can hande cycles through external
				// types, we can't have cycles through external types because
				// no such types are detected yet.
				// TODO(gri) Remove this check once we can detect such cycles,
				//           and adjust cycleError accordingly.
				if t.obj.pkg != check.pkg {
					panic("type cycle via package-external type")
				}
				if tn == t.obj {
					check.cycleError(path[i:])
					check.infoMap[t] = invalid
					// don't modify imported types (leads to race condition, see #35049)
					if t.obj.pkg == check.pkg {
						t.underlying = Typ[Invalid]
					}
					return invalid
				}
			}
			panic("cycle start not found")
		}
		return check.infoMap[t]
	}

	return valid
}
