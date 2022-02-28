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
	check.validType0(typ, nil, nil)
}

type typeInfo uint

// validType0 checks if the given type is valid. If typ is a type parameter
// its value is looked up in the provided environment. The environment is
// nil if typ is not part of (the RHS of) an instantiated type, in that case
// any type parameter encountered must be from an enclosing function and can
// be ignored. The path is the list of type names that lead to the current typ.
func (check *Checker) validType0(typ Type, env *tparamEnv, path []Object) typeInfo {
	const (
		unknown typeInfo = iota
		marked
		valid
		invalid
	)

	switch t := typ.(type) {
	case nil:
		// We should never see a nil type but be conservative and panic
		// only in debug mode.
		if debug {
			panic("validType0(nil)")
		}

	case *Array:
		return check.validType0(t.elem, env, path)

	case *Struct:
		for _, f := range t.fields {
			if check.validType0(f.typ, env, path) == invalid {
				return invalid
			}
		}

	case *Union:
		for _, t := range t.terms {
			if check.validType0(t.typ, env, path) == invalid {
				return invalid
			}
		}

	case *Interface:
		for _, etyp := range t.embeddeds {
			if check.validType0(etyp, env, path) == invalid {
				return invalid
			}
		}

	case *Named:
		// Don't report a 2nd error if we already know the type is invalid
		// (e.g., if a cycle was detected earlier, via under).
		if t.underlying == Typ[Invalid] {
			check.infoMap[t] = invalid
			return invalid
		}

		switch check.infoMap[t] {
		case unknown:
			check.infoMap[t] = marked
			check.infoMap[t] = check.validType0(t.orig.fromRHS, env.push(t), append(path, t.obj))
		case marked:
			// We have seen type t before and thus must have a cycle.
			check.infoMap[t] = invalid
			// t cannot be in an imported package otherwise that package
			// would have reported a type cycle and couldn't have been
			// imported in the first place.
			assert(t.obj.pkg == check.pkg)
			t.underlying = Typ[Invalid] // t is in the current package (no race possibility)
			// Find the starting point of the cycle and report it.
			for i, tn := range path {
				if tn == t.obj {
					check.cycleError(path[i:])
					return invalid
				}
			}
			panic("cycle start not found")
		}
		return check.infoMap[t]

	case *TypeParam:
		// A type parameter stands for the type (argument) it was instantiated with.
		// Check the corresponding type argument for validity if we have one.
		if env != nil {
			if targ := env.tmap[t]; targ != nil {
				// Type arguments found in targ must be looked
				// up in the enclosing environment env.link.
				return check.validType0(targ, env.link, path)
			}
		}
	}

	return valid
}

// A tparamEnv provides the environment for looking up the type arguments
// with which type parameters for a given instance were instantiated.
// If we don't have an instance, the corresponding tparamEnv is nil.
type tparamEnv struct {
	tmap substMap
	link *tparamEnv
}

func (env *tparamEnv) push(typ *Named) *tparamEnv {
	// If typ is not an instantiated type there are no typ-specific
	// type parameters to look up and we don't need an environment.
	targs := typ.TypeArgs()
	if targs == nil {
		return nil // no instance => nil environment
	}

	// Populate tmap: remember the type argument for each type parameter.
	// We cannot use makeSubstMap because the number of type parameters
	// and arguments may not match due to errors in the source (too many
	// or too few type arguments). Populate tmap "manually".
	tparams := typ.TypeParams()
	n, m := targs.Len(), tparams.Len()
	if n > m {
		n = m // too many targs
	}
	tmap := make(substMap, n)
	for i := 0; i < n; i++ {
		tmap[tparams.At(i)] = targs.At(i)
	}

	return &tparamEnv{tmap: tmap, link: env}
}

// TODO(gri) Alternative implementation:
//           We may not need to build a stack of environments to
//           look up the type arguments for type parameters. The
//           same information should be available via the path:
//           We should be able to just walk the path backwards
//           and find the type arguments in the instance objects.
