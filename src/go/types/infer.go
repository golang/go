// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements type parameter inference given
// a list of concrete arguments and a parameter list.

package types

import "go/token"

// infer returns the list of actual type arguments for the given list of type parameters tparams
// by inferring them from the actual arguments args for the parameters params. If infer fails to
// determine all type arguments, an error is reported and the result is nil.
func (check *Checker) infer(pos token.Pos, tparams []*TypeName, params *Tuple, args []*operand) []Type {
	assert(params.Len() == len(args))

	// targs is the list of inferred type parameter types.
	targs := make([]Type, len(tparams))

	// Terminology: TPP = type-parameterized function parameter

	// 1st pass: Unify parameter and argument types for TPPs with typed arguments
	//           and collect the indices of TPPs with untyped arguments.
	var indices []int
	for i, arg := range args {
		par := params.At(i)
		if IsParameterized(par.typ) {
			if arg.mode == invalid {
				// TODO(gri) we might still be able to infer all targs by
				//           simply ignoring (continue) invalid args
				return nil // error was reported earlier
			}
			if isTyped(arg.typ) {
				if !check.identical0(par.typ, arg.typ, true, nil, targs) {
					// Calling subst for an error message can cause problems.
					// TODO(gri) Determine best approach here.
					// check.errorf(arg.pos(), "type %s for %s does not match %s = %s",
					// 	arg.typ, arg.expr, par.typ, check.subst(pos, par.typ, tparams, targs),
					// )
					check.errorf(arg.pos(), "type %s for %s does not match %s", arg.typ, arg.expr, par.typ)
					return nil
				}
			} else {
				indices = append(indices, i)
			}
		}
	}

	// Some of the TPPs with untyped arguments may have been given a type
	// indirectly via a TPP with a typed argument; we can ignore those now.
	j := 0
	for _, i := range indices {
		par := params.At(i)
		// Since untyped types are all basic (i.e., unstructured) types, an
		// untyped argument will never match a structured parameter type; the
		// only parameter type it can possibly match against is a *TypeParam.
		// Thus, only keep the indices of TPPs that are unstructured and which
		// don't have a type inferred yet.
		if tpar, _ := par.typ.(*TypeParam); tpar != nil && targs[tpar.index] == nil {
			indices[j] = i
			j++
		}
	}
	indices = indices[:j]

	// 2nd pass: Unify parameter and default argument types for remaining TPPs.
	for _, i := range indices {
		par := params.At(i)
		arg := args[i]
		targ := Default(arg.typ)
		// The default type for an untyped nil is untyped nil. We must not
		// infer an untyped nil type as type parameter type. Ignore untyped
		// nil by making sure all default argument types are typed.
		if isTyped(targ) && !check.identical0(par.typ, targ, true, nil, targs) {
			// TODO(gri) see TODO comment above
			// check.errorf(arg.pos(), "default type %s for %s does not match %s = %s",
			// 	Default(arg.typ), arg.expr, par.typ, check.subst(pos, par.typ, tparams, targs),
			// )
			check.errorf(arg.pos(), "default type %s for %s does not match %s", Default(arg.typ), arg.expr, par.typ)
			return nil
		}
	}

	// Check if all type parameters have been determined.
	// TODO(gri) consider moving this outside this function and then we won't need to pass in pos
	for i, t := range targs {
		if t == nil {
			tpar := tparams[i]
			ppos := check.fset.Position(tpar.pos).String()
			check.errorf(pos, "cannot infer %s (%s)", tpar.name, ppos)
			return nil
		}
	}

	return targs
}

// IsParameterized reports whether typ contains any type parameters.
func IsParameterized(typ Type) bool {
	return isParameterized(typ, make(map[Type]bool))
}

func isParameterized(typ Type, seen map[Type]bool) (res bool) {
	// detect cycles
	// TODO(gri) can/should this be a Checker map?
	if x, ok := seen[typ]; ok {
		return x
	}
	seen[typ] = false
	defer func() {
		seen[typ] = res
	}()

	switch t := typ.(type) {
	case nil, *Basic: // TODO(gri) should nil be handled here?
		break

	case *Array:
		return isParameterized(t.elem, seen)

	case *Slice:
		return isParameterized(t.elem, seen)

	case *Struct:
		for _, fld := range t.fields {
			if isParameterized(fld.typ, seen) {
				return true
			}
		}

	case *Pointer:
		return isParameterized(t.base, seen)

	case *Tuple:
		n := t.Len()
		for i := 0; i < n; i++ {
			if isParameterized(t.At(i).typ, seen) {
				return true
			}
		}

	case *Signature:
		assert(t.tparams == nil) // TODO(gri) is this correct?
		// TODO(gri) Rethink check below: contract interfaces
		// have methods where the receiver is a contract type
		// parameter, by design.
		//assert(t.recv == nil || !isParameterized(t.recv.typ))
		return isParameterized(t.params, seen) || isParameterized(t.results, seen)

	case *Interface:
		t.assertCompleteness()
		for _, m := range t.allMethods {
			if isParameterized(m.typ, seen) {
				return true
			}
		}

	case *Map:
		return isParameterized(t.key, seen) || isParameterized(t.elem, seen)

	case *Chan:
		return isParameterized(t.elem, seen)

	case *Named:
		return isParameterizedList(t.targs, seen)

	case *TypeParam:
		return true

	default:
		unreachable()
	}

	return false
}

// IsParameterizedList reports whether any type in list is parameterized.
func IsParameterizedList(list []Type) bool {
	return isParameterizedList(list, make(map[Type]bool))
}

func isParameterizedList(list []Type, seen map[Type]bool) bool {
	for _, t := range list {
		if isParameterized(t, seen) {
			return true
		}
	}
	return false
}
