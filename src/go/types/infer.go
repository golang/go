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
		if isParameterized(par.typ) {
			if arg.mode == invalid {
				// TODO(gri) we might still be able to infer all targs by
				//           simply ignoring (continue) invalid args
				return nil // error was reported earlier
			}
			if isTyped(arg.typ) {
				if !check.identical0(par.typ, arg.typ, true, nil, targs) {
					check.errorf(arg.pos(), "type %s for %s does not match %s = %s",
						arg.typ, arg.expr, par.typ, check.subst(par.typ, tparams, targs),
					)
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
			check.errorf(arg.pos(), "default type %s for %s does not match %s = %s",
				Default(arg.typ), arg.expr, par.typ, check.subst(par.typ, tparams, targs),
			)
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

// isParameterized reports whether typ contains any type parameters.
// TODO(gri) do we need to handle cycles here?
func isParameterized(typ Type) bool {
	switch t := typ.(type) {
	case nil, *Basic, *Named: // TODO(gri) should nil be handled here?
		break

	case *Array:
		return isParameterized(t.elem)

	case *Slice:
		return isParameterized(t.elem)

	case *Struct:
		for _, fld := range t.fields {
			if isParameterized(fld.typ) {
				return true
			}
		}

	case *Pointer:
		return isParameterized(t.base)

	case *Tuple:
		n := t.Len()
		for i := 0; i < n; i++ {
			if isParameterized(t.At(i).typ) {
				return true
			}
		}

	case *Signature:
		assert(t.tparams == nil) // TODO(gri) is this correct?
		// TODO(gri) Rethink check below: contract interfaces
		// have methods where the receiver is a contract type
		// parameter, by design.
		//assert(t.recv == nil || !isParameterized(t.recv.typ))
		return isParameterized(t.params) || isParameterized(t.results)

	case *Interface:
		if t.allMethods == nil {
			panic("incomplete method")
		}
		for _, m := range t.allMethods {
			if isParameterized(m.typ) {
				return true
			}
		}

	case *Map:
		return isParameterized(t.key) || isParameterized(t.elem)

	case *Chan:
		return isParameterized(t.elem)

	case *Parameterized:
		return isParameterizedList(t.targs)

	case *TypeParam:
		return true

	default:
		unreachable()
	}

	return false
}

// isParameterizedList reports whether any type in list is parameterized.
func isParameterizedList(list []Type) bool {
	for _, t := range list {
		if isParameterized(t) {
			return true
		}
	}
	return false
}
