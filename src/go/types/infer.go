// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements type parameter inference given
// a list of concrete arguments and a parameter list.

package types

import "go/token"

// infer returns the list of actual type arguments for the given list of type parameters tparams
// by inferring them from the actual arguments args for the parameters pars. If infer fails to
// determine all type arguments, an error is reported and the result is nil.
func (check *Checker) infer(pos token.Pos, tparams []*TypeName, params *Tuple, args []*operand) []Type {
	assert(params.Len() == len(args))

	targs := make([]Type, len(tparams))

	// determine indices of type-parametrized parameters
	var indices []int
	for i := 0; i < params.Len(); i++ {
		par := params.At(i).typ
		if isParametrized(par) {
			indices = append(indices, i)
		}
	}

	// 1st pass: unify parameter and argument types for typed arguments
	for _, i := range indices {
		arg := args[i]
		if arg.mode == invalid {
			// TODO(gri) we might still be able to infer all targs by
			//           simply ignoring (continue) invalid args
			return nil // error was reported earlier
		}
		if isUntyped(arg.typ) {
			continue // handled in 2nd pass
		}
		par := params.At(i)
		if !check.identical0(par.typ, arg.typ, true, nil, targs) {
			check.errorf(arg.pos(), "type %s for %s does not match %s = %s", arg.typ, arg.expr, par.typ, check.subst(par.typ, tparams, targs))
			return nil
		}
	}

	// 2nd pass: unify parameter and default argument types for remaining parametrized parameter types with untyped arguments
	for _, i := range indices {
		arg := args[i]
		if isTyped(arg.typ) {
			continue // handled in 1st pass
		}
		par := params.At(i)
		if !check.identical0(par.typ, Default(arg.typ), true, nil, targs) {
			check.errorf(arg.pos(), "default type %s for %s does not match %s = %s", Default(arg.typ), arg.expr, par.typ, check.subst(par.typ, tparams, targs))
			return nil
		}
	}

	// check if all type parameters have been determined
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

// isParametrized reports whether typ contains any type parameters.
// TODO(gri) do we need to handle cycles here?
func isParametrized(typ Type) bool {
	switch t := typ.(type) {
	case nil, *Basic, *Named: // TODO(gri) should nil be handled here?
		break

	case *Array:
		return isParametrized(t.elem)

	case *Slice:
		return isParametrized(t.elem)

	case *Struct:
		for _, fld := range t.fields {
			if isParametrized(fld.typ) {
				return true
			}
		}

	case *Pointer:
		return isParametrized(t.base)

	case *Tuple:
		n := t.Len()
		for i := 0; i < n; i++ {
			if isParametrized(t.At(i).typ) {
				return true
			}
		}

	case *Signature:
		assert(t.tparams == nil)                             // TODO(gri) is this correct?
		assert(t.recv == nil || !isParametrized(t.recv.typ)) // interface method receiver may not be nil
		return isParametrized(t.params) || isParametrized(t.results)

	case *Interface:
		panic("unimplemented")

	case *Map:
		return isParametrized(t.key) || isParametrized(t.elem)

	case *Chan:
		return isParametrized(t.elem)

	case *TypeParam:
		return true

	default:
		unreachable()
	}

	return false
}
