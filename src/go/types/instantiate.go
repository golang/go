// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements instantiation of generic types
// through substitution of type parameters by type arguments.

package types

import (
	"errors"
	"fmt"
	"go/token"
)

// Instantiate instantiates the type typ with the given type arguments targs.
// typ must be a *Named or a *Signature type, and its number of type parameters
// must match the number of provided type arguments. The result is a new,
// instantiated (not parameterized) type of the same kind (either a *Named or a
// *Signature). Any methods attached to a *Named are simply copied; they are
// not instantiated.
//
// If ctxt is non-nil, it may be used to de-dupe the instance against previous
// instances with the same identity.
//
// If verify is set and constraint satisfaction fails, the returned error may
// wrap an *ArgumentError indicating which type argument did not satisfy its
// corresponding type parameter constraint, and why.
//
// TODO(rfindley): change this function to also return an error if lengths of
// tparams and targs do not match.
func Instantiate(ctxt *Context, typ Type, targs []Type, validate bool) (Type, error) {
	inst := (*Checker)(nil).instance(token.NoPos, typ, targs, ctxt)

	var err error
	if validate {
		var tparams []*TypeParam
		switch t := typ.(type) {
		case *Named:
			tparams = t.TypeParams().list()
		case *Signature:
			tparams = t.TypeParams().list()
		}
		if i, err := (*Checker)(nil).verify(token.NoPos, tparams, targs); err != nil {
			return inst, &ArgumentError{i, err}
		}
	}

	return inst, err
}

// instance creates a type or function instance using the given original type
// typ and arguments targs. For Named types the resulting instance will be
// unexpanded.
func (check *Checker) instance(pos token.Pos, orig Type, targs []Type, ctxt *Context) (res Type) {
	var h string
	if ctxt != nil {
		h = ctxt.instanceHash(orig, targs)
		// typ may already have been instantiated with identical type arguments. In
		// that case, re-use the existing instance.
		if inst := ctxt.lookup(h, orig, targs); inst != nil {
			return inst
		}
	}

	switch orig := orig.(type) {
	case *Named:
		tname := NewTypeName(pos, orig.obj.pkg, orig.obj.name, nil)
		named := check.newNamed(tname, orig, nil, nil, nil) // underlying, tparams, and methods are set when named is resolved
		named.targs = NewTypeList(targs)
		named.resolver = func(ctxt *Context, n *Named) (*TypeParamList, Type, []*Func) {
			return expandNamed(ctxt, n, pos)
		}
		res = named

	case *Signature:
		tparams := orig.TypeParams()
		if !check.validateTArgLen(pos, tparams.Len(), len(targs)) {
			return Typ[Invalid]
		}
		if tparams.Len() == 0 {
			return orig // nothing to do (minor optimization)
		}
		sig := check.subst(pos, orig, makeSubstMap(tparams.list(), targs), ctxt).(*Signature)
		// If the signature doesn't use its type parameters, subst
		// will not make a copy. In that case, make a copy now (so
		// we can set tparams to nil w/o causing side-effects).
		if sig == orig {
			copy := *sig
			sig = &copy
		}
		// After instantiating a generic signature, it is not generic
		// anymore; we need to set tparams to nil.
		sig.tparams = nil
		res = sig
	default:
		// only types and functions can be generic
		panic(fmt.Sprintf("%v: cannot instantiate %v", pos, orig))
	}

	if ctxt != nil {
		// It's possible that we've lost a race to add named to the context.
		// In this case, use whichever instance is recorded in the context.
		res = ctxt.update(h, orig, targs, res)
	}

	return res
}

// validateTArgLen verifies that the length of targs and tparams matches,
// reporting an error if not. If validation fails and check is nil,
// validateTArgLen panics.
func (check *Checker) validateTArgLen(pos token.Pos, ntparams, ntargs int) bool {
	if ntargs != ntparams {
		// TODO(gri) provide better error message
		if check != nil {
			check.errorf(atPos(pos), _Todo, "got %d arguments but %d type parameters", ntargs, ntparams)
			return false
		}
		panic(fmt.Sprintf("%v: got %d arguments but %d type parameters", pos, ntargs, ntparams))
	}
	return true
}

func (check *Checker) verify(pos token.Pos, tparams []*TypeParam, targs []Type) (int, error) {
	smap := makeSubstMap(tparams, targs)
	for i, tpar := range tparams {
		// stop checking bounds after the first failure
		if err := check.satisfies(pos, targs[i], tpar, smap); err != nil {
			return i, err
		}
	}
	return -1, nil
}

// satisfies reports whether the type argument targ satisfies the constraint of type parameter
// parameter tpar (after any of its type parameters have been substituted through smap).
// A suitable error is reported if the result is false.
// TODO(gri) This should be a method of interfaces or type sets.
func (check *Checker) satisfies(pos token.Pos, targ Type, tpar *TypeParam, smap substMap) error {
	iface := tpar.iface()

	// Every type argument satisfies interface{}.
	if iface.Empty() {
		return nil
	}

	// A type argument that is a type parameter with an empty type set satisfies any constraint.
	// (The empty set is a subset of any set.)
	if targ := asTypeParam(targ); targ != nil && targ.iface().typeSet().IsEmpty() {
		return nil
	}

	// TODO(rfindley): it would be great if users could pass in a qualifier here,
	// rather than falling back to verbose qualification. Maybe this can be part
	// of the shared context.
	var qf Qualifier
	if check != nil {
		qf = check.qualifier
	}
	errorf := func(format string, args ...interface{}) error {
		return errors.New(sprintf(nil, qf, false, format, args...))
	}

	// No type argument with non-empty type set satisfies the empty type set.
	if iface.typeSet().IsEmpty() {
		return errorf("%s does not satisfy %s (constraint type set is empty)", targ, tpar.bound)
	}

	// The type parameter bound is parameterized with the same type parameters
	// as the instantiated type; before we can use it for bounds checking we
	// need to instantiate it with the type arguments with which we instantiate
	// the parameterized type.
	iface = check.subst(pos, iface, smap, nil).(*Interface)

	// if iface is comparable, targ must be comparable
	// TODO(gri) the error messages needs to be better, here
	if iface.IsComparable() && !Comparable(targ) {
		if tpar := asTypeParam(targ); tpar != nil && tpar.iface().typeSet().IsAll() {
			return errorf("%s has no constraints", targ)
		}
		return errorf("%s does not satisfy comparable", targ)
	}

	// targ must implement iface (methods)
	// - check only if we have methods
	if iface.NumMethods() > 0 {
		// If the type argument is a pointer to a type parameter, the type argument's
		// method set is empty.
		// TODO(gri) is this what we want? (spec question)
		if base, isPtr := deref(targ); isPtr && asTypeParam(base) != nil {
			return errorf("%s has no methods", targ)
		}
		if m, wrong := check.missingMethod(targ, iface, true); m != nil {
			// TODO(gri) needs to print updated name to avoid major confusion in error message!
			//           (print warning for now)
			// Old warning:
			// check.softErrorf(pos, "%s does not satisfy %s (warning: name not updated) = %s (missing method %s)", targ, tpar.bound, iface, m)
			if wrong != nil {
				// TODO(gri) This can still report uninstantiated types which makes the error message
				//           more difficult to read then necessary.
				// TODO(rFindley) should this use parentheses rather than ':' for qualification?
				return errorf("%s does not satisfy %s: wrong method signature\n\tgot  %s\n\twant %s",
					targ, tpar.bound, wrong, m,
				)
			}
			return errorf("%s does not satisfy %s (missing method %s)", targ, tpar.bound, m.name)
		}
	}

	// targ must also be in the set of types of iface, if any.
	// Constraints with empty type sets were already excluded above.
	if !iface.typeSet().hasTerms() {
		return nil // nothing to do
	}

	// If targ is itself a type parameter, each of its possible types must be in the set
	// of iface types (i.e., the targ type set must be a subset of the iface type set).
	// Type arguments with empty type sets were already excluded above.
	if targ := asTypeParam(targ); targ != nil {
		targBound := targ.iface()
		if !targBound.typeSet().subsetOf(iface.typeSet()) {
			// TODO(gri) report which type is missing
			return errorf("%s does not satisfy %s", targ, tpar.bound)
		}
		return nil
	}

	// Otherwise, targ's type must be included in the iface type set.
	if !iface.typeSet().includes(targ) {
		// TODO(gri) report which type is missing
		return errorf("%s does not satisfy %s", targ, tpar.bound)
	}

	return nil
}
