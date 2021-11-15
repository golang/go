// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements instantiation of generic types
// through substitution of type parameters by type arguments.

package types2

import (
	"cmd/compile/internal/syntax"
	"errors"
	"fmt"
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
// be of dynamic type ArgumentError indicating which type argument did not
// satisfy its corresponding type parameter constraint, and why.
//
// TODO(rfindley): change this function to also return an error if lengths of
// tparams and targs do not match.
func Instantiate(ctxt *Context, typ Type, targs []Type, validate bool) (Type, error) {
	inst := (*Checker)(nil).instance(nopos, typ, targs, ctxt)

	var err error
	if validate {
		var tparams []*TypeParam
		switch t := typ.(type) {
		case *Named:
			tparams = t.TypeParams().list()
		case *Signature:
			tparams = t.TypeParams().list()
		}
		if i, err := (*Checker)(nil).verify(nopos, tparams, targs); err != nil {
			return inst, ArgumentError{i, err}
		}
	}

	return inst, err
}

// instance creates a type or function instance using the given original type
// typ and arguments targs. For Named types the resulting instance will be
// unexpanded.
func (check *Checker) instance(pos syntax.Pos, typ Type, targs []Type, ctxt *Context) Type {
	switch t := typ.(type) {
	case *Named:
		var h string
		if ctxt != nil {
			h = ctxt.typeHash(t, targs)
			// typ may already have been instantiated with identical type arguments. In
			// that case, re-use the existing instance.
			if named := ctxt.typeForHash(h, nil); named != nil {
				return named
			}
		}
		tname := NewTypeName(pos, t.obj.pkg, t.obj.name, nil)
		named := check.newNamed(tname, t, nil, nil, nil) // underlying, tparams, and methods are set when named is resolved
		named.targs = NewTypeList(targs)
		named.resolver = func(ctxt *Context, n *Named) (*TypeParamList, Type, []*Func) {
			return expandNamed(ctxt, n, pos)
		}
		if ctxt != nil {
			// It's possible that we've lost a race to add named to the context.
			// In this case, use whichever instance is recorded in the context.
			named = ctxt.typeForHash(h, named)
		}
		return named

	case *Signature:
		tparams := t.TypeParams()
		if !check.validateTArgLen(pos, tparams.Len(), len(targs)) {
			return Typ[Invalid]
		}
		if tparams.Len() == 0 {
			return typ // nothing to do (minor optimization)
		}
		sig := check.subst(pos, typ, makeSubstMap(tparams.list(), targs), ctxt).(*Signature)
		// If the signature doesn't use its type parameters, subst
		// will not make a copy. In that case, make a copy now (so
		// we can set tparams to nil w/o causing side-effects).
		if sig == t {
			copy := *sig
			sig = &copy
		}
		// After instantiating a generic signature, it is not generic
		// anymore; we need to set tparams to nil.
		sig.tparams = nil
		return sig
	}

	// only types and functions can be generic
	panic(fmt.Sprintf("%v: cannot instantiate %v", pos, typ))
}

// validateTArgLen verifies that the length of targs and tparams matches,
// reporting an error if not. If validation fails and check is nil,
// validateTArgLen panics.
func (check *Checker) validateTArgLen(pos syntax.Pos, ntparams, ntargs int) bool {
	if ntargs != ntparams {
		// TODO(gri) provide better error message
		if check != nil {
			check.errorf(pos, "got %d arguments but %d type parameters", ntargs, ntparams)
			return false
		}
		panic(fmt.Sprintf("%v: got %d arguments but %d type parameters", pos, ntargs, ntparams))
	}
	return true
}

func (check *Checker) verify(pos syntax.Pos, tparams []*TypeParam, targs []Type) (int, error) {
	// TODO(rfindley): it would be great if users could pass in a qualifier here,
	// rather than falling back to verbose qualification. Maybe this can be part
	// of the shared context.
	var qf Qualifier
	if check != nil {
		qf = check.qualifier
	}

	smap := makeSubstMap(tparams, targs)
	for i, tpar := range tparams {
		// The type parameter bound is parameterized with the same type parameters
		// as the instantiated type; before we can use it for bounds checking we
		// need to instantiate it with the type arguments with which we instantiated
		// the parameterized type.
		bound := check.subst(pos, tpar.bound, smap, nil)
		if err := check.implements(targs[i], bound, qf); err != nil {
			return i, err
		}
	}
	return -1, nil
}

// implements checks if V implements T and reports an error if it doesn't.
// If a qualifier is provided, it is used in error formatting.
func (check *Checker) implements(V, T Type, qf Qualifier) error {
	Vu := under(V)
	Tu := under(T)
	if Vu == Typ[Invalid] || Tu == Typ[Invalid] {
		return nil
	}

	errorf := func(format string, args ...interface{}) error {
		return errors.New(sprintf(qf, false, format, args...))
	}

	Ti, _ := Tu.(*Interface)
	if Ti == nil {
		return errorf("%s is not an interface", T)
	}

	// Every type satisfies the empty interface.
	if Ti.Empty() {
		return nil
	}
	// T is not the empty interface (i.e., the type set of T is restricted)

	// An interface V with an empty type set satisfies any interface.
	// (The empty set is a subset of any set.)
	Vi, _ := Vu.(*Interface)
	if Vi != nil && Vi.typeSet().IsEmpty() {
		return nil
	}
	// type set of V is not empty

	// No type with non-empty type set satisfies the empty type set.
	if Ti.typeSet().IsEmpty() {
		return errorf("cannot implement %s (empty type set)", T)
	}

	// If T is comparable, V must be comparable.
	// TODO(gri) the error messages could be better, here
	if Ti.IsComparable() && !Comparable(V) {
		if Vi != nil && Vi.Empty() {
			return errorf("empty interface %s does not implement %s", V, T)
		}
		return errorf("%s does not implement comparable", V)
	}

	// V must implement T (methods)
	// - check only if we have methods
	if Ti.NumMethods() > 0 {
		if m, wrong := check.missingMethod(V, Ti, true); m != nil {
			// TODO(gri) needs to print updated name to avoid major confusion in error message!
			//           (print warning for now)
			// Old warning:
			// check.softErrorf(pos, "%s does not implement %s (warning: name not updated) = %s (missing method %s)", V, T, Ti, m)
			if wrong != nil {
				// TODO(gri) This can still report uninstantiated types which makes the error message
				//           more difficult to read then necessary.
				return errorf("%s does not implement %s: wrong method signature\n\tgot  %s\n\twant %s",
					V, T, wrong, m,
				)
			}
			return errorf("%s does not implement %s (missing method %s)", V, T, m.name)
		}
	}

	// V must also be in the set of types of T, if any.
	// Constraints with empty type sets were already excluded above.
	if !Ti.typeSet().hasTerms() {
		return nil // nothing to do
	}

	// If V is itself an interface, each of its possible types must be in the set
	// of T types (i.e., the V type set must be a subset of the T type set).
	// Interfaces V with empty type sets were already excluded above.
	if Vi != nil {
		if !Vi.typeSet().subsetOf(Ti.typeSet()) {
			// TODO(gri) report which type is missing
			return errorf("%s does not implement %s", V, T)
		}
		return nil
	}

	// Otherwise, V's type must be included in the iface type set.
	if !Ti.typeSet().includes(V) {
		// TODO(gri) report which type is missing
		return errorf("%s does not implement %s", V, T)
	}

	return nil
}
