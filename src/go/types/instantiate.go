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

// Instantiate instantiates the type typ with the given type arguments
// targs. To check type constraint satisfaction, verify must be set.
// pos and posList correspond to the instantiation and type argument
// positions respectively; posList may be nil or shorter than the number
// of type arguments provided.
// typ must be a *Named or a *Signature type, and its number of type
// parameters must match the number of provided type arguments.
// The receiver (check) may be nil if and only if verify is not set.
// The result is a new, instantiated (not generic) type of the same kind
// (either a *Named or a *Signature).
// Any methods attached to a *Named are simply copied; they are not
// instantiated.
func (check *Checker) Instantiate(pos token.Pos, typ Type, targs []Type, posList []token.Pos, verify bool) (res Type) {
	var inst Type
	switch t := typ.(type) {
	case *Named:
		inst = check.instantiateLazy(pos, t, targs)
	case *Signature:
		tparams := t.TParams().list()
		defer func() {
			// If we had an unexpected failure somewhere don't panic below when
			// asserting res.(*Signature). Check for *Signature in case Typ[Invalid]
			// is returned.
			if _, ok := res.(*Signature); !ok {
				return
			}
			// If the signature doesn't use its type parameters, subst
			// will not make a copy. In that case, make a copy now (so
			// we can set tparams to nil w/o causing side-effects).
			if t == res {
				copy := *t
				res = &copy
			}
			// After instantiating a generic signature, it is not generic
			// anymore; we need to set tparams to nil.
			res.(*Signature).tparams = nil
		}()
		inst = check.instantiate(pos, typ, tparams, targs, nil)
	default:
		// only types and functions can be generic
		panic(fmt.Sprintf("%v: cannot instantiate %v", pos, typ))
	}

	if verify {
		if check == nil {
			panic("cannot have nil Checker if verifying constraints")
		}
		assert(len(posList) <= len(targs))
		check.later(func() {
			// Collect tparams again because lazily loaded *Named types may not have
			// had tparams set up above.
			var tparams []*TypeName
			switch t := typ.(type) {
			case *Named:
				tparams = t.TParams().list()
			case *Signature:
				tparams = t.TParams().list()
			}
			// Avoid duplicate errors; instantiate will have complained if tparams
			// and targs do not have the same length.
			if len(tparams) == len(targs) {
				if i, err := check.verify(pos, tparams, targs); err != nil {
					// best position for error reporting
					pos := pos
					if i < len(posList) {
						pos = posList[i]
					}
					check.softErrorf(atPos(pos), _Todo, err.Error())
				}
			}
		})
	}

	return inst
}

func (check *Checker) instantiate(pos token.Pos, typ Type, tparams []*TypeName, targs []Type, typMap map[string]*Named) (res Type) {
	// the number of supplied types must match the number of type parameters
	if len(targs) != len(tparams) {
		// TODO(gri) provide better error message
		if check != nil {
			check.errorf(atPos(pos), _Todo, "got %d arguments but %d type parameters", len(targs), len(tparams))
			return Typ[Invalid]
		}
		panic(fmt.Sprintf("%v: got %d arguments but %d type parameters", pos, len(targs), len(tparams)))
	}

	if check != nil && trace {
		check.trace(pos, "-- instantiating %s with %s", typ, typeListString(targs))
		check.indent++
		defer func() {
			check.indent--
			var under Type
			if res != nil {
				// Calling under() here may lead to endless instantiations.
				// Test case: type T[P any] T[P]
				// TODO(gri) investigate if that's a bug or to be expected.
				under = safeUnderlying(res)
			}
			check.trace(pos, "=> %s (under = %s)", res, under)
		}()
	}

	// TODO(gri) What is better here: work with TypeParams, or work with TypeNames?

	if len(tparams) == 0 {
		return typ // nothing to do (minor optimization)
	}

	return check.subst(pos, typ, makeSubstMap(tparams, targs), typMap)
}

// instantiateLazy avoids actually instantiating the type until needed. typ
// must be a *Named type.
func (check *Checker) instantiateLazy(pos token.Pos, orig *Named, targs []Type) Type {
	h := instantiatedHash(orig, targs)
	if check != nil {
		// typ may already have been instantiated with identical type arguments. In
		// that case, re-use the existing instance.
		if named := check.typMap[h]; named != nil {
			return named
		}
	}

	tname := NewTypeName(pos, orig.obj.pkg, orig.obj.name, nil)
	named := check.newNamed(tname, orig, nil, nil, nil) // methods and tparams are set when named is loaded
	named.targs = targs
	named.instance = &instance{pos}

	if check != nil {
		check.typMap[h] = named
	}

	return named
}

func (check *Checker) verify(pos token.Pos, tparams []*TypeName, targs []Type) (int, error) {
	smap := makeSubstMap(tparams, targs)
	for i, tname := range tparams {
		// stop checking bounds after the first failure
		if err := check.satisfies(pos, targs[i], tname.typ.(*TypeParam), smap); err != nil {
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
	if iface.Empty() {
		return nil // no type bound
	}

	// TODO(rfindley): it would be great if users could pass in a qualifier here,
	// rather than falling back to verbose qualification. Maybe this can be part
	// of a the shared environment.
	var qf Qualifier
	if check != nil {
		qf = check.qualifier
	}
	errorf := func(format string, args ...interface{}) error {
		return errors.New(sprintf(nil, qf, format, args...))
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

	// targ's underlying type must also be one of the interface types listed, if any
	if !iface.typeSet().hasTerms() {
		return nil // nothing to do
	}

	// If targ is itself a type parameter, each of its possible types, but at least one, must be in the
	// list of iface types (i.e., the targ type list must be a non-empty subset of the iface types).
	if targ := asTypeParam(targ); targ != nil {
		targBound := targ.iface()
		if !targBound.typeSet().hasTerms() {
			return errorf("%s does not satisfy %s (%s has no type constraints)", targ, tpar.bound, targ)
		}
		if !targBound.typeSet().subsetOf(iface.typeSet()) {
			// TODO(gri) need better error message
			return errorf("%s does not satisfy %s", targ, tpar.bound)
		}
		return nil
	}

	// Otherwise, targ's type or underlying type must also be one of the interface types listed, if any.
	if !iface.typeSet().includes(targ) {
		// TODO(gri) better error message
		return errorf("%s does not satisfy %s", targ, tpar.bound)
	}

	return nil
}
