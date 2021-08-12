// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements instantiation of generic types
// through substitution of type parameters by type arguments.

package types

import (
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
	var tparams []*TypeName
	switch t := typ.(type) {
	case *Named:
		tparams = t.TParams().list()
	case *Signature:
		tparams = t.TParams().list()
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
	default:
		// only types and functions can be generic
		panic(fmt.Sprintf("%v: cannot instantiate %v", pos, typ))
	}

	inst := check.instantiate(pos, typ, tparams, targs, posList)
	if verify && len(tparams) == len(targs) {
		check.verify(pos, tparams, targs, posList)
	}
	return inst
}

func (check *Checker) instantiate(pos token.Pos, typ Type, tparams []*TypeName, targs []Type, posList []token.Pos) (res Type) {
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
				under = res.Underlying()
			}
			check.trace(pos, "=> %s (under = %s)", res, under)
		}()
	}

	assert(len(posList) <= len(targs))

	// TODO(gri) What is better here: work with TypeParams, or work with TypeNames?

	if len(tparams) == 0 {
		return typ // nothing to do (minor optimization)
	}

	smap := makeSubstMap(tparams, targs)

	return check.subst(pos, typ, smap)
}

// InstantiateLazy is like Instantiate, but avoids actually
// instantiating the type until needed. typ must be a *Named
// type.
func (check *Checker) InstantiateLazy(pos token.Pos, typ Type, targs []Type, posList []token.Pos, verify bool) Type {
	// Don't use asNamed here: we don't want to expand the base during lazy
	// instantiation.
	base := typ.(*Named)
	if base == nil {
		panic(fmt.Sprintf("%v: cannot instantiate %v", pos, typ))
	}
	if verify && base.TParams().Len() == len(targs) {
		check.later(func() {
			check.verify(pos, base.tparams.list(), targs, posList)
		})
	}
	h := instantiatedHash(base, targs)
	if check != nil {
		// typ may already have been instantiated with identical type arguments. In
		// that case, re-use the existing instance.
		if named := check.typMap[h]; named != nil {
			return named
		}
	}

	tname := NewTypeName(pos, base.obj.pkg, base.obj.name, nil)
	named := check.newNamed(tname, base, nil, nil, nil) // methods and tparams are set when named is loaded.
	named.targs = targs
	named.instance = &instance{pos, posList}

	if check != nil {
		check.typMap[h] = named
	}

	return named
}

func (check *Checker) verify(pos token.Pos, tparams []*TypeName, targs []Type, posList []token.Pos) {
	if check == nil {
		panic("cannot have nil Checker if verifying constraints")
	}

	smap := makeSubstMap(tparams, targs)
	for i, tname := range tparams {
		// best position for error reporting
		pos := pos
		if i < len(posList) {
			pos = posList[i]
		}

		// stop checking bounds after the first failure
		if !check.satisfies(pos, targs[i], tname.typ.(*TypeParam), smap) {
			break
		}
	}
}

// satisfies reports whether the type argument targ satisfies the constraint of type parameter
// parameter tpar (after any of its type parameters have been substituted through smap).
// A suitable error is reported if the result is false.
// TODO(gri) This should be a method of interfaces or type sets.
func (check *Checker) satisfies(pos token.Pos, targ Type, tpar *TypeParam, smap *substMap) bool {
	iface := tpar.iface()
	if iface.Empty() {
		return true // no type bound
	}

	// The type parameter bound is parameterized with the same type parameters
	// as the instantiated type; before we can use it for bounds checking we
	// need to instantiate it with the type arguments with which we instantiate
	// the parameterized type.
	iface = check.subst(pos, iface, smap).(*Interface)

	// if iface is comparable, targ must be comparable
	// TODO(gri) the error messages needs to be better, here
	if iface.IsComparable() && !Comparable(targ) {
		if tpar := asTypeParam(targ); tpar != nil && tpar.iface().typeSet().IsTop() {
			check.softErrorf(atPos(pos), _Todo, "%s has no constraints", targ)
			return false
		}
		check.softErrorf(atPos(pos), _Todo, "%s does not satisfy comparable", targ)
		return false
	}

	// targ must implement iface (methods)
	// - check only if we have methods
	if iface.NumMethods() > 0 {
		// If the type argument is a pointer to a type parameter, the type argument's
		// method set is empty.
		// TODO(gri) is this what we want? (spec question)
		if base, isPtr := deref(targ); isPtr && asTypeParam(base) != nil {
			check.errorf(atPos(pos), 0, "%s has no methods", targ)
			return false
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
				check.softErrorf(atPos(pos), _Todo,
					"%s does not satisfy %s: wrong method signature\n\tgot  %s\n\twant %s",
					targ, tpar.bound, wrong, m,
				)
			} else {
				check.softErrorf(atPos(pos), 0, "%s does not satisfy %s (missing method %s)", targ, tpar.bound, m.name)
			}
			return false
		}
	}

	// targ's underlying type must also be one of the interface types listed, if any
	if iface.typeSet().types == nil {
		return true // nothing to do
	}

	// If targ is itself a type parameter, each of its possible types, but at least one, must be in the
	// list of iface types (i.e., the targ type list must be a non-empty subset of the iface types).
	if targ := asTypeParam(targ); targ != nil {
		targBound := targ.iface()
		if targBound.typeSet().types == nil {
			check.softErrorf(atPos(pos), _Todo, "%s does not satisfy %s (%s has no type constraints)", targ, tpar.bound, targ)
			return false
		}
		return iface.is(func(typ Type, tilde bool) bool {
			// TODO(gri) incorporate tilde information!
			if !iface.isSatisfiedBy(typ) {
				// TODO(gri) match this error message with the one below (or vice versa)
				check.softErrorf(atPos(pos), 0, "%s does not satisfy %s (%s type constraint %s not found in %s)", targ, tpar.bound, targ, typ, iface.typeSet().types)
				return false
			}
			return true
		})
	}

	// Otherwise, targ's type or underlying type must also be one of the interface types listed, if any.
	if !iface.isSatisfiedBy(targ) {
		check.softErrorf(atPos(pos), _Todo, "%s does not satisfy %s (%s not found in %s)", targ, tpar.bound, targ, iface.typeSet().types)
		return false
	}

	return true
}
