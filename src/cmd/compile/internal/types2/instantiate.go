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

// Instantiate instantiates the type orig with the given type arguments targs.
// orig must be a *Named or a *Signature type. If there is no error, the
// resulting Type is an instantiated type of the same kind (either a *Named or
// a *Signature). Methods attached to a *Named type are also instantiated, and
// associated with a new *Func that has the same position as the original
// method, but nil function scope.
//
// If ctxt is non-nil, it may be used to de-duplicate the instance against
// previous instances with the same identity. As a special case, generic
// *Signature origin types are only considered identical if they are pointer
// equivalent, so that instantiating distinct (but possibly identical)
// signatures will yield different instances. The use of a shared context does
// not guarantee that identical instances are deduplicated in all cases.
//
// If validate is set, Instantiate verifies that the number of type arguments
// and parameters match, and that the type arguments satisfy their
// corresponding type constraints. If verification fails, the resulting error
// may wrap an *ArgumentError indicating which type argument did not satisfy
// its corresponding type parameter constraint, and why.
//
// If validate is not set, Instantiate does not verify the type argument count
// or whether the type arguments satisfy their constraints. Instantiate is
// guaranteed to not return an error, but may panic. Specifically, for
// *Signature types, Instantiate will panic immediately if the type argument
// count is incorrect; for *Named types, a panic may occur later inside the
// *Named API.
func Instantiate(ctxt *Context, orig Type, targs []Type, validate bool) (Type, error) {
	if ctxt == nil {
		ctxt = NewContext()
	}
	if validate {
		var tparams []*TypeParam
		switch t := orig.(type) {
		case *Named:
			tparams = t.TypeParams().list()
		case *Signature:
			tparams = t.TypeParams().list()
		}
		if len(targs) != len(tparams) {
			return nil, fmt.Errorf("got %d type arguments but %s has %d type parameters", len(targs), orig, len(tparams))
		}
		if i, err := (*Checker)(nil).verify(nopos, tparams, targs, ctxt); err != nil {
			return nil, &ArgumentError{i, err}
		}
	}

	inst := (*Checker)(nil).instance(nopos, orig, targs, nil, ctxt)
	return inst, nil
}

// instance instantiates the given original (generic) function or type with the
// provided type arguments and returns the resulting instance. If an identical
// instance exists already in the given contexts, it returns that instance,
// otherwise it creates a new one.
//
// If expanding is non-nil, it is the Named instance type currently being
// expanded. If ctxt is non-nil, it is the context associated with the current
// type-checking pass or call to Instantiate. At least one of expanding or ctxt
// must be non-nil.
//
// For Named types the resulting instance may be unexpanded.
func (check *Checker) instance(pos syntax.Pos, orig Type, targs []Type, expanding *Named, ctxt *Context) (res Type) {
	// The order of the contexts below matters: we always prefer instances in the
	// expanding instance context in order to preserve reference cycles.
	//
	// Invariant: if expanding != nil, the returned instance will be the instance
	// recorded in expanding.inst.ctxt.
	var ctxts []*Context
	if expanding != nil {
		ctxts = append(ctxts, expanding.inst.ctxt)
	}
	if ctxt != nil {
		ctxts = append(ctxts, ctxt)
	}
	assert(len(ctxts) > 0)

	// Compute all hashes; hashes may differ across contexts due to different
	// unique IDs for Named types within the hasher.
	hashes := make([]string, len(ctxts))
	for i, ctxt := range ctxts {
		hashes[i] = ctxt.instanceHash(orig, targs)
	}

	// If local is non-nil, updateContexts return the type recorded in
	// local.
	updateContexts := func(res Type) Type {
		for i := len(ctxts) - 1; i >= 0; i-- {
			res = ctxts[i].update(hashes[i], orig, targs, res)
		}
		return res
	}

	// typ may already have been instantiated with identical type arguments. In
	// that case, re-use the existing instance.
	for i, ctxt := range ctxts {
		if inst := ctxt.lookup(hashes[i], orig, targs); inst != nil {
			return updateContexts(inst)
		}
	}

	switch orig := orig.(type) {
	case *Named:
		res = check.newNamedInstance(pos, orig, targs, expanding) // substituted lazily

	case *Signature:
		assert(expanding == nil) // function instances cannot be reached from Named types

		tparams := orig.TypeParams()
		if !check.validateTArgLen(pos, tparams.Len(), len(targs)) {
			return Typ[Invalid]
		}
		if tparams.Len() == 0 {
			return orig // nothing to do (minor optimization)
		}
		sig := check.subst(pos, orig, makeSubstMap(tparams.list(), targs), nil, ctxt).(*Signature)
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

	// Update all contexts; it's possible that we've lost a race.
	return updateContexts(res)
}

// validateTArgLen verifies that the length of targs and tparams matches,
// reporting an error if not. If validation fails and check is nil,
// validateTArgLen panics.
func (check *Checker) validateTArgLen(pos syntax.Pos, ntparams, ntargs int) bool {
	if ntargs != ntparams {
		// TODO(gri) provide better error message
		if check != nil {
			check.errorf(pos, _WrongTypeArgCount, "got %d arguments but %d type parameters", ntargs, ntparams)
			return false
		}
		panic(fmt.Sprintf("%v: got %d arguments but %d type parameters", pos, ntargs, ntparams))
	}
	return true
}

func (check *Checker) verify(pos syntax.Pos, tparams []*TypeParam, targs []Type, ctxt *Context) (int, error) {
	smap := makeSubstMap(tparams, targs)
	for i, tpar := range tparams {
		// Ensure that we have a (possibly implicit) interface as type bound (issue #51048).
		tpar.iface()
		// The type parameter bound is parameterized with the same type parameters
		// as the instantiated type; before we can use it for bounds checking we
		// need to instantiate it with the type arguments with which we instantiated
		// the parameterized type.
		bound := check.subst(pos, tpar.bound, smap, nil, ctxt)
		var reason string
		if !check.implements(targs[i], bound, &reason) {
			return i, errors.New(reason)
		}
	}
	return -1, nil
}

// implements checks if V implements T. The receiver may be nil if implements
// is called through an exported API call such as AssignableTo.
//
// If the provided reason is non-nil, it may be set to an error string
// explaining why V does not implement T.
func (check *Checker) implements(V, T Type, reason *string) bool {
	Vu := under(V)
	Tu := under(T)
	if Vu == Typ[Invalid] || Tu == Typ[Invalid] {
		return true // avoid follow-on errors
	}
	if p, _ := Vu.(*Pointer); p != nil && under(p.base) == Typ[Invalid] {
		return true // avoid follow-on errors (see issue #49541 for an example)
	}

	Ti, _ := Tu.(*Interface)
	if Ti == nil {
		var cause string
		if isInterfacePtr(Tu) {
			cause = check.sprintf("type %s is pointer to interface, not interface", T)
		} else {
			cause = check.sprintf("%s is not an interface", T)
		}
		if reason != nil {
			*reason = check.sprintf("%s does not implement %s (%s)", V, T, cause)
		}
		return false
	}

	// Every type satisfies the empty interface.
	if Ti.Empty() {
		return true
	}
	// T is not the empty interface (i.e., the type set of T is restricted)

	// An interface V with an empty type set satisfies any interface.
	// (The empty set is a subset of any set.)
	Vi, _ := Vu.(*Interface)
	if Vi != nil && Vi.typeSet().IsEmpty() {
		return true
	}
	// type set of V is not empty

	// No type with non-empty type set satisfies the empty type set.
	if Ti.typeSet().IsEmpty() {
		if reason != nil {
			*reason = check.sprintf("cannot implement %s (empty type set)", T)
		}
		return false
	}

	// V must implement T's methods, if any.
	if m, wrong := check.missingMethod(V, Ti, true); m != nil /* !Implements(V, Ti) */ {
		if reason != nil {
			*reason = check.sprintf("%s does not implement %s %s", V, T, check.missingMethodReason(V, T, m, wrong))
		}
		return false
	}

	// Only check comparability if we don't have a more specific error.
	checkComparability := func() bool {
		// If T is comparable, V must be comparable.
		if Ti.IsComparable() && !comparable(V, false, nil, nil) {
			if reason != nil {
				*reason = check.sprintf("%s does not implement comparable", V)
			}
			return false
		}
		return true
	}

	// V must also be in the set of types of T, if any.
	// Constraints with empty type sets were already excluded above.
	if !Ti.typeSet().hasTerms() {
		return checkComparability() // nothing to do
	}

	// If V is itself an interface, each of its possible types must be in the set
	// of T types (i.e., the V type set must be a subset of the T type set).
	// Interfaces V with empty type sets were already excluded above.
	if Vi != nil {
		if !Vi.typeSet().subsetOf(Ti.typeSet()) {
			// TODO(gri) report which type is missing
			if reason != nil {
				*reason = check.sprintf("%s does not implement %s", V, T)
			}
			return false
		}
		return checkComparability()
	}

	// Otherwise, V's type must be included in the iface type set.
	var alt Type
	if Ti.typeSet().is(func(t *term) bool {
		if !t.includes(V) {
			// If V ∉ t.typ but V ∈ ~t.typ then remember this type
			// so we can suggest it as an alternative in the error
			// message.
			if alt == nil && !t.tilde && Identical(t.typ, under(t.typ)) {
				tt := *t
				tt.tilde = true
				if tt.includes(V) {
					alt = t.typ
				}
			}
			return true
		}
		return false
	}) {
		if reason != nil {
			if alt != nil {
				*reason = check.sprintf("%s does not implement %s (possibly missing ~ for %s in constraint %s)", V, T, alt, T)
			} else {
				*reason = check.sprintf("%s does not implement %s (%s missing in %s)", V, T, V, Ti.typeSet().terms)
			}
		}
		return false
	}

	return checkComparability()
}
