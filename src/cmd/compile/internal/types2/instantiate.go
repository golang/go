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
	"internal/buildcfg"
	. "internal/types/errors"
)

// A genericType implements access to its type parameters.
type genericType interface {
	Type
	TypeParams() *TypeParamList
}

// Instantiate instantiates the type orig with the given type arguments targs.
// orig must be an *Alias, *Named, or *Signature type. If there is no error,
// the resulting Type is an instantiated type of the same kind (*Alias, *Named
// or *Signature, respectively).
//
// Methods attached to a *Named type are also instantiated, and associated with
// a new *Func that has the same position as the original method, but nil function
// scope.
//
// If ctxt is non-nil, it may be used to de-duplicate the instance against
// previous instances with the same identity. As a special case, generic
// *Signature origin types are only considered identical if they are pointer
// equivalent, so that instantiating distinct (but possibly identical)
// signatures will yield different instances. The use of a shared context does
// not guarantee that identical instances are deduplicated in all cases.
//
// If validate is set, Instantiate verifies that the number of type arguments
// and parameters match, and that the type arguments satisfy their respective
// type constraints. If verification fails, the resulting error may wrap an
// *ArgumentError indicating which type argument did not satisfy its type parameter
// constraint, and why.
//
// If validate is not set, Instantiate does not verify the type argument count
// or whether the type arguments satisfy their constraints. Instantiate is
// guaranteed to not return an error, but may panic. Specifically, for
// *Signature types, Instantiate will panic immediately if the type argument
// count is incorrect; for *Named types, a panic may occur later inside the
// *Named API.
func Instantiate(ctxt *Context, orig Type, targs []Type, validate bool) (Type, error) {
	assert(len(targs) > 0)
	if ctxt == nil {
		ctxt = NewContext()
	}
	orig_ := orig.(genericType) // signature of Instantiate must not change for backward-compatibility

	if validate {
		tparams := orig_.TypeParams().list()
		assert(len(tparams) > 0)
		if len(targs) != len(tparams) {
			return nil, fmt.Errorf("got %d type arguments but %s has %d type parameters", len(targs), orig, len(tparams))
		}
		if i, err := (*Checker)(nil).verify(nopos, tparams, targs, ctxt); err != nil {
			return nil, &ArgumentError{i, err}
		}
	}

	inst := (*Checker)(nil).instance(nopos, orig_, targs, nil, ctxt)
	return inst, nil
}

// instance instantiates the given original (generic) function or type with the
// provided type arguments and returns the resulting instance. If an identical
// instance exists already in the given contexts, it returns that instance,
// otherwise it creates a new one. If there is an error (such as wrong number
// of type arguments), the result is Typ[Invalid].
//
// If expanding is non-nil, it is the Named instance type currently being
// expanded. If ctxt is non-nil, it is the context associated with the current
// type-checking pass or call to Instantiate. At least one of expanding or ctxt
// must be non-nil.
//
// For Named types the resulting instance may be unexpanded.
//
// check may be nil (when not type-checking syntax); pos is used only only if check is non-nil.
func (check *Checker) instance(pos syntax.Pos, orig genericType, targs []Type, expanding *Named, ctxt *Context) (res Type) {
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

	// Record the result in all contexts.
	// Prefer to re-use existing types from expanding context, if it exists, to reduce
	// the memory pinned by the Named type.
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

	case *Alias:
		if !buildcfg.Experiment.AliasTypeParams {
			assert(expanding == nil) // Alias instances cannot be reached from Named types
		}

		// verify type parameter count (see go.dev/issue/71198 for a test case)
		tparams := orig.TypeParams()
		if !check.validateTArgLen(pos, orig.obj.Name(), tparams.Len(), len(targs)) {
			// TODO(gri) Consider returning a valid alias instance with invalid
			//           underlying (aliased) type to match behavior of *Named
			//           types. Then this function will never return an invalid
			//           result.
			return Typ[Invalid]
		}
		if tparams.Len() == 0 {
			return orig // nothing to do (minor optimization)
		}

		res = check.newAliasInstance(pos, orig, targs, expanding, ctxt)

	case *Signature:
		assert(expanding == nil) // function instances cannot be reached from Named types

		tparams := orig.TypeParams()
		// TODO(gri) investigate if this is needed (type argument and parameter count seem to be correct here)
		if !check.validateTArgLen(pos, orig.String(), tparams.Len(), len(targs)) {
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

// validateTArgLen checks that the number of type arguments (got) matches the
// number of type parameters (want); if they don't match an error is reported.
// If validation fails and check is nil, validateTArgLen panics.
func (check *Checker) validateTArgLen(pos syntax.Pos, name string, want, got int) bool {
	var qual string
	switch {
	case got < want:
		qual = "not enough"
	case got > want:
		qual = "too many"
	default:
		return true
	}

	msg := check.sprintf("%s type arguments for type %s: have %d, want %d", qual, name, got, want)
	if check != nil {
		check.error(atPos(pos), WrongTypeArgCount, msg)
		return false
	}

	panic(fmt.Sprintf("%v: %s", pos, msg))
}

// check may be nil; pos is used only if check is non-nil.
func (check *Checker) verify(pos syntax.Pos, tparams []*TypeParam, targs []Type, ctxt *Context) (int, error) {
	smap := makeSubstMap(tparams, targs)
	for i, tpar := range tparams {
		// Ensure that we have a (possibly implicit) interface as type bound (go.dev/issue/51048).
		tpar.iface()
		// The type parameter bound is parameterized with the same type parameters
		// as the instantiated type; before we can use it for bounds checking we
		// need to instantiate it with the type arguments with which we instantiated
		// the parameterized type.
		bound := check.subst(pos, tpar.bound, smap, nil, ctxt)
		var cause string
		if !check.implements(targs[i], bound, true, &cause) {
			return i, errors.New(cause)
		}
	}
	return -1, nil
}

// implements checks if V implements T. The receiver may be nil if implements
// is called through an exported API call such as AssignableTo. If constraint
// is set, T is a type constraint.
//
// If the provided cause is non-nil, it may be set to an error string
// explaining why V does not implement (or satisfy, for constraints) T.
func (check *Checker) implements(V, T Type, constraint bool, cause *string) bool {
	Vu := under(V)
	Tu := under(T)
	if !isValid(Vu) || !isValid(Tu) {
		return true // avoid follow-on errors
	}
	if p, _ := Vu.(*Pointer); p != nil && !isValid(under(p.base)) {
		return true // avoid follow-on errors (see go.dev/issue/49541 for an example)
	}

	verb := "implement"
	if constraint {
		verb = "satisfy"
	}

	Ti, _ := Tu.(*Interface)
	if Ti == nil {
		if cause != nil {
			var detail string
			if isInterfacePtr(Tu) {
				detail = check.sprintf("type %s is pointer to interface, not interface", T)
			} else {
				detail = check.sprintf("%s is not an interface", T)
			}
			*cause = check.sprintf("%s does not %s %s (%s)", V, verb, T, detail)
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
		if cause != nil {
			*cause = check.sprintf("cannot %s %s (empty type set)", verb, T)
		}
		return false
	}

	// V must implement T's methods, if any.
	if !check.hasAllMethods(V, T, true, Identical, cause) /* !Implements(V, T) */ {
		if cause != nil {
			*cause = check.sprintf("%s does not %s %s %s", V, verb, T, *cause)
		}
		return false
	}

	// Only check comparability if we don't have a more specific error.
	checkComparability := func() bool {
		if !Ti.IsComparable() {
			return true
		}
		// If T is comparable, V must be comparable.
		// If V is strictly comparable, we're done.
		if comparableType(V, false /* strict comparability */, nil) == nil {
			return true
		}
		// For constraint satisfaction, use dynamic (spec) comparability
		// so that ordinary, non-type parameter interfaces implement comparable.
		if constraint && comparableType(V, true /* spec comparability */, nil) == nil {
			// V is comparable if we are at Go 1.20 or higher.
			if check == nil || check.allowVersion(go1_20) {
				return true
			}
			if cause != nil {
				*cause = check.sprintf("%s to %s comparable requires go1.20 or later", V, verb)
			}
			return false
		}
		if cause != nil {
			*cause = check.sprintf("%s does not %s comparable", V, verb)
		}
		return false
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
			if cause != nil {
				*cause = check.sprintf("%s does not %s %s", V, verb, T)
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
		if cause != nil {
			var detail string
			switch {
			case alt != nil:
				detail = check.sprintf("possibly missing ~ for %s in %s", alt, T)
			case mentions(Ti, V):
				detail = check.sprintf("%s mentions %s, but %s is not in the type set of %s", T, V, V, T)
			default:
				detail = check.sprintf("%s missing in %s", V, Ti.typeSet().terms)
			}
			*cause = check.sprintf("%s does not %s %s (%s)", V, verb, T, detail)
		}
		return false
	}

	return checkComparability()
}

// mentions reports whether type T "mentions" typ in an (embedded) element or term
// of T (whether typ is in the type set of T or not). For better error messages.
func mentions(T, typ Type) bool {
	switch T := T.(type) {
	case *Interface:
		for _, e := range T.embeddeds {
			if mentions(e, typ) {
				return true
			}
		}
	case *Union:
		for _, t := range T.terms {
			if mentions(t.typ, typ) {
				return true
			}
		}
	default:
		if Identical(T, typ) {
			return true
		}
	}
	return false
}
