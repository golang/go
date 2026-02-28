// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	"sync"
)

// A Named represents a named (defined) type.
type Named struct {
	check      *Checker
	obj        *TypeName      // corresponding declared object for declared types; placeholder for instantiated types
	orig       *Named         // original, uninstantiated type
	fromRHS    Type           // type (on RHS of declaration) this *Named type is derived from (for cycle reporting)
	underlying Type           // possibly a *Named during setup; never a *Named once set up completely
	tparams    *TypeParamList // type parameters, or nil
	targs      *TypeList      // type arguments (after instantiation), or nil

	// methods declared for this type (not the method set of this type).
	// Signatures are type-checked lazily.
	// For non-instantiated types, this is a fully populated list of methods. For
	// instantiated types, this is a 'lazy' list, and methods are instantiated
	// when they are first accessed.
	methods *methodList

	// resolver may be provided to lazily resolve type parameters, underlying, and methods.
	resolver func(*Context, *Named) (tparams *TypeParamList, underlying Type, methods *methodList)
	once     sync.Once // ensures that tparams, underlying, and methods are resolved before accessing
}

// NewNamed returns a new named type for the given type name, underlying type, and associated methods.
// If the given type name obj doesn't have a type yet, its type is set to the returned named type.
// The underlying type must not be a *Named.
func NewNamed(obj *TypeName, underlying Type, methods []*Func) *Named {
	if _, ok := underlying.(*Named); ok {
		panic("underlying type must not be *Named")
	}
	return (*Checker)(nil).newNamed(obj, nil, underlying, nil, newMethodList(methods))
}

func (t *Named) resolve(ctxt *Context) *Named {
	if t.resolver == nil {
		return t
	}

	t.once.Do(func() {
		// TODO(mdempsky): Since we're passing t to the resolver anyway
		// (necessary because types2 expects the receiver type for methods
		// on defined interface types to be the Named rather than the
		// underlying Interface), maybe it should just handle calling
		// SetTypeParams, SetUnderlying, and AddMethod instead?  Those
		// methods would need to support reentrant calls though. It would
		// also make the API more future-proof towards further extensions
		// (like SetTypeParams).
		t.tparams, t.underlying, t.methods = t.resolver(ctxt, t)
		t.fromRHS = t.underlying // for cycle detection
	})
	return t
}

// newNamed is like NewNamed but with a *Checker receiver and additional orig argument.
func (check *Checker) newNamed(obj *TypeName, orig *Named, underlying Type, tparams *TypeParamList, methods *methodList) *Named {
	typ := &Named{check: check, obj: obj, orig: orig, fromRHS: underlying, underlying: underlying, tparams: tparams, methods: methods}
	if typ.orig == nil {
		typ.orig = typ
	}
	if obj.typ == nil {
		obj.typ = typ
	}
	// Ensure that typ is always expanded and sanity-checked.
	if check != nil {
		check.needsCleanup(typ)
	}
	return typ
}

func (t *Named) cleanup() {
	assert(t.orig.orig == t.orig)
	// Ensure that every defined type created in the course of type-checking has
	// either non-*Named underlying, or is unresolved.
	//
	// This guarantees that we don't leak any types whose underlying is *Named,
	// because any unresolved instances will lazily compute their underlying by
	// substituting in the underlying of their origin. The origin must have
	// either been imported or type-checked and expanded here, and in either case
	// its underlying will be fully expanded.
	switch t.underlying.(type) {
	case nil:
		if t.resolver == nil {
			panic("nil underlying")
		}
	case *Named:
		t.under() // t.under may add entries to check.cleaners
	}
	t.check = nil
}

// Obj returns the type name for the declaration defining the named type t. For
// instantiated types, this is same as the type name of the origin type.
func (t *Named) Obj() *TypeName { return t.orig.obj } // for non-instances this is the same as t.obj

// Origin returns the generic type from which the named type t is
// instantiated. If t is not an instantiated type, the result is t.
func (t *Named) Origin() *Named { return t.orig }

// TODO(gri) Come up with a better representation and API to distinguish
// between parameterized instantiated and non-instantiated types.

// TypeParams returns the type parameters of the named type t, or nil.
// The result is non-nil for an (originally) generic type even if it is instantiated.
func (t *Named) TypeParams() *TypeParamList { return t.resolve(nil).tparams }

// SetTypeParams sets the type parameters of the named type t.
// t must not have type arguments.
func (t *Named) SetTypeParams(tparams []*TypeParam) {
	assert(t.targs.Len() == 0)
	t.resolve(nil).tparams = bindTParams(tparams)
}

// TypeArgs returns the type arguments used to instantiate the named type t.
func (t *Named) TypeArgs() *TypeList { return t.targs }

// NumMethods returns the number of explicit methods defined for t.
//
// For an ordinary or instantiated type t, the receiver base type of these
// methods will be the named type t. For an uninstantiated generic type t, each
// method receiver will be instantiated with its receiver type parameters.
func (t *Named) NumMethods() int { return t.resolve(nil).methods.Len() }

// Method returns the i'th method of named type t for 0 <= i < t.NumMethods().
func (t *Named) Method(i int) *Func {
	t.resolve(nil)
	return t.methods.At(i, func() *Func {
		return t.instantiateMethod(i)
	})
}

// instantiateMethod instantiates the i'th method for an instantiated receiver.
func (t *Named) instantiateMethod(i int) *Func {
	assert(t.TypeArgs().Len() > 0) // t must be an instance

	// t.orig.methods is not lazy. origm is the method instantiated with its
	// receiver type parameters (the "origin" method).
	origm := t.orig.Method(i)
	assert(origm != nil)

	check := t.check
	// Ensure that the original method is type-checked.
	if check != nil {
		check.objDecl(origm, nil)
	}

	origSig := origm.typ.(*Signature)
	rbase, _ := deref(origSig.Recv().Type())

	// If rbase is t, then origm is already the instantiated method we're looking
	// for. In this case, we return origm to preserve the invariant that
	// traversing Method->Receiver Type->Method should get back to the same
	// method.
	//
	// This occurs if t is instantiated with the receiver type parameters, as in
	// the use of m in func (r T[_]) m() { r.m() }.
	if rbase == t {
		return origm
	}

	sig := origSig
	// We can only substitute if we have a correspondence between type arguments
	// and type parameters. This check is necessary in the presence of invalid
	// code.
	if origSig.RecvTypeParams().Len() == t.targs.Len() {
		ctxt := check.bestContext(nil)
		smap := makeSubstMap(origSig.RecvTypeParams().list(), t.targs.list())
		sig = check.subst(origm.pos, origSig, smap, ctxt).(*Signature)
	}

	if sig == origSig {
		// No substitution occurred, but we still need to create a new signature to
		// hold the instantiated receiver.
		copy := *origSig
		sig = &copy
	}

	var rtyp Type
	if origm.hasPtrRecv() {
		rtyp = NewPointer(t)
	} else {
		rtyp = t
	}

	sig.recv = substVar(origSig.recv, rtyp)
	return NewFunc(origm.pos, origm.pkg, origm.name, sig)
}

// SetUnderlying sets the underlying type and marks t as complete.
// t must not have type arguments.
func (t *Named) SetUnderlying(underlying Type) {
	assert(t.targs.Len() == 0)
	if underlying == nil {
		panic("underlying type must not be nil")
	}
	if _, ok := underlying.(*Named); ok {
		panic("underlying type must not be *Named")
	}
	t.resolve(nil).underlying = underlying
	if t.fromRHS == nil {
		t.fromRHS = underlying // for cycle detection
	}
}

// AddMethod adds method m unless it is already in the method list.
// t must not have type arguments.
func (t *Named) AddMethod(m *Func) {
	assert(t.targs.Len() == 0)
	t.resolve(nil)
	if t.methods == nil {
		t.methods = newMethodList(nil)
	}
	t.methods.Add(m)
}

func (t *Named) Underlying() Type { return t.resolve(nil).underlying }
func (t *Named) String() string   { return TypeString(t, nil) }

// ----------------------------------------------------------------------------
// Implementation

// under returns the expanded underlying type of n0; possibly by following
// forward chains of named types. If an underlying type is found, resolve
// the chain by setting the underlying type for each defined type in the
// chain before returning it. If no underlying type is found or a cycle
// is detected, the result is Typ[Invalid]. If a cycle is detected and
// n0.check != nil, the cycle is reported.
//
// This is necessary because the underlying type of named may be itself a
// named type that is incomplete:
//
//	type (
//		A B
//		B *C
//		C A
//	)
//
// The type of C is the (named) type of A which is incomplete,
// and which has as its underlying type the named type B.
func (n0 *Named) under() Type {
	u := n0.Underlying()

	// If the underlying type of a defined type is not a defined
	// (incl. instance) type, then that is the desired underlying
	// type.
	var n1 *Named
	switch u1 := u.(type) {
	case nil:
		// After expansion via Underlying(), we should never encounter a nil
		// underlying.
		panic("nil underlying")
	default:
		// common case
		return u
	case *Named:
		// handled below
		n1 = u1
	}

	if n0.check == nil {
		panic("Named.check == nil but type is incomplete")
	}

	// Invariant: after this point n0 as well as any named types in its
	// underlying chain should be set up when this function exits.
	check := n0.check
	n := n0

	seen := make(map[*Named]int) // types that need their underlying resolved
	var path []Object            // objects encountered, for cycle reporting

loop:
	for {
		seen[n] = len(seen)
		path = append(path, n.obj)
		n = n1
		if i, ok := seen[n]; ok {
			// cycle
			check.cycleError(path[i:])
			u = Typ[Invalid]
			break
		}
		u = n.Underlying()
		switch u1 := u.(type) {
		case nil:
			u = Typ[Invalid]
			break loop
		default:
			break loop
		case *Named:
			// Continue collecting *Named types in the chain.
			n1 = u1
		}
	}

	for n := range seen {
		// We should never have to update the underlying type of an imported type;
		// those underlying types should have been resolved during the import.
		// Also, doing so would lead to a race condition (was issue #31749).
		// Do this check always, not just in debug mode (it's cheap).
		if n.obj.pkg != check.pkg {
			panic("imported type with unresolved underlying type")
		}
		n.underlying = u
	}

	return u
}

func (n *Named) setUnderlying(typ Type) {
	if n != nil {
		n.underlying = typ
	}
}

func (n *Named) lookupMethod(pkg *Package, name string, foldCase bool) (int, *Func) {
	n.resolve(nil)
	// If n is an instance, we may not have yet instantiated all of its methods.
	// Look up the method index in orig, and only instantiate method at the
	// matching index (if any).
	i, _ := n.orig.methods.Lookup(pkg, name, foldCase)
	if i < 0 {
		return -1, nil
	}
	// For instances, m.Method(i) will be different from the orig method.
	return i, n.Method(i)
}

// bestContext returns the best available context. In order of preference:
// - the given ctxt, if non-nil
// - check.ctxt, if check is non-nil
// - a new Context
func (check *Checker) bestContext(ctxt *Context) *Context {
	if ctxt != nil {
		return ctxt
	}
	if check != nil {
		if check.ctxt == nil {
			check.ctxt = NewContext()
		}
		return check.ctxt
	}
	return NewContext()
}

// expandNamed ensures that the underlying type of n is instantiated.
// The underlying type will be Typ[Invalid] if there was an error.
func expandNamed(ctxt *Context, n *Named, instPos syntax.Pos) (tparams *TypeParamList, underlying Type, methods *methodList) {
	check := n.check
	if check != nil && check.conf.Trace {
		check.trace(instPos, "-- expandNamed %s", n)
		check.indent++
		defer func() {
			check.indent--
			check.trace(instPos, "=> %s (tparams = %s, under = %s)", n, tparams.list(), underlying)
		}()
	}

	n.orig.resolve(ctxt)
	assert(n.orig.underlying != nil)

	if _, unexpanded := n.orig.underlying.(*Named); unexpanded {
		// We should only get an unexpanded underlying here during type checking
		// (for example, in recursive type declarations).
		assert(check != nil)
	}

	// Mismatching arg and tparam length may be checked elsewhere.
	if n.orig.tparams.Len() == n.targs.Len() {
		// We must always have a context, to avoid infinite recursion.
		ctxt = check.bestContext(ctxt)
		h := ctxt.instanceHash(n.orig, n.targs.list())
		// ensure that an instance is recorded for h to avoid infinite recursion.
		ctxt.update(h, n.orig, n.TypeArgs().list(), n)

		smap := makeSubstMap(n.orig.tparams.list(), n.targs.list())
		underlying = n.check.subst(instPos, n.orig.underlying, smap, ctxt)
		// If the underlying of n is an interface, we need to set the receiver of
		// its methods accurately -- we set the receiver of interface methods on
		// the RHS of a type declaration to the defined type.
		if iface, _ := underlying.(*Interface); iface != nil {
			if methods, copied := replaceRecvType(iface.methods, n.orig, n); copied {
				// If the underlying doesn't actually use type parameters, it's possible
				// that it wasn't substituted. In this case we need to create a new
				// *Interface before modifying receivers.
				if iface == n.orig.underlying {
					old := iface
					iface = check.newInterface()
					iface.embeddeds = old.embeddeds
					iface.complete = old.complete
					iface.implicit = old.implicit // should be false but be conservative
					underlying = iface
				}
				iface.methods = methods
			}
		}
	} else {
		underlying = Typ[Invalid]
	}

	return n.orig.tparams, underlying, newLazyMethodList(n.orig.methods.Len())
}

// safeUnderlying returns the underlying of typ without expanding instances, to
// avoid infinite recursion.
//
// TODO(rfindley): eliminate this function or give it a better name.
func safeUnderlying(typ Type) Type {
	if t, _ := typ.(*Named); t != nil {
		return t.underlying
	}
	return typ.Underlying()
}
