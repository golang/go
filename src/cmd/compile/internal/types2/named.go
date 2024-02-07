// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	"strings"
	"sync"
	"sync/atomic"
)

// Type-checking Named types is subtle, because they may be recursively
// defined, and because their full details may be spread across multiple
// declarations (via methods). For this reason they are type-checked lazily,
// to avoid information being accessed before it is complete.
//
// Conceptually, it is helpful to think of named types as having two distinct
// sets of information:
//  - "LHS" information, defining their identity: Obj() and TypeArgs()
//  - "RHS" information, defining their details: TypeParams(), Underlying(),
//    and methods.
//
// In this taxonomy, LHS information is available immediately, but RHS
// information is lazy. Specifically, a named type N may be constructed in any
// of the following ways:
//  1. type-checked from the source
//  2. loaded eagerly from export data
//  3. loaded lazily from export data (when using unified IR)
//  4. instantiated from a generic type
//
// In cases 1, 3, and 4, it is possible that the underlying type or methods of
// N may not be immediately available.
//  - During type-checking, we allocate N before type-checking its underlying
//    type or methods, so that we may resolve recursive references.
//  - When loading from export data, we may load its methods and underlying
//    type lazily using a provided load function.
//  - After instantiating, we lazily expand the underlying type and methods
//    (note that instances may be created while still in the process of
//    type-checking the original type declaration).
//
// In cases 3 and 4 this lazy construction may also occur concurrently, due to
// concurrent use of the type checker API (after type checking or importing has
// finished). It is critical that we keep track of state, so that Named types
// are constructed exactly once and so that we do not access their details too
// soon.
//
// We achieve this by tracking state with an atomic state variable, and
// guarding potentially concurrent calculations with a mutex. At any point in
// time this state variable determines which data on N may be accessed. As
// state monotonically progresses, any data available at state M may be
// accessed without acquiring the mutex at state N, provided N >= M.
//
// GLOSSARY: Here are a few terms used in this file to describe Named types:
//  - We say that a Named type is "instantiated" if it has been constructed by
//    instantiating a generic named type with type arguments.
//  - We say that a Named type is "declared" if it corresponds to a type
//    declaration in the source. Instantiated named types correspond to a type
//    instantiation in the source, not a declaration. But their Origin type is
//    a declared type.
//  - We say that a Named type is "resolved" if its RHS information has been
//    loaded or fully type-checked. For Named types constructed from export
//    data, this may involve invoking a loader function to extract information
//    from export data. For instantiated named types this involves reading
//    information from their origin.
//  - We say that a Named type is "expanded" if it is an instantiated type and
//    type parameters in its underlying type and methods have been substituted
//    with the type arguments from the instantiation. A type may be partially
//    expanded if some but not all of these details have been substituted.
//    Similarly, we refer to these individual details (underlying type or
//    method) as being "expanded".
//  - When all information is known for a named type, we say it is "complete".
//
// Some invariants to keep in mind: each declared Named type has a single
// corresponding object, and that object's type is the (possibly generic) Named
// type. Declared Named types are identical if and only if their pointers are
// identical. On the other hand, multiple instantiated Named types may be
// identical even though their pointers are not identical. One has to use
// Identical to compare them. For instantiated named types, their obj is a
// synthetic placeholder that records their position of the corresponding
// instantiation in the source (if they were constructed during type checking).
//
// To prevent infinite expansion of named instances that are created outside of
// type-checking, instances share a Context with other instances created during
// their expansion. Via the pidgeonhole principle, this guarantees that in the
// presence of a cycle of named types, expansion will eventually find an
// existing instance in the Context and short-circuit the expansion.
//
// Once an instance is complete, we can nil out this shared Context to unpin
// memory, though this Context may still be held by other incomplete instances
// in its "lineage".

// A Named represents a named (defined) type.
type Named struct {
	check *Checker  // non-nil during type-checking; nil otherwise
	obj   *TypeName // corresponding declared object for declared types; see above for instantiated types

	// fromRHS holds the type (on RHS of declaration) this *Named type is derived
	// from (for cycle reporting). Only used by validType, and therefore does not
	// require synchronization.
	fromRHS Type

	// information for instantiated types; nil otherwise
	inst *instance

	mu         sync.Mutex     // guards all fields below
	state_     uint32         // the current state of this type; must only be accessed atomically
	underlying Type           // possibly a *Named during setup; never a *Named once set up completely
	tparams    *TypeParamList // type parameters, or nil

	// methods declared for this type (not the method set of this type)
	// Signatures are type-checked lazily.
	// For non-instantiated types, this is a fully populated list of methods. For
	// instantiated types, methods are individually expanded when they are first
	// accessed.
	methods []*Func

	// loader may be provided to lazily load type parameters, underlying type, and methods.
	loader func(*Named) (tparams []*TypeParam, underlying Type, methods []*Func)
}

// instance holds information that is only necessary for instantiated named
// types.
type instance struct {
	orig            *Named    // original, uninstantiated type
	targs           *TypeList // type arguments
	expandedMethods int       // number of expanded methods; expandedMethods <= len(orig.methods)
	ctxt            *Context  // local Context; set to nil after full expansion
}

// namedState represents the possible states that a named type may assume.
type namedState uint32

const (
	unresolved namedState = iota // tparams, underlying type and methods might be unavailable
	resolved                     // resolve has run; methods might be incomplete (for instances)
	complete                     // all data is known
)

// NewNamed returns a new named type for the given type name, underlying type, and associated methods.
// If the given type name obj doesn't have a type yet, its type is set to the returned named type.
// The underlying type must not be a *Named.
func NewNamed(obj *TypeName, underlying Type, methods []*Func) *Named {
	if asNamed(underlying) != nil {
		panic("underlying type must not be *Named")
	}
	return (*Checker)(nil).newNamed(obj, underlying, methods)
}

// resolve resolves the type parameters, methods, and underlying type of n.
// This information may be loaded from a provided loader function, or computed
// from an origin type (in the case of instances).
//
// After resolution, the type parameters, methods, and underlying type of n are
// accessible; but if n is an instantiated type, its methods may still be
// unexpanded.
func (n *Named) resolve() *Named {
	if n.state() >= resolved { // avoid locking below
		return n
	}

	// TODO(rfindley): if n.check is non-nil we can avoid locking here, since
	// type-checking is not concurrent. Evaluate if this is worth doing.
	n.mu.Lock()
	defer n.mu.Unlock()

	if n.state() >= resolved {
		return n
	}

	if n.inst != nil {
		assert(n.underlying == nil) // n is an unresolved instance
		assert(n.loader == nil)     // instances are created by instantiation, in which case n.loader is nil

		orig := n.inst.orig
		orig.resolve()
		underlying := n.expandUnderlying()

		n.tparams = orig.tparams
		n.underlying = underlying
		n.fromRHS = orig.fromRHS // for cycle detection

		if len(orig.methods) == 0 {
			n.setState(complete) // nothing further to do
			n.inst.ctxt = nil
		} else {
			n.setState(resolved)
		}
		return n
	}

	// TODO(mdempsky): Since we're passing n to the loader anyway
	// (necessary because types2 expects the receiver type for methods
	// on defined interface types to be the Named rather than the
	// underlying Interface), maybe it should just handle calling
	// SetTypeParams, SetUnderlying, and AddMethod instead?  Those
	// methods would need to support reentrant calls though. It would
	// also make the API more future-proof towards further extensions.
	if n.loader != nil {
		assert(n.underlying == nil)
		assert(n.TypeArgs().Len() == 0) // instances are created by instantiation, in which case n.loader is nil

		tparams, underlying, methods := n.loader(n)

		n.tparams = bindTParams(tparams)
		n.underlying = underlying
		n.fromRHS = underlying // for cycle detection
		n.methods = methods
		n.loader = nil
	}

	n.setState(complete)
	return n
}

// state atomically accesses the current state of the receiver.
func (n *Named) state() namedState {
	return namedState(atomic.LoadUint32(&n.state_))
}

// setState atomically stores the given state for n.
// Must only be called while holding n.mu.
func (n *Named) setState(state namedState) {
	atomic.StoreUint32(&n.state_, uint32(state))
}

// newNamed is like NewNamed but with a *Checker receiver.
func (check *Checker) newNamed(obj *TypeName, underlying Type, methods []*Func) *Named {
	typ := &Named{check: check, obj: obj, fromRHS: underlying, underlying: underlying, methods: methods}
	if obj.typ == nil {
		obj.typ = typ
	}
	// Ensure that typ is always sanity-checked.
	if check != nil {
		check.needsCleanup(typ)
	}
	return typ
}

// newNamedInstance creates a new named instance for the given origin and type
// arguments, recording pos as the position of its synthetic object (for error
// reporting).
//
// If set, expanding is the named type instance currently being expanded, that
// led to the creation of this instance.
func (check *Checker) newNamedInstance(pos syntax.Pos, orig *Named, targs []Type, expanding *Named) *Named {
	assert(len(targs) > 0)

	obj := NewTypeName(pos, orig.obj.pkg, orig.obj.name, nil)
	inst := &instance{orig: orig, targs: newTypeList(targs)}

	// Only pass the expanding context to the new instance if their packages
	// match. Since type reference cycles are only possible within a single
	// package, this is sufficient for the purposes of short-circuiting cycles.
	// Avoiding passing the context in other cases prevents unnecessary coupling
	// of types across packages.
	if expanding != nil && expanding.Obj().pkg == obj.pkg {
		inst.ctxt = expanding.inst.ctxt
	}
	typ := &Named{check: check, obj: obj, inst: inst}
	obj.typ = typ
	// Ensure that typ is always sanity-checked.
	if check != nil {
		check.needsCleanup(typ)
	}
	return typ
}

func (t *Named) cleanup() {
	assert(t.inst == nil || t.inst.orig.inst == nil)
	// Ensure that every defined type created in the course of type-checking has
	// either non-*Named underlying type, or is unexpanded.
	//
	// This guarantees that we don't leak any types whose underlying type is
	// *Named, because any unexpanded instances will lazily compute their
	// underlying type by substituting in the underlying type of their origin.
	// The origin must have either been imported or type-checked and expanded
	// here, and in either case its underlying type will be fully expanded.
	switch t.underlying.(type) {
	case nil:
		if t.TypeArgs().Len() == 0 {
			panic("nil underlying")
		}
	case *Named:
		t.under() // t.under may add entries to check.cleaners
	}
	t.check = nil
}

// Obj returns the type name for the declaration defining the named type t. For
// instantiated types, this is same as the type name of the origin type.
func (t *Named) Obj() *TypeName {
	if t.inst == nil {
		return t.obj
	}
	return t.inst.orig.obj
}

// Origin returns the generic type from which the named type t is
// instantiated. If t is not an instantiated type, the result is t.
func (t *Named) Origin() *Named {
	if t.inst == nil {
		return t
	}
	return t.inst.orig
}

// TypeParams returns the type parameters of the named type t, or nil.
// The result is non-nil for an (originally) generic type even if it is instantiated.
func (t *Named) TypeParams() *TypeParamList { return t.resolve().tparams }

// SetTypeParams sets the type parameters of the named type t.
// t must not have type arguments.
func (t *Named) SetTypeParams(tparams []*TypeParam) {
	assert(t.inst == nil)
	t.resolve().tparams = bindTParams(tparams)
}

// TypeArgs returns the type arguments used to instantiate the named type t.
func (t *Named) TypeArgs() *TypeList {
	if t.inst == nil {
		return nil
	}
	return t.inst.targs
}

// NumMethods returns the number of explicit methods defined for t.
func (t *Named) NumMethods() int {
	return len(t.Origin().resolve().methods)
}

// Method returns the i'th method of named type t for 0 <= i < t.NumMethods().
//
// For an ordinary or instantiated type t, the receiver base type of this
// method is the named type t. For an uninstantiated generic type t, each
// method receiver is instantiated with its receiver type parameters.
//
// Methods are numbered deterministically: given the same list of source files
// presented to the type checker, or the same sequence of NewMethod and AddMethod
// calls, the mapping from method index to corresponding method remains the same.
// But the specific ordering is not specified and must not be relied on as it may
// change in the future.
func (t *Named) Method(i int) *Func {
	t.resolve()

	if t.state() >= complete {
		return t.methods[i]
	}

	assert(t.inst != nil) // only instances should have incomplete methods
	orig := t.inst.orig

	t.mu.Lock()
	defer t.mu.Unlock()

	if len(t.methods) != len(orig.methods) {
		assert(len(t.methods) == 0)
		t.methods = make([]*Func, len(orig.methods))
	}

	if t.methods[i] == nil {
		assert(t.inst.ctxt != nil) // we should still have a context remaining from the resolution phase
		t.methods[i] = t.expandMethod(i)
		t.inst.expandedMethods++

		// Check if we've created all methods at this point. If we have, mark the
		// type as fully expanded.
		if t.inst.expandedMethods == len(orig.methods) {
			t.setState(complete)
			t.inst.ctxt = nil // no need for a context anymore
		}
	}

	return t.methods[i]
}

// expandMethod substitutes type arguments in the i'th method for an
// instantiated receiver.
func (t *Named) expandMethod(i int) *Func {
	// t.orig.methods is not lazy. origm is the method instantiated with its
	// receiver type parameters (the "origin" method).
	origm := t.inst.orig.Method(i)
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
	if origSig.RecvTypeParams().Len() == t.inst.targs.Len() {
		smap := makeSubstMap(origSig.RecvTypeParams().list(), t.inst.targs.list())
		var ctxt *Context
		if check != nil {
			ctxt = check.context()
		}
		sig = check.subst(origm.pos, origSig, smap, t, ctxt).(*Signature)
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
	return substFunc(origm, sig)
}

// SetUnderlying sets the underlying type and marks t as complete.
// t must not have type arguments.
func (t *Named) SetUnderlying(underlying Type) {
	assert(t.inst == nil)
	if underlying == nil {
		panic("underlying type must not be nil")
	}
	if asNamed(underlying) != nil {
		panic("underlying type must not be *Named")
	}
	t.resolve().underlying = underlying
	if t.fromRHS == nil {
		t.fromRHS = underlying // for cycle detection
	}
}

// AddMethod adds method m unless it is already in the method list.
// The method must be in the same package as t, and t must not have
// type arguments.
func (t *Named) AddMethod(m *Func) {
	assert(samePkg(t.obj.pkg, m.pkg))
	assert(t.inst == nil)
	t.resolve()
	if t.methodIndex(m.name, false) < 0 {
		t.methods = append(t.methods, m)
	}
}

// methodIndex returns the index of the method with the given name.
// If foldCase is set, capitalization in the name is ignored.
// The result is negative if no such method exists.
func (t *Named) methodIndex(name string, foldCase bool) int {
	if name == "_" {
		return -1
	}
	if foldCase {
		for i, m := range t.methods {
			if strings.EqualFold(m.name, name) {
				return i
			}
		}
	} else {
		for i, m := range t.methods {
			if m.name == name {
				return i
			}
		}
	}
	return -1
}

// TODO(gri) Investigate if Unalias can be moved to where underlying is set.
func (t *Named) Underlying() Type { return Unalias(t.resolve().underlying) }
func (t *Named) String() string   { return TypeString(t, nil) }

// ----------------------------------------------------------------------------
// Implementation
//
// TODO(rfindley): reorganize the loading and expansion methods under this
// heading.

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

	seen := make(map[*Named]int) // types that need their underlying type resolved
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
		// Also, doing so would lead to a race condition (was go.dev/issue/31749).
		// Do this check always, not just in debug mode (it's cheap).
		if n.obj.pkg != check.pkg {
			panic("imported type with unresolved underlying type")
		}
		n.underlying = u
	}

	return u
}

func (n *Named) lookupMethod(pkg *Package, name string, foldCase bool) (int, *Func) {
	n.resolve()
	if samePkg(n.obj.pkg, pkg) || isExported(name) || foldCase {
		// If n is an instance, we may not have yet instantiated all of its methods.
		// Look up the method index in orig, and only instantiate method at the
		// matching index (if any).
		if i := n.Origin().methodIndex(name, foldCase); i >= 0 {
			// For instances, m.Method(i) will be different from the orig method.
			return i, n.Method(i)
		}
	}
	return -1, nil
}

// context returns the type-checker context.
func (check *Checker) context() *Context {
	if check.ctxt == nil {
		check.ctxt = NewContext()
	}
	return check.ctxt
}

// expandUnderlying substitutes type arguments in the underlying type n.orig,
// returning the result. Returns Typ[Invalid] if there was an error.
func (n *Named) expandUnderlying() Type {
	check := n.check
	if check != nil && check.conf.Trace {
		check.trace(n.obj.pos, "-- Named.expandUnderlying %s", n)
		check.indent++
		defer func() {
			check.indent--
			check.trace(n.obj.pos, "=> %s (tparams = %s, under = %s)", n, n.tparams.list(), n.underlying)
		}()
	}

	assert(n.inst.orig.underlying != nil)
	if n.inst.ctxt == nil {
		n.inst.ctxt = NewContext()
	}

	orig := n.inst.orig
	targs := n.inst.targs

	if asNamed(orig.underlying) != nil {
		// We should only get a Named underlying type here during type checking
		// (for example, in recursive type declarations).
		assert(check != nil)
	}

	if orig.tparams.Len() != targs.Len() {
		// Mismatching arg and tparam length may be checked elsewhere.
		return Typ[Invalid]
	}

	// Ensure that an instance is recorded before substituting, so that we
	// resolve n for any recursive references.
	h := n.inst.ctxt.instanceHash(orig, targs.list())
	n2 := n.inst.ctxt.update(h, orig, n.TypeArgs().list(), n)
	assert(n == n2)

	smap := makeSubstMap(orig.tparams.list(), targs.list())
	var ctxt *Context
	if check != nil {
		ctxt = check.context()
	}
	underlying := n.check.subst(n.obj.pos, orig.underlying, smap, n, ctxt)
	// If the underlying type of n is an interface, we need to set the receiver of
	// its methods accurately -- we set the receiver of interface methods on
	// the RHS of a type declaration to the defined type.
	if iface, _ := underlying.(*Interface); iface != nil {
		if methods, copied := replaceRecvType(iface.methods, orig, n); copied {
			// If the underlying type doesn't actually use type parameters, it's
			// possible that it wasn't substituted. In this case we need to create
			// a new *Interface before modifying receivers.
			if iface == orig.underlying {
				old := iface
				iface = check.newInterface()
				iface.embeddeds = old.embeddeds
				assert(old.complete) // otherwise we are copying incomplete data
				iface.complete = old.complete
				iface.implicit = old.implicit // should be false but be conservative
				underlying = iface
			}
			iface.methods = methods
			iface.tset = nil // recompute type set with new methods

			// If check != nil, check.newInterface will have saved the interface for later completion.
			if check == nil { // golang/go#61561: all newly created interfaces must be fully evaluated
				iface.typeSet()
			}
		}
	}

	return underlying
}

// safeUnderlying returns the underlying type of typ without expanding
// instances, to avoid infinite recursion.
//
// TODO(rfindley): eliminate this function or give it a better name.
func safeUnderlying(typ Type) Type {
	if t := asNamed(typ); t != nil {
		return t.underlying
	}
	return typ.Underlying()
}
