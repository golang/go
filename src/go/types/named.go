// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import "sync"

// TODO(rfindley) Clean up Named struct below; specifically the fromRHS field (can we use underlying?).

// A Named represents a named (defined) type.
type Named struct {
	check      *Checker    // for Named.under implementation; nilled once under has been called
	info       typeInfo    // for cycle detection
	obj        *TypeName   // corresponding declared object
	orig       *Named      // original, uninstantiated type
	fromRHS    Type        // type (on RHS of declaration) this *Named type is derived of (for cycle reporting)
	underlying Type        // possibly a *Named during setup; never a *Named once set up completely
	tparams    []*TypeName // type parameters, or nil
	targs      []Type      // type arguments (after instantiation), or nil
	methods    []*Func     // methods declared for this type (not the method set of this type); signatures are type-checked lazily

	resolve func(*Named) ([]*TypeName, Type, []*Func)
	once    sync.Once
}

// NewNamed returns a new named type for the given type name, underlying type, and associated methods.
// If the given type name obj doesn't have a type yet, its type is set to the returned named type.
// The underlying type must not be a *Named.
func NewNamed(obj *TypeName, underlying Type, methods []*Func) *Named {
	if _, ok := underlying.(*Named); ok {
		panic("types.NewNamed: underlying type must not be *Named")
	}
	return (*Checker)(nil).newNamed(obj, nil, underlying, nil, methods)
}

func (t *Named) expand() *Named {
	if t.resolve == nil {
		return t
	}

	t.once.Do(func() {
		// TODO(mdempsky): Since we're passing t to resolve anyway
		// (necessary because types2 expects the receiver type for methods
		// on defined interface types to be the Named rather than the
		// underlying Interface), maybe it should just handle calling
		// SetTParams, SetUnderlying, and AddMethod instead?  Those
		// methods would need to support reentrant calls though.  It would
		// also make the API more future-proof towards further extensions
		// (like SetTParams).

		tparams, underlying, methods := t.resolve(t)

		switch underlying.(type) {
		case nil, *Named:
			panic("invalid underlying type")
		}

		t.tparams = tparams
		t.underlying = underlying
		t.methods = methods
	})
	return t
}

// newNamed is like NewNamed but with a *Checker receiver and additional orig argument.
func (check *Checker) newNamed(obj *TypeName, orig *Named, underlying Type, tparams []*TypeName, methods []*Func) *Named {
	typ := &Named{check: check, obj: obj, orig: orig, fromRHS: underlying, underlying: underlying, tparams: tparams, methods: methods}
	if typ.orig == nil {
		typ.orig = typ
	}
	if obj.typ == nil {
		obj.typ = typ
	}
	// Ensure that typ is always expanded, at which point the check field can be
	// nilled out.
	//
	// Note that currently we cannot nil out check inside typ.under(), because
	// it's possible that typ is expanded multiple times.
	//
	// TODO(rFindley): clean this up so that under is the only function mutating
	//                 named types.
	if check != nil {
		check.later(func() {
			switch typ.under().(type) {
			case *Named, *instance:
				panic("internal error: unexpanded underlying type")
			}
			typ.check = nil
		})
	}
	return typ
}

// Obj returns the type name for the named type t.
func (t *Named) Obj() *TypeName { return t.obj }

// _Orig returns the original generic type an instantiated type is derived from.
// If t is not an instantiated type, the result is t.
func (t *Named) _Orig() *Named { return t.orig }

// TODO(gri) Come up with a better representation and API to distinguish
//           between parameterized instantiated and non-instantiated types.

// _TParams returns the type parameters of the named type t, or nil.
// The result is non-nil for an (originally) parameterized type even if it is instantiated.
func (t *Named) TParams() []*TypeName { return t.expand().tparams }

// _SetTParams sets the type parameters of the named type t.
func (t *Named) SetTParams(tparams []*TypeName) { t.expand().tparams = tparams }

// _TArgs returns the type arguments after instantiation of the named type t, or nil if not instantiated.
func (t *Named) TArgs() []Type { return t.targs }

// SetTArgs sets the type arguments of the named type t.
func (t *Named) SetTArgs(args []Type) { t.targs = args }

// NumMethods returns the number of explicit methods whose receiver is named type t.
func (t *Named) NumMethods() int { return len(t.expand().methods) }

// Method returns the i'th method of named type t for 0 <= i < t.NumMethods().
func (t *Named) Method(i int) *Func { return t.expand().methods[i] }

// SetUnderlying sets the underlying type and marks t as complete.
func (t *Named) SetUnderlying(underlying Type) {
	if underlying == nil {
		panic("types.Named.SetUnderlying: underlying type must not be nil")
	}
	if _, ok := underlying.(*Named); ok {
		panic("types.Named.SetUnderlying: underlying type must not be *Named")
	}
	t.expand().underlying = underlying
}

// AddMethod adds method m unless it is already in the method list.
func (t *Named) AddMethod(m *Func) {
	t.expand()
	if i, _ := lookupMethod(t.methods, m.pkg, m.name); i < 0 {
		t.methods = append(t.methods, m)
	}
}

func (t *Named) Underlying() Type { return t.expand().underlying }
func (t *Named) String() string   { return TypeString(t, nil) }
