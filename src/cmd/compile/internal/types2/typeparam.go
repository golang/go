// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import "sync/atomic"

// Note: This is a uint32 rather than a uint64 because the
// respective 64 bit atomic instructions are not available
// on all platforms.
var lastID uint32

// nextID returns a value increasing monotonically by 1 with
// each call, starting with 1. It may be called concurrently.
func nextID() uint64 { return uint64(atomic.AddUint32(&lastID, 1)) }

// A TypeParam represents a type parameter type.
type TypeParam struct {
	check *Checker  // for lazy type bound completion
	id    uint64    // unique id, for debugging only
	obj   *TypeName // corresponding type name
	index int       // type parameter index in source order, starting at 0
	bound Type      // any type, but underlying is eventually *Interface for correct programs (see TypeParam.iface)
}

// Obj returns the type name for the type parameter t.
func (t *TypeParam) Obj() *TypeName { return t.obj }

// NewTypeParam returns a new TypeParam. Type parameters may be set on a Named
// or Signature type by calling SetTypeParams. Setting a type parameter on more
// than one type will result in a panic.
//
// The constraint argument can be nil, and set later via SetConstraint. If the
// constraint is non-nil, it must be fully defined.
func NewTypeParam(obj *TypeName, constraint Type) *TypeParam {
	return (*Checker)(nil).newTypeParam(obj, constraint)
}

// check may be nil
func (check *Checker) newTypeParam(obj *TypeName, constraint Type) *TypeParam {
	// Always increment lastID, even if it is not used.
	id := nextID()
	if check != nil {
		check.nextID++
		id = check.nextID
	}
	typ := &TypeParam{check: check, id: id, obj: obj, index: -1, bound: constraint}
	if obj.typ == nil {
		obj.typ = typ
	}
	// iface may mutate typ.bound, so we must ensure that iface() is called
	// at least once before the resulting TypeParam escapes.
	if check != nil {
		check.needsCleanup(typ)
	} else if constraint != nil {
		typ.iface()
	}
	return typ
}

// Index returns the index of the type param within its param list, or -1 if
// the type parameter has not yet been bound to a type.
func (t *TypeParam) Index() int {
	return t.index
}

// Constraint returns the type constraint specified for t.
func (t *TypeParam) Constraint() Type {
	return t.bound
}

// SetConstraint sets the type constraint for t.
//
// It must be called by users of NewTypeParam after the bound's underlying is
// fully defined, and before using the type parameter in any way other than to
// form other types. Once SetConstraint returns the receiver, t is safe for
// concurrent use.
func (t *TypeParam) SetConstraint(bound Type) {
	if bound == nil {
		panic("nil constraint")
	}
	t.bound = bound
	// iface may mutate t.bound (if bound is not an interface), so ensure that
	// this is done before returning.
	t.iface()
}

func (t *TypeParam) Underlying() Type {
	return t.iface()
}

func (t *TypeParam) String() string { return TypeString(t, nil) }

// ----------------------------------------------------------------------------
// Implementation

func (t *TypeParam) cleanup() {
	t.iface()
	t.check = nil
}

// iface returns the constraint interface of t.
func (t *TypeParam) iface() *Interface {
	bound := t.bound

	// determine constraint interface
	var ityp *Interface
	switch u := under(bound).(type) {
	case *Basic:
		if u == Typ[Invalid] {
			// error is reported elsewhere
			return &emptyInterface
		}
	case *Interface:
		if isTypeParam(bound) {
			// error is reported in Checker.collectTypeParams
			return &emptyInterface
		}
		ityp = u
	}

	// If we don't have an interface, wrap constraint into an implicit interface.
	if ityp == nil {
		ityp = NewInterfaceType(nil, []Type{bound})
		ityp.implicit = true
		t.bound = ityp // update t.bound for next time (optimization)
	}

	// compute type set if necessary
	if ityp.tset == nil {
		// pos is used for tracing output; start with the type parameter position.
		pos := t.obj.pos
		// use the (original or possibly instantiated) type bound position if we have one
		if n, _ := bound.(*Named); n != nil {
			pos = n.obj.pos
		}
		computeInterfaceTypeSet(t.check, pos, ityp)
	}

	return ityp
}

// is calls f with the specific type terms of t's constraint and reports whether
// all calls to f returned true. If there are no specific terms, is
// returns the result of f(nil).
func (t *TypeParam) is(f func(*term) bool) bool {
	return t.iface().typeSet().is(f)
}

// underIs calls f with the underlying types of the specific type terms
// of t's constraint and reports whether all calls to f returned true.
// If there are no specific terms, underIs returns the result of f(nil).
func (t *TypeParam) underIs(f func(Type) bool) bool {
	return t.iface().typeSet().underIs(f)
}
