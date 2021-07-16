// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"go/token"
	"sync/atomic"
)

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
	bound Type      // *Named or *Interface; underlying type is always *Interface
}

// NewTypeParam returns a new TypeParam.
func NewTypeParam(obj *TypeName, index int, bound Type) *TypeParam {
	return (*Checker)(nil).newTypeParam(obj, index, bound)
}

// TODO(rfindley): this is factored slightly differently in types2.
func (check *Checker) newTypeParam(obj *TypeName, index int, bound Type) *TypeParam {
	assert(bound != nil)

	// Always increment lastID, even if it is not used.
	id := nextID()
	if check != nil {
		check.nextID++
		id = check.nextID
	}

	typ := &TypeParam{check: check, id: id, obj: obj, index: index, bound: bound}
	if obj.typ == nil {
		obj.typ = typ
	}
	return typ
}

// TODO(rfindley): types2 to has Index and SetID. Should we add them here?

func (t *TypeParam) Bound() *Interface {
	// we may not have an interface (error reported elsewhere)
	iface, _ := under(t.bound).(*Interface)
	if iface == nil {
		return &emptyInterface
	}
	// use the type bound position if we have one
	pos := token.NoPos
	if n, _ := t.bound.(*Named); n != nil {
		pos = n.obj.pos
	}
	// TODO(rFindley) switch this to an unexported method on Checker.
	computeTypeSet(t.check, pos, iface)
	return iface
}

func (t *TypeParam) Underlying() Type { return t }
func (t *TypeParam) String() string   { return TypeString(t, nil) }
