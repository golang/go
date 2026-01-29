// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

// Calling TypeNode converts a *types.Type to a Node shell.

// A typeNode is a Node wrapper for type t.
type typeNode struct {
	miniNode
	typ *types.Type
}

func newTypeNode(typ *types.Type) *typeNode {
	n := &typeNode{typ: typ}
	n.pos = src.NoXPos
	n.op = OTYPE
	n.SetTypecheck(1)
	return n
}

func (n *typeNode) Type() *types.Type { return n.typ }
func (n *typeNode) Sym() *types.Sym   { return n.typ.Sym() }

// TypeNode returns the Node representing the type t.
func TypeNode(t *types.Type) Node {
	if n := t.Obj(); n != nil {
		if n.Type() != t {
			base.Fatalf("type skew: %v has type %v, but expected %v", n, n.Type(), t)
		}
		return n.(*Name)
	}
	return newTypeNode(t)
}

// A DynamicType represents a type expression whose exact type must be
// computed dynamically.
//
// TODO(adonovan): I think "dynamic" is a misnomer here; it's really a
// type with free type parameters that needs to be instantiated to obtain
// a ground type for which an rtype can exist.
type DynamicType struct {
	miniExpr

	// RType is an expression that yields a *runtime._type value
	// representing the asserted type.
	//
	// BUG(mdempsky): If ITab is non-nil, RType may be nil.
	RType Node

	// ITab is an expression that yields a *runtime.itab value
	// representing the asserted type within the assertee expression's
	// original interface type.
	//
	// ITab is only used for assertions (including type switches) from
	// non-empty interface type to a concrete (i.e., non-interface)
	// type. For all other assertions, ITab is nil.
	ITab Node
}

func NewDynamicType(pos src.XPos, rtype Node) *DynamicType {
	n := &DynamicType{RType: rtype}
	n.pos = pos
	n.op = ODYNAMICTYPE
	return n
}

// ToStatic returns static type of dt if it is actually static.
func (dt *DynamicType) ToStatic() Node {
	if dt.Typecheck() == 0 {
		base.Fatalf("missing typecheck: %v", dt)
	}
	if dt.RType != nil && dt.RType.Op() == OADDR {
		addr := dt.RType.(*AddrExpr)
		if addr.X.Op() == OLINKSYMOFFSET {
			return TypeNode(dt.Type())
		}
	}
	if dt.ITab != nil && dt.ITab.Op() == OADDR {
		addr := dt.ITab.(*AddrExpr)
		if addr.X.Op() == OLINKSYMOFFSET {
			return TypeNode(dt.Type())
		}
	}
	return nil
}
