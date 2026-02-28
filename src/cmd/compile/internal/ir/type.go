// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
)

// Nodes that represent the syntax of a type before type-checking.
// After type-checking, they serve only as shells around a *types.Type.
// Calling TypeNode converts a *types.Type to a Node shell.

// An Ntype is a Node that syntactically looks like a type.
// It can be the raw syntax for a type before typechecking,
// or it can be an OTYPE with Type() set to a *types.Type.
// Note that syntax doesn't guarantee it's a type: an expression
// like *fmt is an Ntype (we don't know whether names are types yet),
// but at least 1+1 is not an Ntype.
type Ntype interface {
	Node
	CanBeNtype()
}

// A miniType is a minimal type syntax Node implementation,
// to be embedded as the first field in a larger node implementation.
type miniType struct {
	miniNode
	typ *types.Type
}

func (*miniType) CanBeNtype() {}

func (n *miniType) Type() *types.Type { return n.typ }

// setOTYPE changes n to be an OTYPE node returning t.
// Rewriting the node in place this way should not be strictly
// necessary (we should be able to update the uses with
// proper OTYPE nodes), but it's mostly harmless and easy
// to keep doing for now.
//
// setOTYPE also records t.Nod = self if t.Nod is not already set.
// (Some types are shared by multiple OTYPE nodes, so only
// the first such node is used as t.Nod.)
func (n *miniType) setOTYPE(t *types.Type, self Ntype) {
	if n.typ != nil {
		panic(n.op.String() + " SetType: type already set")
	}
	n.op = OTYPE
	n.typ = t
	t.SetNod(self)
}

func (n *miniType) Sym() *types.Sym { return nil }   // for Format OTYPE
func (n *miniType) Implicit() bool  { return false } // for Format OTYPE

// A FuncType represents a func(Args) Results type syntax.
type FuncType struct {
	miniType
	Recv    *Field
	Params  []*Field
	Results []*Field
}

func NewFuncType(pos src.XPos, rcvr *Field, args, results []*Field) *FuncType {
	n := &FuncType{Recv: rcvr, Params: args, Results: results}
	n.op = OTFUNC
	n.pos = pos
	return n
}

func (n *FuncType) SetOTYPE(t *types.Type) {
	n.setOTYPE(t, n)
	n.Recv = nil
	n.Params = nil
	n.Results = nil
}

// A Field is a declared struct field, interface method, or function argument.
// It is not a Node.
type Field struct {
	Pos      src.XPos
	Sym      *types.Sym
	Ntype    Ntype
	Type     *types.Type
	Embedded bool
	IsDDD    bool
	Note     string
	Decl     *Name
}

func NewField(pos src.XPos, sym *types.Sym, ntyp Ntype, typ *types.Type) *Field {
	return &Field{Pos: pos, Sym: sym, Ntype: ntyp, Type: typ}
}

func (f *Field) String() string {
	var typ string
	if f.Type != nil {
		typ = fmt.Sprint(f.Type)
	} else {
		typ = fmt.Sprint(f.Ntype)
	}
	if f.Sym != nil {
		return fmt.Sprintf("%v %v", f.Sym, typ)
	}
	return typ
}

// TODO(mdempsky): Make Field a Node again so these can be generated?
// Fields are Nodes in go/ast and cmd/compile/internal/syntax.

func copyField(f *Field) *Field {
	if f == nil {
		return nil
	}
	c := *f
	return &c
}
func doField(f *Field, do func(Node) bool) bool {
	if f == nil {
		return false
	}
	if f.Decl != nil && do(f.Decl) {
		return true
	}
	if f.Ntype != nil && do(f.Ntype) {
		return true
	}
	return false
}
func editField(f *Field, edit func(Node) Node) {
	if f == nil {
		return
	}
	if f.Decl != nil {
		f.Decl = edit(f.Decl).(*Name)
	}
	if f.Ntype != nil {
		f.Ntype = edit(f.Ntype).(Ntype)
	}
}

func copyFields(list []*Field) []*Field {
	out := make([]*Field, len(list))
	for i, f := range list {
		out[i] = copyField(f)
	}
	return out
}
func doFields(list []*Field, do func(Node) bool) bool {
	for _, x := range list {
		if doField(x, do) {
			return true
		}
	}
	return false
}
func editFields(list []*Field, edit func(Node) Node) {
	for _, f := range list {
		editField(f, edit)
	}
}

// A typeNode is a Node wrapper for type t.
type typeNode struct {
	miniNode
	typ *types.Type
}

func newTypeNode(pos src.XPos, typ *types.Type) *typeNode {
	n := &typeNode{typ: typ}
	n.pos = pos
	n.op = OTYPE
	return n
}

func (n *typeNode) Type() *types.Type { return n.typ }
func (n *typeNode) Sym() *types.Sym   { return n.typ.Sym() }
func (n *typeNode) CanBeNtype()       {}

// TypeNode returns the Node representing the type t.
func TypeNode(t *types.Type) Ntype {
	return TypeNodeAt(src.NoXPos, t)
}

// TypeNodeAt is like TypeNode, but allows specifying the position
// information if a new OTYPE needs to be constructed.
//
// Deprecated: Use TypeNode instead. For typical use, the position for
// an anonymous OTYPE node should not matter. However, TypeNodeAt is
// available for use with toolstash -cmp to refactor existing code
// that is sensitive to OTYPE position.
func TypeNodeAt(pos src.XPos, t *types.Type) Ntype {
	if n := t.Obj(); n != nil {
		if n.Type() != t {
			base.Fatalf("type skew: %v has type %v, but expected %v", n, n.Type(), t)
		}
		return n.(Ntype)
	}
	return newTypeNode(pos, t)
}

// A DynamicType represents the target type in a type switch.
type DynamicType struct {
	miniExpr
	X    Node // a *runtime._type for the targeted type
	ITab Node // for type switches from nonempty interfaces to non-interfaces, this is the itab for that pair.
}

func NewDynamicType(pos src.XPos, x Node) *DynamicType {
	n := &DynamicType{X: x}
	n.pos = pos
	n.op = ODYNAMICTYPE
	return n
}
