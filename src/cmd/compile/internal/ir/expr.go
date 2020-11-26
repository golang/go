// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
)

// A miniStmt is a miniNode with extra fields common to expressions.
// TODO(rsc): Once we are sure about the contents, compact the bools
// into a bit field and leave extra bits available for implementations
// embedding miniExpr. Right now there are ~60 unused bits sitting here.
type miniExpr struct {
	miniNode
	typ   *types.Type
	init  Nodes       // TODO(rsc): Don't require every Node to have an init
	opt   interface{} // TODO(rsc): Don't require every Node to have an opt?
	flags bitset8
}

const (
	miniExprHasCall = 1 << iota
	miniExprImplicit
	miniExprNonNil
	miniExprTransient
	miniExprBounded
)

func (n *miniExpr) Type() *types.Type     { return n.typ }
func (n *miniExpr) SetType(x *types.Type) { n.typ = x }
func (n *miniExpr) Opt() interface{}      { return n.opt }
func (n *miniExpr) SetOpt(x interface{})  { n.opt = x }
func (n *miniExpr) HasCall() bool         { return n.flags&miniExprHasCall != 0 }
func (n *miniExpr) SetHasCall(b bool)     { n.flags.set(miniExprHasCall, b) }
func (n *miniExpr) Implicit() bool        { return n.flags&miniExprImplicit != 0 }
func (n *miniExpr) SetImplicit(b bool)    { n.flags.set(miniExprImplicit, b) }
func (n *miniExpr) NonNil() bool          { return n.flags&miniExprNonNil != 0 }
func (n *miniExpr) MarkNonNil()           { n.flags |= miniExprNonNil }
func (n *miniExpr) Transient() bool       { return n.flags&miniExprTransient != 0 }
func (n *miniExpr) SetTransient(b bool)   { n.flags.set(miniExprTransient, b) }
func (n *miniExpr) Bounded() bool         { return n.flags&miniExprBounded != 0 }
func (n *miniExpr) SetBounded(b bool)     { n.flags.set(miniExprBounded, b) }
func (n *miniExpr) Init() Nodes           { return n.init }
func (n *miniExpr) PtrInit() *Nodes       { return &n.init }
func (n *miniExpr) SetInit(x Nodes)       { n.init = x }

// A ClosureExpr is a function literal expression.
type ClosureExpr struct {
	miniExpr
	fn *Func
}

func NewClosureExpr(pos src.XPos, fn *Func) *ClosureExpr {
	n := &ClosureExpr{fn: fn}
	n.op = OCLOSURE
	n.pos = pos
	return n
}

func (n *ClosureExpr) String() string                { return fmt.Sprint(n) }
func (n *ClosureExpr) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *ClosureExpr) RawCopy() Node                 { c := *n; return &c }
func (n *ClosureExpr) Func() *Func                   { return n.fn }

// A ClosureRead denotes reading a variable stored within a closure struct.
type ClosureRead struct {
	miniExpr
	offset int64
}

func NewClosureRead(typ *types.Type, offset int64) *ClosureRead {
	n := &ClosureRead{offset: offset}
	n.typ = typ
	n.op = OCLOSUREREAD
	return n
}

func (n *ClosureRead) String() string                { return fmt.Sprint(n) }
func (n *ClosureRead) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *ClosureRead) RawCopy() Node                 { c := *n; return &c }
func (n *ClosureRead) Type() *types.Type             { return n.typ }
func (n *ClosureRead) Offset() int64                 { return n.offset }

// A CallPartExpr is a method expression X.Method (uncalled).
type CallPartExpr struct {
	miniExpr
	fn     *Func
	X      Node
	Method *Name
}

func NewCallPartExpr(pos src.XPos, x Node, method *Name, fn *Func) *CallPartExpr {
	n := &CallPartExpr{fn: fn, X: x, Method: method}
	n.op = OCALLPART
	n.pos = pos
	n.typ = fn.Type()
	n.fn = fn
	return n
}

func (n *CallPartExpr) String() string                { return fmt.Sprint(n) }
func (n *CallPartExpr) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *CallPartExpr) RawCopy() Node                 { c := *n; return &c }
func (n *CallPartExpr) Func() *Func                   { return n.fn }
func (n *CallPartExpr) Left() Node                    { return n.X }
func (n *CallPartExpr) Right() Node                   { return n.Method }
func (n *CallPartExpr) SetLeft(x Node)                { n.X = x }
func (n *CallPartExpr) SetRight(x Node)               { n.Method = x.(*Name) }

// A StarExpr is a dereference expression *X.
// It may end up being a value or a type.
type StarExpr struct {
	miniExpr
	X Node
}

func NewStarExpr(pos src.XPos, x Node) *StarExpr {
	n := &StarExpr{X: x}
	n.op = ODEREF
	n.pos = pos
	return n
}

func (n *StarExpr) String() string                { return fmt.Sprint(n) }
func (n *StarExpr) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *StarExpr) RawCopy() Node                 { c := *n; return &c }
func (n *StarExpr) Left() Node                    { return n.X }
func (n *StarExpr) SetLeft(x Node)                { n.X = x }

func (*StarExpr) CanBeNtype() {}

// SetOTYPE changes n to be an OTYPE node returning t,
// like all the type nodes in type.go.
func (n *StarExpr) SetOTYPE(t *types.Type) {
	n.op = OTYPE
	n.X = nil
	n.typ = t
	if t.Nod == nil {
		t.Nod = n
	}
}

func (n *StarExpr) DeepCopy(pos src.XPos) Node {
	if n.op == OTYPE {
		// Can't change types and no node references left.
		return n
	}
	c := SepCopy(n).(*StarExpr)
	c.pos = n.posOr(pos)
	c.X = DeepCopy(pos, n.X)
	return c
}
