// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
	"go/constant"
)

// A miniNode is a minimal node implementation,
// meant to be embedded as the first field in a larger node implementation,
// at a cost of 8 bytes.
//
// A miniNode is NOT a valid Node by itself: the embedding struct
// must at the least provide:
//
//	func (n *MyNode) String() string { return fmt.Sprint(n) }
//	func (n *MyNode) RawCopy() Node { c := *n; return &c }
//	func (n *MyNode) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
//
// The embedding struct should also fill in n.op in its constructor,
// for more useful panic messages when invalid methods are called,
// instead of implementing Op itself.
//
type miniNode struct {
	pos  src.XPos // uint32
	op   Op       // uint8
	bits bitset8
	esc  uint16
}

// op can be read, but not written.
// An embedding implementation can provide a SetOp if desired.
// (The panicking SetOp is with the other panics below.)
func (n *miniNode) Op() Op            { return n.op }
func (n *miniNode) Pos() src.XPos     { return n.pos }
func (n *miniNode) SetPos(x src.XPos) { n.pos = x }
func (n *miniNode) Esc() uint16       { return n.esc }
func (n *miniNode) SetEsc(x uint16)   { n.esc = x }

const (
	miniWalkdefShift   = 0
	miniTypecheckShift = 2
	miniInitorderShift = 4
	miniDiag           = 1 << 6
	miniHasCall        = 1 << 7 // for miniStmt
)

func (n *miniNode) Walkdef() uint8   { return n.bits.get2(miniWalkdefShift) }
func (n *miniNode) Typecheck() uint8 { return n.bits.get2(miniTypecheckShift) }
func (n *miniNode) Initorder() uint8 { return n.bits.get2(miniInitorderShift) }
func (n *miniNode) SetWalkdef(x uint8) {
	if x > 3 {
		panic(fmt.Sprintf("cannot SetWalkdef %d", x))
	}
	n.bits.set2(miniWalkdefShift, x)
}
func (n *miniNode) SetTypecheck(x uint8) {
	if x > 3 {
		panic(fmt.Sprintf("cannot SetTypecheck %d", x))
	}
	n.bits.set2(miniTypecheckShift, x)
}
func (n *miniNode) SetInitorder(x uint8) {
	if x > 3 {
		panic(fmt.Sprintf("cannot SetInitorder %d", x))
	}
	n.bits.set2(miniInitorderShift, x)
}

func (n *miniNode) Diag() bool     { return n.bits&miniDiag != 0 }
func (n *miniNode) SetDiag(x bool) { n.bits.set(miniDiag, x) }

// Empty, immutable graph structure.

func (n *miniNode) Left() Node       { return nil }
func (n *miniNode) Right() Node      { return nil }
func (n *miniNode) Init() Nodes      { return Nodes{} }
func (n *miniNode) PtrInit() *Nodes  { return &immutableEmptyNodes }
func (n *miniNode) Body() Nodes      { return Nodes{} }
func (n *miniNode) PtrBody() *Nodes  { return &immutableEmptyNodes }
func (n *miniNode) List() Nodes      { return Nodes{} }
func (n *miniNode) PtrList() *Nodes  { return &immutableEmptyNodes }
func (n *miniNode) Rlist() Nodes     { return Nodes{} }
func (n *miniNode) PtrRlist() *Nodes { return &immutableEmptyNodes }
func (n *miniNode) SetLeft(x Node) {
	if x != nil {
		panic(n.no("SetLeft"))
	}
}
func (n *miniNode) SetRight(x Node) {
	if x != nil {
		panic(n.no("SetRight"))
	}
}
func (n *miniNode) SetInit(x Nodes) {
	if x != (Nodes{}) {
		panic(n.no("SetInit"))
	}
}
func (n *miniNode) SetBody(x Nodes) {
	if x != (Nodes{}) {
		panic(n.no("SetBody"))
	}
}
func (n *miniNode) SetList(x Nodes) {
	if x != (Nodes{}) {
		panic(n.no("SetList"))
	}
}
func (n *miniNode) SetRlist(x Nodes) {
	if x != (Nodes{}) {
		panic(n.no("SetRlist"))
	}
}

// Additional functionality unavailable.

func (n *miniNode) no(name string) string { return "cannot " + name + " on " + n.op.String() }

func (n *miniNode) SetOp(Op)            { panic(n.no("SetOp")) }
func (n *miniNode) SubOp() Op           { panic(n.no("SubOp")) }
func (n *miniNode) SetSubOp(Op)         { panic(n.no("SetSubOp")) }
func (n *miniNode) Type() *types.Type   { return nil }
func (n *miniNode) SetType(*types.Type) { panic(n.no("SetType")) }
func (n *miniNode) Func() *Func         { panic(n.no("Func")) }
func (n *miniNode) SetFunc(*Func)       { panic(n.no("SetFunc")) }
func (n *miniNode) Name() *Name         { return nil }
func (n *miniNode) SetName(*Name)       { panic(n.no("SetName")) }
func (n *miniNode) Sym() *types.Sym     { return nil }
func (n *miniNode) SetSym(*types.Sym)   { panic(n.no("SetSym")) }
func (n *miniNode) Offset() int64       { return types.BADWIDTH }
func (n *miniNode) SetOffset(x int64)   { panic(n.no("SetOffset")) }
func (n *miniNode) Class() Class        { return Pxxx }
func (n *miniNode) SetClass(Class)      { panic(n.no("SetClass")) }
func (n *miniNode) Likely() bool        { panic(n.no("Likely")) }
func (n *miniNode) SetLikely(bool)      { panic(n.no("SetLikely")) }
func (n *miniNode) SliceBounds() (low, high, max Node) {
	panic(n.no("SliceBounds"))
}
func (n *miniNode) SetSliceBounds(low, high, max Node) {
	panic(n.no("SetSliceBounds"))
}
func (n *miniNode) Iota() int64               { panic(n.no("Iota")) }
func (n *miniNode) SetIota(int64)             { panic(n.no("SetIota")) }
func (n *miniNode) Colas() bool               { return false }
func (n *miniNode) SetColas(bool)             { panic(n.no("SetColas")) }
func (n *miniNode) NoInline() bool            { panic(n.no("NoInline")) }
func (n *miniNode) SetNoInline(bool)          { panic(n.no("SetNoInline")) }
func (n *miniNode) Transient() bool           { panic(n.no("Transient")) }
func (n *miniNode) SetTransient(bool)         { panic(n.no("SetTransient")) }
func (n *miniNode) Implicit() bool            { return false }
func (n *miniNode) SetImplicit(bool)          { panic(n.no("SetImplicit")) }
func (n *miniNode) IsDDD() bool               { return false }
func (n *miniNode) SetIsDDD(bool)             { panic(n.no("SetIsDDD")) }
func (n *miniNode) Embedded() bool            { return false }
func (n *miniNode) SetEmbedded(bool)          { panic(n.no("SetEmbedded")) }
func (n *miniNode) IndexMapLValue() bool      { panic(n.no("IndexMapLValue")) }
func (n *miniNode) SetIndexMapLValue(bool)    { panic(n.no("SetIndexMapLValue")) }
func (n *miniNode) ResetAux()                 { panic(n.no("ResetAux")) }
func (n *miniNode) HasBreak() bool            { panic(n.no("HasBreak")) }
func (n *miniNode) SetHasBreak(bool)          { panic(n.no("SetHasBreak")) }
func (n *miniNode) HasVal() bool              { return false }
func (n *miniNode) Val() constant.Value       { panic(n.no("Val")) }
func (n *miniNode) SetVal(v constant.Value)   { panic(n.no("SetVal")) }
func (n *miniNode) Int64Val() int64           { panic(n.no("Int64Val")) }
func (n *miniNode) Uint64Val() uint64         { panic(n.no("Uint64Val")) }
func (n *miniNode) CanInt64() bool            { panic(n.no("CanInt64")) }
func (n *miniNode) BoolVal() bool             { panic(n.no("BoolVal")) }
func (n *miniNode) StringVal() string         { panic(n.no("StringVal")) }
func (n *miniNode) HasCall() bool             { panic(n.no("HasCall")) }
func (n *miniNode) SetHasCall(bool)           { panic(n.no("SetHasCall")) }
func (n *miniNode) NonNil() bool              { return false }
func (n *miniNode) MarkNonNil()               { panic(n.no("MarkNonNil")) }
func (n *miniNode) Bounded() bool             { return false }
func (n *miniNode) SetBounded(bool)           { panic(n.no("SetBounded")) }
func (n *miniNode) Opt() interface{}          { return nil }
func (n *miniNode) SetOpt(interface{})        { panic(n.no("SetOpt")) }
func (n *miniNode) MarkReadonly()             { panic(n.no("MarkReadonly")) }
func (n *miniNode) TChanDir() types.ChanDir   { panic(n.no("TChanDir")) }
func (n *miniNode) SetTChanDir(types.ChanDir) { panic(n.no("SetTChanDir")) }

// TODO: Delete when CanBeAnSSASym is removed from Node itself.
func (*miniNode) CanBeAnSSASym() {}
