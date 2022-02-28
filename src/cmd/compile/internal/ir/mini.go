// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:generate go run -mod=mod mknode.go

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
//	func (n *MyNode) rawCopy() Node { c := *n; return &c }
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

// posOr returns pos if known, or else n.pos.
// For use in DeepCopy.
func (n *miniNode) posOr(pos src.XPos) src.XPos {
	if pos.IsKnown() {
		return pos
	}
	return n.pos
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
	miniWalkdefShift   = 0 // TODO(mdempsky): Move to Name.flags.
	miniTypecheckShift = 2
	miniDiag           = 1 << 4
	miniWalked         = 1 << 5 // to prevent/catch re-walking
)

func (n *miniNode) Typecheck() uint8 { return n.bits.get2(miniTypecheckShift) }
func (n *miniNode) SetTypecheck(x uint8) {
	if x > 2 {
		panic(fmt.Sprintf("cannot SetTypecheck %d", x))
	}
	n.bits.set2(miniTypecheckShift, x)
}

func (n *miniNode) Diag() bool     { return n.bits&miniDiag != 0 }
func (n *miniNode) SetDiag(x bool) { n.bits.set(miniDiag, x) }

func (n *miniNode) Walked() bool     { return n.bits&miniWalked != 0 }
func (n *miniNode) SetWalked(x bool) { n.bits.set(miniWalked, x) }

// Empty, immutable graph structure.

func (n *miniNode) Init() Nodes { return Nodes{} }

// Additional functionality unavailable.

func (n *miniNode) no(name string) string { return "cannot " + name + " on " + n.op.String() }

func (n *miniNode) Type() *types.Type       { return nil }
func (n *miniNode) SetType(*types.Type)     { panic(n.no("SetType")) }
func (n *miniNode) Name() *Name             { return nil }
func (n *miniNode) Sym() *types.Sym         { return nil }
func (n *miniNode) Val() constant.Value     { panic(n.no("Val")) }
func (n *miniNode) SetVal(v constant.Value) { panic(n.no("SetVal")) }
func (n *miniNode) NonNil() bool            { return false }
func (n *miniNode) MarkNonNil()             { panic(n.no("MarkNonNil")) }
