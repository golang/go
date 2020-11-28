// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
)

// A miniStmt is a miniNode with extra fields common to statements.
type miniStmt struct {
	miniNode
	init Nodes
}

func (n *miniStmt) Init() Nodes       { return n.init }
func (n *miniStmt) SetInit(x Nodes)   { n.init = x }
func (n *miniStmt) PtrInit() *Nodes   { return &n.init }
func (n *miniStmt) HasCall() bool     { return n.bits&miniHasCall != 0 }
func (n *miniStmt) SetHasCall(b bool) { n.bits.set(miniHasCall, b) }

// A BranchStmt is a break, continue, fallthrough, or goto statement.
type BranchStmt struct {
	miniStmt
	Label *types.Sym // label if present
}

func NewBranchStmt(pos src.XPos, op Op, label *types.Sym) *BranchStmt {
	switch op {
	case OBREAK, OCONTINUE, OFALL, OGOTO:
		// ok
	default:
		panic("NewBranch " + op.String())
	}
	n := &BranchStmt{Label: label}
	n.pos = pos
	n.op = op
	return n
}

func (n *BranchStmt) String() string                { return fmt.Sprint(n) }
func (n *BranchStmt) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *BranchStmt) RawCopy() Node                 { c := *n; return &c }
func (n *BranchStmt) Sym() *types.Sym               { return n.Label }
func (n *BranchStmt) SetSym(sym *types.Sym)         { n.Label = sym }

// An EmptyStmt is an empty statement
type EmptyStmt struct {
	miniStmt
}

func NewEmptyStmt(pos src.XPos) *EmptyStmt {
	n := &EmptyStmt{}
	n.pos = pos
	n.op = OEMPTY
	return n
}

func (n *EmptyStmt) String() string                { return fmt.Sprint(n) }
func (n *EmptyStmt) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *EmptyStmt) RawCopy() Node                 { c := *n; return &c }

// A LabelStmt is a label statement (just the label, not including the statement it labels).
type LabelStmt struct {
	miniStmt
	Label *types.Sym // "Label:"
}

func NewLabelStmt(pos src.XPos, label *types.Sym) *LabelStmt {
	n := &LabelStmt{Label: label}
	n.pos = pos
	n.op = OLABEL
	return n
}

func (n *LabelStmt) String() string                { return fmt.Sprint(n) }
func (n *LabelStmt) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *LabelStmt) RawCopy() Node                 { c := *n; return &c }
func (n *LabelStmt) Sym() *types.Sym               { return n.Label }
func (n *LabelStmt) SetSym(x *types.Sym)           { n.Label = x }
