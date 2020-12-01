// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
)

// A Decl is a declaration of a const, type, or var. (A declared func is a Func.)
// (This is not technically a statement but it's not worth its own file.)
type Decl struct {
	miniNode
	X Node // the thing being declared
}

func NewDecl(pos src.XPos, op Op, x Node) *Decl {
	n := &Decl{X: x}
	n.pos = pos
	switch op {
	default:
		panic("invalid Decl op " + op.String())
	case ODCL, ODCLCONST, ODCLTYPE:
		n.op = op
	}
	return n
}

func (n *Decl) String() string                { return fmt.Sprint(n) }
func (n *Decl) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *Decl) rawCopy() Node                 { c := *n; return &c }
func (n *Decl) Left() Node                    { return n.X }
func (n *Decl) SetLeft(x Node)                { n.X = x }

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

// An AssignListStmt is an assignment statement with
// more than one item on at least one side: Lhs = Rhs.
// If Def is true, the assignment is a :=.
type AssignListStmt struct {
	miniStmt
	Lhs    Nodes
	Def    bool
	Rhs    Nodes
	offset int64 // for initorder
}

func NewAssignListStmt(pos src.XPos, lhs, rhs []Node) *AssignListStmt {
	n := &AssignListStmt{}
	n.pos = pos
	n.op = OAS2
	n.Lhs.Set(lhs)
	n.Rhs.Set(rhs)
	n.offset = types.BADWIDTH
	return n
}

func (n *AssignListStmt) String() string                { return fmt.Sprint(n) }
func (n *AssignListStmt) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *AssignListStmt) rawCopy() Node                 { c := *n; return &c }

func (n *AssignListStmt) List() Nodes       { return n.Lhs }
func (n *AssignListStmt) PtrList() *Nodes   { return &n.Lhs }
func (n *AssignListStmt) SetList(x Nodes)   { n.Lhs = x }
func (n *AssignListStmt) Rlist() Nodes      { return n.Rhs }
func (n *AssignListStmt) PtrRlist() *Nodes  { return &n.Rhs }
func (n *AssignListStmt) SetRlist(x Nodes)  { n.Rhs = x }
func (n *AssignListStmt) Colas() bool       { return n.Def }
func (n *AssignListStmt) SetColas(x bool)   { n.Def = x }
func (n *AssignListStmt) Offset() int64     { return n.offset }
func (n *AssignListStmt) SetOffset(x int64) { n.offset = x }

func (n *AssignListStmt) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OAS2, OAS2DOTTYPE, OAS2FUNC, OAS2MAPR, OAS2RECV, OSELRECV2:
		n.op = op
	}
}

// An AssignStmt is a simple assignment statement: X = Y.
// If Def is true, the assignment is a :=.
type AssignStmt struct {
	miniStmt
	X      Node
	Def    bool
	Y      Node
	offset int64 // for initorder
}

func NewAssignStmt(pos src.XPos, x, y Node) *AssignStmt {
	n := &AssignStmt{X: x, Y: y}
	n.pos = pos
	n.op = OAS
	n.offset = types.BADWIDTH
	return n
}

func (n *AssignStmt) String() string                { return fmt.Sprint(n) }
func (n *AssignStmt) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *AssignStmt) rawCopy() Node                 { c := *n; return &c }

func (n *AssignStmt) Left() Node        { return n.X }
func (n *AssignStmt) SetLeft(x Node)    { n.X = x }
func (n *AssignStmt) Right() Node       { return n.Y }
func (n *AssignStmt) SetRight(y Node)   { n.Y = y }
func (n *AssignStmt) Colas() bool       { return n.Def }
func (n *AssignStmt) SetColas(x bool)   { n.Def = x }
func (n *AssignStmt) Offset() int64     { return n.offset }
func (n *AssignStmt) SetOffset(x int64) { n.offset = x }

func (n *AssignStmt) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OAS, OSELRECV:
		n.op = op
	}
}

// An AssignOpStmt is an AsOp= assignment statement: X AsOp= Y.
type AssignOpStmt struct {
	miniStmt
	typ    *types.Type
	X      Node
	AsOp   Op // OADD etc
	Y      Node
	IncDec bool // actually ++ or --
}

func NewAssignOpStmt(pos src.XPos, op Op, x, y Node) *AssignOpStmt {
	n := &AssignOpStmt{AsOp: op, X: x, Y: y}
	n.pos = pos
	n.op = OASOP
	return n
}

func (n *AssignOpStmt) String() string                { return fmt.Sprint(n) }
func (n *AssignOpStmt) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *AssignOpStmt) rawCopy() Node                 { c := *n; return &c }

func (n *AssignOpStmt) Left() Node            { return n.X }
func (n *AssignOpStmt) SetLeft(x Node)        { n.X = x }
func (n *AssignOpStmt) Right() Node           { return n.Y }
func (n *AssignOpStmt) SetRight(y Node)       { n.Y = y }
func (n *AssignOpStmt) SubOp() Op             { return n.AsOp }
func (n *AssignOpStmt) SetSubOp(x Op)         { n.AsOp = x }
func (n *AssignOpStmt) Implicit() bool        { return n.IncDec }
func (n *AssignOpStmt) SetImplicit(b bool)    { n.IncDec = b }
func (n *AssignOpStmt) Type() *types.Type     { return n.typ }
func (n *AssignOpStmt) SetType(x *types.Type) { n.typ = x }

// A BlockStmt is a block: { List }.
type BlockStmt struct {
	miniStmt
	list Nodes
}

func NewBlockStmt(pos src.XPos, list []Node) *BlockStmt {
	n := &BlockStmt{}
	n.pos = pos
	n.op = OBLOCK
	n.list.Set(list)
	return n
}

func (n *BlockStmt) String() string                { return fmt.Sprint(n) }
func (n *BlockStmt) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *BlockStmt) rawCopy() Node                 { c := *n; return &c }
func (n *BlockStmt) List() Nodes                   { return n.list }
func (n *BlockStmt) PtrList() *Nodes               { return &n.list }
func (n *BlockStmt) SetList(x Nodes)               { n.list = x }

// A BranchStmt is a break, continue, fallthrough, or goto statement.
//
// For back-end code generation, Op may also be RETJMP (return+jump),
// in which case the label names another function entirely.
type BranchStmt struct {
	miniStmt
	Label *types.Sym // label if present
}

func NewBranchStmt(pos src.XPos, op Op, label *types.Sym) *BranchStmt {
	switch op {
	case OBREAK, OCONTINUE, OFALL, OGOTO, ORETJMP:
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
func (n *BranchStmt) rawCopy() Node                 { c := *n; return &c }
func (n *BranchStmt) Sym() *types.Sym               { return n.Label }
func (n *BranchStmt) SetSym(sym *types.Sym)         { n.Label = sym }

// A CaseStmt is a case statement in a switch or select: case List: Body.
type CaseStmt struct {
	miniStmt
	Vars Nodes // declared variable for this case in type switch
	list Nodes // list of expressions for switch, early select
	Comm Node  // communication case (Exprs[0]) after select is type-checked
	body Nodes
}

func NewCaseStmt(pos src.XPos, list, body []Node) *CaseStmt {
	n := &CaseStmt{}
	n.pos = pos
	n.op = OCASE
	n.list.Set(list)
	n.body.Set(body)
	return n
}

func (n *CaseStmt) String() string                { return fmt.Sprint(n) }
func (n *CaseStmt) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *CaseStmt) rawCopy() Node                 { c := *n; return &c }
func (n *CaseStmt) List() Nodes                   { return n.list }
func (n *CaseStmt) PtrList() *Nodes               { return &n.list }
func (n *CaseStmt) SetList(x Nodes)               { n.list = x }
func (n *CaseStmt) Body() Nodes                   { return n.body }
func (n *CaseStmt) PtrBody() *Nodes               { return &n.body }
func (n *CaseStmt) SetBody(x Nodes)               { n.body = x }
func (n *CaseStmt) Rlist() Nodes                  { return n.Vars }
func (n *CaseStmt) PtrRlist() *Nodes              { return &n.Vars }
func (n *CaseStmt) SetRlist(x Nodes)              { n.Vars = x }
func (n *CaseStmt) Left() Node                    { return n.Comm }
func (n *CaseStmt) SetLeft(x Node)                { n.Comm = x }

// A DeferStmt is a defer statement: defer Call.
type DeferStmt struct {
	miniStmt
	Call Node
}

func NewDeferStmt(pos src.XPos, call Node) *DeferStmt {
	n := &DeferStmt{Call: call}
	n.pos = pos
	n.op = ODEFER
	return n
}

func (n *DeferStmt) String() string                { return fmt.Sprint(n) }
func (n *DeferStmt) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *DeferStmt) rawCopy() Node                 { c := *n; return &c }

func (n *DeferStmt) Left() Node     { return n.Call }
func (n *DeferStmt) SetLeft(x Node) { n.Call = x }

// A ForStmt is a non-range for loop: for Init; Cond; Post { Body }
// Op can be OFOR or OFORUNTIL (!Cond).
type ForStmt struct {
	miniStmt
	Label    *types.Sym
	Cond     Node
	Post     Node
	Late     Nodes
	body     Nodes
	hasBreak bool
}

func NewForStmt(pos src.XPos, init []Node, cond, post Node, body []Node) *ForStmt {
	n := &ForStmt{Cond: cond, Post: post}
	n.pos = pos
	n.op = OFOR
	n.init.Set(init)
	n.body.Set(body)
	return n
}

func (n *ForStmt) String() string                { return fmt.Sprint(n) }
func (n *ForStmt) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *ForStmt) rawCopy() Node                 { c := *n; return &c }
func (n *ForStmt) Sym() *types.Sym               { return n.Label }
func (n *ForStmt) SetSym(x *types.Sym)           { n.Label = x }
func (n *ForStmt) Left() Node                    { return n.Cond }
func (n *ForStmt) SetLeft(x Node)                { n.Cond = x }
func (n *ForStmt) Right() Node                   { return n.Post }
func (n *ForStmt) SetRight(x Node)               { n.Post = x }
func (n *ForStmt) Body() Nodes                   { return n.body }
func (n *ForStmt) PtrBody() *Nodes               { return &n.body }
func (n *ForStmt) SetBody(x Nodes)               { n.body = x }
func (n *ForStmt) List() Nodes                   { return n.Late }
func (n *ForStmt) PtrList() *Nodes               { return &n.Late }
func (n *ForStmt) SetList(x Nodes)               { n.Late = x }
func (n *ForStmt) HasBreak() bool                { return n.hasBreak }
func (n *ForStmt) SetHasBreak(b bool)            { n.hasBreak = b }

func (n *ForStmt) SetOp(op Op) {
	if op != OFOR && op != OFORUNTIL {
		panic(n.no("SetOp " + op.String()))
	}
	n.op = op
}

// A GoStmt is a go statement: go Call.
type GoStmt struct {
	miniStmt
	Call Node
}

func NewGoStmt(pos src.XPos, call Node) *GoStmt {
	n := &GoStmt{Call: call}
	n.pos = pos
	n.op = OGO
	return n
}

func (n *GoStmt) String() string                { return fmt.Sprint(n) }
func (n *GoStmt) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *GoStmt) rawCopy() Node                 { c := *n; return &c }

func (n *GoStmt) Left() Node     { return n.Call }
func (n *GoStmt) SetLeft(x Node) { n.Call = x }

// A IfStmt is a return statement: if Init; Cond { Then } else { Else }.
type IfStmt struct {
	miniStmt
	Cond   Node
	body   Nodes
	Else   Nodes
	likely bool // code layout hint
}

func NewIfStmt(pos src.XPos, cond Node, body, els []Node) *IfStmt {
	n := &IfStmt{Cond: cond}
	n.pos = pos
	n.op = OIF
	n.body.Set(body)
	n.Else.Set(els)
	return n
}

func (n *IfStmt) String() string                { return fmt.Sprint(n) }
func (n *IfStmt) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *IfStmt) rawCopy() Node                 { c := *n; return &c }
func (n *IfStmt) Left() Node                    { return n.Cond }
func (n *IfStmt) SetLeft(x Node)                { n.Cond = x }
func (n *IfStmt) Body() Nodes                   { return n.body }
func (n *IfStmt) PtrBody() *Nodes               { return &n.body }
func (n *IfStmt) SetBody(x Nodes)               { n.body = x }
func (n *IfStmt) Rlist() Nodes                  { return n.Else }
func (n *IfStmt) PtrRlist() *Nodes              { return &n.Else }
func (n *IfStmt) SetRlist(x Nodes)              { n.Else = x }
func (n *IfStmt) Likely() bool                  { return n.likely }
func (n *IfStmt) SetLikely(x bool)              { n.likely = x }

// An InlineMarkStmt is a marker placed just before an inlined body.
type InlineMarkStmt struct {
	miniStmt
	Index int64
}

func NewInlineMarkStmt(pos src.XPos, index int64) *InlineMarkStmt {
	n := &InlineMarkStmt{Index: index}
	n.pos = pos
	n.op = OINLMARK
	return n
}

func (n *InlineMarkStmt) String() string                { return fmt.Sprint(n) }
func (n *InlineMarkStmt) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *InlineMarkStmt) rawCopy() Node                 { c := *n; return &c }
func (n *InlineMarkStmt) Offset() int64                 { return n.Index }
func (n *InlineMarkStmt) SetOffset(x int64)             { n.Index = x }

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
func (n *LabelStmt) rawCopy() Node                 { c := *n; return &c }
func (n *LabelStmt) Sym() *types.Sym               { return n.Label }
func (n *LabelStmt) SetSym(x *types.Sym)           { n.Label = x }

// A RangeStmt is a range loop: for Vars = range X { Stmts }
// Op can be OFOR or OFORUNTIL (!Cond).
type RangeStmt struct {
	miniStmt
	Label    *types.Sym
	Vars     Nodes // TODO(rsc): Replace with Key, Value Node
	Def      bool
	X        Node
	body     Nodes
	hasBreak bool
	typ      *types.Type // TODO(rsc): Remove - use X.Type() instead
}

func NewRangeStmt(pos src.XPos, vars []Node, x Node, body []Node) *RangeStmt {
	n := &RangeStmt{X: x}
	n.pos = pos
	n.op = ORANGE
	n.Vars.Set(vars)
	n.body.Set(body)
	return n
}

func (n *RangeStmt) String() string                { return fmt.Sprint(n) }
func (n *RangeStmt) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *RangeStmt) rawCopy() Node                 { c := *n; return &c }
func (n *RangeStmt) Sym() *types.Sym               { return n.Label }
func (n *RangeStmt) SetSym(x *types.Sym)           { n.Label = x }
func (n *RangeStmt) Right() Node                   { return n.X }
func (n *RangeStmt) SetRight(x Node)               { n.X = x }
func (n *RangeStmt) Body() Nodes                   { return n.body }
func (n *RangeStmt) PtrBody() *Nodes               { return &n.body }
func (n *RangeStmt) SetBody(x Nodes)               { n.body = x }
func (n *RangeStmt) List() Nodes                   { return n.Vars }
func (n *RangeStmt) PtrList() *Nodes               { return &n.Vars }
func (n *RangeStmt) SetList(x Nodes)               { n.Vars = x }
func (n *RangeStmt) HasBreak() bool                { return n.hasBreak }
func (n *RangeStmt) SetHasBreak(b bool)            { n.hasBreak = b }
func (n *RangeStmt) Colas() bool                   { return n.Def }
func (n *RangeStmt) SetColas(b bool)               { n.Def = b }
func (n *RangeStmt) Type() *types.Type             { return n.typ }
func (n *RangeStmt) SetType(x *types.Type)         { n.typ = x }

// A ReturnStmt is a return statement.
type ReturnStmt struct {
	miniStmt
	orig    Node  // for typecheckargs rewrite
	Results Nodes // return list
}

func NewReturnStmt(pos src.XPos, results []Node) *ReturnStmt {
	n := &ReturnStmt{}
	n.pos = pos
	n.op = ORETURN
	n.orig = n
	n.Results.Set(results)
	return n
}

func (n *ReturnStmt) String() string                { return fmt.Sprint(n) }
func (n *ReturnStmt) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *ReturnStmt) rawCopy() Node                 { c := *n; return &c }
func (n *ReturnStmt) Orig() Node                    { return n.orig }
func (n *ReturnStmt) SetOrig(x Node)                { n.orig = x }
func (n *ReturnStmt) List() Nodes                   { return n.Results }
func (n *ReturnStmt) PtrList() *Nodes               { return &n.Results }
func (n *ReturnStmt) SetList(x Nodes)               { n.Results = x }
func (n *ReturnStmt) IsDDD() bool                   { return false } // typecheckargs asks

// A SelectStmt is a block: { Cases }.
type SelectStmt struct {
	miniStmt
	Label    *types.Sym
	Cases    Nodes
	hasBreak bool

	// TODO(rsc): Instead of recording here, replace with a block?
	Compiled Nodes // compiled form, after walkswitch
}

func NewSelectStmt(pos src.XPos, cases []Node) *SelectStmt {
	n := &SelectStmt{}
	n.pos = pos
	n.op = OSELECT
	n.Cases.Set(cases)
	return n
}

func (n *SelectStmt) String() string                { return fmt.Sprint(n) }
func (n *SelectStmt) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *SelectStmt) rawCopy() Node                 { c := *n; return &c }
func (n *SelectStmt) List() Nodes                   { return n.Cases }
func (n *SelectStmt) PtrList() *Nodes               { return &n.Cases }
func (n *SelectStmt) SetList(x Nodes)               { n.Cases = x }
func (n *SelectStmt) Sym() *types.Sym               { return n.Label }
func (n *SelectStmt) SetSym(x *types.Sym)           { n.Label = x }
func (n *SelectStmt) HasBreak() bool                { return n.hasBreak }
func (n *SelectStmt) SetHasBreak(x bool)            { n.hasBreak = x }
func (n *SelectStmt) Body() Nodes                   { return n.Compiled }
func (n *SelectStmt) PtrBody() *Nodes               { return &n.Compiled }
func (n *SelectStmt) SetBody(x Nodes)               { n.Compiled = x }

// A SendStmt is a send statement: X <- Y.
type SendStmt struct {
	miniStmt
	Chan  Node
	Value Node
}

func NewSendStmt(pos src.XPos, ch, value Node) *SendStmt {
	n := &SendStmt{Chan: ch, Value: value}
	n.pos = pos
	n.op = OSEND
	return n
}

func (n *SendStmt) String() string                { return fmt.Sprint(n) }
func (n *SendStmt) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *SendStmt) rawCopy() Node                 { c := *n; return &c }

func (n *SendStmt) Left() Node      { return n.Chan }
func (n *SendStmt) SetLeft(x Node)  { n.Chan = x }
func (n *SendStmt) Right() Node     { return n.Value }
func (n *SendStmt) SetRight(y Node) { n.Value = y }

// A SwitchStmt is a switch statement: switch Init; Expr { Cases }.
type SwitchStmt struct {
	miniStmt
	Tag      Node
	Cases    Nodes // list of *CaseStmt
	Label    *types.Sym
	hasBreak bool

	// TODO(rsc): Instead of recording here, replace with a block?
	Compiled Nodes // compiled form, after walkswitch
}

func NewSwitchStmt(pos src.XPos, tag Node, cases []Node) *SwitchStmt {
	n := &SwitchStmt{Tag: tag}
	n.pos = pos
	n.op = OSWITCH
	n.Cases.Set(cases)
	return n
}

func (n *SwitchStmt) String() string                { return fmt.Sprint(n) }
func (n *SwitchStmt) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *SwitchStmt) rawCopy() Node                 { c := *n; return &c }
func (n *SwitchStmt) Left() Node                    { return n.Tag }
func (n *SwitchStmt) SetLeft(x Node)                { n.Tag = x }
func (n *SwitchStmt) List() Nodes                   { return n.Cases }
func (n *SwitchStmt) PtrList() *Nodes               { return &n.Cases }
func (n *SwitchStmt) SetList(x Nodes)               { n.Cases = x }
func (n *SwitchStmt) Body() Nodes                   { return n.Compiled }
func (n *SwitchStmt) PtrBody() *Nodes               { return &n.Compiled }
func (n *SwitchStmt) SetBody(x Nodes)               { n.Compiled = x }
func (n *SwitchStmt) Sym() *types.Sym               { return n.Label }
func (n *SwitchStmt) SetSym(x *types.Sym)           { n.Label = x }
func (n *SwitchStmt) HasBreak() bool                { return n.hasBreak }
func (n *SwitchStmt) SetHasBreak(x bool)            { n.hasBreak = x }

// A TypeSwitchGuard is the [Name :=] X.(type) in a type switch.
type TypeSwitchGuard struct {
	miniNode
	name *Name
	X    Node
}

func NewTypeSwitchGuard(pos src.XPos, name, x Node) *TypeSwitchGuard {
	n := &TypeSwitchGuard{X: x}
	if name != nil {
		n.name = name.(*Name)
	}
	n.pos = pos
	n.op = OTYPESW
	return n
}

func (n *TypeSwitchGuard) String() string                { return fmt.Sprint(n) }
func (n *TypeSwitchGuard) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *TypeSwitchGuard) rawCopy() Node                 { c := *n; return &c }

func (n *TypeSwitchGuard) Left() Node {
	if n.name == nil {
		return nil
	}
	return n.name
}
func (n *TypeSwitchGuard) SetLeft(x Node) {
	if x == nil {
		n.name = nil
		return
	}
	n.name = x.(*Name)
}
func (n *TypeSwitchGuard) Right() Node     { return n.X }
func (n *TypeSwitchGuard) SetRight(x Node) { n.X = x }
