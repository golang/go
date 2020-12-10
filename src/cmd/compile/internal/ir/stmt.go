// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

// A Decl is a declaration of a const, type, or var. (A declared func is a Func.)
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

func (*Decl) isStmt() {}

func (n *Decl) Left() Node     { return n.X }
func (n *Decl) SetLeft(x Node) { n.X = x }

// A Stmt is a Node that can appear as a statement.
// This includes statement-like expressions such as f().
//
// (It's possible it should include <-c, but that would require
// splitting ORECV out of UnaryExpr, which hasn't yet been
// necessary. Maybe instead we will introduce ExprStmt at
// some point.)
type Stmt interface {
	Node
	isStmt()
}

// A miniStmt is a miniNode with extra fields common to statements.
type miniStmt struct {
	miniNode
	init Nodes
}

func (*miniStmt) isStmt() {}

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
	Lhs Nodes
	Def bool
	Rhs Nodes
}

func NewAssignListStmt(pos src.XPos, op Op, lhs, rhs []Node) *AssignListStmt {
	n := &AssignListStmt{}
	n.pos = pos
	n.SetOp(op)
	n.Lhs.Set(lhs)
	n.Rhs.Set(rhs)
	return n
}

func (n *AssignListStmt) List() Nodes      { return n.Lhs }
func (n *AssignListStmt) PtrList() *Nodes  { return &n.Lhs }
func (n *AssignListStmt) SetList(x Nodes)  { n.Lhs = x }
func (n *AssignListStmt) Rlist() Nodes     { return n.Rhs }
func (n *AssignListStmt) PtrRlist() *Nodes { return &n.Rhs }
func (n *AssignListStmt) SetRlist(x Nodes) { n.Rhs = x }
func (n *AssignListStmt) Colas() bool      { return n.Def }
func (n *AssignListStmt) SetColas(x bool)  { n.Def = x }

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
	X   Node
	Def bool
	Y   Node
}

func NewAssignStmt(pos src.XPos, x, y Node) *AssignStmt {
	n := &AssignStmt{X: x, Y: y}
	n.pos = pos
	n.op = OAS
	return n
}

func (n *AssignStmt) Left() Node      { return n.X }
func (n *AssignStmt) SetLeft(x Node)  { n.X = x }
func (n *AssignStmt) Right() Node     { return n.Y }
func (n *AssignStmt) SetRight(y Node) { n.Y = y }
func (n *AssignStmt) Colas() bool     { return n.Def }
func (n *AssignStmt) SetColas(x bool) { n.Def = x }

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

func NewAssignOpStmt(pos src.XPos, asOp Op, x, y Node) *AssignOpStmt {
	n := &AssignOpStmt{AsOp: asOp, X: x, Y: y}
	n.pos = pos
	n.op = OASOP
	return n
}

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
	List_ Nodes
}

func NewBlockStmt(pos src.XPos, list []Node) *BlockStmt {
	n := &BlockStmt{}
	n.pos = pos
	n.op = OBLOCK
	n.List_.Set(list)
	return n
}

func (n *BlockStmt) List() Nodes     { return n.List_ }
func (n *BlockStmt) PtrList() *Nodes { return &n.List_ }
func (n *BlockStmt) SetList(x Nodes) { n.List_ = x }

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

func (n *BranchStmt) Sym() *types.Sym       { return n.Label }
func (n *BranchStmt) SetSym(sym *types.Sym) { n.Label = sym }

// A CaseStmt is a case statement in a switch or select: case List: Body.
type CaseStmt struct {
	miniStmt
	Vars  Nodes // declared variable for this case in type switch
	List_ Nodes // list of expressions for switch, early select
	Comm  Node  // communication case (Exprs[0]) after select is type-checked
	Body_ Nodes
}

func NewCaseStmt(pos src.XPos, list, body []Node) *CaseStmt {
	n := &CaseStmt{}
	n.pos = pos
	n.op = OCASE
	n.List_.Set(list)
	n.Body_.Set(body)
	return n
}

func (n *CaseStmt) List() Nodes      { return n.List_ }
func (n *CaseStmt) PtrList() *Nodes  { return &n.List_ }
func (n *CaseStmt) SetList(x Nodes)  { n.List_ = x }
func (n *CaseStmt) Body() Nodes      { return n.Body_ }
func (n *CaseStmt) PtrBody() *Nodes  { return &n.Body_ }
func (n *CaseStmt) SetBody(x Nodes)  { n.Body_ = x }
func (n *CaseStmt) Rlist() Nodes     { return n.Vars }
func (n *CaseStmt) PtrRlist() *Nodes { return &n.Vars }
func (n *CaseStmt) SetRlist(x Nodes) { n.Vars = x }
func (n *CaseStmt) Left() Node       { return n.Comm }
func (n *CaseStmt) SetLeft(x Node)   { n.Comm = x }

// A ForStmt is a non-range for loop: for Init; Cond; Post { Body }
// Op can be OFOR or OFORUNTIL (!Cond).
type ForStmt struct {
	miniStmt
	Label     *types.Sym
	Cond      Node
	Late      Nodes
	Post      Node
	Body_     Nodes
	HasBreak_ bool
}

func NewForStmt(pos src.XPos, init []Node, cond, post Node, body []Node) *ForStmt {
	n := &ForStmt{Cond: cond, Post: post}
	n.pos = pos
	n.op = OFOR
	n.init.Set(init)
	n.Body_.Set(body)
	return n
}

func (n *ForStmt) Sym() *types.Sym     { return n.Label }
func (n *ForStmt) SetSym(x *types.Sym) { n.Label = x }
func (n *ForStmt) Left() Node          { return n.Cond }
func (n *ForStmt) SetLeft(x Node)      { n.Cond = x }
func (n *ForStmt) Right() Node         { return n.Post }
func (n *ForStmt) SetRight(x Node)     { n.Post = x }
func (n *ForStmt) Body() Nodes         { return n.Body_ }
func (n *ForStmt) PtrBody() *Nodes     { return &n.Body_ }
func (n *ForStmt) SetBody(x Nodes)     { n.Body_ = x }
func (n *ForStmt) List() Nodes         { return n.Late }
func (n *ForStmt) PtrList() *Nodes     { return &n.Late }
func (n *ForStmt) SetList(x Nodes)     { n.Late = x }
func (n *ForStmt) HasBreak() bool      { return n.HasBreak_ }
func (n *ForStmt) SetHasBreak(b bool)  { n.HasBreak_ = b }

func (n *ForStmt) SetOp(op Op) {
	if op != OFOR && op != OFORUNTIL {
		panic(n.no("SetOp " + op.String()))
	}
	n.op = op
}

// A GoDeferStmt is a go or defer statement: go Call / defer Call.
//
// The two opcodes use a signle syntax because the implementations
// are very similar: both are concerned with saving Call and running it
// in a different context (a separate goroutine or a later time).
type GoDeferStmt struct {
	miniStmt
	Call Node
}

func NewGoDeferStmt(pos src.XPos, op Op, call Node) *GoDeferStmt {
	n := &GoDeferStmt{Call: call}
	n.pos = pos
	switch op {
	case ODEFER, OGO:
		n.op = op
	default:
		panic("NewGoDeferStmt " + op.String())
	}
	return n
}

func (n *GoDeferStmt) Left() Node     { return n.Call }
func (n *GoDeferStmt) SetLeft(x Node) { n.Call = x }

// A IfStmt is a return statement: if Init; Cond { Then } else { Else }.
type IfStmt struct {
	miniStmt
	Cond    Node
	Body_   Nodes
	Else    Nodes
	Likely_ bool // code layout hint
}

func NewIfStmt(pos src.XPos, cond Node, body, els []Node) *IfStmt {
	n := &IfStmt{Cond: cond}
	n.pos = pos
	n.op = OIF
	n.Body_.Set(body)
	n.Else.Set(els)
	return n
}

func (n *IfStmt) Left() Node       { return n.Cond }
func (n *IfStmt) SetLeft(x Node)   { n.Cond = x }
func (n *IfStmt) Body() Nodes      { return n.Body_ }
func (n *IfStmt) PtrBody() *Nodes  { return &n.Body_ }
func (n *IfStmt) SetBody(x Nodes)  { n.Body_ = x }
func (n *IfStmt) Rlist() Nodes     { return n.Else }
func (n *IfStmt) PtrRlist() *Nodes { return &n.Else }
func (n *IfStmt) SetRlist(x Nodes) { n.Else = x }
func (n *IfStmt) Likely() bool     { return n.Likely_ }
func (n *IfStmt) SetLikely(x bool) { n.Likely_ = x }

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

func (n *InlineMarkStmt) Offset() int64     { return n.Index }
func (n *InlineMarkStmt) SetOffset(x int64) { n.Index = x }

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

func (n *LabelStmt) Sym() *types.Sym     { return n.Label }
func (n *LabelStmt) SetSym(x *types.Sym) { n.Label = x }

// A RangeStmt is a range loop: for Vars = range X { Stmts }
// Op can be OFOR or OFORUNTIL (!Cond).
type RangeStmt struct {
	miniStmt
	Label     *types.Sym
	Vars      Nodes // TODO(rsc): Replace with Key, Value Node
	Def       bool
	X         Node
	Body_     Nodes
	HasBreak_ bool
	typ       *types.Type // TODO(rsc): Remove - use X.Type() instead
}

func NewRangeStmt(pos src.XPos, vars []Node, x Node, body []Node) *RangeStmt {
	n := &RangeStmt{X: x}
	n.pos = pos
	n.op = ORANGE
	n.Vars.Set(vars)
	n.Body_.Set(body)
	return n
}

func (n *RangeStmt) Sym() *types.Sym       { return n.Label }
func (n *RangeStmt) SetSym(x *types.Sym)   { n.Label = x }
func (n *RangeStmt) Right() Node           { return n.X }
func (n *RangeStmt) SetRight(x Node)       { n.X = x }
func (n *RangeStmt) Body() Nodes           { return n.Body_ }
func (n *RangeStmt) PtrBody() *Nodes       { return &n.Body_ }
func (n *RangeStmt) SetBody(x Nodes)       { n.Body_ = x }
func (n *RangeStmt) List() Nodes           { return n.Vars }
func (n *RangeStmt) PtrList() *Nodes       { return &n.Vars }
func (n *RangeStmt) SetList(x Nodes)       { n.Vars = x }
func (n *RangeStmt) HasBreak() bool        { return n.HasBreak_ }
func (n *RangeStmt) SetHasBreak(b bool)    { n.HasBreak_ = b }
func (n *RangeStmt) Colas() bool           { return n.Def }
func (n *RangeStmt) SetColas(b bool)       { n.Def = b }
func (n *RangeStmt) Type() *types.Type     { return n.typ }
func (n *RangeStmt) SetType(x *types.Type) { n.typ = x }

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

func (n *ReturnStmt) Orig() Node      { return n.orig }
func (n *ReturnStmt) SetOrig(x Node)  { n.orig = x }
func (n *ReturnStmt) List() Nodes     { return n.Results }
func (n *ReturnStmt) PtrList() *Nodes { return &n.Results }
func (n *ReturnStmt) SetList(x Nodes) { n.Results = x }
func (n *ReturnStmt) IsDDD() bool     { return false } // typecheckargs asks

// A SelectStmt is a block: { Cases }.
type SelectStmt struct {
	miniStmt
	Label     *types.Sym
	Cases     Nodes
	HasBreak_ bool

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

func (n *SelectStmt) List() Nodes         { return n.Cases }
func (n *SelectStmt) PtrList() *Nodes     { return &n.Cases }
func (n *SelectStmt) SetList(x Nodes)     { n.Cases = x }
func (n *SelectStmt) Sym() *types.Sym     { return n.Label }
func (n *SelectStmt) SetSym(x *types.Sym) { n.Label = x }
func (n *SelectStmt) HasBreak() bool      { return n.HasBreak_ }
func (n *SelectStmt) SetHasBreak(x bool)  { n.HasBreak_ = x }
func (n *SelectStmt) Body() Nodes         { return n.Compiled }
func (n *SelectStmt) PtrBody() *Nodes     { return &n.Compiled }
func (n *SelectStmt) SetBody(x Nodes)     { n.Compiled = x }

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

func (n *SendStmt) Left() Node      { return n.Chan }
func (n *SendStmt) SetLeft(x Node)  { n.Chan = x }
func (n *SendStmt) Right() Node     { return n.Value }
func (n *SendStmt) SetRight(y Node) { n.Value = y }

// A SwitchStmt is a switch statement: switch Init; Expr { Cases }.
type SwitchStmt struct {
	miniStmt
	Tag       Node
	Cases     Nodes // list of *CaseStmt
	Label     *types.Sym
	HasBreak_ bool

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

func (n *SwitchStmt) Left() Node          { return n.Tag }
func (n *SwitchStmt) SetLeft(x Node)      { n.Tag = x }
func (n *SwitchStmt) List() Nodes         { return n.Cases }
func (n *SwitchStmt) PtrList() *Nodes     { return &n.Cases }
func (n *SwitchStmt) SetList(x Nodes)     { n.Cases = x }
func (n *SwitchStmt) Body() Nodes         { return n.Compiled }
func (n *SwitchStmt) PtrBody() *Nodes     { return &n.Compiled }
func (n *SwitchStmt) SetBody(x Nodes)     { n.Compiled = x }
func (n *SwitchStmt) Sym() *types.Sym     { return n.Label }
func (n *SwitchStmt) SetSym(x *types.Sym) { n.Label = x }
func (n *SwitchStmt) HasBreak() bool      { return n.HasBreak_ }
func (n *SwitchStmt) SetHasBreak(x bool)  { n.HasBreak_ = x }

// A TypeSwitchGuard is the [Name :=] X.(type) in a type switch.
type TypeSwitchGuard struct {
	miniNode
	Tag  *Ident
	X    Node
	Used bool
}

func NewTypeSwitchGuard(pos src.XPos, tag *Ident, x Node) *TypeSwitchGuard {
	n := &TypeSwitchGuard{Tag: tag, X: x}
	n.pos = pos
	n.op = OTYPESW
	return n
}

func (n *TypeSwitchGuard) Left() Node {
	if n.Tag == nil {
		return nil
	}
	return n.Tag
}
func (n *TypeSwitchGuard) SetLeft(x Node) {
	if x == nil {
		n.Tag = nil
		return
	}
	n.Tag = x.(*Ident)
}
func (n *TypeSwitchGuard) Right() Node     { return n.X }
func (n *TypeSwitchGuard) SetRight(x Node) { n.X = x }
