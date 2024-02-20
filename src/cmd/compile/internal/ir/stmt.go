// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
	"go/constant"
)

// A Decl is a declaration of a const, type, or var. (A declared func is a Func.)
type Decl struct {
	miniNode
	X *Name // the thing being declared
}

func NewDecl(pos src.XPos, op Op, x *Name) *Decl {
	n := &Decl{X: x}
	n.pos = pos
	switch op {
	default:
		panic("invalid Decl op " + op.String())
	case ODCL:
		n.op = op
	}
	return n
}

func (*Decl) isStmt() {}

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

func (n *miniStmt) Init() Nodes     { return n.init }
func (n *miniStmt) SetInit(x Nodes) { n.init = x }
func (n *miniStmt) PtrInit() *Nodes { return &n.init }

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
	n.Lhs = lhs
	n.Rhs = rhs
	return n
}

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

func (n *AssignStmt) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OAS:
		n.op = op
	}
}

// An AssignOpStmt is an AsOp= assignment statement: X AsOp= Y.
type AssignOpStmt struct {
	miniStmt
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

// A BlockStmt is a block: { List }.
type BlockStmt struct {
	miniStmt
	List Nodes
}

func NewBlockStmt(pos src.XPos, list []Node) *BlockStmt {
	n := &BlockStmt{}
	n.pos = pos
	if !pos.IsKnown() {
		n.pos = base.Pos
		if len(list) > 0 {
			n.pos = list[0].Pos()
		}
	}
	n.op = OBLOCK
	n.List = list
	return n
}

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

func (n *BranchStmt) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OBREAK, OCONTINUE, OFALL, OGOTO:
		n.op = op
	}
}

func (n *BranchStmt) Sym() *types.Sym { return n.Label }

// A CaseClause is a case statement in a switch or select: case List: Body.
type CaseClause struct {
	miniStmt
	Var  *Name // declared variable for this case in type switch
	List Nodes // list of expressions for switch, early select

	// RTypes is a list of RType expressions, which are copied to the
	// corresponding OEQ nodes that are emitted when switch statements
	// are desugared. RTypes[i] must be non-nil if the emitted
	// comparison for List[i] will be a mixed interface/concrete
	// comparison; see reflectdata.CompareRType for details.
	//
	// Because mixed interface/concrete switch cases are rare, we allow
	// len(RTypes) < len(List). Missing entries are implicitly nil.
	RTypes Nodes

	Body Nodes
}

func NewCaseStmt(pos src.XPos, list, body []Node) *CaseClause {
	n := &CaseClause{List: list, Body: body}
	n.pos = pos
	n.op = OCASE
	return n
}

type CommClause struct {
	miniStmt
	Comm Node // communication case
	Body Nodes
}

func NewCommStmt(pos src.XPos, comm Node, body []Node) *CommClause {
	n := &CommClause{Comm: comm, Body: body}
	n.pos = pos
	n.op = OCASE
	return n
}

// A ForStmt is a non-range for loop: for Init; Cond; Post { Body }
type ForStmt struct {
	miniStmt
	Label        *types.Sym
	Cond         Node
	Post         Node
	Body         Nodes
	DistinctVars bool
}

func NewForStmt(pos src.XPos, init Node, cond, post Node, body []Node, distinctVars bool) *ForStmt {
	n := &ForStmt{Cond: cond, Post: post}
	n.pos = pos
	n.op = OFOR
	if init != nil {
		n.init = []Node{init}
	}
	n.Body = body
	n.DistinctVars = distinctVars
	return n
}

type FourStmt struct {
	miniStmt
	Label *types.Sym
	Cond  Node
	Post  Node
	Body  Nodes
}

func NewFourStmt(pos src.XPos, init Node, cond, post Node, body []Node) *FourStmt {
	n := &FourStmt{Cond: cond, Post: post}
	n.pos = pos
	n.op = OFOUR
	if init != nil {
		n.init = []Node{init}
	}
	n.Body = body
	return n
}

type UnlessStmt struct {
	miniStmt
	Cond   Node
	Body   Nodes
	Likely bool // code layout hint
}

func NewUnlessStmt(pos src.XPos, cond Node, body []Node) *UnlessStmt {
	n := &UnlessStmt{Cond: cond}
	n.pos = pos
	n.op = OUNLESS
	n.Body = body
	return n
}

// A GoDeferStmt is a go or defer statement: go Call / defer Call.
//
// The two opcodes use a single syntax because the implementations
// are very similar: both are concerned with saving Call and running it
// in a different context (a separate goroutine or a later time).
type GoDeferStmt struct {
	miniStmt
	Call    Node
	DeferAt Expr
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

// An IfStmt is a return statement: if Init; Cond { Body } else { Else }.
type IfStmt struct {
	miniStmt
	Cond   Node
	Body   Nodes
	Else   Nodes
	Likely bool // code layout hint
}

func NewIfStmt(pos src.XPos, cond Node, body, els []Node) *IfStmt {
	n := &IfStmt{Cond: cond}
	n.pos = pos
	n.op = OIF
	n.Body = body
	n.Else = els
	return n
}

// A JumpTableStmt is used to implement switches. Its semantics are:
//
//	tmp := jt.Idx
//	if tmp == Cases[0] goto Targets[0]
//	if tmp == Cases[1] goto Targets[1]
//	...
//	if tmp == Cases[n] goto Targets[n]
//
// Note that a JumpTableStmt is more like a multiway-goto than
// a multiway-if. In particular, the case bodies are just
// labels to jump to, not full Nodes lists.
type JumpTableStmt struct {
	miniStmt

	// Value used to index the jump table.
	// We support only integer types that
	// are at most the size of a uintptr.
	Idx Node

	// If Idx is equal to Cases[i], jump to Targets[i].
	// Cases entries must be distinct and in increasing order.
	// The length of Cases and Targets must be equal.
	Cases   []constant.Value
	Targets []*types.Sym
}

func NewJumpTableStmt(pos src.XPos, idx Node) *JumpTableStmt {
	n := &JumpTableStmt{Idx: idx}
	n.pos = pos
	n.op = OJUMPTABLE
	return n
}

// An InterfaceSwitchStmt is used to implement type switches.
// Its semantics are:
//
//	if RuntimeType implements Descriptor.Cases[0] {
//	    Case, Itab = 0, itab<RuntimeType, Descriptor.Cases[0]>
//	} else if RuntimeType implements Descriptor.Cases[1] {
//	    Case, Itab = 1, itab<RuntimeType, Descriptor.Cases[1]>
//	...
//	} else if RuntimeType implements Descriptor.Cases[N-1] {
//	    Case, Itab = N-1, itab<RuntimeType, Descriptor.Cases[N-1]>
//	} else {
//	    Case, Itab = len(cases), nil
//	}
//
// RuntimeType must be a non-nil *runtime._type.
// Hash must be the hash field of RuntimeType (or its copy loaded from an itab).
// Descriptor must represent an abi.InterfaceSwitch global variable.
type InterfaceSwitchStmt struct {
	miniStmt

	Case        Node
	Itab        Node
	RuntimeType Node
	Hash        Node
	Descriptor  *obj.LSym
}

func NewInterfaceSwitchStmt(pos src.XPos, case_, itab, runtimeType, hash Node, descriptor *obj.LSym) *InterfaceSwitchStmt {
	n := &InterfaceSwitchStmt{
		Case:        case_,
		Itab:        itab,
		RuntimeType: runtimeType,
		Hash:        hash,
		Descriptor:  descriptor,
	}
	n.pos = pos
	n.op = OINTERFACESWITCH
	return n
}

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

func (n *LabelStmt) Sym() *types.Sym { return n.Label }

// A RangeStmt is a range loop: for Key, Value = range X { Body }
type RangeStmt struct {
	miniStmt
	Label        *types.Sym
	Def          bool
	X            Node
	RType        Node `mknode:"-"` // see reflectdata/helpers.go
	Key          Node
	Value        Node
	Body         Nodes
	DistinctVars bool
	Prealloc     *Name

	// When desugaring the RangeStmt during walk, the assignments to Key
	// and Value may require OCONVIFACE operations. If so, these fields
	// will be copied to their respective ConvExpr fields.
	KeyTypeWord   Node `mknode:"-"`
	KeySrcRType   Node `mknode:"-"`
	ValueTypeWord Node `mknode:"-"`
	ValueSrcRType Node `mknode:"-"`
}

func NewRangeStmt(pos src.XPos, key, value, x Node, body []Node, distinctVars bool) *RangeStmt {
	n := &RangeStmt{X: x, Key: key, Value: value}
	n.pos = pos
	n.op = ORANGE
	n.Body = body
	n.DistinctVars = distinctVars
	return n
}

// A ReturnStmt is a return statement.
type ReturnStmt struct {
	miniStmt
	Results Nodes // return list
}

func NewReturnStmt(pos src.XPos, results []Node) *ReturnStmt {
	n := &ReturnStmt{}
	n.pos = pos
	n.op = ORETURN
	n.Results = results
	return n
}

// A SelectStmt is a block: { Cases }.
type SelectStmt struct {
	miniStmt
	Label *types.Sym
	Cases []*CommClause

	// TODO(rsc): Instead of recording here, replace with a block?
	Compiled Nodes // compiled form, after walkSelect
}

func NewSelectStmt(pos src.XPos, cases []*CommClause) *SelectStmt {
	n := &SelectStmt{Cases: cases}
	n.pos = pos
	n.op = OSELECT
	return n
}

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

// A SwitchStmt is a switch statement: switch Init; Tag { Cases }.
type SwitchStmt struct {
	miniStmt
	Tag   Node
	Cases []*CaseClause
	Label *types.Sym

	// TODO(rsc): Instead of recording here, replace with a block?
	Compiled Nodes // compiled form, after walkSwitch
}

func NewSwitchStmt(pos src.XPos, tag Node, cases []*CaseClause) *SwitchStmt {
	n := &SwitchStmt{Tag: tag, Cases: cases}
	n.pos = pos
	n.op = OSWITCH
	return n
}

// A TailCallStmt is a tail call statement, which is used for back-end
// code generation to jump directly to another function entirely.
type TailCallStmt struct {
	miniStmt
	Call *CallExpr // the underlying call
}

func NewTailCallStmt(pos src.XPos, call *CallExpr) *TailCallStmt {
	n := &TailCallStmt{Call: call}
	n.pos = pos
	n.op = OTAILCALL
	return n
}

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
