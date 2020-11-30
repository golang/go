// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"cmd/compile/internal/base"
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

func toNtype(x Node) Ntype {
	if x == nil {
		return nil
	}
	if _, ok := x.(Ntype); !ok {
		Dump("not Ntype", x)
	}
	return x.(Ntype)
}

// An AddStringExpr is a string concatenation Expr[0] + Exprs[1] + ... + Expr[len(Expr)-1].
type AddStringExpr struct {
	miniExpr
	list Nodes
}

func NewAddStringExpr(pos src.XPos, list []Node) *AddStringExpr {
	n := &AddStringExpr{}
	n.pos = pos
	n.op = OADDSTR
	n.list.Set(list)
	return n
}

func (n *AddStringExpr) String() string                { return fmt.Sprint(n) }
func (n *AddStringExpr) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *AddStringExpr) RawCopy() Node                 { c := *n; return &c }
func (n *AddStringExpr) List() Nodes                   { return n.list }
func (n *AddStringExpr) PtrList() *Nodes               { return &n.list }
func (n *AddStringExpr) SetList(x Nodes)               { n.list = x }

// An AddrExpr is an address-of expression &X.
// It may end up being a normal address-of or an allocation of a composite literal.
type AddrExpr struct {
	miniExpr
	X     Node
	Alloc Node // preallocated storage if any
}

func NewAddrExpr(pos src.XPos, x Node) *AddrExpr {
	n := &AddrExpr{X: x}
	n.op = OADDR
	n.pos = pos
	return n
}

func (n *AddrExpr) String() string                { return fmt.Sprint(n) }
func (n *AddrExpr) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *AddrExpr) RawCopy() Node                 { c := *n; return &c }
func (n *AddrExpr) Left() Node                    { return n.X }
func (n *AddrExpr) SetLeft(x Node)                { n.X = x }
func (n *AddrExpr) Right() Node                   { return n.Alloc }
func (n *AddrExpr) SetRight(x Node)               { n.Alloc = x }

func (n *AddrExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OADDR, OPTRLIT:
		n.op = op
	}
}

// A BinaryExpr is a binary expression X Op Y,
// or Op(X, Y) for builtin functions that do not become calls.
type BinaryExpr struct {
	miniExpr
	X Node
	Y Node
}

func NewBinaryExpr(pos src.XPos, op Op, x, y Node) *BinaryExpr {
	n := &BinaryExpr{X: x, Y: y}
	n.pos = pos
	n.SetOp(op)
	return n
}

func (n *BinaryExpr) String() string                { return fmt.Sprint(n) }
func (n *BinaryExpr) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *BinaryExpr) RawCopy() Node                 { c := *n; return &c }
func (n *BinaryExpr) Left() Node                    { return n.X }
func (n *BinaryExpr) SetLeft(x Node)                { n.X = x }
func (n *BinaryExpr) Right() Node                   { return n.Y }
func (n *BinaryExpr) SetRight(y Node)               { n.Y = y }

func (n *BinaryExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OADD, OADDSTR, OAND, OANDAND, OANDNOT, ODIV, OEQ, OGE, OGT, OLE,
		OLSH, OLT, OMOD, OMUL, ONE, OOR, OOROR, ORSH, OSUB, OXOR,
		OCOPY, OCOMPLEX,
		OEFACE:
		n.op = op
	}
}

// A CallExpr is a function call X(Args).
type CallExpr struct {
	miniExpr
	orig     Node
	X        Node
	Args     Nodes
	Rargs    Nodes // TODO(rsc): Delete.
	body     Nodes // TODO(rsc): Delete.
	DDD      bool
	noInline bool
}

func NewCallExpr(pos src.XPos, fun Node, args []Node) *CallExpr {
	n := &CallExpr{X: fun}
	n.pos = pos
	n.orig = n
	n.op = OCALL
	n.Args.Set(args)
	return n
}

func (n *CallExpr) String() string                { return fmt.Sprint(n) }
func (n *CallExpr) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *CallExpr) RawCopy() Node                 { c := *n; return &c }
func (n *CallExpr) Orig() Node                    { return n.orig }
func (n *CallExpr) SetOrig(x Node)                { n.orig = x }
func (n *CallExpr) Left() Node                    { return n.X }
func (n *CallExpr) SetLeft(x Node)                { n.X = x }
func (n *CallExpr) List() Nodes                   { return n.Args }
func (n *CallExpr) PtrList() *Nodes               { return &n.Args }
func (n *CallExpr) SetList(x Nodes)               { n.Args = x }
func (n *CallExpr) Rlist() Nodes                  { return n.Rargs }
func (n *CallExpr) PtrRlist() *Nodes              { return &n.Rargs }
func (n *CallExpr) SetRlist(x Nodes)              { n.Rargs = x }
func (n *CallExpr) IsDDD() bool                   { return n.DDD }
func (n *CallExpr) SetIsDDD(x bool)               { n.DDD = x }
func (n *CallExpr) NoInline() bool                { return n.noInline }
func (n *CallExpr) SetNoInline(x bool)            { n.noInline = x }
func (n *CallExpr) Body() Nodes                   { return n.body }
func (n *CallExpr) PtrBody() *Nodes               { return &n.body }
func (n *CallExpr) SetBody(x Nodes)               { n.body = x }

func (n *CallExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OCALL, OCALLFUNC, OCALLINTER, OCALLMETH,
		OAPPEND, ODELETE, OGETG, OMAKE, OPRINT, OPRINTN, ORECOVER:
		n.op = op
	}
}

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

// A CompLitExpr is a composite literal Type{Vals}.
// Before type-checking, the type is Ntype.
type CompLitExpr struct {
	miniExpr
	orig  Node
	Ntype Ntype
	list  Nodes // initialized values
}

func NewCompLitExpr(pos src.XPos, typ Ntype, list []Node) *CompLitExpr {
	n := &CompLitExpr{Ntype: typ}
	n.pos = pos
	n.op = OCOMPLIT
	n.list.Set(list)
	n.orig = n
	return n
}

func (n *CompLitExpr) String() string                { return fmt.Sprint(n) }
func (n *CompLitExpr) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *CompLitExpr) RawCopy() Node                 { c := *n; return &c }
func (n *CompLitExpr) Orig() Node                    { return n.orig }
func (n *CompLitExpr) SetOrig(x Node)                { n.orig = x }
func (n *CompLitExpr) Right() Node                   { return n.Ntype }
func (n *CompLitExpr) SetRight(x Node)               { n.Ntype = toNtype(x) }
func (n *CompLitExpr) List() Nodes                   { return n.list }
func (n *CompLitExpr) PtrList() *Nodes               { return &n.list }
func (n *CompLitExpr) SetList(x Nodes)               { n.list = x }

func (n *CompLitExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OARRAYLIT, OCOMPLIT, OMAPLIT, OSTRUCTLIT, OSLICELIT:
		n.op = op
	}
}

// A ConvExpr is a conversion Type(X).
// It may end up being a value or a type.
type ConvExpr struct {
	miniExpr
	orig Node
	X    Node
}

func NewConvExpr(pos src.XPos, op Op, typ *types.Type, x Node) *ConvExpr {
	n := &ConvExpr{X: x}
	n.pos = pos
	n.typ = typ
	n.SetOp(op)
	n.orig = n
	return n
}

func (n *ConvExpr) String() string                { return fmt.Sprint(n) }
func (n *ConvExpr) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *ConvExpr) RawCopy() Node                 { c := *n; return &c }
func (n *ConvExpr) Orig() Node                    { return n.orig }
func (n *ConvExpr) SetOrig(x Node)                { n.orig = x }
func (n *ConvExpr) Left() Node                    { return n.X }
func (n *ConvExpr) SetLeft(x Node)                { n.X = x }

func (n *ConvExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OCONV, OCONVIFACE, OCONVNOP, OBYTES2STR, OBYTES2STRTMP, ORUNES2STR, OSTR2BYTES, OSTR2BYTESTMP, OSTR2RUNES, ORUNESTR:
		n.op = op
	}
}

// An IndexExpr is an index expression X[Y].
type IndexExpr struct {
	miniExpr
	X        Node
	Index    Node
	Assigned bool
}

func NewIndexExpr(pos src.XPos, x, index Node) *IndexExpr {
	n := &IndexExpr{X: x, Index: index}
	n.pos = pos
	n.op = OINDEX
	return n
}

func (n *IndexExpr) String() string                { return fmt.Sprint(n) }
func (n *IndexExpr) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *IndexExpr) RawCopy() Node                 { c := *n; return &c }
func (n *IndexExpr) Left() Node                    { return n.X }
func (n *IndexExpr) SetLeft(x Node)                { n.X = x }
func (n *IndexExpr) Right() Node                   { return n.Index }
func (n *IndexExpr) SetRight(y Node)               { n.Index = y }
func (n *IndexExpr) IndexMapLValue() bool          { return n.Assigned }
func (n *IndexExpr) SetIndexMapLValue(x bool)      { n.Assigned = x }

func (n *IndexExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OINDEX, OINDEXMAP:
		n.op = op
	}
}

// A KeyExpr is an X:Y composite literal key.
// After type-checking, a key for a struct sets Sym to the field.
type KeyExpr struct {
	miniExpr
	Key    Node
	sym    *types.Sym
	Value  Node
	offset int64
}

func NewKeyExpr(pos src.XPos, key, value Node) *KeyExpr {
	n := &KeyExpr{Key: key, Value: value}
	n.pos = pos
	n.op = OKEY
	n.offset = types.BADWIDTH
	return n
}

func (n *KeyExpr) String() string                { return fmt.Sprint(n) }
func (n *KeyExpr) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *KeyExpr) RawCopy() Node                 { c := *n; return &c }
func (n *KeyExpr) Left() Node                    { return n.Key }
func (n *KeyExpr) SetLeft(x Node)                { n.Key = x }
func (n *KeyExpr) Right() Node                   { return n.Value }
func (n *KeyExpr) SetRight(y Node)               { n.Value = y }
func (n *KeyExpr) Sym() *types.Sym               { return n.sym }
func (n *KeyExpr) SetSym(x *types.Sym)           { n.sym = x }
func (n *KeyExpr) Offset() int64                 { return n.offset }
func (n *KeyExpr) SetOffset(x int64)             { n.offset = x }

func (n *KeyExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OKEY, OSTRUCTKEY:
		n.op = op
	}
}

// An InlinedCallExpr is an inlined function call.
type InlinedCallExpr struct {
	miniExpr
	body       Nodes
	ReturnVars Nodes
}

func NewInlinedCallExpr(pos src.XPos, body, retvars []Node) *InlinedCallExpr {
	n := &InlinedCallExpr{}
	n.pos = pos
	n.op = OINLCALL
	n.body.Set(body)
	n.ReturnVars.Set(retvars)
	return n
}

func (n *InlinedCallExpr) String() string                { return fmt.Sprint(n) }
func (n *InlinedCallExpr) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *InlinedCallExpr) RawCopy() Node                 { c := *n; return &c }
func (n *InlinedCallExpr) Body() Nodes                   { return n.body }
func (n *InlinedCallExpr) PtrBody() *Nodes               { return &n.body }
func (n *InlinedCallExpr) SetBody(x Nodes)               { n.body = x }
func (n *InlinedCallExpr) Rlist() Nodes                  { return n.ReturnVars }
func (n *InlinedCallExpr) PtrRlist() *Nodes              { return &n.ReturnVars }
func (n *InlinedCallExpr) SetRlist(x Nodes)              { n.ReturnVars = x }

// A MakeExpr is a make expression: make(Type[, Len[, Cap]]).
// Op is OMAKECHAN, OMAKEMAP, OMAKESLICE, or OMAKESLICECOPY,
// but *not* OMAKE (that's a pre-typechecking CallExpr).
type MakeExpr struct {
	miniExpr
	Len Node
	Cap Node
}

func NewMakeExpr(pos src.XPos, op Op, len, cap Node) *MakeExpr {
	n := &MakeExpr{Len: len, Cap: cap}
	n.pos = pos
	n.SetOp(op)
	return n
}

func (n *MakeExpr) String() string                { return fmt.Sprint(n) }
func (n *MakeExpr) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *MakeExpr) RawCopy() Node                 { c := *n; return &c }
func (n *MakeExpr) Left() Node                    { return n.Len }
func (n *MakeExpr) SetLeft(x Node)                { n.Len = x }
func (n *MakeExpr) Right() Node                   { return n.Cap }
func (n *MakeExpr) SetRight(x Node)               { n.Cap = x }

func (n *MakeExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OMAKECHAN, OMAKEMAP, OMAKESLICE, OMAKESLICECOPY:
		n.op = op
	}
}

// A MethodExpr is a method expression X.M (where X is an expression, not a type).
type MethodExpr struct {
	miniExpr
	X      Node
	M      Node
	sym    *types.Sym
	offset int64
	class  Class
}

func NewMethodExpr(pos src.XPos, op Op, x, m Node) *MethodExpr {
	n := &MethodExpr{X: x, M: m}
	n.pos = pos
	n.op = OMETHEXPR
	n.offset = types.BADWIDTH
	return n
}

func (n *MethodExpr) String() string                { return fmt.Sprint(n) }
func (n *MethodExpr) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *MethodExpr) RawCopy() Node                 { c := *n; return &c }
func (n *MethodExpr) Left() Node                    { return n.X }
func (n *MethodExpr) SetLeft(x Node)                { n.X = x }
func (n *MethodExpr) Right() Node                   { return n.M }
func (n *MethodExpr) SetRight(y Node)               { n.M = y }
func (n *MethodExpr) Sym() *types.Sym               { return n.sym }
func (n *MethodExpr) SetSym(x *types.Sym)           { n.sym = x }
func (n *MethodExpr) Offset() int64                 { return n.offset }
func (n *MethodExpr) SetOffset(x int64)             { n.offset = x }
func (n *MethodExpr) Class() Class                  { return n.class }
func (n *MethodExpr) SetClass(x Class)              { n.class = x }

// A NilExpr represents the predefined untyped constant nil.
// (It may be copied and assigned a type, though.)
type NilExpr struct {
	miniExpr
	sym *types.Sym // TODO: Remove
}

func NewNilExpr(pos src.XPos) *NilExpr {
	n := &NilExpr{}
	n.pos = pos
	n.op = ONIL
	return n
}

func (n *NilExpr) String() string                { return fmt.Sprint(n) }
func (n *NilExpr) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *NilExpr) RawCopy() Node                 { c := *n; return &c }
func (n *NilExpr) Sym() *types.Sym               { return n.sym }
func (n *NilExpr) SetSym(x *types.Sym)           { n.sym = x }

// A ParenExpr is a parenthesized expression (X).
// It may end up being a value or a type.
type ParenExpr struct {
	miniExpr
	X Node
}

func NewParenExpr(pos src.XPos, x Node) *ParenExpr {
	n := &ParenExpr{X: x}
	n.op = OPAREN
	n.pos = pos
	return n
}

func (n *ParenExpr) String() string                { return fmt.Sprint(n) }
func (n *ParenExpr) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *ParenExpr) RawCopy() Node                 { c := *n; return &c }
func (n *ParenExpr) Left() Node                    { return n.X }
func (n *ParenExpr) SetLeft(x Node)                { n.X = x }

func (*ParenExpr) CanBeNtype() {}

// SetOTYPE changes n to be an OTYPE node returning t,
// like all the type nodes in type.go.
func (n *ParenExpr) SetOTYPE(t *types.Type) {
	n.op = OTYPE
	n.typ = t
	if t.Nod == nil {
		t.Nod = n
	}
}

// A ResultExpr represents a direct access to a result slot on the stack frame.
type ResultExpr struct {
	miniExpr
	offset int64
}

func NewResultExpr(pos src.XPos, typ *types.Type, offset int64) *ResultExpr {
	n := &ResultExpr{offset: offset}
	n.pos = pos
	n.op = ORESULT
	n.typ = typ
	return n
}

func (n *ResultExpr) String() string                { return fmt.Sprint(n) }
func (n *ResultExpr) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *ResultExpr) RawCopy() Node                 { c := *n; return &c }
func (n *ResultExpr) Offset() int64                 { return n.offset }
func (n *ResultExpr) SetOffset(x int64)             { n.offset = x }

// A SelectorExpr is a selector expression X.Sym.
type SelectorExpr struct {
	miniExpr
	X      Node
	Sel    *types.Sym
	offset int64
}

func NewSelectorExpr(pos src.XPos, x Node, sel *types.Sym) *SelectorExpr {
	n := &SelectorExpr{X: x, Sel: sel}
	n.pos = pos
	n.op = OXDOT
	n.offset = types.BADWIDTH
	return n
}

func (n *SelectorExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case ODOT, ODOTPTR, ODOTMETH, ODOTINTER, OXDOT:
		n.op = op
	}
}

func (n *SelectorExpr) String() string                { return fmt.Sprint(n) }
func (n *SelectorExpr) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *SelectorExpr) RawCopy() Node                 { c := *n; return &c }
func (n *SelectorExpr) Left() Node                    { return n.X }
func (n *SelectorExpr) SetLeft(x Node)                { n.X = x }
func (n *SelectorExpr) Sym() *types.Sym               { return n.Sel }
func (n *SelectorExpr) SetSym(x *types.Sym)           { n.Sel = x }
func (n *SelectorExpr) Offset() int64                 { return n.offset }
func (n *SelectorExpr) SetOffset(x int64)             { n.offset = x }

// Before type-checking, bytes.Buffer is a SelectorExpr.
// After type-checking it becomes a Name.
func (*SelectorExpr) CanBeNtype() {}

// A SliceExpr is a slice expression X[Low:High] or X[Low:High:Max].
type SliceExpr struct {
	miniExpr
	X    Node
	list Nodes // TODO(rsc): Use separate Nodes
}

func NewSliceExpr(pos src.XPos, op Op, x Node) *SliceExpr {
	n := &SliceExpr{X: x}
	n.pos = pos
	n.op = op
	return n
}

func (n *SliceExpr) String() string                { return fmt.Sprint(n) }
func (n *SliceExpr) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *SliceExpr) RawCopy() Node                 { c := *n; return &c }
func (n *SliceExpr) Left() Node                    { return n.X }
func (n *SliceExpr) SetLeft(x Node)                { n.X = x }
func (n *SliceExpr) List() Nodes                   { return n.list }
func (n *SliceExpr) PtrList() *Nodes               { return &n.list }
func (n *SliceExpr) SetList(x Nodes)               { n.list = x }

func (n *SliceExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OSLICE, OSLICEARR, OSLICESTR, OSLICE3, OSLICE3ARR:
		n.op = op
	}
}

// SliceBounds returns n's slice bounds: low, high, and max in expr[low:high:max].
// n must be a slice expression. max is nil if n is a simple slice expression.
func (n *SliceExpr) SliceBounds() (low, high, max Node) {
	if n.list.Len() == 0 {
		return nil, nil, nil
	}

	switch n.Op() {
	case OSLICE, OSLICEARR, OSLICESTR:
		s := n.list.Slice()
		return s[0], s[1], nil
	case OSLICE3, OSLICE3ARR:
		s := n.list.Slice()
		return s[0], s[1], s[2]
	}
	base.Fatalf("SliceBounds op %v: %v", n.Op(), n)
	return nil, nil, nil
}

// SetSliceBounds sets n's slice bounds, where n is a slice expression.
// n must be a slice expression. If max is non-nil, n must be a full slice expression.
func (n *SliceExpr) SetSliceBounds(low, high, max Node) {
	switch n.Op() {
	case OSLICE, OSLICEARR, OSLICESTR:
		if max != nil {
			base.Fatalf("SetSliceBounds %v given three bounds", n.Op())
		}
		s := n.list.Slice()
		if s == nil {
			if low == nil && high == nil {
				return
			}
			n.list.Set2(low, high)
			return
		}
		s[0] = low
		s[1] = high
		return
	case OSLICE3, OSLICE3ARR:
		s := n.list.Slice()
		if s == nil {
			if low == nil && high == nil && max == nil {
				return
			}
			n.list.Set3(low, high, max)
			return
		}
		s[0] = low
		s[1] = high
		s[2] = max
		return
	}
	base.Fatalf("SetSliceBounds op %v: %v", n.Op(), n)
}

// IsSlice3 reports whether o is a slice3 op (OSLICE3, OSLICE3ARR).
// o must be a slicing op.
func (o Op) IsSlice3() bool {
	switch o {
	case OSLICE, OSLICEARR, OSLICESTR:
		return false
	case OSLICE3, OSLICE3ARR:
		return true
	}
	base.Fatalf("IsSlice3 op %v", o)
	return false
}

// A SliceHeader expression constructs a slice header from its parts.
type SliceHeaderExpr struct {
	miniExpr
	Ptr    Node
	lenCap Nodes // TODO(rsc): Split into two Node fields
}

func NewSliceHeaderExpr(pos src.XPos, typ *types.Type, ptr, len, cap Node) *SliceHeaderExpr {
	n := &SliceHeaderExpr{Ptr: ptr}
	n.pos = pos
	n.op = OSLICEHEADER
	n.typ = typ
	n.lenCap.Set2(len, cap)
	return n
}

func (n *SliceHeaderExpr) String() string                { return fmt.Sprint(n) }
func (n *SliceHeaderExpr) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *SliceHeaderExpr) RawCopy() Node                 { c := *n; return &c }
func (n *SliceHeaderExpr) Left() Node                    { return n.Ptr }
func (n *SliceHeaderExpr) SetLeft(x Node)                { n.Ptr = x }
func (n *SliceHeaderExpr) List() Nodes                   { return n.lenCap }
func (n *SliceHeaderExpr) PtrList() *Nodes               { return &n.lenCap }
func (n *SliceHeaderExpr) SetList(x Nodes)               { n.lenCap = x }

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

// A TypeAssertionExpr is a selector expression X.(Type).
// Before type-checking, the type is Ntype.
type TypeAssertExpr struct {
	miniExpr
	X     Node
	Ntype Node  // TODO: Should be Ntype, but reused as address of type structure
	Itab  Nodes // Itab[0] is itab
}

func NewTypeAssertExpr(pos src.XPos, x Node, typ Ntype) *TypeAssertExpr {
	n := &TypeAssertExpr{X: x, Ntype: typ}
	n.pos = pos
	n.op = ODOTTYPE
	return n
}

func (n *TypeAssertExpr) String() string                { return fmt.Sprint(n) }
func (n *TypeAssertExpr) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *TypeAssertExpr) RawCopy() Node                 { c := *n; return &c }
func (n *TypeAssertExpr) Left() Node                    { return n.X }
func (n *TypeAssertExpr) SetLeft(x Node)                { n.X = x }
func (n *TypeAssertExpr) Right() Node                   { return n.Ntype }
func (n *TypeAssertExpr) SetRight(x Node)               { n.Ntype = x } // TODO: toNtype(x)
func (n *TypeAssertExpr) List() Nodes                   { return n.Itab }
func (n *TypeAssertExpr) PtrList() *Nodes               { return &n.Itab }
func (n *TypeAssertExpr) SetList(x Nodes)               { n.Itab = x }

func (n *TypeAssertExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case ODOTTYPE, ODOTTYPE2:
		n.op = op
	}
}

// A UnaryExpr is a unary expression Op X,
// or Op(X) for a builtin function that does not end up being a call.
type UnaryExpr struct {
	miniExpr
	X Node
}

func NewUnaryExpr(pos src.XPos, op Op, x Node) *UnaryExpr {
	n := &UnaryExpr{X: x}
	n.pos = pos
	n.SetOp(op)
	return n
}

func (n *UnaryExpr) String() string                { return fmt.Sprint(n) }
func (n *UnaryExpr) Format(s fmt.State, verb rune) { FmtNode(n, s, verb) }
func (n *UnaryExpr) RawCopy() Node                 { c := *n; return &c }
func (n *UnaryExpr) Left() Node                    { return n.X }
func (n *UnaryExpr) SetLeft(x Node)                { n.X = x }

func (n *UnaryExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OBITNOT, ONEG, ONOT, OPLUS, ORECV,
		OALIGNOF, OCAP, OCLOSE, OIMAG, OLEN, ONEW,
		OOFFSETOF, OPANIC, OREAL, OSIZEOF,
		OCHECKNIL, OCFUNC, OIDATA, OITAB, ONEWOBJ, OSPTR, OVARDEF, OVARKILL, OVARLIVE:
		n.op = op
	}
}
