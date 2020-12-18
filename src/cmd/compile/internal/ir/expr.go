// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"go/constant"
)

func maybeDo(x Node, err error, do func(Node) error) error {
	if x != nil && err == nil {
		err = do(x)
	}
	return err
}

func maybeDoList(x Nodes, err error, do func(Node) error) error {
	if err == nil {
		err = DoList(x, do)
	}
	return err
}

func maybeEdit(x Node, edit func(Node) Node) Node {
	if x == nil {
		return x
	}
	return edit(x)
}

// An Expr is a Node that can appear as an expression.
type Expr interface {
	Node
	isExpr()
}

// A miniExpr is a miniNode with extra fields common to expressions.
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
	miniExprNonNil
	miniExprTransient
	miniExprBounded
	miniExprImplicit // for use by implementations; not supported by every Expr
)

func (*miniExpr) isExpr() {}

func (n *miniExpr) Type() *types.Type     { return n.typ }
func (n *miniExpr) SetType(x *types.Type) { n.typ = x }
func (n *miniExpr) Opt() interface{}      { return n.opt }
func (n *miniExpr) SetOpt(x interface{})  { n.opt = x }
func (n *miniExpr) HasCall() bool         { return n.flags&miniExprHasCall != 0 }
func (n *miniExpr) SetHasCall(b bool)     { n.flags.set(miniExprHasCall, b) }
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
	List_    Nodes
	Prealloc *Name
}

func NewAddStringExpr(pos src.XPos, list []Node) *AddStringExpr {
	n := &AddStringExpr{}
	n.pos = pos
	n.op = OADDSTR
	n.List_.Set(list)
	return n
}

func (n *AddStringExpr) List() Nodes     { return n.List_ }
func (n *AddStringExpr) PtrList() *Nodes { return &n.List_ }
func (n *AddStringExpr) SetList(x Nodes) { n.List_ = x }

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

func (n *AddrExpr) Left() Node         { return n.X }
func (n *AddrExpr) SetLeft(x Node)     { n.X = x }
func (n *AddrExpr) Right() Node        { return n.Alloc }
func (n *AddrExpr) SetRight(x Node)    { n.Alloc = x }
func (n *AddrExpr) Implicit() bool     { return n.flags&miniExprImplicit != 0 }
func (n *AddrExpr) SetImplicit(b bool) { n.flags.set(miniExprImplicit, b) }

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

func (n *BinaryExpr) Left() Node      { return n.X }
func (n *BinaryExpr) SetLeft(x Node)  { n.X = x }
func (n *BinaryExpr) Right() Node     { return n.Y }
func (n *BinaryExpr) SetRight(y Node) { n.Y = y }

func (n *BinaryExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OADD, OADDSTR, OAND, OANDNOT, ODIV, OEQ, OGE, OGT, OLE,
		OLSH, OLT, OMOD, OMUL, ONE, OOR, ORSH, OSUB, OXOR,
		OCOPY, OCOMPLEX,
		OEFACE:
		n.op = op
	}
}

// A CallUse records how the result of the call is used:
type CallUse int

const (
	_ CallUse = iota

	CallUseExpr // single expression result is used
	CallUseList // list of results are used
	CallUseStmt // results not used - call is a statement
)

// A CallExpr is a function call X(Args).
type CallExpr struct {
	miniExpr
	orig      Node
	X         Node
	Args      Nodes
	Rargs     Nodes // TODO(rsc): Delete.
	Body_     Nodes // TODO(rsc): Delete.
	DDD       bool
	Use       CallUse
	NoInline_ bool
}

func NewCallExpr(pos src.XPos, op Op, fun Node, args []Node) *CallExpr {
	n := &CallExpr{X: fun}
	n.pos = pos
	n.orig = n
	n.SetOp(op)
	n.Args.Set(args)
	return n
}

func (*CallExpr) isStmt() {}

func (n *CallExpr) Orig() Node         { return n.orig }
func (n *CallExpr) SetOrig(x Node)     { n.orig = x }
func (n *CallExpr) Left() Node         { return n.X }
func (n *CallExpr) SetLeft(x Node)     { n.X = x }
func (n *CallExpr) List() Nodes        { return n.Args }
func (n *CallExpr) PtrList() *Nodes    { return &n.Args }
func (n *CallExpr) SetList(x Nodes)    { n.Args = x }
func (n *CallExpr) Rlist() Nodes       { return n.Rargs }
func (n *CallExpr) PtrRlist() *Nodes   { return &n.Rargs }
func (n *CallExpr) SetRlist(x Nodes)   { n.Rargs = x }
func (n *CallExpr) IsDDD() bool        { return n.DDD }
func (n *CallExpr) SetIsDDD(x bool)    { n.DDD = x }
func (n *CallExpr) NoInline() bool     { return n.NoInline_ }
func (n *CallExpr) SetNoInline(x bool) { n.NoInline_ = x }
func (n *CallExpr) Body() Nodes        { return n.Body_ }
func (n *CallExpr) PtrBody() *Nodes    { return &n.Body_ }
func (n *CallExpr) SetBody(x Nodes)    { n.Body_ = x }

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
	Func_    *Func
	X        Node
	Method   *types.Field
	Prealloc *Name
}

func NewCallPartExpr(pos src.XPos, x Node, method *types.Field, fn *Func) *CallPartExpr {
	n := &CallPartExpr{Func_: fn, X: x, Method: method}
	n.op = OCALLPART
	n.pos = pos
	n.typ = fn.Type()
	n.Func_ = fn
	return n
}

func (n *CallPartExpr) Func() *Func     { return n.Func_ }
func (n *CallPartExpr) Left() Node      { return n.X }
func (n *CallPartExpr) Sym() *types.Sym { return n.Method.Sym }
func (n *CallPartExpr) SetLeft(x Node)  { n.X = x }

// A ClosureExpr is a function literal expression.
type ClosureExpr struct {
	miniExpr
	Func_    *Func
	Prealloc *Name
}

func NewClosureExpr(pos src.XPos, fn *Func) *ClosureExpr {
	n := &ClosureExpr{Func_: fn}
	n.op = OCLOSURE
	n.pos = pos
	return n
}

func (n *ClosureExpr) Func() *Func { return n.Func_ }

// A ClosureRead denotes reading a variable stored within a closure struct.
type ClosureReadExpr struct {
	miniExpr
	Offset_ int64
}

func NewClosureRead(typ *types.Type, offset int64) *ClosureReadExpr {
	n := &ClosureReadExpr{Offset_: offset}
	n.typ = typ
	n.op = OCLOSUREREAD
	return n
}

func (n *ClosureReadExpr) Type() *types.Type { return n.typ }
func (n *ClosureReadExpr) Offset() int64     { return n.Offset_ }

// A CompLitExpr is a composite literal Type{Vals}.
// Before type-checking, the type is Ntype.
type CompLitExpr struct {
	miniExpr
	orig     Node
	Ntype    Ntype
	List_    Nodes // initialized values
	Prealloc *Name
	Len      int64 // backing array length for OSLICELIT
}

func NewCompLitExpr(pos src.XPos, op Op, typ Ntype, list []Node) *CompLitExpr {
	n := &CompLitExpr{Ntype: typ}
	n.pos = pos
	n.SetOp(op)
	n.List_.Set(list)
	n.orig = n
	return n
}

func (n *CompLitExpr) Orig() Node         { return n.orig }
func (n *CompLitExpr) SetOrig(x Node)     { n.orig = x }
func (n *CompLitExpr) Right() Node        { return n.Ntype }
func (n *CompLitExpr) SetRight(x Node)    { n.Ntype = toNtype(x) }
func (n *CompLitExpr) List() Nodes        { return n.List_ }
func (n *CompLitExpr) PtrList() *Nodes    { return &n.List_ }
func (n *CompLitExpr) SetList(x Nodes)    { n.List_ = x }
func (n *CompLitExpr) Implicit() bool     { return n.flags&miniExprImplicit != 0 }
func (n *CompLitExpr) SetImplicit(b bool) { n.flags.set(miniExprImplicit, b) }

func (n *CompLitExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OARRAYLIT, OCOMPLIT, OMAPLIT, OSTRUCTLIT, OSLICELIT:
		n.op = op
	}
}

type ConstExpr struct {
	miniExpr
	val  constant.Value
	orig Node
}

func NewConstExpr(val constant.Value, orig Node) Node {
	n := &ConstExpr{orig: orig, val: val}
	n.op = OLITERAL
	n.pos = orig.Pos()
	n.SetType(orig.Type())
	n.SetTypecheck(orig.Typecheck())
	n.SetDiag(orig.Diag())
	return n
}

func (n *ConstExpr) Sym() *types.Sym     { return n.orig.Sym() }
func (n *ConstExpr) Orig() Node          { return n.orig }
func (n *ConstExpr) SetOrig(orig Node)   { panic(n.no("SetOrig")) }
func (n *ConstExpr) Val() constant.Value { return n.val }

// A ConvExpr is a conversion Type(X).
// It may end up being a value or a type.
type ConvExpr struct {
	miniExpr
	X Node
}

func NewConvExpr(pos src.XPos, op Op, typ *types.Type, x Node) *ConvExpr {
	n := &ConvExpr{X: x}
	n.pos = pos
	n.typ = typ
	n.SetOp(op)
	return n
}

func (n *ConvExpr) Left() Node         { return n.X }
func (n *ConvExpr) SetLeft(x Node)     { n.X = x }
func (n *ConvExpr) Implicit() bool     { return n.flags&miniExprImplicit != 0 }
func (n *ConvExpr) SetImplicit(b bool) { n.flags.set(miniExprImplicit, b) }

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

func (n *IndexExpr) Left() Node               { return n.X }
func (n *IndexExpr) SetLeft(x Node)           { n.X = x }
func (n *IndexExpr) Right() Node              { return n.Index }
func (n *IndexExpr) SetRight(y Node)          { n.Index = y }
func (n *IndexExpr) IndexMapLValue() bool     { return n.Assigned }
func (n *IndexExpr) SetIndexMapLValue(x bool) { n.Assigned = x }

func (n *IndexExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OINDEX, OINDEXMAP:
		n.op = op
	}
}

// A KeyExpr is a Key: Value composite literal key.
type KeyExpr struct {
	miniExpr
	Key   Node
	Value Node
}

func NewKeyExpr(pos src.XPos, key, value Node) *KeyExpr {
	n := &KeyExpr{Key: key, Value: value}
	n.pos = pos
	n.op = OKEY
	return n
}

func (n *KeyExpr) Left() Node      { return n.Key }
func (n *KeyExpr) SetLeft(x Node)  { n.Key = x }
func (n *KeyExpr) Right() Node     { return n.Value }
func (n *KeyExpr) SetRight(y Node) { n.Value = y }

// A StructKeyExpr is an Field: Value composite literal key.
type StructKeyExpr struct {
	miniExpr
	Field   *types.Sym
	Value   Node
	Offset_ int64
}

func NewStructKeyExpr(pos src.XPos, field *types.Sym, value Node) *StructKeyExpr {
	n := &StructKeyExpr{Field: field, Value: value}
	n.pos = pos
	n.op = OSTRUCTKEY
	n.Offset_ = types.BADWIDTH
	return n
}

func (n *StructKeyExpr) Sym() *types.Sym     { return n.Field }
func (n *StructKeyExpr) SetSym(x *types.Sym) { n.Field = x }
func (n *StructKeyExpr) Left() Node          { return n.Value }
func (n *StructKeyExpr) SetLeft(x Node)      { n.Value = x }
func (n *StructKeyExpr) Offset() int64       { return n.Offset_ }
func (n *StructKeyExpr) SetOffset(x int64)   { n.Offset_ = x }

// An InlinedCallExpr is an inlined function call.
type InlinedCallExpr struct {
	miniExpr
	Body_      Nodes
	ReturnVars Nodes
}

func NewInlinedCallExpr(pos src.XPos, body, retvars []Node) *InlinedCallExpr {
	n := &InlinedCallExpr{}
	n.pos = pos
	n.op = OINLCALL
	n.Body_.Set(body)
	n.ReturnVars.Set(retvars)
	return n
}

func (n *InlinedCallExpr) Body() Nodes      { return n.Body_ }
func (n *InlinedCallExpr) PtrBody() *Nodes  { return &n.Body_ }
func (n *InlinedCallExpr) SetBody(x Nodes)  { n.Body_ = x }
func (n *InlinedCallExpr) Rlist() Nodes     { return n.ReturnVars }
func (n *InlinedCallExpr) PtrRlist() *Nodes { return &n.ReturnVars }
func (n *InlinedCallExpr) SetRlist(x Nodes) { n.ReturnVars = x }

// A LogicalExpr is a expression X Op Y where Op is && or ||.
// It is separate from BinaryExpr to make room for statements
// that must be executed before Y but after X.
type LogicalExpr struct {
	miniExpr
	X Node
	Y Node
}

func NewLogicalExpr(pos src.XPos, op Op, x, y Node) *LogicalExpr {
	n := &LogicalExpr{X: x, Y: y}
	n.pos = pos
	n.SetOp(op)
	return n
}

func (n *LogicalExpr) Left() Node      { return n.X }
func (n *LogicalExpr) SetLeft(x Node)  { n.X = x }
func (n *LogicalExpr) Right() Node     { return n.Y }
func (n *LogicalExpr) SetRight(y Node) { n.Y = y }

func (n *LogicalExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OANDAND, OOROR:
		n.op = op
	}
}

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

func (n *MakeExpr) Left() Node      { return n.Len }
func (n *MakeExpr) SetLeft(x Node)  { n.Len = x }
func (n *MakeExpr) Right() Node     { return n.Cap }
func (n *MakeExpr) SetRight(x Node) { n.Cap = x }

func (n *MakeExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OMAKECHAN, OMAKEMAP, OMAKESLICE, OMAKESLICECOPY:
		n.op = op
	}
}

// A MethodExpr is a method expression T.M (where T is a type).
type MethodExpr struct {
	miniExpr
	T         *types.Type
	Method    *types.Field
	FuncName_ *Name
}

func NewMethodExpr(pos src.XPos, t *types.Type, method *types.Field) *MethodExpr {
	n := &MethodExpr{T: t, Method: method}
	n.pos = pos
	n.op = OMETHEXPR
	return n
}

func (n *MethodExpr) FuncName() *Name   { return n.FuncName_ }
func (n *MethodExpr) Left() Node        { panic("MethodExpr.Left") }
func (n *MethodExpr) SetLeft(x Node)    { panic("MethodExpr.SetLeft") }
func (n *MethodExpr) Right() Node       { panic("MethodExpr.Right") }
func (n *MethodExpr) SetRight(x Node)   { panic("MethodExpr.SetRight") }
func (n *MethodExpr) Sym() *types.Sym   { panic("MethodExpr.Sym") }
func (n *MethodExpr) Offset() int64     { panic("MethodExpr.Offset") }
func (n *MethodExpr) SetOffset(x int64) { panic("MethodExpr.SetOffset") }
func (n *MethodExpr) Class() Class      { panic("MethodExpr.Class") }
func (n *MethodExpr) SetClass(x Class)  { panic("MethodExpr.SetClass") }

// A NilExpr represents the predefined untyped constant nil.
// (It may be copied and assigned a type, though.)
type NilExpr struct {
	miniExpr
	Sym_ *types.Sym // TODO: Remove
}

func NewNilExpr(pos src.XPos) *NilExpr {
	n := &NilExpr{}
	n.pos = pos
	n.op = ONIL
	return n
}

func (n *NilExpr) Sym() *types.Sym     { return n.Sym_ }
func (n *NilExpr) SetSym(x *types.Sym) { n.Sym_ = x }

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

func (n *ParenExpr) Left() Node         { return n.X }
func (n *ParenExpr) SetLeft(x Node)     { n.X = x }
func (n *ParenExpr) Implicit() bool     { return n.flags&miniExprImplicit != 0 }
func (n *ParenExpr) SetImplicit(b bool) { n.flags.set(miniExprImplicit, b) }

func (*ParenExpr) CanBeNtype() {}

// SetOTYPE changes n to be an OTYPE node returning t,
// like all the type nodes in type.go.
func (n *ParenExpr) SetOTYPE(t *types.Type) {
	n.op = OTYPE
	n.typ = t
	t.SetNod(n)
}

// A ResultExpr represents a direct access to a result slot on the stack frame.
type ResultExpr struct {
	miniExpr
	Offset_ int64
}

func NewResultExpr(pos src.XPos, typ *types.Type, offset int64) *ResultExpr {
	n := &ResultExpr{Offset_: offset}
	n.pos = pos
	n.op = ORESULT
	n.typ = typ
	return n
}

func (n *ResultExpr) Offset() int64     { return n.Offset_ }
func (n *ResultExpr) SetOffset(x int64) { n.Offset_ = x }

// A NameOffsetExpr refers to an offset within a variable.
// It is like a SelectorExpr but without the field name.
type NameOffsetExpr struct {
	miniExpr
	Name_   *Name
	Offset_ int64
}

func NewNameOffsetExpr(pos src.XPos, name *Name, offset int64, typ *types.Type) *NameOffsetExpr {
	n := &NameOffsetExpr{Name_: name, Offset_: offset}
	n.typ = typ
	n.op = ONAMEOFFSET
	return n
}

// A SelectorExpr is a selector expression X.Sym.
type SelectorExpr struct {
	miniExpr
	X         Node
	Sel       *types.Sym
	Offset_   int64
	Selection *types.Field
}

func NewSelectorExpr(pos src.XPos, op Op, x Node, sel *types.Sym) *SelectorExpr {
	n := &SelectorExpr{X: x, Sel: sel}
	n.pos = pos
	n.Offset_ = types.BADWIDTH
	n.SetOp(op)
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

func (n *SelectorExpr) Left() Node          { return n.X }
func (n *SelectorExpr) SetLeft(x Node)      { n.X = x }
func (n *SelectorExpr) Sym() *types.Sym     { return n.Sel }
func (n *SelectorExpr) SetSym(x *types.Sym) { n.Sel = x }
func (n *SelectorExpr) Offset() int64       { return n.Offset_ }
func (n *SelectorExpr) SetOffset(x int64)   { n.Offset_ = x }
func (n *SelectorExpr) Implicit() bool      { return n.flags&miniExprImplicit != 0 }
func (n *SelectorExpr) SetImplicit(b bool)  { n.flags.set(miniExprImplicit, b) }

// Before type-checking, bytes.Buffer is a SelectorExpr.
// After type-checking it becomes a Name.
func (*SelectorExpr) CanBeNtype() {}

// A SliceExpr is a slice expression X[Low:High] or X[Low:High:Max].
type SliceExpr struct {
	miniExpr
	X     Node
	List_ Nodes // TODO(rsc): Use separate Nodes
}

func NewSliceExpr(pos src.XPos, op Op, x Node) *SliceExpr {
	n := &SliceExpr{X: x}
	n.pos = pos
	n.op = op
	return n
}

func (n *SliceExpr) Left() Node      { return n.X }
func (n *SliceExpr) SetLeft(x Node)  { n.X = x }
func (n *SliceExpr) List() Nodes     { return n.List_ }
func (n *SliceExpr) PtrList() *Nodes { return &n.List_ }
func (n *SliceExpr) SetList(x Nodes) { n.List_ = x }

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
	if n.List_.Len() == 0 {
		return nil, nil, nil
	}

	switch n.Op() {
	case OSLICE, OSLICEARR, OSLICESTR:
		s := n.List_.Slice()
		return s[0], s[1], nil
	case OSLICE3, OSLICE3ARR:
		s := n.List_.Slice()
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
		s := n.List_.Slice()
		if s == nil {
			if low == nil && high == nil {
				return
			}
			n.List_.Set2(low, high)
			return
		}
		s[0] = low
		s[1] = high
		return
	case OSLICE3, OSLICE3ARR:
		s := n.List_.Slice()
		if s == nil {
			if low == nil && high == nil && max == nil {
				return
			}
			n.List_.Set3(low, high, max)
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
	Ptr     Node
	LenCap_ Nodes // TODO(rsc): Split into two Node fields
}

func NewSliceHeaderExpr(pos src.XPos, typ *types.Type, ptr, len, cap Node) *SliceHeaderExpr {
	n := &SliceHeaderExpr{Ptr: ptr}
	n.pos = pos
	n.op = OSLICEHEADER
	n.typ = typ
	n.LenCap_.Set2(len, cap)
	return n
}

func (n *SliceHeaderExpr) Left() Node      { return n.Ptr }
func (n *SliceHeaderExpr) SetLeft(x Node)  { n.Ptr = x }
func (n *SliceHeaderExpr) List() Nodes     { return n.LenCap_ }
func (n *SliceHeaderExpr) PtrList() *Nodes { return &n.LenCap_ }
func (n *SliceHeaderExpr) SetList(x Nodes) { n.LenCap_ = x }

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

func (n *StarExpr) Left() Node         { return n.X }
func (n *StarExpr) SetLeft(x Node)     { n.X = x }
func (n *StarExpr) Implicit() bool     { return n.flags&miniExprImplicit != 0 }
func (n *StarExpr) SetImplicit(b bool) { n.flags.set(miniExprImplicit, b) }

func (*StarExpr) CanBeNtype() {}

// SetOTYPE changes n to be an OTYPE node returning t,
// like all the type nodes in type.go.
func (n *StarExpr) SetOTYPE(t *types.Type) {
	n.op = OTYPE
	n.X = nil
	n.typ = t
	t.SetNod(n)
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

func (n *TypeAssertExpr) Left() Node      { return n.X }
func (n *TypeAssertExpr) SetLeft(x Node)  { n.X = x }
func (n *TypeAssertExpr) Right() Node     { return n.Ntype }
func (n *TypeAssertExpr) SetRight(x Node) { n.Ntype = x } // TODO: toNtype(x)
func (n *TypeAssertExpr) List() Nodes     { return n.Itab }
func (n *TypeAssertExpr) PtrList() *Nodes { return &n.Itab }
func (n *TypeAssertExpr) SetList(x Nodes) { n.Itab = x }

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

func (n *UnaryExpr) Left() Node     { return n.X }
func (n *UnaryExpr) SetLeft(x Node) { n.X = x }

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
