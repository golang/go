// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ir

import (
	"bytes"
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/src"
	"fmt"
	"go/constant"
	"go/token"
)

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
	init  Nodes // TODO(rsc): Don't require every Node to have an init
	flags bitset8
}

const (
	miniExprNonNil = 1 << iota
	miniExprTransient
	miniExprBounded
	miniExprImplicit // for use by implementations; not supported by every Expr
	miniExprCheckPtr
)

func (*miniExpr) isExpr() {}

func (n *miniExpr) Type() *types.Type     { return n.typ }
func (n *miniExpr) SetType(x *types.Type) { n.typ = x }
func (n *miniExpr) NonNil() bool          { return n.flags&miniExprNonNil != 0 }
func (n *miniExpr) MarkNonNil()           { n.flags |= miniExprNonNil }
func (n *miniExpr) Transient() bool       { return n.flags&miniExprTransient != 0 }
func (n *miniExpr) SetTransient(b bool)   { n.flags.set(miniExprTransient, b) }
func (n *miniExpr) Bounded() bool         { return n.flags&miniExprBounded != 0 }
func (n *miniExpr) SetBounded(b bool)     { n.flags.set(miniExprBounded, b) }
func (n *miniExpr) Init() Nodes           { return n.init }
func (n *miniExpr) PtrInit() *Nodes       { return &n.init }
func (n *miniExpr) SetInit(x Nodes)       { n.init = x }

// An AddStringExpr is a string concatenation Expr[0] + Exprs[1] + ... + Expr[len(Expr)-1].
type AddStringExpr struct {
	miniExpr
	List     Nodes
	Prealloc *Name
}

func NewAddStringExpr(pos src.XPos, list []Node) *AddStringExpr {
	n := &AddStringExpr{}
	n.pos = pos
	n.op = OADDSTR
	n.List = list
	return n
}

// An AddrExpr is an address-of expression &X.
// It may end up being a normal address-of or an allocation of a composite literal.
type AddrExpr struct {
	miniExpr
	X        Node
	Prealloc *Name // preallocated storage if any
}

func NewAddrExpr(pos src.XPos, x Node) *AddrExpr {
	n := &AddrExpr{X: x}
	n.op = OADDR
	n.pos = pos
	return n
}

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

// A BasicLit is a literal of basic type.
type BasicLit struct {
	miniExpr
	val constant.Value
}

func NewBasicLit(pos src.XPos, val constant.Value) Node {
	n := &BasicLit{val: val}
	n.op = OLITERAL
	n.pos = pos
	if k := val.Kind(); k != constant.Unknown {
		n.SetType(idealType(k))
	}
	return n
}

func (n *BasicLit) Val() constant.Value       { return n.val }
func (n *BasicLit) SetVal(val constant.Value) { n.val = val }

// A BinaryExpr is a binary expression X Op Y,
// or Op(X, Y) for builtin functions that do not become calls.
type BinaryExpr struct {
	miniExpr
	X     Node
	Y     Node
	RType Node `mknode:"-"` // see reflectdata/helpers.go
}

func NewBinaryExpr(pos src.XPos, op Op, x, y Node) *BinaryExpr {
	n := &BinaryExpr{X: x, Y: y}
	n.pos = pos
	n.SetOp(op)
	return n
}

func (n *BinaryExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OADD, OADDSTR, OAND, OANDNOT, ODIV, OEQ, OGE, OGT, OLE,
		OLSH, OLT, OMOD, OMUL, ONE, OOR, ORSH, OSUB, OXOR,
		OCOPY, OCOMPLEX, OUNSAFEADD, OUNSAFESLICE,
		OEFACE:
		n.op = op
	}
}

// A CallExpr is a function call X(Args).
type CallExpr struct {
	miniExpr
	origNode
	X         Node
	Args      Nodes
	RType     Node    `mknode:"-"` // see reflectdata/helpers.go
	KeepAlive []*Name // vars to be kept alive until call returns
	IsDDD     bool
	NoInline  bool
}

func NewCallExpr(pos src.XPos, op Op, fun Node, args []Node) *CallExpr {
	n := &CallExpr{X: fun}
	n.pos = pos
	n.orig = n
	n.SetOp(op)
	n.Args = args
	return n
}

func (*CallExpr) isStmt() {}

func (n *CallExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OAPPEND,
		OCALL, OCALLFUNC, OCALLINTER, OCALLMETH,
		ODELETE,
		OGETG, OGETCALLERPC, OGETCALLERSP,
		OMAKE, OPRINT, OPRINTN,
		ORECOVER, ORECOVERFP:
		n.op = op
	}
}

// A ClosureExpr is a function literal expression.
type ClosureExpr struct {
	miniExpr
	Func     *Func `mknode:"-"`
	Prealloc *Name
	IsGoWrap bool // whether this is wrapper closure of a go statement
}

// A CompLitExpr is a composite literal Type{Vals}.
// Before type-checking, the type is Ntype.
type CompLitExpr struct {
	miniExpr
	origNode
	List     Nodes // initialized values
	RType    Node  `mknode:"-"` // *runtime._type for OMAPLIT map types
	Prealloc *Name
	// For OSLICELIT, Len is the backing array length.
	// For OMAPLIT, Len is the number of entries that we've removed from List and
	// generated explicit mapassign calls for. This is used to inform the map alloc hint.
	Len int64
}

func NewCompLitExpr(pos src.XPos, op Op, typ *types.Type, list []Node) *CompLitExpr {
	n := &CompLitExpr{List: list}
	n.pos = pos
	n.SetOp(op)
	if typ != nil {
		n.SetType(typ)
	}
	n.orig = n
	return n
}

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
	origNode
	val constant.Value
}

func NewConstExpr(val constant.Value, orig Node) Node {
	n := &ConstExpr{val: val}
	n.op = OLITERAL
	n.pos = orig.Pos()
	n.orig = orig
	n.SetType(orig.Type())
	n.SetTypecheck(orig.Typecheck())
	return n
}

func (n *ConstExpr) Sym() *types.Sym     { return n.orig.Sym() }
func (n *ConstExpr) Val() constant.Value { return n.val }

// A ConvExpr is a conversion Type(X).
// It may end up being a value or a type.
type ConvExpr struct {
	miniExpr
	X Node

	// For implementing OCONVIFACE expressions.
	//
	// TypeWord is an expression yielding a *runtime._type or
	// *runtime.itab value to go in the type word of the iface/eface
	// result. See reflectdata.ConvIfaceTypeWord for further details.
	//
	// SrcRType is an expression yielding a *runtime._type value for X,
	// if it's not pointer-shaped and needs to be heap allocated.
	TypeWord Node `mknode:"-"`
	SrcRType Node `mknode:"-"`

	// For -d=checkptr instrumentation of conversions from
	// unsafe.Pointer to *Elem or *[Len]Elem.
	//
	// TODO(mdempsky): We only ever need one of these, but currently we
	// don't decide which one until walk. Longer term, it probably makes
	// sense to have a dedicated IR op for `(*[Len]Elem)(ptr)[:n:m]`
	// expressions.
	ElemRType     Node `mknode:"-"`
	ElemElemRType Node `mknode:"-"`
}

func NewConvExpr(pos src.XPos, op Op, typ *types.Type, x Node) *ConvExpr {
	n := &ConvExpr{X: x}
	n.pos = pos
	n.typ = typ
	n.SetOp(op)
	return n
}

func (n *ConvExpr) Implicit() bool     { return n.flags&miniExprImplicit != 0 }
func (n *ConvExpr) SetImplicit(b bool) { n.flags.set(miniExprImplicit, b) }
func (n *ConvExpr) CheckPtr() bool     { return n.flags&miniExprCheckPtr != 0 }
func (n *ConvExpr) SetCheckPtr(b bool) { n.flags.set(miniExprCheckPtr, b) }

func (n *ConvExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OCONV, OCONVIFACE, OCONVIDATA, OCONVNOP, OBYTES2STR, OBYTES2STRTMP, ORUNES2STR, OSTR2BYTES, OSTR2BYTESTMP, OSTR2RUNES, ORUNESTR, OSLICE2ARRPTR:
		n.op = op
	}
}

// An IndexExpr is an index expression X[Index].
type IndexExpr struct {
	miniExpr
	X        Node
	Index    Node
	RType    Node `mknode:"-"` // see reflectdata/helpers.go
	Assigned bool
}

func NewIndexExpr(pos src.XPos, x, index Node) *IndexExpr {
	n := &IndexExpr{X: x, Index: index}
	n.pos = pos
	n.op = OINDEX
	return n
}

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

// A StructKeyExpr is an Field: Value composite literal key.
type StructKeyExpr struct {
	miniExpr
	Field *types.Field
	Value Node
}

func NewStructKeyExpr(pos src.XPos, field *types.Field, value Node) *StructKeyExpr {
	n := &StructKeyExpr{Field: field, Value: value}
	n.pos = pos
	n.op = OSTRUCTKEY
	return n
}

func (n *StructKeyExpr) Sym() *types.Sym { return n.Field.Sym }

// An InlinedCallExpr is an inlined function call.
type InlinedCallExpr struct {
	miniExpr
	Body       Nodes
	ReturnVars Nodes // must be side-effect free
}

func NewInlinedCallExpr(pos src.XPos, body, retvars []Node) *InlinedCallExpr {
	n := &InlinedCallExpr{}
	n.pos = pos
	n.op = OINLCALL
	n.Body = body
	n.ReturnVars = retvars
	return n
}

func (n *InlinedCallExpr) SingleResult() Node {
	if have := len(n.ReturnVars); have != 1 {
		base.FatalfAt(n.Pos(), "inlined call has %v results, expected 1", have)
	}
	if !n.Type().HasShape() && n.ReturnVars[0].Type().HasShape() {
		// If the type of the call is not a shape, but the type of the return value
		// is a shape, we need to do an implicit conversion, so the real type
		// of n is maintained.
		r := NewConvExpr(n.Pos(), OCONVNOP, n.Type(), n.ReturnVars[0])
		r.SetTypecheck(1)
		return r
	}
	return n.ReturnVars[0]
}

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
	RType Node `mknode:"-"` // see reflectdata/helpers.go
	Len   Node
	Cap   Node
}

func NewMakeExpr(pos src.XPos, op Op, len, cap Node) *MakeExpr {
	n := &MakeExpr{Len: len, Cap: cap}
	n.pos = pos
	n.SetOp(op)
	return n
}

func (n *MakeExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OMAKECHAN, OMAKEMAP, OMAKESLICE, OMAKESLICECOPY:
		n.op = op
	}
}

// A NilExpr represents the predefined untyped constant nil.
// (It may be copied and assigned a type, though.)
type NilExpr struct {
	miniExpr
}

func NewNilExpr(pos src.XPos) *NilExpr {
	n := &NilExpr{}
	n.pos = pos
	n.op = ONIL
	return n
}

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

func (n *ParenExpr) Implicit() bool     { return n.flags&miniExprImplicit != 0 }
func (n *ParenExpr) SetImplicit(b bool) { n.flags.set(miniExprImplicit, b) }

// A RawOrigExpr represents an arbitrary Go expression as a string value.
// When printed in diagnostics, the string value is written out exactly as-is.
type RawOrigExpr struct {
	miniExpr
	Raw string
}

func NewRawOrigExpr(pos src.XPos, op Op, raw string) *RawOrigExpr {
	n := &RawOrigExpr{Raw: raw}
	n.pos = pos
	n.op = op
	return n
}

// A ResultExpr represents a direct access to a result.
type ResultExpr struct {
	miniExpr
	Index int64 // index of the result expr.
}

func NewResultExpr(pos src.XPos, typ *types.Type, index int64) *ResultExpr {
	n := &ResultExpr{Index: index}
	n.pos = pos
	n.op = ORESULT
	n.typ = typ
	return n
}

// A LinksymOffsetExpr refers to an offset within a global variable.
// It is like a SelectorExpr but without the field name.
type LinksymOffsetExpr struct {
	miniExpr
	Linksym *obj.LSym
	Offset_ int64
}

func NewLinksymOffsetExpr(pos src.XPos, lsym *obj.LSym, offset int64, typ *types.Type) *LinksymOffsetExpr {
	n := &LinksymOffsetExpr{Linksym: lsym, Offset_: offset}
	n.typ = typ
	n.op = OLINKSYMOFFSET
	return n
}

// NewLinksymExpr is NewLinksymOffsetExpr, but with offset fixed at 0.
func NewLinksymExpr(pos src.XPos, lsym *obj.LSym, typ *types.Type) *LinksymOffsetExpr {
	return NewLinksymOffsetExpr(pos, lsym, 0, typ)
}

// NewNameOffsetExpr is NewLinksymOffsetExpr, but taking a *Name
// representing a global variable instead of an *obj.LSym directly.
func NewNameOffsetExpr(pos src.XPos, name *Name, offset int64, typ *types.Type) *LinksymOffsetExpr {
	if name == nil || IsBlank(name) || !(name.Op() == ONAME && name.Class == PEXTERN) {
		base.FatalfAt(pos, "cannot take offset of nil, blank name or non-global variable: %v", name)
	}
	return NewLinksymOffsetExpr(pos, name.Linksym(), offset, typ)
}

// A SelectorExpr is a selector expression X.Sel.
type SelectorExpr struct {
	miniExpr
	X Node
	// Sel is the name of the field or method being selected, without (in the
	// case of methods) any preceding type specifier. If the field/method is
	// exported, than the Sym uses the local package regardless of the package
	// of the containing type.
	Sel *types.Sym
	// The actual selected field - may not be filled in until typechecking.
	Selection *types.Field
	Prealloc  *Name // preallocated storage for OMETHVALUE, if any
}

func NewSelectorExpr(pos src.XPos, op Op, x Node, sel *types.Sym) *SelectorExpr {
	n := &SelectorExpr{X: x, Sel: sel}
	n.pos = pos
	n.SetOp(op)
	return n
}

func (n *SelectorExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OXDOT, ODOT, ODOTPTR, ODOTMETH, ODOTINTER, OMETHVALUE, OMETHEXPR:
		n.op = op
	}
}

func (n *SelectorExpr) Sym() *types.Sym    { return n.Sel }
func (n *SelectorExpr) Implicit() bool     { return n.flags&miniExprImplicit != 0 }
func (n *SelectorExpr) SetImplicit(b bool) { n.flags.set(miniExprImplicit, b) }
func (n *SelectorExpr) Offset() int64      { return n.Selection.Offset }

func (n *SelectorExpr) FuncName() *Name {
	if n.Op() != OMETHEXPR {
		panic(n.no("FuncName"))
	}
	fn := NewNameAt(n.Selection.Pos, MethodSym(n.X.Type(), n.Sel))
	fn.Class = PFUNC
	fn.SetType(n.Type())
	if n.Selection.Nname != nil {
		// TODO(austin): Nname is nil for interface method
		// expressions (I.M), so we can't attach a Func to
		// those here. reflectdata.methodWrapper generates the
		// Func.
		fn.Func = n.Selection.Nname.(*Name).Func
	}
	return fn
}

// A SliceExpr is a slice expression X[Low:High] or X[Low:High:Max].
type SliceExpr struct {
	miniExpr
	X    Node
	Low  Node
	High Node
	Max  Node
}

func NewSliceExpr(pos src.XPos, op Op, x, low, high, max Node) *SliceExpr {
	n := &SliceExpr{X: x, Low: low, High: high, Max: max}
	n.pos = pos
	n.op = op
	return n
}

func (n *SliceExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OSLICE, OSLICEARR, OSLICESTR, OSLICE3, OSLICE3ARR:
		n.op = op
	}
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
	Ptr Node
	Len Node
	Cap Node
}

func NewSliceHeaderExpr(pos src.XPos, typ *types.Type, ptr, len, cap Node) *SliceHeaderExpr {
	n := &SliceHeaderExpr{Ptr: ptr, Len: len, Cap: cap}
	n.pos = pos
	n.op = OSLICEHEADER
	n.typ = typ
	return n
}

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

func (n *StarExpr) Implicit() bool     { return n.flags&miniExprImplicit != 0 }
func (n *StarExpr) SetImplicit(b bool) { n.flags.set(miniExprImplicit, b) }

// A TypeAssertionExpr is a selector expression X.(Type).
// Before type-checking, the type is Ntype.
type TypeAssertExpr struct {
	miniExpr
	X Node

	// Runtime type information provided by walkDotType for
	// assertions from non-empty interface to concrete type.
	ITab Node `mknode:"-"` // *runtime.itab for Type implementing X's type
}

func NewTypeAssertExpr(pos src.XPos, x Node, typ *types.Type) *TypeAssertExpr {
	n := &TypeAssertExpr{X: x}
	n.pos = pos
	n.op = ODOTTYPE
	if typ != nil {
		n.SetType(typ)
	}
	return n
}

func (n *TypeAssertExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case ODOTTYPE, ODOTTYPE2:
		n.op = op
	}
}

// A DynamicTypeAssertExpr asserts that X is of dynamic type RType.
type DynamicTypeAssertExpr struct {
	miniExpr
	X Node

	// SrcRType is an expression that yields a *runtime._type value
	// representing X's type. It's used in failed assertion panic
	// messages.
	SrcRType Node

	// RType is an expression that yields a *runtime._type value
	// representing the asserted type.
	//
	// BUG(mdempsky): If ITab is non-nil, RType may be nil.
	RType Node

	// ITab is an expression that yields a *runtime.itab value
	// representing the asserted type within the assertee expression's
	// original interface type.
	//
	// ITab is only used for assertions from non-empty interface type to
	// a concrete (i.e., non-interface) type. For all other assertions,
	// ITab is nil.
	ITab Node
}

func NewDynamicTypeAssertExpr(pos src.XPos, op Op, x, rtype Node) *DynamicTypeAssertExpr {
	n := &DynamicTypeAssertExpr{X: x, RType: rtype}
	n.pos = pos
	n.op = op
	return n
}

func (n *DynamicTypeAssertExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case ODYNAMICDOTTYPE, ODYNAMICDOTTYPE2:
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

func (n *UnaryExpr) SetOp(op Op) {
	switch op {
	default:
		panic(n.no("SetOp " + op.String()))
	case OBITNOT, ONEG, ONOT, OPLUS, ORECV,
		OALIGNOF, OCAP, OCLOSE, OIMAG, OLEN, ONEW,
		OOFFSETOF, OPANIC, OREAL, OSIZEOF,
		OCHECKNIL, OCFUNC, OIDATA, OITAB, OSPTR:
		n.op = op
	}
}

// Probably temporary: using Implicit() flag to mark generic function nodes that
// are called to make getGfInfo analysis easier in one pre-order pass.
func (n *InstExpr) Implicit() bool     { return n.flags&miniExprImplicit != 0 }
func (n *InstExpr) SetImplicit(b bool) { n.flags.set(miniExprImplicit, b) }

// An InstExpr is a generic function or type instantiation.
type InstExpr struct {
	miniExpr
	X     Node
	Targs []Ntype
}

func NewInstExpr(pos src.XPos, op Op, x Node, targs []Ntype) *InstExpr {
	n := &InstExpr{X: x, Targs: targs}
	n.pos = pos
	n.op = op
	return n
}

func IsZero(n Node) bool {
	switch n.Op() {
	case ONIL:
		return true

	case OLITERAL:
		switch u := n.Val(); u.Kind() {
		case constant.String:
			return constant.StringVal(u) == ""
		case constant.Bool:
			return !constant.BoolVal(u)
		default:
			return constant.Sign(u) == 0
		}

	case OARRAYLIT:
		n := n.(*CompLitExpr)
		for _, n1 := range n.List {
			if n1.Op() == OKEY {
				n1 = n1.(*KeyExpr).Value
			}
			if !IsZero(n1) {
				return false
			}
		}
		return true

	case OSTRUCTLIT:
		n := n.(*CompLitExpr)
		for _, n1 := range n.List {
			n1 := n1.(*StructKeyExpr)
			if !IsZero(n1.Value) {
				return false
			}
		}
		return true
	}

	return false
}

// lvalue etc
func IsAddressable(n Node) bool {
	switch n.Op() {
	case OINDEX:
		n := n.(*IndexExpr)
		if n.X.Type() != nil && n.X.Type().IsArray() {
			return IsAddressable(n.X)
		}
		if n.X.Type() != nil && n.X.Type().IsString() {
			return false
		}
		fallthrough
	case ODEREF, ODOTPTR:
		return true

	case ODOT:
		n := n.(*SelectorExpr)
		return IsAddressable(n.X)

	case ONAME:
		n := n.(*Name)
		if n.Class == PFUNC {
			return false
		}
		return true

	case OLINKSYMOFFSET:
		return true
	}

	return false
}

func StaticValue(n Node) Node {
	for {
		if n.Op() == OCONVNOP {
			n = n.(*ConvExpr).X
			continue
		}

		if n.Op() == OINLCALL {
			n = n.(*InlinedCallExpr).SingleResult()
			continue
		}

		n1 := staticValue1(n)
		if n1 == nil {
			return n
		}
		n = n1
	}
}

// staticValue1 implements a simple SSA-like optimization. If n is a local variable
// that is initialized and never reassigned, staticValue1 returns the initializer
// expression. Otherwise, it returns nil.
func staticValue1(nn Node) Node {
	if nn.Op() != ONAME {
		return nil
	}
	n := nn.(*Name)
	if n.Class != PAUTO {
		return nil
	}

	defn := n.Defn
	if defn == nil {
		return nil
	}

	var rhs Node
FindRHS:
	switch defn.Op() {
	case OAS:
		defn := defn.(*AssignStmt)
		rhs = defn.Y
	case OAS2:
		defn := defn.(*AssignListStmt)
		for i, lhs := range defn.Lhs {
			if lhs == n {
				rhs = defn.Rhs[i]
				break FindRHS
			}
		}
		base.Fatalf("%v missing from LHS of %v", n, defn)
	default:
		return nil
	}
	if rhs == nil {
		base.Fatalf("RHS is nil: %v", defn)
	}

	if reassigned(n) {
		return nil
	}

	return rhs
}

// reassigned takes an ONAME node, walks the function in which it is defined, and returns a boolean
// indicating whether the name has any assignments other than its declaration.
// The second return value is the first such assignment encountered in the walk, if any. It is mostly
// useful for -m output documenting the reason for inhibited optimizations.
// NB: global variables are always considered to be re-assigned.
// TODO: handle initial declaration not including an assignment and followed by a single assignment?
func reassigned(name *Name) bool {
	if name.Op() != ONAME {
		base.Fatalf("reassigned %v", name)
	}
	// no way to reliably check for no-reassignment of globals, assume it can be
	if name.Curfn == nil {
		return true
	}

	// TODO(mdempsky): This is inefficient and becoming increasingly
	// unwieldy. Figure out a way to generalize escape analysis's
	// reassignment detection for use by inlining and devirtualization.

	// isName reports whether n is a reference to name.
	isName := func(x Node) bool {
		n, ok := x.(*Name)
		return ok && n.Canonical() == name
	}

	var do func(n Node) bool
	do = func(n Node) bool {
		switch n.Op() {
		case OAS:
			n := n.(*AssignStmt)
			if isName(n.X) && n != name.Defn {
				return true
			}
		case OAS2, OAS2FUNC, OAS2MAPR, OAS2DOTTYPE, OAS2RECV, OSELRECV2:
			n := n.(*AssignListStmt)
			for _, p := range n.Lhs {
				if isName(p) && n != name.Defn {
					return true
				}
			}
		case OADDR:
			n := n.(*AddrExpr)
			if isName(OuterValue(n.X)) {
				return true
			}
		case OCLOSURE:
			n := n.(*ClosureExpr)
			if Any(n.Func, do) {
				return true
			}
		}
		return false
	}
	return Any(name.Curfn, do)
}

// IsIntrinsicCall reports whether the compiler back end will treat the call as an intrinsic operation.
var IsIntrinsicCall = func(*CallExpr) bool { return false }

// SameSafeExpr checks whether it is safe to reuse one of l and r
// instead of computing both. SameSafeExpr assumes that l and r are
// used in the same statement or expression. In order for it to be
// safe to reuse l or r, they must:
//   - be the same expression
//   - not have side-effects (no function calls, no channel ops);
//     however, panics are ok
//   - not cause inappropriate aliasing; e.g. two string to []byte
//     conversions, must result in two distinct slices
//
// The handling of OINDEXMAP is subtle. OINDEXMAP can occur both
// as an lvalue (map assignment) and an rvalue (map access). This is
// currently OK, since the only place SameSafeExpr gets used on an
// lvalue expression is for OSLICE and OAPPEND optimizations, and it
// is correct in those settings.
func SameSafeExpr(l Node, r Node) bool {
	for l.Op() == OCONVNOP {
		l = l.(*ConvExpr).X
	}
	for r.Op() == OCONVNOP {
		r = r.(*ConvExpr).X
	}
	if l.Op() != r.Op() || !types.Identical(l.Type(), r.Type()) {
		return false
	}

	switch l.Op() {
	case ONAME:
		return l == r

	case ODOT, ODOTPTR:
		l := l.(*SelectorExpr)
		r := r.(*SelectorExpr)
		return l.Sel != nil && r.Sel != nil && l.Sel == r.Sel && SameSafeExpr(l.X, r.X)

	case ODEREF:
		l := l.(*StarExpr)
		r := r.(*StarExpr)
		return SameSafeExpr(l.X, r.X)

	case ONOT, OBITNOT, OPLUS, ONEG:
		l := l.(*UnaryExpr)
		r := r.(*UnaryExpr)
		return SameSafeExpr(l.X, r.X)

	case OCONV:
		l := l.(*ConvExpr)
		r := r.(*ConvExpr)
		// Some conversions can't be reused, such as []byte(str).
		// Allow only numeric-ish types. This is a bit conservative.
		return types.IsSimple[l.Type().Kind()] && SameSafeExpr(l.X, r.X)

	case OINDEX, OINDEXMAP:
		l := l.(*IndexExpr)
		r := r.(*IndexExpr)
		return SameSafeExpr(l.X, r.X) && SameSafeExpr(l.Index, r.Index)

	case OADD, OSUB, OOR, OXOR, OMUL, OLSH, ORSH, OAND, OANDNOT, ODIV, OMOD:
		l := l.(*BinaryExpr)
		r := r.(*BinaryExpr)
		return SameSafeExpr(l.X, r.X) && SameSafeExpr(l.Y, r.Y)

	case OLITERAL:
		return constant.Compare(l.Val(), token.EQL, r.Val())

	case ONIL:
		return true
	}

	return false
}

// ShouldCheckPtr reports whether pointer checking should be enabled for
// function fn at a given level. See debugHelpFooter for defined
// levels.
func ShouldCheckPtr(fn *Func, level int) bool {
	return base.Debug.Checkptr >= level && fn.Pragma&NoCheckPtr == 0
}

// ShouldAsanCheckPtr reports whether pointer checking should be enabled for
// function fn when -asan is enabled.
func ShouldAsanCheckPtr(fn *Func) bool {
	return base.Flag.ASan && fn.Pragma&NoCheckPtr == 0
}

// IsReflectHeaderDataField reports whether l is an expression p.Data
// where p has type reflect.SliceHeader or reflect.StringHeader.
func IsReflectHeaderDataField(l Node) bool {
	if l.Type() != types.Types[types.TUINTPTR] {
		return false
	}

	var tsym *types.Sym
	switch l.Op() {
	case ODOT:
		l := l.(*SelectorExpr)
		tsym = l.X.Type().Sym()
	case ODOTPTR:
		l := l.(*SelectorExpr)
		tsym = l.X.Type().Elem().Sym()
	default:
		return false
	}

	if tsym == nil || l.Sym().Name != "Data" || tsym.Pkg.Path != "reflect" {
		return false
	}
	return tsym.Name == "SliceHeader" || tsym.Name == "StringHeader"
}

func ParamNames(ft *types.Type) []Node {
	args := make([]Node, ft.NumParams())
	for i, f := range ft.Params().FieldSlice() {
		args[i] = AsNode(f.Nname)
	}
	return args
}

// MethodSym returns the method symbol representing a method name
// associated with a specific receiver type.
//
// Method symbols can be used to distinguish the same method appearing
// in different method sets. For example, T.M and (*T).M have distinct
// method symbols.
//
// The returned symbol will be marked as a function.
func MethodSym(recv *types.Type, msym *types.Sym) *types.Sym {
	sym := MethodSymSuffix(recv, msym, "")
	sym.SetFunc(true)
	return sym
}

// MethodSymSuffix is like methodsym, but allows attaching a
// distinguisher suffix. To avoid collisions, the suffix must not
// start with a letter, number, or period.
func MethodSymSuffix(recv *types.Type, msym *types.Sym, suffix string) *types.Sym {
	if msym.IsBlank() {
		base.Fatalf("blank method name")
	}

	rsym := recv.Sym()
	if recv.IsPtr() {
		if rsym != nil {
			base.Fatalf("declared pointer receiver type: %v", recv)
		}
		rsym = recv.Elem().Sym()
	}

	// Find the package the receiver type appeared in. For
	// anonymous receiver types (i.e., anonymous structs with
	// embedded fields), use the "go" pseudo-package instead.
	rpkg := Pkgs.Go
	if rsym != nil {
		rpkg = rsym.Pkg
	}

	var b bytes.Buffer
	if recv.IsPtr() {
		// The parentheses aren't really necessary, but
		// they're pretty traditional at this point.
		fmt.Fprintf(&b, "(%-S)", recv)
	} else {
		fmt.Fprintf(&b, "%-S", recv)
	}

	// A particular receiver type may have multiple non-exported
	// methods with the same name. To disambiguate them, include a
	// package qualifier for names that came from a different
	// package than the receiver type.
	if !types.IsExported(msym.Name) && msym.Pkg != rpkg {
		b.WriteString(".")
		b.WriteString(msym.Pkg.Prefix)
	}

	b.WriteString(".")
	b.WriteString(msym.Name)
	b.WriteString(suffix)
	return rpkg.LookupBytes(b.Bytes())
}

// MethodExprName returns the ONAME representing the method
// referenced by expression n, which must be a method selector,
// method expression, or method value.
func MethodExprName(n Node) *Name {
	name, _ := MethodExprFunc(n).Nname.(*Name)
	return name
}

// MethodExprFunc is like MethodExprName, but returns the types.Field instead.
func MethodExprFunc(n Node) *types.Field {
	switch n.Op() {
	case ODOTMETH, OMETHEXPR, OMETHVALUE:
		return n.(*SelectorExpr).Selection
	}
	base.Fatalf("unexpected node: %v (%v)", n, n.Op())
	panic("unreachable")
}
