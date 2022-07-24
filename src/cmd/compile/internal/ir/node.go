// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// “Abstract” syntax representation.

package ir

import (
	"fmt"
	"go/constant"
	"sort"

	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

// A Node is the abstract interface to an IR node.
type Node interface {
	// Formatting
	Format(s fmt.State, verb rune)

	// Source position.
	Pos() src.XPos
	SetPos(x src.XPos)

	// For making copies. For Copy and SepCopy.
	copy() Node

	doChildren(func(Node) bool) bool
	editChildren(func(Node) Node)

	// Abstract graph structure, for generic traversals.
	Op() Op
	Init() Nodes

	// Fields specific to certain Ops only.
	Type() *types.Type
	SetType(t *types.Type)
	Name() *Name
	Sym() *types.Sym
	Val() constant.Value
	SetVal(v constant.Value)

	// Storage for analysis passes.
	Esc() uint16
	SetEsc(x uint16)

	// Typecheck values:
	//  0 means the node is not typechecked
	//  1 means the node is completely typechecked
	//  2 means typechecking of the node is in progress
	//  3 means the node has its type from types2, but may need transformation
	Typecheck() uint8
	SetTypecheck(x uint8)
	NonNil() bool
	MarkNonNil()
}

// Line returns n's position as a string. If n has been inlined,
// it uses the outermost position where n has been inlined.
func Line(n Node) string {
	return base.FmtPos(n.Pos())
}

func IsSynthetic(n Node) bool {
	name := n.Sym().Name
	return name[0] == '.' || name[0] == '~'
}

// IsAutoTmp indicates if n was created by the compiler as a temporary,
// based on the setting of the .AutoTemp flag in n's Name.
func IsAutoTmp(n Node) bool {
	if n == nil || n.Op() != ONAME {
		return false
	}
	return n.Name().AutoTemp()
}

// MayBeShared reports whether n may occur in multiple places in the AST.
// Extra care must be taken when mutating such a node.
func MayBeShared(n Node) bool {
	switch n.Op() {
	case ONAME, OLITERAL, ONIL, OTYPE:
		return true
	}
	return false
}

type InitNode interface {
	Node
	PtrInit() *Nodes
	SetInit(x Nodes)
}

func TakeInit(n Node) Nodes {
	init := n.Init()
	if len(init) != 0 {
		n.(InitNode).SetInit(nil)
	}
	return init
}

//go:generate stringer -type=Op -trimprefix=O node.go

type Op uint8

// Node ops.
const (
	OXXX Op = iota

	// names
	ONAME // var or func name
	// Unnamed arg or return value: f(int, string) (int, error) { etc }
	// Also used for a qualified package identifier that hasn't been resolved yet.
	ONONAME
	OTYPE    // type name
	OLITERAL // literal
	ONIL     // nil

	// expressions
	OADD          // X + Y
	OSUB          // X - Y
	OOR           // X | Y
	OXOR          // X ^ Y
	OADDSTR       // +{List} (string addition, list elements are strings)
	OADDR         // &X
	OANDAND       // X && Y
	OAPPEND       // append(Args); after walk, X may contain elem type descriptor
	OBYTES2STR    // Type(X) (Type is string, X is a []byte)
	OBYTES2STRTMP // Type(X) (Type is string, X is a []byte, ephemeral)
	ORUNES2STR    // Type(X) (Type is string, X is a []rune)
	OSTR2BYTES    // Type(X) (Type is []byte, X is a string)
	OSTR2BYTESTMP // Type(X) (Type is []byte, X is a string, ephemeral)
	OSTR2RUNES    // Type(X) (Type is []rune, X is a string)
	OSLICE2ARRPTR // Type(X) (Type is *[N]T, X is a []T)
	// X = Y or (if Def=true) X := Y
	// If Def, then Init includes a DCL node for X.
	OAS
	// Lhs = Rhs (x, y, z = a, b, c) or (if Def=true) Lhs := Rhs
	// If Def, then Init includes DCL nodes for Lhs
	OAS2
	OAS2DOTTYPE // Lhs = Rhs (x, ok = I.(int))
	OAS2FUNC    // Lhs = Rhs (x, y = f())
	OAS2MAPR    // Lhs = Rhs (x, ok = m["foo"])
	OAS2RECV    // Lhs = Rhs (x, ok = <-c)
	OASOP       // X AsOp= Y (x += y)
	OCALL       // X(Args) (function call, method call or type conversion)

	// OCALLFUNC, OCALLMETH, and OCALLINTER have the same structure.
	// Prior to walk, they are: X(Args), where Args is all regular arguments.
	// After walk, if any argument whose evaluation might requires temporary variable,
	// that temporary variable will be pushed to Init, Args will contains an updated
	// set of arguments.
	OCALLFUNC  // X(Args) (function call f(args))
	OCALLMETH  // X(Args) (direct method call x.Method(args))
	OCALLINTER // X(Args) (interface method call x.Method(args))
	OCAP       // cap(X)
	OCLOSE     // close(X)
	OCLOSURE   // func Type { Func.Closure.Body } (func literal)
	OCOMPLIT   // Type{List} (composite literal, not yet lowered to specific form)
	OMAPLIT    // Type{List} (composite literal, Type is map)
	OSTRUCTLIT // Type{List} (composite literal, Type is struct)
	OARRAYLIT  // Type{List} (composite literal, Type is array)
	OSLICELIT  // Type{List} (composite literal, Type is slice), Len is slice length.
	OPTRLIT    // &X (X is composite literal)
	OCONV      // Type(X) (type conversion)
	OCONVIFACE // Type(X) (type conversion, to interface)
	OCONVIDATA // Builds a data word to store X in an interface. Equivalent to IDATA(CONVIFACE(X)). Is an ir.ConvExpr.
	OCONVNOP   // Type(X) (type conversion, no effect)
	OCOPY      // copy(X, Y)
	ODCL       // var X (declares X of type X.Type)

	// Used during parsing but don't last.
	ODCLFUNC  // func f() or func (r) f()
	ODCLCONST // const pi = 3.14
	ODCLTYPE  // type Int int or type Int = int

	ODELETE        // delete(Args)
	ODOT           // X.Sel (X is of struct type)
	ODOTPTR        // X.Sel (X is of pointer to struct type)
	ODOTMETH       // X.Sel (X is non-interface, Sel is method name)
	ODOTINTER      // X.Sel (X is interface, Sel is method name)
	OXDOT          // X.Sel (before rewrite to one of the preceding)
	ODOTTYPE       // X.Ntype or X.Type (.Ntype during parsing, .Type once resolved); after walk, Itab contains address of interface type descriptor and Itab.X contains address of concrete type descriptor
	ODOTTYPE2      // X.Ntype or X.Type (.Ntype during parsing, .Type once resolved; on rhs of OAS2DOTTYPE); after walk, Itab contains address of interface type descriptor
	OEQ            // X == Y
	ONE            // X != Y
	OLT            // X < Y
	OLE            // X <= Y
	OGE            // X >= Y
	OGT            // X > Y
	ODEREF         // *X
	OINDEX         // X[Index] (index of array or slice)
	OINDEXMAP      // X[Index] (index of map)
	OKEY           // Key:Value (key:value in struct/array/map literal)
	OSTRUCTKEY     // Field:Value (key:value in struct literal, after type checking)
	OLEN           // len(X)
	OMAKE          // make(Args) (before type checking converts to one of the following)
	OMAKECHAN      // make(Type[, Len]) (type is chan)
	OMAKEMAP       // make(Type[, Len]) (type is map)
	OMAKESLICE     // make(Type[, Len[, Cap]]) (type is slice)
	OMAKESLICECOPY // makeslicecopy(Type, Len, Cap) (type is slice; Len is length and Cap is the copied from slice)
	// OMAKESLICECOPY is created by the order pass and corresponds to:
	//  s = make(Type, Len); copy(s, Cap)
	//
	// Bounded can be set on the node when Len == len(Cap) is known at compile time.
	//
	// This node is created so the walk pass can optimize this pattern which would
	// otherwise be hard to detect after the order pass.
	OMUL         // X * Y
	ODIV         // X / Y
	OMOD         // X % Y
	OLSH         // X << Y
	ORSH         // X >> Y
	OAND         // X & Y
	OANDNOT      // X &^ Y
	ONEW         // new(X); corresponds to calls to new in source code
	ONOT         // !X
	OBITNOT      // ^X
	OPLUS        // +X
	ONEG         // -X
	OOROR        // X || Y
	OPANIC       // panic(X)
	OPRINT       // print(List)
	OPRINTN      // println(List)
	OPAREN       // (X)
	OSEND        // Chan <- Value
	OSLICE       // X[Low : High] (X is untypechecked or slice)
	OSLICEARR    // X[Low : High] (X is pointer to array)
	OSLICESTR    // X[Low : High] (X is string)
	OSLICE3      // X[Low : High : Max] (X is untypedchecked or slice)
	OSLICE3ARR   // X[Low : High : Max] (X is pointer to array)
	OSLICEHEADER // sliceheader{Ptr, Len, Cap} (Ptr is unsafe.Pointer, Len is length, Cap is capacity)
	ORECOVER     // recover()
	ORECOVERFP   // recover(Args) w/ explicit FP argument
	ORECV        // <-X
	ORUNESTR     // Type(X) (Type is string, X is rune)
	OSELRECV2    // like OAS2: Lhs = Rhs where len(Lhs)=2, len(Rhs)=1, Rhs[0].Op = ORECV (appears as .Var of OCASE)
	OREAL        // real(X)
	OIMAG        // imag(X)
	OCOMPLEX     // complex(X, Y)
	OALIGNOF     // unsafe.Alignof(X)
	OOFFSETOF    // unsafe.Offsetof(X)
	OSIZEOF      // unsafe.Sizeof(X)
	OUNSAFEADD   // unsafe.Add(X, Y)
	OUNSAFESLICE // unsafe.Slice(X, Y)
	OMETHEXPR    // X(Args) (method expression T.Method(args), first argument is the method receiver)
	OMETHVALUE   // X.Sel   (method expression t.Method, not called)

	// statements
	OBLOCK // { List } (block of code)
	OBREAK // break [Label]
	// OCASE:  case List: Body (List==nil means default)
	//   For OTYPESW, List is a OTYPE node for the specified type (or OLITERAL
	//   for nil) or an ODYNAMICTYPE indicating a runtime type for generics.
	//   If a type-switch variable is specified, Var is an
	//   ONAME for the version of the type-switch variable with the specified
	//   type.
	OCASE
	OCONTINUE // continue [Label]
	ODEFER    // defer Call
	OFALL     // fallthrough
	OFOR      // for Init; Cond; Post { Body }
	OGOTO     // goto Label
	OIF       // if Init; Cond { Then } else { Else }
	OLABEL    // Label:
	OGO       // go Call
	ORANGE    // for Key, Value = range X { Body }
	ORETURN   // return Results
	OSELECT   // select { Cases }
	OSWITCH   // switch Init; Expr { Cases }
	// OTYPESW:  X := Y.(type) (appears as .Tag of OSWITCH)
	//   X is nil if there is no type-switch variable
	OTYPESW
	OFUNCINST // instantiation of a generic function

	// misc
	// intermediate representation of an inlined call.  Uses Init (assignments
	// for the captured variables, parameters, retvars, & INLMARK op),
	// Body (body of the inlined function), and ReturnVars (list of
	// return values)
	OINLCALL       // intermediary representation of an inlined call.
	OEFACE         // itable and data words of an empty-interface value.
	OITAB          // itable word of an interface value.
	OIDATA         // data word of an interface value in X
	OSPTR          // base pointer of a slice or string.
	OCFUNC         // reference to c function pointer (not go func value)
	OCHECKNIL      // emit code to ensure pointer/interface not nil
	ORESULT        // result of a function call; Xoffset is stack offset
	OINLMARK       // start of an inlined body, with file/line of caller. Xoffset is an index into the inline tree.
	OLINKSYMOFFSET // offset within a name
	OJUMPTABLE     // A jump table structure for implementing dense expression switches

	// opcodes for generics
	ODYNAMICDOTTYPE  // x = i.(T) where T is a type parameter (or derived from a type parameter)
	ODYNAMICDOTTYPE2 // x, ok = i.(T) where T is a type parameter (or derived from a type parameter)
	ODYNAMICTYPE     // a type node for type switches (represents a dynamic target type for a type switch)

	// arch-specific opcodes
	OTAILCALL    // tail call to another function
	OGETG        // runtime.getg() (read g pointer)
	OGETCALLERPC // runtime.getcallerpc() (continuation PC in caller frame)
	OGETCALLERSP // runtime.getcallersp() (stack pointer in caller frame)

	OEND
)

// IsCmp reports whether op is a comparison operation (==, !=, <, <=,
// >, or >=).
func (op Op) IsCmp() bool {
	switch op {
	case OEQ, ONE, OLT, OLE, OGT, OGE:
		return true
	}
	return false
}

// Nodes is a pointer to a slice of *Node.
// For fields that are not used in most nodes, this is used instead of
// a slice to save space.
type Nodes []Node

// Append appends entries to Nodes.
func (n *Nodes) Append(a ...Node) {
	if len(a) == 0 {
		return
	}
	*n = append(*n, a...)
}

// Prepend prepends entries to Nodes.
// If a slice is passed in, this will take ownership of it.
func (n *Nodes) Prepend(a ...Node) {
	if len(a) == 0 {
		return
	}
	*n = append(a, *n...)
}

// Take clears n, returning its former contents.
func (n *Nodes) Take() []Node {
	ret := *n
	*n = nil
	return ret
}

// Copy returns a copy of the content of the slice.
func (n Nodes) Copy() Nodes {
	if n == nil {
		return nil
	}
	c := make(Nodes, len(n))
	copy(c, n)
	return c
}

// NameQueue is a FIFO queue of *Name. The zero value of NameQueue is
// a ready-to-use empty queue.
type NameQueue struct {
	ring       []*Name
	head, tail int
}

// Empty reports whether q contains no Names.
func (q *NameQueue) Empty() bool {
	return q.head == q.tail
}

// PushRight appends n to the right of the queue.
func (q *NameQueue) PushRight(n *Name) {
	if len(q.ring) == 0 {
		q.ring = make([]*Name, 16)
	} else if q.head+len(q.ring) == q.tail {
		// Grow the ring.
		nring := make([]*Name, len(q.ring)*2)
		// Copy the old elements.
		part := q.ring[q.head%len(q.ring):]
		if q.tail-q.head <= len(part) {
			part = part[:q.tail-q.head]
			copy(nring, part)
		} else {
			pos := copy(nring, part)
			copy(nring[pos:], q.ring[:q.tail%len(q.ring)])
		}
		q.ring, q.head, q.tail = nring, 0, q.tail-q.head
	}

	q.ring[q.tail%len(q.ring)] = n
	q.tail++
}

// PopLeft pops a Name from the left of the queue. It panics if q is
// empty.
func (q *NameQueue) PopLeft() *Name {
	if q.Empty() {
		panic("dequeue empty")
	}
	n := q.ring[q.head%len(q.ring)]
	q.head++
	return n
}

// NameSet is a set of Names.
type NameSet map[*Name]struct{}

// Has reports whether s contains n.
func (s NameSet) Has(n *Name) bool {
	_, isPresent := s[n]
	return isPresent
}

// Add adds n to s.
func (s *NameSet) Add(n *Name) {
	if *s == nil {
		*s = make(map[*Name]struct{})
	}
	(*s)[n] = struct{}{}
}

// Sorted returns s sorted according to less.
func (s NameSet) Sorted(less func(*Name, *Name) bool) []*Name {
	var res []*Name
	for n := range s {
		res = append(res, n)
	}
	sort.Slice(res, func(i, j int) bool { return less(res[i], res[j]) })
	return res
}

type PragmaFlag uint16

const (
	// Func pragmas.
	Nointerface      PragmaFlag = 1 << iota
	Noescape                    // func parameters don't escape
	Norace                      // func must not have race detector annotations
	Nosplit                     // func should not execute on separate stack
	Noinline                    // func should not be inlined
	NoCheckPtr                  // func should not be instrumented by checkptr
	CgoUnsafeArgs               // treat a pointer to one arg as a pointer to them all
	UintptrKeepAlive            // pointers converted to uintptr must be kept alive
	UintptrEscapes              // pointers converted to uintptr escape

	// Runtime-only func pragmas.
	// See ../../../../runtime/HACKING.md for detailed descriptions.
	Systemstack        // func must run on system stack
	Nowritebarrier     // emit compiler error instead of write barrier
	Nowritebarrierrec  // error on write barrier in this or recursive callees
	Yeswritebarrierrec // cancels Nowritebarrierrec in this function and callees

	// Runtime and cgo type pragmas
	NotInHeap // values of this type must not be heap allocated

	// Go command pragmas
	GoBuildPragma

	RegisterParams // TODO(register args) remove after register abi is working

)

func AsNode(n types.Object) Node {
	if n == nil {
		return nil
	}
	return n.(Node)
}

var BlankNode Node

func IsConst(n Node, ct constant.Kind) bool {
	return ConstType(n) == ct
}

// IsNil reports whether n represents the universal untyped zero value "nil".
func IsNil(n Node) bool {
	// Check n.Orig because constant propagation may produce typed nil constants,
	// which don't exist in the Go spec.
	return n != nil && Orig(n).Op() == ONIL
}

func IsBlank(n Node) bool {
	if n == nil {
		return false
	}
	return n.Sym().IsBlank()
}

// IsMethod reports whether n is a method.
// n must be a function or a method.
func IsMethod(n Node) bool {
	return n.Type().Recv() != nil
}

func HasNamedResults(fn *Func) bool {
	typ := fn.Type()
	return typ.NumResults() > 0 && types.OrigSym(typ.Results().Field(0).Sym) != nil
}

// HasUniquePos reports whether n has a unique position that can be
// used for reporting error messages.
//
// It's primarily used to distinguish references to named objects,
// whose Pos will point back to their declaration position rather than
// their usage position.
func HasUniquePos(n Node) bool {
	switch n.Op() {
	case ONAME:
		return false
	case OLITERAL, ONIL, OTYPE:
		if n.Sym() != nil {
			return false
		}
	}

	if !n.Pos().IsKnown() {
		if base.Flag.K != 0 {
			base.Warn("setlineno: unknown position (line 0)")
		}
		return false
	}

	return true
}

func SetPos(n Node) src.XPos {
	lno := base.Pos
	if n != nil && HasUniquePos(n) {
		base.Pos = n.Pos()
	}
	return lno
}

// The result of InitExpr MUST be assigned back to n, e.g.
//
//	n.X = InitExpr(init, n.X)
func InitExpr(init []Node, expr Node) Node {
	if len(init) == 0 {
		return expr
	}

	n, ok := expr.(InitNode)
	if !ok || MayBeShared(n) {
		// Introduce OCONVNOP to hold init list.
		n = NewConvExpr(base.Pos, OCONVNOP, nil, expr)
		n.SetType(expr.Type())
		n.SetTypecheck(1)
	}

	n.PtrInit().Prepend(init...)
	return n
}

// what's the outer value that a write to n affects?
// outer value means containing struct or array.
func OuterValue(n Node) Node {
	for {
		switch nn := n; nn.Op() {
		case OXDOT:
			base.FatalfAt(n.Pos(), "OXDOT in OuterValue: %v", n)
		case ODOT:
			nn := nn.(*SelectorExpr)
			n = nn.X
			continue
		case OPAREN:
			nn := nn.(*ParenExpr)
			n = nn.X
			continue
		case OCONVNOP:
			nn := nn.(*ConvExpr)
			n = nn.X
			continue
		case OINDEX:
			nn := nn.(*IndexExpr)
			if nn.X.Type() == nil {
				base.Fatalf("OuterValue needs type for %v", nn.X)
			}
			if nn.X.Type().IsArray() {
				n = nn.X
				continue
			}
		}

		return n
	}
}

const (
	EscUnknown = iota
	EscNone    // Does not escape to heap, result, or parameters.
	EscHeap    // Reachable from the heap
	EscNever   // By construction will not escape.
)
