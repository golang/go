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
	Diag() bool
	SetDiag(x bool)

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

// mayBeShared reports whether n may occur in multiple places in the AST.
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
	OPACK    // import
	OLITERAL // literal
	ONIL     // nil

	// expressions
	OADD          // Left + Right
	OSUB          // Left - Right
	OOR           // Left | Right
	OXOR          // Left ^ Right
	OADDSTR       // +{List} (string addition, list elements are strings)
	OADDR         // &Left
	OANDAND       // Left && Right
	OAPPEND       // append(List); after walk, Left may contain elem type descriptor
	OBYTES2STR    // Type(Left) (Type is string, Left is a []byte)
	OBYTES2STRTMP // Type(Left) (Type is string, Left is a []byte, ephemeral)
	ORUNES2STR    // Type(Left) (Type is string, Left is a []rune)
	OSTR2BYTES    // Type(Left) (Type is []byte, Left is a string)
	OSTR2BYTESTMP // Type(Left) (Type is []byte, Left is a string, ephemeral)
	OSTR2RUNES    // Type(Left) (Type is []rune, Left is a string)
	// Left = Right or (if Colas=true) Left := Right
	// If Colas, then Ninit includes a DCL node for Left.
	OAS
	// List = Rlist (x, y, z = a, b, c) or (if Colas=true) List := Rlist
	// If Colas, then Ninit includes DCL nodes for List
	OAS2
	OAS2DOTTYPE // List = Right (x, ok = I.(int))
	OAS2FUNC    // List = Right (x, y = f())
	OAS2MAPR    // List = Right (x, ok = m["foo"])
	OAS2RECV    // List = Right (x, ok = <-c)
	OASOP       // Left Etype= Right (x += y)
	OCALL       // Left(List) (function call, method call or type conversion)

	// OCALLFUNC, OCALLMETH, and OCALLINTER have the same structure.
	// Prior to walk, they are: Left(List), where List is all regular arguments.
	// After walk, List is a series of assignments to temporaries,
	// and Rlist is an updated set of arguments.
	// Nbody is all OVARLIVE nodes that are attached to OCALLxxx.
	// TODO(josharian/khr): Use Ninit instead of List for the assignments to temporaries. See CL 114797.
	OCALLFUNC  // Left(List/Rlist) (function call f(args))
	OCALLMETH  // Left(List/Rlist) (direct method call x.Method(args))
	OCALLINTER // Left(List/Rlist) (interface method call x.Method(args))
	OCALLPART  // Left.Right (method expression x.Method, not called)
	OCAP       // cap(Left)
	OCLOSE     // close(Left)
	OCLOSURE   // func Type { Func.Closure.Nbody } (func literal)
	OCOMPLIT   // Right{List} (composite literal, not yet lowered to specific form)
	OMAPLIT    // Type{List} (composite literal, Type is map)
	OSTRUCTLIT // Type{List} (composite literal, Type is struct)
	OARRAYLIT  // Type{List} (composite literal, Type is array)
	OSLICELIT  // Type{List} (composite literal, Type is slice) Right.Int64() = slice length.
	OPTRLIT    // &Left (left is composite literal)
	OCONV      // Type(Left) (type conversion)
	OCONVIFACE // Type(Left) (type conversion, to interface)
	OCONVNOP   // Type(Left) (type conversion, no effect)
	OCOPY      // copy(Left, Right)
	ODCL       // var Left (declares Left of type Left.Type)

	// Used during parsing but don't last.
	ODCLFUNC  // func f() or func (r) f()
	ODCLCONST // const pi = 3.14
	ODCLTYPE  // type Int int or type Int = int

	ODELETE        // delete(List)
	ODOT           // Left.Sym (Left is of struct type)
	ODOTPTR        // Left.Sym (Left is of pointer to struct type)
	ODOTMETH       // Left.Sym (Left is non-interface, Right is method name)
	ODOTINTER      // Left.Sym (Left is interface, Right is method name)
	OXDOT          // Left.Sym (before rewrite to one of the preceding)
	ODOTTYPE       // Left.Right or Left.Type (.Right during parsing, .Type once resolved); after walk, .Right contains address of interface type descriptor and .Right.Right contains address of concrete type descriptor
	ODOTTYPE2      // Left.Right or Left.Type (.Right during parsing, .Type once resolved; on rhs of OAS2DOTTYPE); after walk, .Right contains address of interface type descriptor
	OEQ            // Left == Right
	ONE            // Left != Right
	OLT            // Left < Right
	OLE            // Left <= Right
	OGE            // Left >= Right
	OGT            // Left > Right
	ODEREF         // *Left
	OINDEX         // Left[Right] (index of array or slice)
	OINDEXMAP      // Left[Right] (index of map)
	OKEY           // Left:Right (key:value in struct/array/map literal)
	OSTRUCTKEY     // Sym:Left (key:value in struct literal, after type checking)
	OLEN           // len(Left)
	OMAKE          // make(List) (before type checking converts to one of the following)
	OMAKECHAN      // make(Type, Left) (type is chan)
	OMAKEMAP       // make(Type, Left) (type is map)
	OMAKESLICE     // make(Type, Left, Right) (type is slice)
	OMAKESLICECOPY // makeslicecopy(Type, Left, Right) (type is slice; Left is length and Right is the copied from slice)
	// OMAKESLICECOPY is created by the order pass and corresponds to:
	//  s = make(Type, Left); copy(s, Right)
	//
	// Bounded can be set on the node when Left == len(Right) is known at compile time.
	//
	// This node is created so the walk pass can optimize this pattern which would
	// otherwise be hard to detect after the order pass.
	OMUL         // Left * Right
	ODIV         // Left / Right
	OMOD         // Left % Right
	OLSH         // Left << Right
	ORSH         // Left >> Right
	OAND         // Left & Right
	OANDNOT      // Left &^ Right
	ONEW         // new(Left); corresponds to calls to new in source code
	ONOT         // !Left
	OBITNOT      // ^Left
	OPLUS        // +Left
	ONEG         // -Left
	OOROR        // Left || Right
	OPANIC       // panic(Left)
	OPRINT       // print(List)
	OPRINTN      // println(List)
	OPAREN       // (Left)
	OSEND        // Left <- Right
	OSLICE       // Left[List[0] : List[1]] (Left is untypechecked or slice)
	OSLICEARR    // Left[List[0] : List[1]] (Left is pointer to array)
	OSLICESTR    // Left[List[0] : List[1]] (Left is string)
	OSLICE3      // Left[List[0] : List[1] : List[2]] (Left is untypedchecked or slice)
	OSLICE3ARR   // Left[List[0] : List[1] : List[2]] (Left is pointer to array)
	OSLICEHEADER // sliceheader{Left, List[0], List[1]} (Left is unsafe.Pointer, List[0] is length, List[1] is capacity)
	ORECOVER     // recover()
	ORECV        // <-Left
	ORUNESTR     // Type(Left) (Type is string, Left is rune)
	OSELRECV2    // like OAS2: List = Rlist where len(List)=2, len(Rlist)=1, Rlist[0].Op = ORECV (appears as .Left of OCASE)
	OIOTA        // iota
	OREAL        // real(Left)
	OIMAG        // imag(Left)
	OCOMPLEX     // complex(Left, Right) or complex(List[0]) where List[0] is a 2-result function call
	OALIGNOF     // unsafe.Alignof(Left)
	OOFFSETOF    // unsafe.Offsetof(Left)
	OSIZEOF      // unsafe.Sizeof(Left)
	OMETHEXPR    // method expression
	OSTMTEXPR    // statement expression (Init; Left)

	// statements
	OBLOCK // { List } (block of code)
	OBREAK // break [Sym]
	// OCASE:  case List: Nbody (List==nil means default)
	//   For OTYPESW, List is a OTYPE node for the specified type (or OLITERAL
	//   for nil), and, if a type-switch variable is specified, Rlist is an
	//   ONAME for the version of the type-switch variable with the specified
	//   type.
	OCASE
	OCONTINUE // continue [Sym]
	ODEFER    // defer Left (Left must be call)
	OFALL     // fallthrough
	OFOR      // for Ninit; Left; Right { Nbody }
	// OFORUNTIL is like OFOR, but the test (Left) is applied after the body:
	// 	Ninit
	// 	top: { Nbody }   // Execute the body at least once
	// 	cont: Right
	// 	if Left {        // And then test the loop condition
	// 		List     // Before looping to top, execute List
	// 		goto top
	// 	}
	// OFORUNTIL is created by walk. There's no way to write this in Go code.
	OFORUNTIL
	OGOTO   // goto Sym
	OIF     // if Ninit; Left { Nbody } else { Rlist }
	OLABEL  // Sym:
	OGO     // go Left (Left must be call)
	ORANGE  // for List = range Right { Nbody }
	ORETURN // return List
	OSELECT // select { List } (List is list of OCASE)
	OSWITCH // switch Ninit; Left { List } (List is a list of OCASE)
	// OTYPESW:  Left := Right.(type) (appears as .Left of OSWITCH)
	//   Left is nil if there is no type-switch variable
	OTYPESW
	OFUNCINST // instantiation of a generic function
	OTYPEINST // instantiation of a generic type

	// types
	OTCHAN   // chan int
	OTMAP    // map[string]int
	OTSTRUCT // struct{}
	OTINTER  // interface{}
	// OTFUNC: func() - Left is receiver field, List is list of param fields, Rlist is
	// list of result fields.
	OTFUNC
	OTARRAY // [8]int or [...]int
	OTSLICE // []int

	// misc
	// intermediate representation of an inlined call.  Uses Init (assignments
	// for the captured variables, parameters, retvars, & INLMARK op),
	// Body (body of the inlined function), and ReturnVars (list of
	// return values)
	OINLCALL       // intermediary representation of an inlined call.
	OEFACE         // itable and data words of an empty-interface value.
	OITAB          // itable word of an interface value.
	OIDATA         // data word of an interface value in Left
	OSPTR          // base pointer of a slice or string.
	OCFUNC         // reference to c function pointer (not go func value)
	OCHECKNIL      // emit code to ensure pointer/interface not nil
	OVARDEF        // variable is about to be fully initialized
	OVARKILL       // variable is dead
	OVARLIVE       // variable is alive
	ORESULT        // result of a function call; Xoffset is stack offset
	OINLMARK       // start of an inlined body, with file/line of caller. Xoffset is an index into the inline tree.
	OLINKSYMOFFSET // offset within a name

	// arch-specific opcodes
	OTAILCALL // tail call to another function
	OGETG     // runtime.getg() (read g pointer)

	OEND
)

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

type PragmaFlag int16

const (
	// Func pragmas.
	Nointerface    PragmaFlag = 1 << iota
	Noescape                  // func parameters don't escape
	Norace                    // func must not have race detector annotations
	Nosplit                   // func should not execute on separate stack
	Noinline                  // func should not be inlined
	NoCheckPtr                // func should not be instrumented by checkptr
	CgoUnsafeArgs             // treat a pointer to one arg as a pointer to them all
	UintptrEscapes            // pointers converted to uintptr escape

	// Runtime-only func pragmas.
	// See ../../../../runtime/README.md for detailed descriptions.
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

// isNil reports whether n represents the universal untyped zero value "nil".
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
	case ONAME, OPACK:
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
// 	n.Left = InitExpr(init, n.Left)
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
			base.Fatalf("OXDOT in walk")
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
