// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// “Abstract” syntax representation.

package gc

// A Node is a single node in the syntax tree.
// Actually the syntax tree is a syntax DAG, because there is only one
// node with Op=ONAME for a given instance of a variable x.
// The same is true for Op=OTYPE and Op=OLITERAL.
type Node struct {
	// Tree structure.
	// Generic recursive walks should follow these fields.
	Left  *Node
	Right *Node
	Ninit *NodeList
	Nbody *NodeList
	List  *NodeList
	Rlist *NodeList

	// most nodes
	Type *Type
	Orig *Node // original form, for printing, and tracking copies of ONAMEs

	// func
	Func *Func

	// ONAME
	Name *Name

	Sym *Sym        // various
	E   interface{} // Opt or Val, see methods below

	Xoffset int64

	Lineno int32

	// OREGISTER, OINDREG
	Reg int16

	Esc uint16 // EscXXX

	Op          Op
	Nointerface bool
	Ullman      uint8 // sethi/ullman number
	Addable     bool  // addressable
	Etype       EType // op for OASOP, etype for OTYPE, exclam for export, 6g saved reg
	Bounded     bool  // bounds check unnecessary
	Class       Class // PPARAM, PAUTO, PEXTERN, etc
	Embedded    uint8 // ODCLFIELD embedded type
	Colas       bool  // OAS resulting from :=
	Diag        uint8 // already printed error about this
	Noescape    bool  // func arguments do not escape; TODO(rsc): move Noescape to Func struct (see CL 7360)
	Walkdef     uint8
	Typecheck   uint8
	Local       bool
	Dodata      uint8
	Initorder   uint8
	Used        bool
	Isddd       bool // is the argument variadic
	Implicit    bool
	Addrtaken   bool // address taken, even if not moved to heap
	Assigned    bool // is the variable ever assigned to
	Likely      int8 // likeliness of if statement
	Hasbreak    bool // has break statement
	hasVal      int8 // +1 for Val, -1 for Opt, 0 for not yet set
}

// Val returns the Val for the node.
func (n *Node) Val() Val {
	if n.hasVal != +1 {
		return Val{}
	}
	return Val{n.E}
}

// SetVal sets the Val for the node, which must not have been used with SetOpt.
func (n *Node) SetVal(v Val) {
	if n.hasVal == -1 {
		Debug['h'] = 1
		Dump("have Opt", n)
		Fatalf("have Opt")
	}
	n.hasVal = +1
	n.E = v.U
}

// Opt returns the optimizer data for the node.
func (n *Node) Opt() interface{} {
	if n.hasVal != -1 {
		return nil
	}
	return n.E
}

// SetOpt sets the optimizer data for the node, which must not have been used with SetVal.
// SetOpt(nil) is ignored for Vals to simplify call sites that are clearing Opts.
func (n *Node) SetOpt(x interface{}) {
	if x == nil && n.hasVal >= 0 {
		return
	}
	if n.hasVal == +1 {
		Debug['h'] = 1
		Dump("have Val", n)
		Fatalf("have Val")
	}
	n.hasVal = -1
	n.E = x
}

// Name holds Node fields used only by named nodes (ONAME, OPACK, some OLITERAL).
type Name struct {
	Pack      *Node // real package for import . names
	Pkg       *Pkg  // pkg for OPACK nodes
	Heapaddr  *Node // temp holding heap address of param
	Inlvar    *Node // ONAME substitute while inlining
	Defn      *Node // initializing assignment
	Curfn     *Node // function for local variables
	Param     *Param
	Decldepth int32 // declaration loop depth, increased for every loop or label
	Vargen    int32 // unique name for ONAME within a function.  Function outputs are numbered starting at one.
	Iota      int32 // value if this name is iota
	Funcdepth int32
	Method    bool // OCALLMETH name
	Readonly  bool
	Captured  bool // is the variable captured by a closure
	Byval     bool // is the variable captured by value or by reference
	Needzero  bool // if it contains pointers, needs to be zeroed on function entry
	Keepalive bool // mark value live across unknown assembly call
}

type Param struct {
	Ntype *Node

	// ONAME func param with PHEAP
	Outerexpr  *Node // expression copied into closure for variable
	Stackparam *Node // OPARAM node referring to stack copy of param

	// ONAME PPARAM
	Field *Type // TFIELD in arg struct

	// ONAME closure param with PPARAMREF
	Outer   *Node // outer PPARAMREF in nested closure
	Closure *Node // ONAME/PHEAP <-> ONAME/PPARAMREF
}

// Func holds Node fields used only with function-like nodes.
type Func struct {
	Shortname  *Node
	Enter      *NodeList
	Exit       *NodeList
	Cvars      *NodeList // closure params
	Dcl        *NodeList // autodcl for this func/closure
	Inldcl     *NodeList // copy of dcl for use in inlining
	Closgen    int
	Outerfunc  *Node
	Fieldtrack []*Type
	Outer      *Node // outer func for closure
	Ntype      *Node // signature
	Top        int   // top context (Ecall, Eproc, etc)
	Closure    *Node // OCLOSURE <-> ODCLFUNC
	FCurfn     *Node
	Nname      *Node

	Inl     *NodeList // copy of the body for use in inlining
	InlCost int32
	Depth   int32

	Endlineno int32

	Norace            bool // func must not have race detector annotations
	Nosplit           bool // func should not execute on separate stack
	Noinline          bool // func should not be inlined
	Nowritebarrier    bool // emit compiler error instead of write barrier
	Nowritebarrierrec bool // error on write barrier in this or recursive callees
	Dupok             bool // duplicate definitions ok
	Wrapper           bool // is method wrapper
	Needctxt          bool // function uses context register (has closure variables)
	Systemstack       bool // must run on system stack

	WBLineno int32 // line number of first write barrier
}

type Op uint8

// Node ops.
const (
	OXXX = Op(iota)

	// names
	ONAME    // var, const or func name
	ONONAME  // unnamed arg or return value: f(int, string) (int, error) { etc }
	OTYPE    // type name
	OPACK    // import
	OLITERAL // literal

	// expressions
	OADD             // Left + Right
	OSUB             // Left - Right
	OOR              // Left | Right
	OXOR             // Left ^ Right
	OADDSTR          // Left + Right (string addition)
	OADDR            // &Left
	OANDAND          // Left && Right
	OAPPEND          // append(List)
	OARRAYBYTESTR    // Type(Left) (Type is string, Left is a []byte)
	OARRAYBYTESTRTMP // Type(Left) (Type is string, Left is a []byte, ephemeral)
	OARRAYRUNESTR    // Type(Left) (Type is string, Left is a []rune)
	OSTRARRAYBYTE    // Type(Left) (Type is []byte, Left is a string)
	OSTRARRAYBYTETMP // Type(Left) (Type is []byte, Left is a string, ephemeral)
	OSTRARRAYRUNE    // Type(Left) (Type is []rune, Left is a string)
	OAS              // Left = Right or (if Colas=true) Left := Right
	OAS2             // List = Rlist (x, y, z = a, b, c)
	OAS2FUNC         // List = Rlist (x, y = f())
	OAS2RECV         // List = Rlist (x, ok = <-c)
	OAS2MAPR         // List = Rlist (x, ok = m["foo"])
	OAS2DOTTYPE      // List = Rlist (x, ok = I.(int))
	OASOP            // Left Etype= Right (x += y)
	OASWB            // Left = Right (with write barrier)
	OCALL            // Left(List) (function call, method call or type conversion)
	OCALLFUNC        // Left(List) (function call f(args))
	OCALLMETH        // Left(List) (direct method call x.Method(args))
	OCALLINTER       // Left(List) (interface method call x.Method(args))
	OCALLPART        // Left.Right (method expression x.Method, not called)
	OCAP             // cap(Left)
	OCLOSE           // close(Left)
	OCLOSURE         // func Type { Body } (func literal)
	OCMPIFACE        // Left Etype Right (interface comparison, x == y or x != y)
	OCMPSTR          // Left Etype Right (string comparison, x == y, x < y, etc)
	OCOMPLIT         // Right{List} (composite literal, not yet lowered to specific form)
	OMAPLIT          // Type{List} (composite literal, Type is map)
	OSTRUCTLIT       // Type{List} (composite literal, Type is struct)
	OARRAYLIT        // Type{List} (composite literal, Type is array or slice)
	OPTRLIT          // &Left (left is composite literal)
	OCONV            // Type(Left) (type conversion)
	OCONVIFACE       // Type(Left) (type conversion, to interface)
	OCONVNOP         // Type(Left) (type conversion, no effect)
	OCOPY            // copy(Left, Right)
	ODCL             // var Left (declares Left of type Left.Type)

	// Used during parsing but don't last.
	ODCLFUNC  // func f() or func (r) f()
	ODCLFIELD // struct field, interface field, or func/method argument/return value.
	ODCLCONST // const pi = 3.14
	ODCLTYPE  // type Int int

	ODELETE    // delete(Left, Right)
	ODOT       // Left.Right (Left is of struct type)
	ODOTPTR    // Left.Right (Left is of pointer to struct type)
	ODOTMETH   // Left.Right (Left is non-interface, Right is method name)
	ODOTINTER  // Left.Right (Left is interface, Right is method name)
	OXDOT      // Left.Right (before rewrite to one of the preceding)
	ODOTTYPE   // Left.Right or Left.Type (.Right during parsing, .Type once resolved)
	ODOTTYPE2  // Left.Right or Left.Type (.Right during parsing, .Type once resolved; on rhs of OAS2DOTTYPE)
	OEQ        // Left == Right
	ONE        // Left != Right
	OLT        // Left < Right
	OLE        // Left <= Right
	OGE        // Left >= Right
	OGT        // Left > Right
	OIND       // *Left
	OINDEX     // Left[Right] (index of array or slice)
	OINDEXMAP  // Left[Right] (index of map)
	OKEY       // Left:Right (key:value in struct/array/map literal, or slice index pair)
	OPARAM     // variant of ONAME for on-stack copy of a parameter or return value that escapes.
	OLEN       // len(Left)
	OMAKE      // make(List) (before type checking converts to one of the following)
	OMAKECHAN  // make(Type, Left) (type is chan)
	OMAKEMAP   // make(Type, Left) (type is map)
	OMAKESLICE // make(Type, Left, Right) (type is slice)
	OMUL       // Left * Right
	ODIV       // Left / Right
	OMOD       // Left % Right
	OLSH       // Left << Right
	ORSH       // Left >> Right
	OAND       // Left & Right
	OANDNOT    // Left &^ Right
	ONEW       // new(Left)
	ONOT       // !Left
	OCOM       // ^Left
	OPLUS      // +Left
	OMINUS     // -Left
	OOROR      // Left || Right
	OPANIC     // panic(Left)
	OPRINT     // print(List)
	OPRINTN    // println(List)
	OPAREN     // (Left)
	OSEND      // Left <- Right
	OSLICE     // Left[Right.Left : Right.Right] (Left is untypechecked or slice; Right.Op==OKEY)
	OSLICEARR  // Left[Right.Left : Right.Right] (Left is array)
	OSLICESTR  // Left[Right.Left : Right.Right] (Left is string)
	OSLICE3    // Left[R.Left : R.R.Left : R.R.R] (R=Right; Left is untypedchecked or slice; R.Op and R.R.Op==OKEY)
	OSLICE3ARR // Left[R.Left : R.R.Left : R.R.R] (R=Right; Left is array; R.Op and R.R.Op==OKEY)
	ORECOVER   // recover()
	ORECV      // <-Left
	ORUNESTR   // Type(Left) (Type is string, Left is rune)
	OSELRECV   // Left = <-Right.Left: (appears as .Left of OCASE; Right.Op == ORECV)
	OSELRECV2  // List = <-Right.Left: (apperas as .Left of OCASE; count(List) == 2, Right.Op == ORECV)
	OIOTA      // iota
	OREAL      // real(Left)
	OIMAG      // imag(Left)
	OCOMPLEX   // complex(Left, Right)

	// statements
	OBLOCK    // { List } (block of code)
	OBREAK    // break
	OCASE     // case List: Nbody (select case after processing; List==nil means default)
	OXCASE    // case List: Nbody (select case before processing; List==nil means default)
	OCONTINUE // continue
	ODEFER    // defer Left (Left must be call)
	OEMPTY    // no-op (empty statement)
	OFALL     // fallthrough (after processing)
	OXFALL    // fallthrough (before processing)
	OFOR      // for Ninit; Left; Right { Nbody }
	OGOTO     // goto Left
	OIF       // if Ninit; Left { Nbody } else { Rlist }
	OLABEL    // Left:
	OPROC     // go Left (Left must be call)
	ORANGE    // for List = range Right { Nbody }
	ORETURN   // return List
	OSELECT   // select { List } (List is list of OXCASE or OCASE)
	OSWITCH   // switch Ninit; Left { List } (List is a list of OXCASE or OCASE)
	OTYPESW   // List = Left.(type) (appears as .Left of OSWITCH)

	// types
	OTCHAN   // chan int
	OTMAP    // map[string]int
	OTSTRUCT // struct{}
	OTINTER  // interface{}
	OTFUNC   // func()
	OTARRAY  // []int, [8]int, [N]int or [...]int

	// misc
	ODDD        // func f(args ...int) or f(l...) or var a = [...]int{0, 1, 2}.
	ODDDARG     // func f(args ...int), introduced by escape analysis.
	OINLCALL    // intermediary representation of an inlined call.
	OEFACE      // itable and data words of an empty-interface value.
	OITAB       // itable word of an interface value.
	OSPTR       // base pointer of a slice or string.
	OCLOSUREVAR // variable reference at beginning of closure function
	OCFUNC      // reference to c function pointer (not go func value)
	OCHECKNIL   // emit code to ensure pointer/interface not nil
	OVARKILL    // variable is dead
	OVARLIVE    // variable is alive

	// thearch-specific registers
	OREGISTER // a register, such as AX.
	OINDREG   // offset plus indirect of a register, such as 8(SP).

	// arch-specific opcodes
	OCMP    // compare: ACMP.
	ODEC    // decrement: ADEC.
	OINC    // increment: AINC.
	OEXTEND // extend: ACWD/ACDQ/ACQO.
	OHMUL   // high mul: AMUL/AIMUL for unsigned/signed (OMUL uses AIMUL for both).
	OLROT   // left rotate: AROL.
	ORROTC  // right rotate-carry: ARCR.
	ORETJMP // return to other function
	OPS     // compare parity set (for x86 NaN check)
	OPC     // compare parity clear (for x86 NaN check)
	OSQRT   // sqrt(float64), on systems that have hw support
	OGETG   // runtime.getg() (read g pointer)

	OEND
)

// A NodeList is a linked list of nodes.
// TODO(rsc): Some uses of NodeList should be made into slices.
// The remaining ones probably just need a simple linked list,
// not one with concatenation support.
type NodeList struct {
	N    *Node
	Next *NodeList
	End  *NodeList
}

// concat returns the concatenation of the lists a and b.
// The storage taken by both is reused for the result.
func concat(a *NodeList, b *NodeList) *NodeList {
	if a == nil {
		return b
	}
	if b == nil {
		return a
	}

	a.End.Next = b
	a.End = b.End
	b.End = nil
	return a
}

// list1 returns a one-element list containing n.
func list1(n *Node) *NodeList {
	if n == nil {
		return nil
	}
	if n.Op == OBLOCK && n.Ninit == nil {
		// Flatten list and steal storage.
		// Poison pointer to catch errant uses.
		l := n.List

		n.List = nil
		return l
	}

	l := new(NodeList)
	l.N = n
	l.End = l
	return l
}

// list returns the result of appending n to l.
func list(l *NodeList, n *Node) *NodeList {
	return concat(l, list1(n))
}

// listsort sorts *l in place according to the comparison function lt.
// The algorithm expects lt(a, b) to be equivalent to a < b.
// The algorithm is mergesort, so it is guaranteed to be O(n log n).
func listsort(l **NodeList, lt func(*Node, *Node) bool) {
	if *l == nil || (*l).Next == nil {
		return
	}

	l1 := *l
	l2 := *l
	for {
		l2 = l2.Next
		if l2 == nil {
			break
		}
		l2 = l2.Next
		if l2 == nil {
			break
		}
		l1 = l1.Next
	}

	l2 = l1.Next
	l1.Next = nil
	l2.End = (*l).End
	(*l).End = l1

	l1 = *l
	listsort(&l1, lt)
	listsort(&l2, lt)

	if lt(l1.N, l2.N) {
		*l = l1
	} else {
		*l = l2
		l2 = l1
		l1 = *l
	}

	// now l1 == *l; and l1 < l2

	var le *NodeList
	for (l1 != nil) && (l2 != nil) {
		for (l1.Next != nil) && lt(l1.Next.N, l2.N) {
			l1 = l1.Next
		}

		// l1 is last one from l1 that is < l2
		le = l1.Next // le is the rest of l1, first one that is >= l2
		if le != nil {
			le.End = (*l).End
		}

		(*l).End = l1       // cut *l at l1
		*l = concat(*l, l2) // glue l2 to *l's tail

		l1 = l2 // l1 is the first element of *l that is < the new l2
		l2 = le // ... because l2 now is the old tail of l1
	}

	*l = concat(*l, l2) // any remainder
}

// count returns the length of the list l.
func count(l *NodeList) int {
	n := int64(0)
	for ; l != nil; l = l.Next {
		n++
	}
	if int64(int(n)) != n { // Overflow.
		Yyerror("too many elements in list")
	}
	return int(n)
}
