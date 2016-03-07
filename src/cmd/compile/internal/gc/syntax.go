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
	Nbody Nodes
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
	Enter      Nodes // for example, allocate and initialize memory for escaping parameters
	Exit       Nodes
	Cvars      Nodes    // closure params
	Dcl        []*Node  // autodcl for this func/closure
	Inldcl     *[]*Node // copy of dcl for use in inlining
	Closgen    int
	Outerfunc  *Node
	Fieldtrack []*Type
	Outer      *Node // outer func for closure
	Ntype      *Node // signature
	Top        int   // top context (Ecall, Eproc, etc)
	Closure    *Node // OCLOSURE <-> ODCLFUNC
	FCurfn     *Node
	Nname      *Node

	Inl     Nodes // copy of the body for use in inlining
	InlCost int32
	Depth   int32

	Endlineno int32
	WBLineno  int32 // line number of first write barrier

	Pragma   Pragma // go:xxx function annotations
	Dupok    bool   // duplicate definitions ok
	Wrapper  bool   // is method wrapper
	Needctxt bool   // function uses context register (has closure variables)
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
	OADDSTR          // +{List} (string addition, list elements are strings)
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
	if n.Op == OBLOCK && nodeSeqLen(n.Ninit) == 0 {
		// Flatten list and steal storage.
		// Poison pointer to catch errant uses.
		l := n.List

		setNodeSeq(&n.List, nil)
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

// Nodes is a pointer to a slice of *Node.
// For fields that are not used in most nodes, this is used instead of
// a slice to save space.
type Nodes struct{ slice *[]*Node }

// Slice returns the entries in Nodes as a slice.
// Changes to the slice entries (as in s[i] = n) will be reflected in
// the Nodes.
func (n *Nodes) Slice() []*Node {
	if n.slice == nil {
		return nil
	}
	return *n.slice
}

// NodeList returns the entries in Nodes as a NodeList.
// Changes to the NodeList entries (as in l.N = n) will *not* be
// reflected in the Nodes.
// This wastes memory and should be used as little as possible.
func (n *Nodes) NodeList() *NodeList {
	if n.slice == nil {
		return nil
	}
	var ret *NodeList
	for _, n := range *n.slice {
		ret = list(ret, n)
	}
	return ret
}

// Set sets Nodes to a slice.
// This takes ownership of the slice.
func (n *Nodes) Set(s []*Node) {
	if len(s) == 0 {
		n.slice = nil
	} else {
		n.slice = &s
	}
}

// Append appends entries to Nodes.
// If a slice is passed in, this will take ownership of it.
func (n *Nodes) Append(a ...*Node) {
	if n.slice == nil {
		if len(a) > 0 {
			n.slice = &a
		}
	} else {
		*n.slice = append(*n.slice, a...)
	}
}

// SetToNodeList sets Nodes to the contents of a NodeList.
func (n *Nodes) SetToNodeList(l *NodeList) {
	s := make([]*Node, 0, count(l))
	for ; l != nil; l = l.Next {
		s = append(s, l.N)
	}
	n.Set(s)
}

// AppendNodeList appends the contents of a NodeList.
func (n *Nodes) AppendNodeList(l *NodeList) {
	if n.slice == nil {
		n.SetToNodeList(l)
	} else {
		for ; l != nil; l = l.Next {
			*n.slice = append(*n.slice, l.N)
		}
	}
}

// nodesOrNodeList must be either type Nodes or type *NodeList, or, in
// some cases, []*Node. It exists during the transition from NodeList
// to Nodes only and then should be deleted. See nodeSeqIterate to
// return an iterator from a nodesOrNodeList.
type nodesOrNodeList interface{}

// nodesOrNodeListPtr must be type *Nodes or type **NodeList, or, in
// some cases, *[]*Node. It exists during the transition from NodeList
// to Nodes only, and then should be deleted. See setNodeSeq to assign
// to a generic value.
type nodesOrNodeListPtr interface{}

// nodeSeqIterator is an interface used to iterate over a sequence of nodes.
// TODO(iant): Remove after conversion from NodeList to Nodes is complete.
type nodeSeqIterator interface {
	// Return whether iteration is complete.
	Done() bool
	// Advance to the next node.
	Next()
	// Return the current node.
	N() *Node
	// Return the address of the current node.
	P() **Node
	// Return the number of items remaining in the iteration.
	Len() int
	// Return the remaining items as a sequence.
	// This will have the same type as that passed to nodeSeqIterate.
	Seq() nodesOrNodeList
}

// nodeListIterator is a type that implements nodeSeqIterator using a
// *NodeList.
type nodeListIterator struct {
	l *NodeList
}

func (nli *nodeListIterator) Done() bool {
	return nli.l == nil
}

func (nli *nodeListIterator) Next() {
	nli.l = nli.l.Next
}

func (nli *nodeListIterator) N() *Node {
	return nli.l.N
}

func (nli *nodeListIterator) P() **Node {
	return &nli.l.N
}

func (nli *nodeListIterator) Len() int {
	return count(nli.l)
}

func (nli *nodeListIterator) Seq() nodesOrNodeList {
	return nli.l
}

// nodesIterator implements nodeSeqIterator using a Nodes.
type nodesIterator struct {
	n Nodes
	i int
}

func (ni *nodesIterator) Done() bool {
	return ni.i >= len(ni.n.Slice())
}

func (ni *nodesIterator) Next() {
	ni.i++
}

func (ni *nodesIterator) N() *Node {
	return ni.n.Slice()[ni.i]
}

func (ni *nodesIterator) P() **Node {
	return &ni.n.Slice()[ni.i]
}

func (ni *nodesIterator) Len() int {
	return len(ni.n.Slice()[ni.i:])
}

func (ni *nodesIterator) Seq() nodesOrNodeList {
	var r Nodes
	r.Set(ni.n.Slice()[ni.i:])
	return r
}

// nodeSeqIterate returns an iterator over a *NodeList, a Nodes,
// a []*Node, or nil.
func nodeSeqIterate(ns nodesOrNodeList) nodeSeqIterator {
	switch ns := ns.(type) {
	case *NodeList:
		return &nodeListIterator{ns}
	case Nodes:
		return &nodesIterator{ns, 0}
	case []*Node:
		var r Nodes
		r.Set(ns)
		return &nodesIterator{r, 0}
	case nil:
		var r Nodes
		return &nodesIterator{r, 0}
	default:
		panic("can't happen")
	}
}

// nodeSeqLen returns the length of a *NodeList, a Nodes, a []*Node, or nil.
func nodeSeqLen(ns nodesOrNodeList) int {
	switch ns := ns.(type) {
	case *NodeList:
		return count(ns)
	case Nodes:
		return len(ns.Slice())
	case []*Node:
		return len(ns)
	case nil:
		return 0
	default:
		panic("can't happen")
	}
}

// nodeSeqFirst returns the first element of a *NodeList, a Nodes,
// or a []*Node. It panics if the sequence is empty.
func nodeSeqFirst(ns nodesOrNodeList) *Node {
	switch ns := ns.(type) {
	case *NodeList:
		return ns.N
	case Nodes:
		return ns.Slice()[0]
	case []*Node:
		return ns[0]
	default:
		panic("can't happen")
	}
}

// nodeSeqSecond returns the second element of a *NodeList, a Nodes,
// or a []*Node. It panics if the sequence has fewer than two elements.
func nodeSeqSecond(ns nodesOrNodeList) *Node {
	switch ns := ns.(type) {
	case *NodeList:
		return ns.Next.N
	case Nodes:
		return ns.Slice()[1]
	case []*Node:
		return ns[1]
	default:
		panic("can't happen")
	}
}

// nodeSeqSlice returns a []*Node containing the contents of a
// *NodeList, a Nodes, or a []*Node.
// This is an interim function during the transition from NodeList to Nodes.
// TODO(iant): Remove when transition is complete.
func nodeSeqSlice(ns nodesOrNodeList) []*Node {
	switch ns := ns.(type) {
	case *NodeList:
		var s []*Node
		for l := ns; l != nil; l = l.Next {
			s = append(s, l.N)
		}
		return s
	case Nodes:
		return ns.Slice()
	case []*Node:
		return ns
	default:
		panic("can't happen")
	}
}

// setNodeSeq implements *a = b.
// a must have type **NodeList, *Nodes, or *[]*Node.
// b must have type *NodeList, Nodes, []*Node, or nil.
// This is an interim function during the transition from NodeList to Nodes.
// TODO(iant): Remove when transition is complete.
func setNodeSeq(a nodesOrNodeListPtr, b nodesOrNodeList) {
	if b == nil {
		switch a := a.(type) {
		case **NodeList:
			*a = nil
		case *Nodes:
			a.Set(nil)
		case *[]*Node:
			*a = nil
		default:
			panic("can't happen")
		}
		return
	}

	// Simplify b to either *NodeList or []*Node.
	if n, ok := b.(Nodes); ok {
		b = n.Slice()
	}

	if l, ok := a.(**NodeList); ok {
		switch b := b.(type) {
		case *NodeList:
			*l = b
		case []*Node:
			var ll *NodeList
			for _, n := range b {
				ll = list(ll, n)
			}
			*l = ll
		default:
			panic("can't happen")
		}
	} else {
		var s []*Node
		switch b := b.(type) {
		case *NodeList:
			for l := b; l != nil; l = l.Next {
				s = append(s, l.N)
			}
		case []*Node:
			s = b
		default:
			panic("can't happen")
		}

		switch a := a.(type) {
		case *Nodes:
			a.Set(s)
		case *[]*Node:
			*a = s
		default:
			panic("can't happen")
		}
	}
}

// setNodeSeqNode sets the node sequence a to the node n.
// a must have type **NodeList, *Nodes, or *[]*Node.
// This is an interim function during the transition from NodeList to Nodes.
// TODO(iant): Remove when transition is complete.
func setNodeSeqNode(a nodesOrNodeListPtr, n *Node) {
	// This is what the old list1 function did;
	// the rest of the compiler has come to expect it.
	if n.Op == OBLOCK && nodeSeqLen(n.Ninit) == 0 {
		l := n.List
		setNodeSeq(&n.List, nil)
		setNodeSeq(a, l)
		return
	}

	switch a := a.(type) {
	case **NodeList:
		*a = list1(n)
	case *Nodes:
		a.Set([]*Node{n})
	case *[]*Node:
		*a = []*Node{n}
	default:
		panic("can't happen")
	}
}

// appendNodeSeq appends the node sequence b to the node sequence a.
// a must have type **NodeList, *Nodes, or *[]*Node.
// b must have type *NodeList, Nodes, or []*Node.
// This is an interim function during the transition from NodeList to Nodes.
// TODO(iant): Remove when transition is complete.
func appendNodeSeq(a nodesOrNodeListPtr, b nodesOrNodeList) {
	// Simplify b to either *NodeList or []*Node.
	if n, ok := b.(Nodes); ok {
		b = n.Slice()
	}

	if l, ok := a.(**NodeList); ok {
		switch b := b.(type) {
		case *NodeList:
			*l = concat(*l, b)
		case []*Node:
			for _, n := range b {
				*l = list(*l, n)
			}
		default:
			panic("can't happen")
		}
	} else {
		var s []*Node
		switch a := a.(type) {
		case *Nodes:
			s = a.Slice()
		case *[]*Node:
			s = *a
		default:
			panic("can't happen")
		}

		switch b := b.(type) {
		case *NodeList:
			for l := b; l != nil; l = l.Next {
				s = append(s, l.N)
			}
		case []*Node:
			s = append(s, b...)
		default:
			panic("can't happen")
		}

		switch a := a.(type) {
		case *Nodes:
			a.Set(s)
		case *[]*Node:
			*a = s
		default:
			panic("can't happen")
		}
	}
}

// appendNodeSeqNode appends n to the node sequence a.
// a must have type **NodeList, *Nodes, or *[]*Node.
// This is an interim function during the transition from NodeList to Nodes.
// TODO(iant): Remove when transition is complete.
func appendNodeSeqNode(a nodesOrNodeListPtr, n *Node) {
	// This is what the old list1 function did;
	// the rest of the compiler has come to expect it.
	if n.Op == OBLOCK && nodeSeqLen(n.Ninit) == 0 {
		l := n.List
		setNodeSeq(&n.List, nil)
		appendNodeSeq(a, l)
		return
	}

	switch a := a.(type) {
	case **NodeList:
		*a = list(*a, n)
	case *Nodes:
		a.Append(n)
	case *[]*Node:
		*a = append(*a, n)
	default:
		panic("can't happen")
	}
}
