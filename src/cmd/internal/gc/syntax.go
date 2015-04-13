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
	Ntest *Node
	Nincr *Node
	Ninit *NodeList
	Nbody *NodeList
	Nelse *NodeList
	List  *NodeList
	Rlist *NodeList

	Op          uint8
	Nointerface bool
	Ullman      uint8 // sethi/ullman number
	Addable     bool  // addressable
	Etype       uint8 // op for OASOP, etype for OTYPE, exclam for export
	Bounded     bool  // bounds check unnecessary
	Class       uint8 // PPARAM, PAUTO, PEXTERN, etc
	Method      bool  // OCALLMETH is direct method call
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
	Readonly    bool
	Implicit    bool
	Addrtaken   bool  // address taken, even if not moved to heap
	Assigned    bool  // is the variable ever assigned to
	Captured    bool  // is the variable captured by a closure
	Byval       bool  // is the variable captured by value or by reference
	Reslice     bool  // this is a reslice x = x[0:y] or x = append(x, ...)
	Likely      int8  // likeliness of if statement
	Hasbreak    bool  // has break statement
	Needzero    bool  // if it contains pointers, needs to be zeroed on function entry
	Esc         uint8 // EscXXX
	Funcdepth   int32

	// most nodes
	Type  *Type
	Orig  *Node // original form, for printing, and tracking copies of ONAMEs
	Nname *Node

	// func
	Func *Func

	// OLITERAL
	Val Val

	// OREGISTER, OINDREG
	Reg int16

	// ONAME
	Ntype     *Node
	Defn      *Node // ONAME: initializing assignment; OLABEL: labeled statement
	Pack      *Node // real package for import . names
	Curfn     *Node // function for local variables
	Paramfld  *Type // TFIELD for this PPARAM; also for ODOT, curfn
	Decldepth int   // declaration loop depth, increased for every loop or label

	// ONAME func param with PHEAP
	Heapaddr   *Node // temp holding heap address of param
	Outerexpr  *Node // expression copied into closure for variable
	Stackparam *Node // OPARAM node referring to stack copy of param
	Alloc      *Node // allocation call

	// ONAME closure param with PPARAMREF
	Outer   *Node // outer PPARAMREF in nested closure
	Closure *Node // ONAME/PHEAP <-> ONAME/PPARAMREF
	Top     int   // top context (Ecall, Eproc, etc)

	// ONAME substitute while inlining
	Inlvar *Node

	// OPACK
	Pkg *Pkg

	// OARRAYLIT, OMAPLIT, OSTRUCTLIT.
	Initplan *InitPlan

	// Escape analysis.
	Escflowsrc   *NodeList // flow(this, src)
	Escretval    *NodeList // on OCALLxxx, list of dummy return values
	Escloopdepth int       // -1: global, 0: return variables, 1:function top level, increased inside function for every loop or label to mark scopes

	Sym      *Sym  // various
	Vargen   int32 // unique name for OTYPE/ONAME
	Lineno   int32
	Xoffset  int64
	Stkdelta int64 // offset added by stack frame compaction phase.
	Ostk     int32 // 6g only
	Iota     int32
	Walkgen  uint32
	Esclevel int32
	Opt      interface{} // for optimization passes
}

// Func holds Node fields used only with function-like nodes.
type Func struct {
	Shortname *Node
	Enter     *NodeList
	Exit      *NodeList
	Cvars     *NodeList // closure params
	Dcl       *NodeList // autodcl for this func/closure
	Inldcl    *NodeList // copy of dcl for use in inlining
	Closgen   int
	Outerfunc *Node

	Inl     *NodeList // copy of the body for use in inlining
	InlCost int32

	Endlineno int32

	Nosplit        bool // func should not execute on separate stack
	Nowritebarrier bool // emit compiler error instead of write barrier
	Dupok          bool // duplicate definitions ok
	Wrapper        bool // is method wrapper
	Needctxt       bool // function uses context register (has closure variables)
}

// Node ops.
const (
	OXXX = iota

	// names
	ONAME    // var, const or func name
	ONONAME  // unnamed arg or return value: f(int, string) (int, error) { etc }
	OTYPE    // type name
	OPACK    // import
	OLITERAL // literal

	// expressions
	OADD             // x + y
	OSUB             // x - y
	OOR              // x | y
	OXOR             // x ^ y
	OADDSTR          // s + "foo"
	OADDR            // &x
	OANDAND          // b0 && b1
	OAPPEND          // append
	OARRAYBYTESTR    // string(bytes)
	OARRAYBYTESTRTMP // string(bytes) ephemeral
	OARRAYRUNESTR    // string(runes)
	OSTRARRAYBYTE    // []byte(s)
	OSTRARRAYBYTETMP // []byte(s) ephemeral
	OSTRARRAYRUNE    // []rune(s)
	OAS              // x = y or x := y
	OAS2             // x, y, z = xx, yy, zz
	OAS2FUNC         // x, y = f()
	OAS2RECV         // x, ok = <-c
	OAS2MAPR         // x, ok = m["foo"]
	OAS2DOTTYPE      // x, ok = I.(int)
	OASOP            // x += y
	OCALL            // function call, method call or type conversion, possibly preceded by defer or go.
	OCALLFUNC        // f()
	OCALLMETH        // t.Method()
	OCALLINTER       // err.Error()
	OCALLPART        // t.Method (without ())
	OCAP             // cap
	OCLOSE           // close
	OCLOSURE         // f = func() { etc }
	OCMPIFACE        // err1 == err2
	OCMPSTR          // s1 == s2
	OCOMPLIT         // composite literal, typechecking may convert to a more specific OXXXLIT.
	OMAPLIT          // M{"foo":3, "bar":4}
	OSTRUCTLIT       // T{x:3, y:4}
	OARRAYLIT        // [2]int{3, 4}
	OPTRLIT          // &T{x:3, y:4}
	OCONV            // var i int; var u uint; i = int(u)
	OCONVIFACE       // I(t)
	OCONVNOP         // type Int int; var i int; var j Int; i = int(j)
	OCOPY            // copy
	ODCL             // var x int
	ODCLFUNC         // func f() or func (r) f()
	ODCLFIELD        // struct field, interface field, or func/method argument/return value.
	ODCLCONST        // const pi = 3.14
	ODCLTYPE         // type Int int
	ODELETE          // delete
	ODOT             // t.x
	ODOTPTR          // p.x that is implicitly (*p).x
	ODOTMETH         // t.Method
	ODOTINTER        // err.Error
	OXDOT            // t.x, typechecking may convert to a more specific ODOTXXX.
	ODOTTYPE         // e = err.(MyErr)
	ODOTTYPE2        // e, ok = err.(MyErr)
	OEQ              // x == y
	ONE              // x != y
	OLT              // x < y
	OLE              // x <= y
	OGE              // x >= y
	OGT              // x > y
	OIND             // *p
	OINDEX           // a[i]
	OINDEXMAP        // m[s]
	OKEY             // The x:3 in t{x:3, y:4}, the 1:2 in a[1:2], the 2:20 in [3]int{2:20}, etc.
	OPARAM           // The on-stack copy of a parameter or return value that escapes.
	OLEN             // len
	OMAKE            // make, typechecking may convert to a more specific OMAKEXXX.
	OMAKECHAN        // make(chan int)
	OMAKEMAP         // make(map[string]int)
	OMAKESLICE       // make([]int, 0)
	OMUL             // *
	ODIV             // x / y
	OMOD             // x % y
	OLSH             // x << u
	ORSH             // x >> u
	OAND             // x & y
	OANDNOT          // x &^ y
	ONEW             // new
	ONOT             // !b
	OCOM             // ^x
	OPLUS            // +x
	OMINUS           // -y
	OOROR            // b1 || b2
	OPANIC           // panic
	OPRINT           // print
	OPRINTN          // println
	OPAREN           // (x)
	OSEND            // c <- x
	OSLICE           // v[1:2], typechecking may convert to a more specific OSLICEXXX.
	OSLICEARR        // a[1:2]
	OSLICESTR        // s[1:2]
	OSLICE3          // v[1:2:3], typechecking may convert to OSLICE3ARR.
	OSLICE3ARR       // a[1:2:3]
	ORECOVER         // recover
	ORECV            // <-c
	ORUNESTR         // string(i)
	OSELRECV         // case x = <-c:
	OSELRECV2        // case x, ok = <-c:
	OIOTA            // iota
	OREAL            // real
	OIMAG            // imag
	OCOMPLEX         // complex

	// statements
	OBLOCK    // block of code
	OBREAK    // break
	OCASE     // case, after being verified by swt.c's casebody.
	OXCASE    // case, before verification.
	OCONTINUE // continue
	ODEFER    // defer
	OEMPTY    // no-op
	OFALL     // fallthrough, after being verified by swt.c's casebody.
	OXFALL    // fallthrough, before verification.
	OFOR      // for
	OGOTO     // goto
	OIF       // if
	OLABEL    // label:
	OPROC     // go
	ORANGE    // range
	ORETURN   // return
	OSELECT   // select
	OSWITCH   // switch x
	OTYPESW   // switch err.(type)

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
	OSQRT   // sqrt(float64), on systems that have hw support
	OGETG   // runtime.getg() (read g pointer)

	OEND
)

/*
 * Every node has a walkgen field.
 * If you want to do a traversal of a node graph that
 * might contain duplicates and want to avoid
 * visiting the same nodes twice, increment walkgen
 * before starting.  Then before processing a node, do
 *
 *	if(n->walkgen == walkgen)
 *		return;
 *	n->walkgen = walkgen;
 *
 * Such a walk cannot call another such walk recursively,
 * because of the use of the global walkgen.
 */
var walkgen uint32

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

// listsort sorts *l in place according to the 3-way comparison function f.
// The algorithm is mergesort, so it is guaranteed to be O(n log n).
func listsort(l **NodeList, f func(*Node, *Node) int) {
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
	listsort(&l1, f)
	listsort(&l2, f)

	if f(l1.N, l2.N) < 0 {
		*l = l1
	} else {
		*l = l2
		l2 = l1
		l1 = *l
	}

	// now l1 == *l; and l1 < l2

	var le *NodeList
	for (l1 != nil) && (l2 != nil) {
		for (l1.Next != nil) && f(l1.Next.N, l2.N) < 0 {
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
