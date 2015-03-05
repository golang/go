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
	Left           *Node
	Right          *Node
	Ntest          *Node
	Nincr          *Node
	Ninit          *NodeList
	Nbody          *NodeList
	Nelse          *NodeList
	List           *NodeList
	Rlist          *NodeList
	Op             uint8
	Nointerface    bool
	Ullman         uint8
	Addable        uint8
	Trecur         uint8
	Etype          uint8
	Bounded        bool
	Class          uint8
	Method         uint8
	Embedded       uint8
	Colas          uint8
	Diag           uint8
	Noescape       bool
	Nosplit        bool
	Builtin        uint8
	Nowritebarrier bool
	Walkdef        uint8
	Typecheck      uint8
	Local          uint8
	Dodata         uint8
	Initorder      uint8
	Used           uint8
	Isddd          uint8
	Readonly       uint8
	Implicit       uint8
	Addrtaken      uint8
	Assigned       uint8
	Captured       uint8
	Byval          uint8
	Dupok          uint8
	Wrapper        uint8
	Reslice        uint8
	Likely         int8
	Hasbreak       uint8
	Needzero       bool
	Needctxt       bool
	Esc            uint
	Funcdepth      int
	Type           *Type
	Orig           *Node
	Nname          *Node
	Shortname      *Node
	Enter          *NodeList
	Exit           *NodeList
	Cvars          *NodeList
	Dcl            *NodeList
	Inl            *NodeList
	Inldcl         *NodeList
	Closgen        int
	Outerfunc      *Node
	Val            Val
	Ntype          *Node
	Defn           *Node
	Pack           *Node
	Curfn          *Node
	Paramfld       *Type
	Decldepth      int
	Heapaddr       *Node
	Outerexpr      *Node
	Stackparam     *Node
	Alloc          *Node
	Outer          *Node
	Closure        *Node
	Top            int
	Inlvar         *Node
	Pkg            *Pkg
	Initplan       *InitPlan
	Escflowsrc     *NodeList
	Escretval      *NodeList
	Escloopdepth   int
	Sym            *Sym
	InlCost        int32
	Vargen         int32
	Lineno         int32
	Endlineno      int32
	Xoffset        int64
	Stkdelta       int64
	Ostk           int32
	Iota           int32
	Walkgen        uint32
	Esclevel       int32
	Opt            interface{}
}

// Node ops.
const (
	OXXX = iota
	ONAME
	ONONAME
	OTYPE
	OPACK
	OLITERAL
	OADD
	OSUB
	OOR
	OXOR
	OADDSTR
	OADDR
	OANDAND
	OAPPEND
	OARRAYBYTESTR
	OARRAYBYTESTRTMP
	OARRAYRUNESTR
	OSTRARRAYBYTE
	OSTRARRAYBYTETMP
	OSTRARRAYRUNE
	OAS
	OAS2
	OAS2FUNC
	OAS2RECV
	OAS2MAPR
	OAS2DOTTYPE
	OASOP
	OCALL
	OCALLFUNC
	OCALLMETH
	OCALLINTER
	OCALLPART
	OCAP
	OCLOSE
	OCLOSURE
	OCMPIFACE
	OCMPSTR
	OCOMPLIT
	OMAPLIT
	OSTRUCTLIT
	OARRAYLIT
	OPTRLIT
	OCONV
	OCONVIFACE
	OCONVNOP
	OCOPY
	ODCL
	ODCLFUNC
	ODCLFIELD
	ODCLCONST
	ODCLTYPE
	ODELETE
	ODOT
	ODOTPTR
	ODOTMETH
	ODOTINTER
	OXDOT
	ODOTTYPE
	ODOTTYPE2
	OEQ
	ONE
	OLT
	OLE
	OGE
	OGT
	OIND
	OINDEX
	OINDEXMAP
	OKEY
	OPARAM
	OLEN
	OMAKE
	OMAKECHAN
	OMAKEMAP
	OMAKESLICE
	OMUL
	ODIV
	OMOD
	OLSH
	ORSH
	OAND
	OANDNOT
	ONEW
	ONOT
	OCOM
	OPLUS
	OMINUS
	OOROR
	OPANIC
	OPRINT
	OPRINTN
	OPAREN
	OSEND
	OSLICE
	OSLICEARR
	OSLICESTR
	OSLICE3
	OSLICE3ARR
	ORECOVER
	ORECV
	ORUNESTR
	OSELRECV
	OSELRECV2
	OIOTA
	OREAL
	OIMAG
	OCOMPLEX
	OBLOCK
	OBREAK
	OCASE
	OXCASE
	OCONTINUE
	ODEFER
	OEMPTY
	OFALL
	OXFALL
	OFOR
	OGOTO
	OIF
	OLABEL
	OPROC
	ORANGE
	ORETURN
	OSELECT
	OSWITCH
	OTYPESW
	OTCHAN
	OTMAP
	OTSTRUCT
	OTINTER
	OTFUNC
	OTARRAY
	ODDD
	ODDDARG
	OINLCALL
	OEFACE
	OITAB
	OSPTR
	OCLOSUREVAR
	OCFUNC
	OCHECKNIL
	OVARKILL
	OREGISTER
	OINDREG
	OCMP
	ODEC
	OINC
	OEXTEND
	OHMUL
	OLROT
	ORROTC
	ORETJMP
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
