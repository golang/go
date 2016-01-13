// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"fmt"
	"strings"
)

// The instrument pass modifies the code tree for instrumentation.
//
// For flag_race it modifies the function as follows:
//
// 1. It inserts a call to racefuncenter at the beginning of each function.
// 2. It inserts a call to racefuncexit at the end of each function.
// 3. It inserts a call to raceread before each memory read.
// 4. It inserts a call to racewrite before each memory write.
//
// For flag_msan:
//
// 1. It inserts a call to msanread before each memory read.
// 2. It inserts a call to msanwrite before each memory write.
//
// The rewriting is not yet complete. Certain nodes are not rewritten
// but should be.

// TODO(dvyukov): do not instrument initialization as writes:
// a := make([]int, 10)

// Do not instrument the following packages at all,
// at best instrumentation would cause infinite recursion.
var omit_pkgs = []string{"runtime/internal/atomic", "runtime/internal/sys", "runtime", "runtime/race", "runtime/msan"}

// Only insert racefuncenter/racefuncexit into the following packages.
// Memory accesses in the packages are either uninteresting or will cause false positives.
var norace_inst_pkgs = []string{"sync", "sync/atomic"}

func ispkgin(pkgs []string) bool {
	if myimportpath != "" {
		for _, p := range pkgs {
			if myimportpath == p {
				return true
			}
		}
	}

	return false
}

func instrument(fn *Node) {
	if ispkgin(omit_pkgs) || fn.Func.Norace {
		return
	}

	if flag_race == 0 || !ispkgin(norace_inst_pkgs) {
		instrumentlist(fn.Nbody, nil)

		// nothing interesting for race detector in fn->enter
		instrumentlist(fn.Func.Exit, nil)
	}

	if flag_race != 0 {
		// nodpc is the PC of the caller as extracted by
		// getcallerpc. We use -widthptr(FP) for x86.
		// BUG: this will not work on arm.
		nodpc := Nod(OXXX, nil, nil)

		*nodpc = *nodfp
		nodpc.Type = Types[TUINTPTR]
		nodpc.Xoffset = int64(-Widthptr)
		nd := mkcall("racefuncenter", nil, nil, nodpc)
		fn.Func.Enter = concat(list1(nd), fn.Func.Enter)
		nd = mkcall("racefuncexit", nil, nil)
		fn.Func.Exit = list(fn.Func.Exit, nd)
	}

	if Debug['W'] != 0 {
		s := fmt.Sprintf("after instrument %v", fn.Func.Nname.Sym)
		dumplist(s, fn.Nbody)
		s = fmt.Sprintf("enter %v", fn.Func.Nname.Sym)
		dumplist(s, fn.Func.Enter)
		s = fmt.Sprintf("exit %v", fn.Func.Nname.Sym)
		dumplist(s, fn.Func.Exit)
	}
}

func instrumentlist(l *NodeList, init **NodeList) {
	var instr *NodeList

	for ; l != nil; l = l.Next {
		instr = nil
		instrumentnode(&l.N, &instr, 0, 0)
		if init == nil {
			l.N.Ninit = concat(l.N.Ninit, instr)
		} else {
			*init = concat(*init, instr)
		}
	}
}

// walkexpr and walkstmt combined
// walks the tree and adds calls to the
// instrumentation code to top-level (statement) nodes' init
func instrumentnode(np **Node, init **NodeList, wr int, skip int) {
	n := *np

	if n == nil {
		return
	}

	if Debug['w'] > 1 {
		Dump("instrument-before", n)
	}
	setlineno(n)
	if init == nil {
		Fatalf("instrument: bad init list")
	}
	if init == &n.Ninit {
		// If init == &n->ninit and n->ninit is non-nil,
		// instrumentnode might append it to itself.
		// nil it out and handle it separately before putting it back.
		l := n.Ninit

		n.Ninit = nil
		instrumentlist(l, nil)
		instrumentnode(&n, &l, wr, skip) // recurse with nil n->ninit
		appendinit(&n, l)
		*np = n
		return
	}

	instrumentlist(n.Ninit, nil)

	switch n.Op {
	default:
		Fatalf("instrument: unknown node type %v", Oconv(int(n.Op), 0))

	case OAS, OASWB, OAS2FUNC:
		instrumentnode(&n.Left, init, 1, 0)
		instrumentnode(&n.Right, init, 0, 0)
		goto ret

		// can't matter
	case OCFUNC, OVARKILL, OVARLIVE:
		goto ret

	case OBLOCK:
		var out *NodeList
		for l := n.List; l != nil; l = l.Next {
			switch l.N.Op {
			case OCALLFUNC, OCALLMETH, OCALLINTER:
				instrumentnode(&l.N, &l.N.Ninit, 0, 0)
				out = list(out, l.N)
				// Scan past OAS nodes copying results off stack.
				// Those must not be instrumented, because the
				// instrumentation calls will smash the results.
				// The assignments are to temporaries, so they cannot
				// be involved in races and need not be instrumented.
				for l.Next != nil && l.Next.N.Op == OAS && iscallret(l.Next.N.Right) {
					l = l.Next
					out = list(out, l.N)
				}
			default:
				instrumentnode(&l.N, &out, 0, 0)
				out = list(out, l.N)
			}
		}
		n.List = out
		goto ret

	case ODEFER:
		instrumentnode(&n.Left, init, 0, 0)
		goto ret

	case OPROC:
		instrumentnode(&n.Left, init, 0, 0)
		goto ret

	case OCALLINTER:
		instrumentnode(&n.Left, init, 0, 0)
		goto ret

		// Instrument dst argument of runtime.writebarrier* calls
	// as we do not instrument runtime code.
	// typedslicecopy is instrumented in runtime.
	case OCALLFUNC:
		instrumentnode(&n.Left, init, 0, 0)
		goto ret

	case ONOT,
		OMINUS,
		OPLUS,
		OREAL,
		OIMAG,
		OCOM,
		OSQRT:
		instrumentnode(&n.Left, init, wr, 0)
		goto ret

	case ODOTINTER:
		instrumentnode(&n.Left, init, 0, 0)
		goto ret

	case ODOT:
		instrumentnode(&n.Left, init, 0, 1)
		callinstr(&n, init, wr, skip)
		goto ret

	case ODOTPTR: // dst = (*x).f with implicit *; otherwise it's ODOT+OIND
		instrumentnode(&n.Left, init, 0, 0)

		callinstr(&n, init, wr, skip)
		goto ret

	case OIND: // *p
		instrumentnode(&n.Left, init, 0, 0)

		callinstr(&n, init, wr, skip)
		goto ret

	case OSPTR, OLEN, OCAP:
		instrumentnode(&n.Left, init, 0, 0)
		if Istype(n.Left.Type, TMAP) {
			n1 := Nod(OCONVNOP, n.Left, nil)
			n1.Type = Ptrto(Types[TUINT8])
			n1 = Nod(OIND, n1, nil)
			typecheck(&n1, Erv)
			callinstr(&n1, init, 0, skip)
		}

		goto ret

	case OLSH,
		ORSH,
		OLROT,
		OAND,
		OANDNOT,
		OOR,
		OXOR,
		OSUB,
		OMUL,
		OHMUL,
		OEQ,
		ONE,
		OLT,
		OLE,
		OGE,
		OGT,
		OADD,
		OCOMPLEX:
		instrumentnode(&n.Left, init, wr, 0)
		instrumentnode(&n.Right, init, wr, 0)
		goto ret

	case OANDAND, OOROR:
		instrumentnode(&n.Left, init, wr, 0)

		// walk has ensured the node has moved to a location where
		// side effects are safe.
		// n->right may not be executed,
		// so instrumentation goes to n->right->ninit, not init.
		instrumentnode(&n.Right, &n.Right.Ninit, wr, 0)

		goto ret

	case ONAME:
		callinstr(&n, init, wr, skip)
		goto ret

	case OCONV:
		instrumentnode(&n.Left, init, wr, 0)
		goto ret

	case OCONVNOP:
		instrumentnode(&n.Left, init, wr, 0)
		goto ret

	case ODIV, OMOD:
		instrumentnode(&n.Left, init, wr, 0)
		instrumentnode(&n.Right, init, wr, 0)
		goto ret

	case OINDEX:
		if !Isfixedarray(n.Left.Type) {
			instrumentnode(&n.Left, init, 0, 0)
		} else if !islvalue(n.Left) {
			// index of unaddressable array, like Map[k][i].
			instrumentnode(&n.Left, init, wr, 0)

			instrumentnode(&n.Right, init, 0, 0)
			goto ret
		}

		instrumentnode(&n.Right, init, 0, 0)
		if n.Left.Type.Etype != TSTRING {
			callinstr(&n, init, wr, skip)
		}
		goto ret

	case OSLICE, OSLICEARR, OSLICE3, OSLICE3ARR, OSLICESTR:
		instrumentnode(&n.Left, init, 0, 0)
		instrumentnode(&n.Right, init, 0, 0)
		goto ret

	case OKEY:
		instrumentnode(&n.Left, init, 0, 0)
		instrumentnode(&n.Right, init, 0, 0)
		goto ret

	case OADDR:
		instrumentnode(&n.Left, init, 0, 1)
		goto ret

		// n->left is Type* which is not interesting.
	case OEFACE:
		instrumentnode(&n.Right, init, 0, 0)

		goto ret

	case OITAB:
		instrumentnode(&n.Left, init, 0, 0)
		goto ret

		// should not appear in AST by now
	case OSEND,
		ORECV,
		OCLOSE,
		ONEW,
		OXCASE,
		OXFALL,
		OCASE,
		OPANIC,
		ORECOVER,
		OCONVIFACE,
		OCMPIFACE,
		OMAKECHAN,
		OMAKEMAP,
		OMAKESLICE,
		OCALL,
		OCOPY,
		OAPPEND,
		ORUNESTR,
		OARRAYBYTESTR,
		OARRAYRUNESTR,
		OSTRARRAYBYTE,
		OSTRARRAYRUNE,
		OINDEXMAP,
		// lowered to call
		OCMPSTR,
		OADDSTR,
		ODOTTYPE,
		ODOTTYPE2,
		OAS2DOTTYPE,
		OCALLPART,
		// lowered to PTRLIT
		OCLOSURE,  // lowered to PTRLIT
		ORANGE,    // lowered to ordinary for loop
		OARRAYLIT, // lowered to assignments
		OMAPLIT,
		OSTRUCTLIT,
		OAS2,
		OAS2RECV,
		OAS2MAPR,
		OASOP:
		Yyerror("instrument: %v must be lowered by now", Oconv(int(n.Op), 0))

		goto ret

		// impossible nodes: only appear in backend.
	case ORROTC, OEXTEND:
		Yyerror("instrument: %v cannot exist now", Oconv(int(n.Op), 0))
		goto ret

	case OGETG:
		Yyerror("instrument: OGETG can happen only in runtime which we don't instrument")
		goto ret

	case OFOR:
		if n.Left != nil {
			instrumentnode(&n.Left, &n.Left.Ninit, 0, 0)
		}
		if n.Right != nil {
			instrumentnode(&n.Right, &n.Right.Ninit, 0, 0)
		}
		goto ret

	case OIF, OSWITCH:
		if n.Left != nil {
			instrumentnode(&n.Left, &n.Left.Ninit, 0, 0)
		}
		goto ret

		// just do generic traversal
	case OCALLMETH,
		ORETURN,
		ORETJMP,
		OSELECT,
		OEMPTY,
		OBREAK,
		OCONTINUE,
		OFALL,
		OGOTO,
		OLABEL:
		goto ret

		// does not require instrumentation
	case OPRINT, // don't bother instrumenting it
		OPRINTN,     // don't bother instrumenting it
		OCHECKNIL,   // always followed by a read.
		OPARAM,      // it appears only in fn->exit to copy heap params back
		OCLOSUREVAR, // immutable pointer to captured variable
		ODOTMETH,    // either part of CALLMETH or CALLPART (lowered to PTRLIT)
		OINDREG,     // at this stage, only n(SP) nodes from nodarg
		ODCL,        // declarations (without value) cannot be races
		ODCLCONST,
		ODCLTYPE,
		OTYPE,
		ONONAME,
		OLITERAL,
		OTYPESW: // ignored by code generation, do not instrument.
		goto ret
	}

ret:
	if n.Op != OBLOCK { // OBLOCK is handled above in a special way.
		instrumentlist(n.List, init)
	}
	instrumentlist(n.Nbody, nil)
	instrumentlist(n.Rlist, nil)
	*np = n
}

func isartificial(n *Node) bool {
	// compiler-emitted artificial things that we do not want to instrument,
	// can't possibly participate in a data race.
	// can't be seen by C/C++ and therefore irrelevant for msan.
	if n.Op == ONAME && n.Sym != nil && n.Sym.Name != "" {
		if n.Sym.Name == "_" {
			return true
		}

		// autotmp's are always local
		if strings.HasPrefix(n.Sym.Name, "autotmp_") {
			return true
		}

		// statictmp's are read-only
		if strings.HasPrefix(n.Sym.Name, "statictmp_") {
			return true
		}

		// go.itab is accessed only by the compiler and runtime (assume safe)
		if n.Sym.Pkg != nil && n.Sym.Pkg.Name != "" && n.Sym.Pkg.Name == "go.itab" {
			return true
		}
	}

	return false
}

func callinstr(np **Node, init **NodeList, wr int, skip int) bool {
	n := *np

	//print("callinstr for %+N [ %O ] etype=%E class=%d\n",
	//	  n, n->op, n->type ? n->type->etype : -1, n->class);

	if skip != 0 || n.Type == nil || n.Type.Etype >= TIDEAL {
		return false
	}
	t := n.Type
	if isartificial(n) {
		return false
	}

	b := outervalue(n)

	// it skips e.g. stores to ... parameter array
	if isartificial(b) {
		return false
	}
	class := b.Class

	// BUG: we _may_ want to instrument PAUTO sometimes
	// e.g. if we've got a local variable/method receiver
	// that has got a pointer inside. Whether it points to
	// the heap or not is impossible to know at compile time
	if (class&PHEAP != 0) || class == PPARAMREF || class == PEXTERN || b.Op == OINDEX || b.Op == ODOTPTR || b.Op == OIND {
		hascalls := 0
		foreach(n, hascallspred, &hascalls)
		if hascalls != 0 {
			n = detachexpr(n, init)
			*np = n
		}

		n = treecopy(n, 0)
		makeaddable(n)
		var f *Node
		if flag_msan != 0 {
			name := "msanread"
			if wr != 0 {
				name = "msanwrite"
			}
			// dowidth may not have been called for PEXTERN.
			dowidth(t)
			w := t.Width
			if w == BADWIDTH {
				Fatalf("instrument: %v badwidth", t)
			}
			f = mkcall(name, nil, init, uintptraddr(n), Nodintconst(w))
		} else if flag_race != 0 && (t.Etype == TSTRUCT || Isfixedarray(t)) {
			name := "racereadrange"
			if wr != 0 {
				name = "racewriterange"
			}
			// dowidth may not have been called for PEXTERN.
			dowidth(t)
			w := t.Width
			if w == BADWIDTH {
				Fatalf("instrument: %v badwidth", t)
			}
			f = mkcall(name, nil, init, uintptraddr(n), Nodintconst(w))
		} else if flag_race != 0 {
			name := "raceread"
			if wr != 0 {
				name = "racewrite"
			}
			f = mkcall(name, nil, init, uintptraddr(n))
		}

		*init = list(*init, f)
		return true
	}

	return false
}

// makeaddable returns a node whose memory location is the
// same as n, but which is addressable in the Go language
// sense.
// This is different from functions like cheapexpr that may make
// a copy of their argument.
func makeaddable(n *Node) {
	// The arguments to uintptraddr technically have an address but
	// may not be addressable in the Go sense: for example, in the case
	// of T(v).Field where T is a struct type and v is
	// an addressable value.
	switch n.Op {
	case OINDEX:
		if Isfixedarray(n.Left.Type) {
			makeaddable(n.Left)
		}

		// Turn T(v).Field into v.Field
	case ODOT, OXDOT:
		if n.Left.Op == OCONVNOP {
			n.Left = n.Left.Left
		}
		makeaddable(n.Left)

		// nothing to do
	case ODOTPTR:
		fallthrough
	default:
		break
	}
}

func uintptraddr(n *Node) *Node {
	r := Nod(OADDR, n, nil)
	r.Bounded = true
	r = conv(r, Types[TUNSAFEPTR])
	r = conv(r, Types[TUINTPTR])
	return r
}

func detachexpr(n *Node, init **NodeList) *Node {
	addr := Nod(OADDR, n, nil)
	l := temp(Ptrto(n.Type))
	as := Nod(OAS, l, addr)
	typecheck(&as, Etop)
	walkexpr(&as, init)
	*init = list(*init, as)
	ind := Nod(OIND, l, nil)
	typecheck(&ind, Erv)
	walkexpr(&ind, init)
	return ind
}

func foreachnode(n *Node, f func(*Node, interface{}), c interface{}) {
	if n != nil {
		f(n, c)
	}
}

func foreachlist(l *NodeList, f func(*Node, interface{}), c interface{}) {
	for ; l != nil; l = l.Next {
		foreachnode(l.N, f, c)
	}
}

func foreach(n *Node, f func(*Node, interface{}), c interface{}) {
	foreachlist(n.Ninit, f, c)
	foreachnode(n.Left, f, c)
	foreachnode(n.Right, f, c)
	foreachlist(n.List, f, c)
	foreachlist(n.Nbody, f, c)
	foreachlist(n.Rlist, f, c)
}

func hascallspred(n *Node, c interface{}) {
	switch n.Op {
	case OCALL, OCALLFUNC, OCALLMETH, OCALLINTER:
		(*c.(*int))++
	}
}

// appendinit is like addinit in subr.go
// but appends rather than prepends.
func appendinit(np **Node, init *NodeList) {
	if init == nil {
		return
	}

	n := *np
	switch n.Op {
	// There may be multiple refs to this node;
	// introduce OCONVNOP to hold init list.
	case ONAME, OLITERAL:
		n = Nod(OCONVNOP, n, nil)

		n.Type = n.Left.Type
		n.Typecheck = 1
		*np = n
	}

	n.Ninit = concat(n.Ninit, init)
	n.Ullman = UINF
}
