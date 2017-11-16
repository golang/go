// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"fmt"
	"strings"
)

// The instrument pass modifies the code tree for instrumentation.
//
// For flag_race it modifies the function as follows:
//
// 1. It inserts a call to racefuncenterfp at the beginning of each function.
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

// Only insert racefuncenterfp/racefuncexit into the following packages.
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
	if ispkgin(omit_pkgs) || fn.Func.Pragma&Norace != 0 {
		return
	}

	if !flag_race || !ispkgin(norace_inst_pkgs) {
		instrumentlist(fn.Nbody, nil)

		// nothing interesting for race detector in fn->enter
		instrumentlist(fn.Func.Exit, nil)
	}

	if flag_race {
		// nodpc is the PC of the caller as extracted by
		// getcallerpc. We use -widthptr(FP) for x86.
		// BUG: this will not work on arm.
		nodpc := *nodfp
		nodpc.Type = types.Types[TUINTPTR]
		nodpc.Xoffset = int64(-Widthptr)
		savedLineno := lineno
		lineno = src.NoXPos
		nd := mkcall("racefuncenter", nil, nil, &nodpc)

		fn.Func.Enter.Prepend(nd)
		nd = mkcall("racefuncexit", nil, nil)
		fn.Func.Exit.Append(nd)
		fn.Func.Dcl = append(fn.Func.Dcl, &nodpc)
		lineno = savedLineno
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

func instrumentlist(l Nodes, init *Nodes) {
	s := l.Slice()
	for i := range s {
		var instr Nodes
		instrumentnode(&s[i], &instr, 0, 0)
		if init == nil {
			s[i].Ninit.AppendNodes(&instr)
		} else {
			init.AppendNodes(&instr)
		}
	}
}

// walkexpr and walkstmt combined
// walks the tree and adds calls to the
// instrumentation code to top-level (statement) nodes' init
func instrumentnode(np **Node, init *Nodes, wr int, skip int) {
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

		n.Ninit.Set(nil)
		instrumentlist(l, nil)
		instrumentnode(&n, &l, wr, skip) // recurse with nil n->ninit
		appendinit(&n, l)
		*np = n
		return
	}

	instrumentlist(n.Ninit, nil)

	switch n.Op {
	default:
		Fatalf("instrument: unknown node type %v", n.Op)

	case OAS, OAS2FUNC:
		instrumentnode(&n.Left, init, 1, 0)
		instrumentnode(&n.Right, init, 0, 0)

		// can't matter
	case OCFUNC, OVARKILL, OVARLIVE:

	case OBLOCK:
		ls := n.List.Slice()
		afterCall := false
		for i := range ls {
			op := ls[i].Op
			// Scan past OAS nodes copying results off stack.
			// Those must not be instrumented, because the
			// instrumentation calls will smash the results.
			// The assignments are to temporaries, so they cannot
			// be involved in races and need not be instrumented.
			if afterCall && op == OAS && iscallret(ls[i].Right) {
				continue
			}
			instrumentnode(&ls[i], &ls[i].Ninit, 0, 0)
			afterCall = (op == OCALLFUNC || op == OCALLMETH || op == OCALLINTER)
		}

	case ODEFER:
		instrumentnode(&n.Left, init, 0, 0)

	case OPROC:
		instrumentnode(&n.Left, init, 0, 0)

	case OCALLINTER:
		instrumentnode(&n.Left, init, 0, 0)

	case OCALLFUNC:
		// Note that runtime.typedslicecopy is the only
		// assignment-like function call in the AST at this
		// point (between walk and SSA); since we don't
		// instrument it here, typedslicecopy is manually
		// instrumented in runtime. Calls to the write barrier
		// and typedmemmove are created later by SSA, so those
		// still appear as OAS nodes at this point.
		instrumentnode(&n.Left, init, 0, 0)

	case ONOT,
		OMINUS,
		OPLUS,
		OREAL,
		OIMAG,
		OCOM:
		instrumentnode(&n.Left, init, wr, 0)

	case ODOTINTER:
		instrumentnode(&n.Left, init, 0, 0)

	case ODOT:
		instrumentnode(&n.Left, init, 0, 1)
		callinstr(&n, init, wr, skip)

	case ODOTPTR: // dst = (*x).f with implicit *; otherwise it's ODOT+OIND
		instrumentnode(&n.Left, init, 0, 0)

		callinstr(&n, init, wr, skip)

	case OIND: // *p
		instrumentnode(&n.Left, init, 0, 0)

		callinstr(&n, init, wr, skip)

	case OSPTR, OLEN, OCAP:
		instrumentnode(&n.Left, init, 0, 0)
		if n.Left.Type.IsMap() {
			n1 := nod(OCONVNOP, n.Left, nil)
			n1.Type = types.NewPtr(types.Types[TUINT8])
			n1 = nod(OIND, n1, nil)
			n1 = typecheck(n1, Erv)
			callinstr(&n1, init, 0, skip)
		}

	case OLSH,
		ORSH,
		OAND,
		OANDNOT,
		OOR,
		OXOR,
		OSUB,
		OMUL,
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

	case OANDAND, OOROR:
		instrumentnode(&n.Left, init, wr, 0)

		// walk has ensured the node has moved to a location where
		// side effects are safe.
		// n->right may not be executed,
		// so instrumentation goes to n->right->ninit, not init.
		instrumentnode(&n.Right, &n.Right.Ninit, wr, 0)

	case ONAME:
		callinstr(&n, init, wr, skip)

	case OCONV:
		instrumentnode(&n.Left, init, wr, 0)

	case OCONVNOP:
		instrumentnode(&n.Left, init, wr, 0)

	case ODIV, OMOD:
		instrumentnode(&n.Left, init, wr, 0)
		instrumentnode(&n.Right, init, wr, 0)

	case OINDEX:
		if !n.Left.Type.IsArray() {
			instrumentnode(&n.Left, init, 0, 0)
		} else if !islvalue(n.Left) {
			// index of unaddressable array, like Map[k][i].
			instrumentnode(&n.Left, init, wr, 0)

			instrumentnode(&n.Right, init, 0, 0)
			break
		}

		instrumentnode(&n.Right, init, 0, 0)
		if !n.Left.Type.IsString() {
			callinstr(&n, init, wr, skip)
		}

	case OSLICE, OSLICEARR, OSLICE3, OSLICE3ARR, OSLICESTR:
		instrumentnode(&n.Left, init, 0, 0)
		low, high, max := n.SliceBounds()
		instrumentnode(&low, init, 0, 0)
		instrumentnode(&high, init, 0, 0)
		instrumentnode(&max, init, 0, 0)
		n.SetSliceBounds(low, high, max)

	case OADDR:
		instrumentnode(&n.Left, init, 0, 1)

		// n->left is Type* which is not interesting.
	case OEFACE:
		instrumentnode(&n.Right, init, 0, 0)

	case OITAB, OIDATA:
		instrumentnode(&n.Left, init, 0, 0)

	case OSTRARRAYBYTETMP:
		instrumentnode(&n.Left, init, 0, 0)

	case OAS2DOTTYPE:
		instrumentnode(&n.Left, init, 1, 0)
		instrumentnode(&n.Right, init, 0, 0)

	case ODOTTYPE, ODOTTYPE2:
		instrumentnode(&n.Left, init, 0, 0)

		// should not appear in AST by now
	case OSEND,
		ORECV,
		OCLOSE,
		ONEW,
		OXCASE,
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
		OCALLPART,
		// lowered to PTRLIT
		OCLOSURE,  // lowered to PTRLIT
		ORANGE,    // lowered to ordinary for loop
		OARRAYLIT, // lowered to assignments
		OSLICELIT,
		OMAPLIT,
		OSTRUCTLIT,
		OAS2,
		OAS2RECV,
		OAS2MAPR,
		OASOP:
		Fatalf("instrument: %v must be lowered by now", n.Op)

	case OGETG:
		Fatalf("instrument: OGETG can happen only in runtime which we don't instrument")

	case OFOR, OFORUNTIL:
		if n.Left != nil {
			instrumentnode(&n.Left, &n.Left.Ninit, 0, 0)
		}
		if n.Right != nil {
			instrumentnode(&n.Right, &n.Right.Ninit, 0, 0)
		}

	case OIF, OSWITCH:
		if n.Left != nil {
			instrumentnode(&n.Left, &n.Left.Ninit, 0, 0)
		}

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

		// does not require instrumentation
	case OPRINT, // don't bother instrumenting it
		OPRINTN,     // don't bother instrumenting it
		OCHECKNIL,   // always followed by a read.
		OCLOSUREVAR, // immutable pointer to captured variable
		ODOTMETH,    // either part of CALLMETH or CALLPART (lowered to PTRLIT)
		OINDREGSP,   // at this stage, only n(SP) nodes from nodarg
		ODCL,        // declarations (without value) cannot be races
		ODCLCONST,
		ODCLTYPE,
		OTYPE,
		ONONAME,
		OLITERAL,
		OTYPESW: // ignored by code generation, do not instrument.
	}

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
		if n.IsAutoTmp() {
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

func callinstr(np **Node, init *Nodes, wr int, skip int) bool {
	n := *np

	//fmt.Printf("callinstr for %v [ %v ] etype=%v class=%v\n",
	//	n, n.Op, n.Type.Etype, n.Class)

	if skip != 0 || n.Type == nil || n.Type.Etype >= TIDEAL {
		return false
	}
	t := n.Type
	// dowidth may not have been called for PEXTERN.
	dowidth(t)
	w := t.Width
	if w == BADWIDTH {
		Fatalf("instrument: %v badwidth", t)
	}
	if w == 0 {
		return false // can't race on zero-sized things
	}
	if isartificial(n) {
		return false
	}

	b := outervalue(n)

	// it skips e.g. stores to ... parameter array
	if isartificial(b) {
		return false
	}
	class := b.Class()

	// BUG: we _may_ want to instrument PAUTO sometimes
	// e.g. if we've got a local variable/method receiver
	// that has got a pointer inside. Whether it points to
	// the heap or not is impossible to know at compile time
	if class == PAUTOHEAP || class == PEXTERN || b.Op == OINDEX || b.Op == ODOTPTR || b.Op == OIND {
		hasCalls := false
		inspect(n, func(n *Node) bool {
			switch n.Op {
			case OCALL, OCALLFUNC, OCALLMETH, OCALLINTER:
				hasCalls = true
			}
			return !hasCalls
		})
		if hasCalls {
			n = detachexpr(n, init)
			*np = n
		}

		n = treecopy(n, src.NoXPos)
		makeaddable(n)
		var f *Node
		if flag_msan {
			name := "msanread"
			if wr != 0 {
				name = "msanwrite"
			}
			f = mkcall(name, nil, init, uintptraddr(n), nodintconst(w))
		} else if flag_race && t.NumComponents() > 1 {
			// for composite objects we have to write every address
			// because a write might happen to any subobject.
			// composites with only one element don't have subobjects, though.
			name := "racereadrange"
			if wr != 0 {
				name = "racewriterange"
			}
			f = mkcall(name, nil, init, uintptraddr(n), nodintconst(w))
		} else if flag_race {
			// for non-composite objects we can write just the start
			// address, as any write must write the first byte.
			name := "raceread"
			if wr != 0 {
				name = "racewrite"
			}
			f = mkcall(name, nil, init, uintptraddr(n))
		}

		init.Append(f)
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
		if n.Left.Type.IsArray() {
			makeaddable(n.Left)
		}

		// Turn T(v).Field into v.Field
	case ODOT, OXDOT:
		if n.Left.Op == OCONVNOP {
			n.Left = n.Left.Left
		}
		makeaddable(n.Left)

		// nothing to do
	}
}

func uintptraddr(n *Node) *Node {
	r := nod(OADDR, n, nil)
	r.SetBounded(true)
	r = conv(r, types.Types[TUNSAFEPTR])
	r = conv(r, types.Types[TUINTPTR])
	return r
}

func detachexpr(n *Node, init *Nodes) *Node {
	addr := nod(OADDR, n, nil)
	l := temp(types.NewPtr(n.Type))
	as := nod(OAS, l, addr)
	as = typecheck(as, Etop)
	as = walkexpr(as, init)
	init.Append(as)
	ind := nod(OIND, l, nil)
	ind = typecheck(ind, Erv)
	ind = walkexpr(ind, init)
	return ind
}

// appendinit is like addinit in subr.go
// but appends rather than prepends.
func appendinit(np **Node, init Nodes) {
	if init.Len() == 0 {
		return
	}

	n := *np
	switch n.Op {
	// There may be multiple refs to this node;
	// introduce OCONVNOP to hold init list.
	case ONAME, OLITERAL:
		n = nod(OCONVNOP, n, nil)

		n.Type = n.Left.Type
		n.SetTypecheck(1)
		*np = n
	}

	n.Ninit.AppendNodes(&init)
	n.SetHasCall(true)
}
