// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/internal/obj"
	"fmt"
)

// static initialization
const (
	InitNotStarted = 0
	InitDone       = 1
	InitPending    = 2
)

var (
	initlist  []*Node
	initplans map[*Node]*InitPlan
	inittemps = make(map[*Node]*Node)
)

// init1 walks the AST starting at n, and accumulates in out
// the list of definitions needing init code in dependency order.
func init1(n *Node, out **NodeList) {
	if n == nil {
		return
	}
	init1(n.Left, out)
	init1(n.Right, out)
	for l := n.List; l != nil; l = l.Next {
		init1(l.N, out)
	}

	if n.Left != nil && n.Type != nil && n.Left.Op == OTYPE && n.Class == PFUNC {
		// Methods called as Type.Method(receiver, ...).
		// Definitions for method expressions are stored in type->nname.
		init1(n.Type.Nname, out)
	}

	if n.Op != ONAME {
		return
	}
	switch n.Class {
	case PEXTERN, PFUNC:
	default:
		if isblank(n) && n.Name.Curfn == nil && n.Name.Defn != nil && n.Name.Defn.Initorder == InitNotStarted {
			// blank names initialization is part of init() but not
			// when they are inside a function.
			break
		}
		return
	}

	if n.Initorder == InitDone {
		return
	}
	if n.Initorder == InitPending {
		// Since mutually recursive sets of functions are allowed,
		// we don't necessarily raise an error if n depends on a node
		// which is already waiting for its dependencies to be visited.
		//
		// initlist contains a cycle of identifiers referring to each other.
		// If this cycle contains a variable, then this variable refers to itself.
		// Conversely, if there exists an initialization cycle involving
		// a variable in the program, the tree walk will reach a cycle
		// involving that variable.
		if n.Class != PFUNC {
			foundinitloop(n, n)
		}

		for i := len(initlist) - 1; i >= 0; i-- {
			x := initlist[i]
			if x == n {
				break
			}
			if x.Class != PFUNC {
				foundinitloop(n, x)
			}
		}

		// The loop involves only functions, ok.
		return
	}

	// reached a new unvisited node.
	n.Initorder = InitPending
	initlist = append(initlist, n)

	// make sure that everything n depends on is initialized.
	// n->defn is an assignment to n
	if defn := n.Name.Defn; defn != nil {
		switch defn.Op {
		default:
			Dump("defn", defn)
			Fatalf("init1: bad defn")

		case ODCLFUNC:
			init2list(defn.Nbody, out)

		case OAS:
			if defn.Left != n {
				Dump("defn", defn)
				Fatalf("init1: bad defn")
			}
			if isblank(defn.Left) && candiscard(defn.Right) {
				defn.Op = OEMPTY
				defn.Left = nil
				defn.Right = nil
				break
			}

			init2(defn.Right, out)
			if Debug['j'] != 0 {
				fmt.Printf("%v\n", n.Sym)
			}
			if isblank(n) || !staticinit(n, out) {
				if Debug['%'] != 0 {
					Dump("nonstatic", defn)
				}
				*out = list(*out, defn)
			}

		case OAS2FUNC, OAS2MAPR, OAS2DOTTYPE, OAS2RECV:
			if defn.Initorder == InitDone {
				break
			}
			defn.Initorder = InitPending
			for l := defn.Rlist; l != nil; l = l.Next {
				init1(l.N, out)
			}
			if Debug['%'] != 0 {
				Dump("nonstatic", defn)
			}
			*out = list(*out, defn)
			defn.Initorder = InitDone
		}
	}

	last := len(initlist) - 1
	if initlist[last] != n {
		Fatalf("bad initlist %v", initlist)
	}
	initlist[last] = nil // allow GC
	initlist = initlist[:last]

	n.Initorder = InitDone
	return
}

// foundinitloop prints an init loop error and exits.
func foundinitloop(node, visited *Node) {
	// If there have already been errors printed,
	// those errors probably confused us and
	// there might not be a loop. Let the user
	// fix those first.
	Flusherrors()
	if nerrors > 0 {
		errorexit()
	}

	// Find the index of node and visited in the initlist.
	var nodeindex, visitedindex int
	for ; initlist[nodeindex] != node; nodeindex++ {
	}
	for ; initlist[visitedindex] != visited; visitedindex++ {
	}

	// There is a loop involving visited. We know about node and
	// initlist = n1 <- ... <- visited <- ... <- node <- ...
	fmt.Printf("%v: initialization loop:\n", visited.Line())

	// Print visited -> ... -> n1 -> node.
	for _, n := range initlist[visitedindex:] {
		fmt.Printf("\t%v %v refers to\n", n.Line(), n.Sym)
	}

	// Print node -> ... -> visited.
	for _, n := range initlist[nodeindex:visitedindex] {
		fmt.Printf("\t%v %v refers to\n", n.Line(), n.Sym)
	}

	fmt.Printf("\t%v %v\n", visited.Line(), visited.Sym)
	errorexit()
}

// recurse over n, doing init1 everywhere.
func init2(n *Node, out **NodeList) {
	if n == nil || n.Initorder == InitDone {
		return
	}

	if n.Op == ONAME && n.Ninit != nil {
		Fatalf("name %v with ninit: %v\n", n.Sym, Nconv(n, obj.FmtSign))
	}

	init1(n, out)
	init2(n.Left, out)
	init2(n.Right, out)
	init2list(n.Ninit, out)
	init2list(n.List, out)
	init2list(n.Rlist, out)
	init2list(n.Nbody, out)

	if n.Op == OCLOSURE {
		init2list(n.Func.Closure.Nbody, out)
	}
	if n.Op == ODOTMETH || n.Op == OCALLPART {
		init2(n.Type.Nname, out)
	}
}

func init2list(l *NodeList, out **NodeList) {
	for ; l != nil; l = l.Next {
		init2(l.N, out)
	}
}

func initreorder(l *NodeList, out **NodeList) {
	var n *Node

	for ; l != nil; l = l.Next {
		n = l.N
		switch n.Op {
		case ODCLFUNC, ODCLCONST, ODCLTYPE:
			continue
		}

		initreorder(n.Ninit, out)
		n.Ninit = nil
		init1(n, out)
	}
}

// initfix computes initialization order for a list l of top-level
// declarations and outputs the corresponding list of statements
// to include in the init() function body.
func initfix(l *NodeList) *NodeList {
	var lout *NodeList
	initplans = make(map[*Node]*InitPlan)
	lno := int(lineno)
	initreorder(l, &lout)
	lineno = int32(lno)
	initplans = nil
	return lout
}

// compilation of top-level (static) assignments
// into DATA statements if at all possible.
func staticinit(n *Node, out **NodeList) bool {
	if n.Op != ONAME || n.Class != PEXTERN || n.Name.Defn == nil || n.Name.Defn.Op != OAS {
		Fatalf("staticinit")
	}

	lineno = n.Lineno
	l := n.Name.Defn.Left
	r := n.Name.Defn.Right
	return staticassign(l, r, out)
}

// like staticassign but we are copying an already
// initialized value r.
func staticcopy(l *Node, r *Node, out **NodeList) bool {
	if r.Op != ONAME {
		return false
	}
	if r.Class == PFUNC {
		gdata(l, r, Widthptr)
		return true
	}
	if r.Class != PEXTERN || r.Sym.Pkg != localpkg {
		return false
	}
	if r.Name.Defn == nil { // probably zeroed but perhaps supplied externally and of unknown value
		return false
	}
	if r.Name.Defn.Op != OAS {
		return false
	}
	orig := r
	r = r.Name.Defn.Right

	for r.Op == OCONVNOP {
		r = r.Left
	}

	switch r.Op {
	case ONAME:
		if staticcopy(l, r, out) {
			return true
		}
		*out = list(*out, Nod(OAS, l, r))
		return true

	case OLITERAL:
		if iszero(r) {
			return true
		}
		gdata(l, r, int(l.Type.Width))
		return true

	case OADDR:
		switch r.Left.Op {
		case ONAME:
			gdata(l, r, int(l.Type.Width))
			return true
		}

	case OPTRLIT:
		switch r.Left.Op {
		//dump("not static addr", r);
		default:
			break

			// copy pointer
		case OARRAYLIT, OSTRUCTLIT, OMAPLIT:
			gdata(l, Nod(OADDR, inittemps[r], nil), int(l.Type.Width))

			return true
		}

	case OARRAYLIT:
		if Isslice(r.Type) {
			// copy slice
			a := inittemps[r]

			n := *l
			n.Xoffset = l.Xoffset + int64(Array_array)
			gdata(&n, Nod(OADDR, a, nil), Widthptr)
			n.Xoffset = l.Xoffset + int64(Array_nel)
			gdata(&n, r.Right, Widthint)
			n.Xoffset = l.Xoffset + int64(Array_cap)
			gdata(&n, r.Right, Widthint)
			return true
		}
		fallthrough

		// fall through
	case OSTRUCTLIT:
		p := initplans[r]

		n := *l
		for i := range p.E {
			e := &p.E[i]
			n.Xoffset = l.Xoffset + e.Xoffset
			n.Type = e.Expr.Type
			if e.Expr.Op == OLITERAL {
				gdata(&n, e.Expr, int(n.Type.Width))
			} else {
				ll := Nod(OXXX, nil, nil)
				*ll = n
				ll.Orig = ll // completely separate copy
				if !staticassign(ll, e.Expr, out) {
					// Requires computation, but we're
					// copying someone else's computation.
					rr := Nod(OXXX, nil, nil)

					*rr = *orig
					rr.Orig = rr // completely separate copy
					rr.Type = ll.Type
					rr.Xoffset += e.Xoffset
					setlineno(rr)
					*out = list(*out, Nod(OAS, ll, rr))
				}
			}
		}

		return true
	}

	return false
}

func staticassign(l *Node, r *Node, out **NodeList) bool {
	for r.Op == OCONVNOP {
		r = r.Left
	}

	switch r.Op {
	case ONAME:
		return staticcopy(l, r, out)

	case OLITERAL:
		if iszero(r) {
			return true
		}
		gdata(l, r, int(l.Type.Width))
		return true

	case OADDR:
		var nam Node
		if stataddr(&nam, r.Left) {
			n := *r
			n.Left = &nam
			gdata(l, &n, int(l.Type.Width))
			return true
		}
		fallthrough

	case OPTRLIT:
		switch r.Left.Op {
		case OARRAYLIT, OMAPLIT, OSTRUCTLIT:
			// Init pointer.
			a := staticname(r.Left.Type, 1)

			inittemps[r] = a
			gdata(l, Nod(OADDR, a, nil), int(l.Type.Width))

			// Init underlying literal.
			if !staticassign(a, r.Left, out) {
				*out = list(*out, Nod(OAS, a, r.Left))
			}
			return true
		}
		//dump("not static ptrlit", r);

	case OSTRARRAYBYTE:
		if l.Class == PEXTERN && r.Left.Op == OLITERAL {
			sval := r.Left.Val().U.(string)
			slicebytes(l, sval, len(sval))
			return true
		}

	case OARRAYLIT:
		initplan(r)
		if Isslice(r.Type) {
			// Init slice.
			ta := typ(TARRAY)

			ta.Type = r.Type.Type
			ta.Bound = Mpgetfix(r.Right.Val().U.(*Mpint))
			a := staticname(ta, 1)
			inittemps[r] = a
			n := *l
			n.Xoffset = l.Xoffset + int64(Array_array)
			gdata(&n, Nod(OADDR, a, nil), Widthptr)
			n.Xoffset = l.Xoffset + int64(Array_nel)
			gdata(&n, r.Right, Widthint)
			n.Xoffset = l.Xoffset + int64(Array_cap)
			gdata(&n, r.Right, Widthint)

			// Fall through to init underlying array.
			l = a
		}
		fallthrough

	case OSTRUCTLIT:
		initplan(r)

		p := initplans[r]
		n := *l
		for i := range p.E {
			e := &p.E[i]
			n.Xoffset = l.Xoffset + e.Xoffset
			n.Type = e.Expr.Type
			if e.Expr.Op == OLITERAL {
				gdata(&n, e.Expr, int(n.Type.Width))
			} else {
				setlineno(e.Expr)
				a := Nod(OXXX, nil, nil)
				*a = n
				a.Orig = a // completely separate copy
				if !staticassign(a, e.Expr, out) {
					*out = list(*out, Nod(OAS, a, e.Expr))
				}
			}
		}

		return true

	case OMAPLIT:
		// TODO: Table-driven map insert.
		break

	case OCLOSURE:
		if r.Func.Cvars == nil {
			// Closures with no captured variables are globals,
			// so the assignment can be done at link time.
			n := *l
			gdata(&n, r.Func.Closure.Func.Nname, Widthptr)
			return true
		}
	}

	//dump("not static", r);
	return false
}

// from here down is the walk analysis
// of composite literals.
// most of the work is to generate
// data statements for the constant
// part of the composite literal.
func staticname(t *Type, ctxt int) *Node {
	n := newname(Lookupf("statictmp_%.4d", statuniqgen))
	statuniqgen++
	if ctxt == 0 {
		n.Name.Readonly = true
	}
	addvar(n, t, PEXTERN)
	return n
}

func isliteral(n *Node) bool {
	if n.Op == OLITERAL {
		if n.Val().Ctype() != CTNIL {
			return true
		}
	}
	return false
}

func simplename(n *Node) bool {
	if n.Op != ONAME {
		return false
	}
	if !n.Addable {
		return false
	}
	if n.Class&PHEAP != 0 {
		return false
	}
	if n.Class == PPARAMREF {
		return false
	}
	return true
}

func litas(l *Node, r *Node, init **NodeList) {
	a := Nod(OAS, l, r)
	typecheck(&a, Etop)
	walkexpr(&a, init)
	*init = list(*init, a)
}

const (
	MODEDYNAM = 1
	MODECONST = 2
)

func getdyn(n *Node, top int) int {
	mode := 0
	switch n.Op {
	default:
		if isliteral(n) {
			return MODECONST
		}
		return MODEDYNAM

	case OARRAYLIT:
		if top == 0 && n.Type.Bound < 0 {
			return MODEDYNAM
		}
		fallthrough

	case OSTRUCTLIT:
		break
	}

	for nl := n.List; nl != nil; nl = nl.Next {
		value := nl.N.Right
		mode |= getdyn(value, 0)
		if mode == MODEDYNAM|MODECONST {
			break
		}
	}

	return mode
}

func structlit(ctxt int, pass int, n *Node, var_ *Node, init **NodeList) {
	for nl := n.List; nl != nil; nl = nl.Next {
		r := nl.N
		if r.Op != OKEY {
			Fatalf("structlit: rhs not OKEY: %v", r)
		}
		index := r.Left
		value := r.Right

		var a *Node

		switch value.Op {
		case OARRAYLIT:
			if value.Type.Bound < 0 {
				if pass == 1 && ctxt != 0 {
					a = Nod(ODOT, var_, newname(index.Sym))
					slicelit(ctxt, value, a, init)
				} else if pass == 2 && ctxt == 0 {
					a = Nod(ODOT, var_, newname(index.Sym))
					slicelit(ctxt, value, a, init)
				} else if pass == 3 {
					break
				}
				continue
			}

			a = Nod(ODOT, var_, newname(index.Sym))
			arraylit(ctxt, pass, value, a, init)
			continue

		case OSTRUCTLIT:
			a = Nod(ODOT, var_, newname(index.Sym))
			structlit(ctxt, pass, value, a, init)
			continue
		}

		if isliteral(value) {
			if pass == 2 {
				continue
			}
		} else if pass == 1 {
			continue
		}

		// build list of var.field = expr
		setlineno(value)
		a = Nod(ODOT, var_, newname(index.Sym))

		a = Nod(OAS, a, value)
		typecheck(&a, Etop)
		if pass == 1 {
			walkexpr(&a, init) // add any assignments in r to top
			if a.Op != OAS {
				Fatalf("structlit: not as")
			}
			a.Dodata = 2
		} else {
			orderstmtinplace(&a)
			walkstmt(&a)
		}

		*init = list(*init, a)
	}
}

func arraylit(ctxt int, pass int, n *Node, var_ *Node, init **NodeList) {
	for l := n.List; l != nil; l = l.Next {
		r := l.N
		if r.Op != OKEY {
			Fatalf("arraylit: rhs not OKEY: %v", r)
		}
		index := r.Left
		value := r.Right

		var a *Node

		switch value.Op {
		case OARRAYLIT:
			if value.Type.Bound < 0 {
				if pass == 1 && ctxt != 0 {
					a = Nod(OINDEX, var_, index)
					slicelit(ctxt, value, a, init)
				} else if pass == 2 && ctxt == 0 {
					a = Nod(OINDEX, var_, index)
					slicelit(ctxt, value, a, init)
				} else if pass == 3 {
					break
				}
				continue
			}

			a = Nod(OINDEX, var_, index)
			arraylit(ctxt, pass, value, a, init)
			continue

		case OSTRUCTLIT:
			a = Nod(OINDEX, var_, index)
			structlit(ctxt, pass, value, a, init)
			continue
		}

		if isliteral(index) && isliteral(value) {
			if pass == 2 {
				continue
			}
		} else if pass == 1 {
			continue
		}

		// build list of var[index] = value
		setlineno(value)
		a = Nod(OINDEX, var_, index)

		a = Nod(OAS, a, value)
		typecheck(&a, Etop)
		if pass == 1 {
			walkexpr(&a, init)
			if a.Op != OAS {
				Fatalf("arraylit: not as")
			}
			a.Dodata = 2
		} else {
			orderstmtinplace(&a)
			walkstmt(&a)
		}

		*init = list(*init, a)
	}
}

func slicelit(ctxt int, n *Node, var_ *Node, init **NodeList) {
	// make an array type
	t := shallow(n.Type)

	t.Bound = Mpgetfix(n.Right.Val().U.(*Mpint))
	t.Width = 0
	t.Sym = nil
	t.Haspointers = 0
	dowidth(t)

	if ctxt != 0 {
		// put everything into static array
		vstat := staticname(t, ctxt)

		arraylit(ctxt, 1, n, vstat, init)
		arraylit(ctxt, 2, n, vstat, init)

		// copy static to slice
		a := Nod(OSLICE, vstat, Nod(OKEY, nil, nil))

		a = Nod(OAS, var_, a)
		typecheck(&a, Etop)
		a.Dodata = 2
		*init = list(*init, a)
		return
	}

	// recipe for var = []t{...}
	// 1. make a static array
	//	var vstat [...]t
	// 2. assign (data statements) the constant part
	//	vstat = constpart{}
	// 3. make an auto pointer to array and allocate heap to it
	//	var vauto *[...]t = new([...]t)
	// 4. copy the static array to the auto array
	//	*vauto = vstat
	// 5. assign slice of allocated heap to var
	//	var = [0:]*auto
	// 6. for each dynamic part assign to the slice
	//	var[i] = dynamic part
	//
	// an optimization is done if there is no constant part
	//	3. var vauto *[...]t = new([...]t)
	//	5. var = [0:]*auto
	//	6. var[i] = dynamic part

	// if the literal contains constants,
	// make static initialized array (1),(2)
	var vstat *Node

	mode := getdyn(n, 1)
	if mode&MODECONST != 0 {
		vstat = staticname(t, ctxt)
		arraylit(ctxt, 1, n, vstat, init)
	}

	// make new auto *array (3 declare)
	vauto := temp(Ptrto(t))

	// set auto to point at new temp or heap (3 assign)
	var a *Node
	if x := prealloc[n]; x != nil {
		// temp allocated during order.go for dddarg
		x.Type = t

		if vstat == nil {
			a = Nod(OAS, x, nil)
			typecheck(&a, Etop)
			*init = list(*init, a) // zero new temp
		}

		a = Nod(OADDR, x, nil)
	} else if n.Esc == EscNone {
		a = temp(t)
		if vstat == nil {
			a = Nod(OAS, temp(t), nil)
			typecheck(&a, Etop)
			*init = list(*init, a) // zero new temp
			a = a.Left
		}

		a = Nod(OADDR, a, nil)
	} else {
		a = Nod(ONEW, nil, nil)
		a.List = list1(typenod(t))
	}

	a = Nod(OAS, vauto, a)
	typecheck(&a, Etop)
	walkexpr(&a, init)
	*init = list(*init, a)

	if vstat != nil {
		// copy static to heap (4)
		a = Nod(OIND, vauto, nil)

		a = Nod(OAS, a, vstat)
		typecheck(&a, Etop)
		walkexpr(&a, init)
		*init = list(*init, a)
	}

	// make slice out of heap (5)
	a = Nod(OAS, var_, Nod(OSLICE, vauto, Nod(OKEY, nil, nil)))

	typecheck(&a, Etop)
	orderstmtinplace(&a)
	walkstmt(&a)
	*init = list(*init, a)

	// put dynamics into slice (6)
	for l := n.List; l != nil; l = l.Next {
		r := l.N
		if r.Op != OKEY {
			Fatalf("slicelit: rhs not OKEY: %v", r)
		}
		index := r.Left
		value := r.Right
		a := Nod(OINDEX, var_, index)
		a.Bounded = true

		// TODO need to check bounds?

		switch value.Op {
		case OARRAYLIT:
			if value.Type.Bound < 0 {
				break
			}
			arraylit(ctxt, 2, value, a, init)
			continue

		case OSTRUCTLIT:
			structlit(ctxt, 2, value, a, init)
			continue
		}

		if isliteral(index) && isliteral(value) {
			continue
		}

		// build list of var[c] = expr
		setlineno(value)
		a = Nod(OAS, a, value)

		typecheck(&a, Etop)
		orderstmtinplace(&a)
		walkstmt(&a)
		*init = list(*init, a)
	}
}

func maplit(ctxt int, n *Node, var_ *Node, init **NodeList) {
	ctxt = 0

	// make the map var
	nerr := nerrors

	a := Nod(OMAKE, nil, nil)
	a.List = list1(typenod(n.Type))
	litas(var_, a, init)

	// count the initializers
	b := int64(0)

	for l := n.List; l != nil; l = l.Next {
		r := l.N
		if r.Op != OKEY {
			Fatalf("maplit: rhs not OKEY: %v", r)
		}
		index := r.Left
		value := r.Right

		if isliteral(index) && isliteral(value) {
			b++
		}
	}

	if b != 0 {
		// build type [count]struct { a Tindex, b Tvalue }
		t := n.Type

		tk := t.Down
		tv := t.Type

		symb := Lookup("b")
		t = typ(TFIELD)
		t.Type = tv
		t.Sym = symb

		syma := Lookup("a")
		t1 := t
		t = typ(TFIELD)
		t.Type = tk
		t.Sym = syma
		t.Down = t1

		t1 = t
		t = typ(TSTRUCT)
		t.Type = t1

		t1 = t
		t = typ(TARRAY)
		t.Bound = b
		t.Type = t1

		dowidth(t)

		// make and initialize static array
		vstat := staticname(t, ctxt)

		b := int64(0)
		for l := n.List; l != nil; l = l.Next {
			r := l.N

			if r.Op != OKEY {
				Fatalf("maplit: rhs not OKEY: %v", r)
			}
			index := r.Left
			value := r.Right

			if isliteral(index) && isliteral(value) {
				// build vstat[b].a = key;
				setlineno(index)
				a = Nodintconst(b)

				a = Nod(OINDEX, vstat, a)
				a = Nod(ODOT, a, newname(syma))
				a = Nod(OAS, a, index)
				typecheck(&a, Etop)
				walkexpr(&a, init)
				a.Dodata = 2
				*init = list(*init, a)

				// build vstat[b].b = value;
				setlineno(value)
				a = Nodintconst(b)

				a = Nod(OINDEX, vstat, a)
				a = Nod(ODOT, a, newname(symb))
				a = Nod(OAS, a, value)
				typecheck(&a, Etop)
				walkexpr(&a, init)
				a.Dodata = 2
				*init = list(*init, a)

				b++
			}
		}

		// loop adding structure elements to map
		// for i = 0; i < len(vstat); i++ {
		//	map[vstat[i].a] = vstat[i].b
		// }
		index := temp(Types[TINT])

		a = Nod(OINDEX, vstat, index)
		a.Bounded = true
		a = Nod(ODOT, a, newname(symb))

		r := Nod(OINDEX, vstat, index)
		r.Bounded = true
		r = Nod(ODOT, r, newname(syma))
		r = Nod(OINDEX, var_, r)

		r = Nod(OAS, r, a)

		a = Nod(OFOR, nil, nil)
		a.Nbody = list1(r)

		a.Ninit = list1(Nod(OAS, index, Nodintconst(0)))
		a.Left = Nod(OLT, index, Nodintconst(t.Bound))
		a.Right = Nod(OAS, index, Nod(OADD, index, Nodintconst(1)))

		typecheck(&a, Etop)
		walkstmt(&a)
		*init = list(*init, a)
	}

	// put in dynamic entries one-at-a-time
	var key *Node

	var val *Node
	for l := n.List; l != nil; l = l.Next {
		r := l.N

		if r.Op != OKEY {
			Fatalf("maplit: rhs not OKEY: %v", r)
		}
		index := r.Left
		value := r.Right

		if isliteral(index) && isliteral(value) {
			continue
		}

		// build list of var[c] = expr.
		// use temporary so that mapassign1 can have addressable key, val.
		if key == nil {
			key = temp(var_.Type.Down)
			val = temp(var_.Type.Type)
		}

		setlineno(r.Left)
		a = Nod(OAS, key, r.Left)
		typecheck(&a, Etop)
		walkstmt(&a)
		*init = list(*init, a)
		setlineno(r.Right)
		a = Nod(OAS, val, r.Right)
		typecheck(&a, Etop)
		walkstmt(&a)
		*init = list(*init, a)

		setlineno(val)
		a = Nod(OAS, Nod(OINDEX, var_, key), val)
		typecheck(&a, Etop)
		walkstmt(&a)
		*init = list(*init, a)

		if nerr != nerrors {
			break
		}
	}

	if key != nil {
		a = Nod(OVARKILL, key, nil)
		typecheck(&a, Etop)
		*init = list(*init, a)
		a = Nod(OVARKILL, val, nil)
		typecheck(&a, Etop)
		*init = list(*init, a)
	}
}

func anylit(ctxt int, n *Node, var_ *Node, init **NodeList) {
	t := n.Type
	switch n.Op {
	default:
		Fatalf("anylit: not lit")

	case OPTRLIT:
		if !Isptr[t.Etype] {
			Fatalf("anylit: not ptr")
		}

		var r *Node
		if n.Right != nil {
			r = Nod(OADDR, n.Right, nil)
			typecheck(&r, Erv)
		} else {
			r = Nod(ONEW, nil, nil)
			r.Typecheck = 1
			r.Type = t
			r.Esc = n.Esc
		}

		walkexpr(&r, init)
		a := Nod(OAS, var_, r)

		typecheck(&a, Etop)
		*init = list(*init, a)

		var_ = Nod(OIND, var_, nil)
		typecheck(&var_, Erv|Easgn)
		anylit(ctxt, n.Left, var_, init)

	case OSTRUCTLIT:
		if t.Etype != TSTRUCT {
			Fatalf("anylit: not struct")
		}

		if simplename(var_) && count(n.List) > 4 {
			if ctxt == 0 {
				// lay out static data
				vstat := staticname(t, ctxt)

				structlit(ctxt, 1, n, vstat, init)

				// copy static to var
				a := Nod(OAS, var_, vstat)

				typecheck(&a, Etop)
				walkexpr(&a, init)
				*init = list(*init, a)

				// add expressions to automatic
				structlit(ctxt, 2, n, var_, init)

				break
			}

			structlit(ctxt, 1, n, var_, init)
			structlit(ctxt, 2, n, var_, init)
			break
		}

		// initialize of not completely specified
		if simplename(var_) || count(n.List) < structcount(t) {
			a := Nod(OAS, var_, nil)
			typecheck(&a, Etop)
			walkexpr(&a, init)
			*init = list(*init, a)
		}

		structlit(ctxt, 3, n, var_, init)

	case OARRAYLIT:
		if t.Etype != TARRAY {
			Fatalf("anylit: not array")
		}
		if t.Bound < 0 {
			slicelit(ctxt, n, var_, init)
			break
		}

		if simplename(var_) && count(n.List) > 4 {
			if ctxt == 0 {
				// lay out static data
				vstat := staticname(t, ctxt)

				arraylit(1, 1, n, vstat, init)

				// copy static to automatic
				a := Nod(OAS, var_, vstat)

				typecheck(&a, Etop)
				walkexpr(&a, init)
				*init = list(*init, a)

				// add expressions to automatic
				arraylit(ctxt, 2, n, var_, init)

				break
			}

			arraylit(ctxt, 1, n, var_, init)
			arraylit(ctxt, 2, n, var_, init)
			break
		}

		// initialize of not completely specified
		if simplename(var_) || int64(count(n.List)) < t.Bound {
			a := Nod(OAS, var_, nil)
			typecheck(&a, Etop)
			walkexpr(&a, init)
			*init = list(*init, a)
		}

		arraylit(ctxt, 3, n, var_, init)

	case OMAPLIT:
		if t.Etype != TMAP {
			Fatalf("anylit: not map")
		}
		maplit(ctxt, n, var_, init)
	}
}

func oaslit(n *Node, init **NodeList) bool {
	if n.Left == nil || n.Right == nil {
		// not a special composit literal assignment
		return false
	}
	if n.Left.Type == nil || n.Right.Type == nil {
		// not a special composit literal assignment
		return false
	}
	if !simplename(n.Left) {
		// not a special composit literal assignment
		return false
	}
	if !Eqtype(n.Left.Type, n.Right.Type) {
		// not a special composit literal assignment
		return false
	}

	// context is init() function.
	// implies generated data executed
	// exactly once and not subject to races.
	ctxt := 0

	//	if(n->dodata == 1)
	//		ctxt = 1;

	switch n.Right.Op {
	default:
		// not a special composit literal assignment
		return false

	case OSTRUCTLIT, OARRAYLIT, OMAPLIT:
		if vmatch1(n.Left, n.Right) {
			// not a special composit literal assignment
			return false
		}
		anylit(ctxt, n.Right, n.Left, init)
	}

	n.Op = OEMPTY
	n.Right = nil
	return true
}

func getlit(lit *Node) int {
	if Smallintconst(lit) {
		return int(Mpgetfix(lit.Val().U.(*Mpint)))
	}
	return -1
}

func stataddr(nam *Node, n *Node) bool {
	if n == nil {
		return false
	}

	switch n.Op {
	case ONAME:
		*nam = *n
		return n.Addable

	case ODOT:
		if !stataddr(nam, n.Left) {
			break
		}
		nam.Xoffset += n.Xoffset
		nam.Type = n.Type
		return true

	case OINDEX:
		if n.Left.Type.Bound < 0 {
			break
		}
		if !stataddr(nam, n.Left) {
			break
		}
		l := getlit(n.Right)
		if l < 0 {
			break
		}

		// Check for overflow.
		if n.Type.Width != 0 && Thearch.MAXWIDTH/n.Type.Width <= int64(l) {
			break
		}
		nam.Xoffset += int64(l) * n.Type.Width
		nam.Type = n.Type
		return true
	}

	return false
}

func initplan(n *Node) {
	if initplans[n] != nil {
		return
	}
	p := new(InitPlan)
	initplans[n] = p
	switch n.Op {
	default:
		Fatalf("initplan")

	case OARRAYLIT:
		for l := n.List; l != nil; l = l.Next {
			a := l.N
			if a.Op != OKEY || !Smallintconst(a.Left) {
				Fatalf("initplan arraylit")
			}
			addvalue(p, n.Type.Type.Width*Mpgetfix(a.Left.Val().U.(*Mpint)), nil, a.Right)
		}

	case OSTRUCTLIT:
		for l := n.List; l != nil; l = l.Next {
			a := l.N
			if a.Op != OKEY || a.Left.Type == nil {
				Fatalf("initplan structlit")
			}
			addvalue(p, a.Left.Type.Width, nil, a.Right)
		}

	case OMAPLIT:
		for l := n.List; l != nil; l = l.Next {
			a := l.N
			if a.Op != OKEY {
				Fatalf("initplan maplit")
			}
			addvalue(p, -1, a.Left, a.Right)
		}
	}
}

func addvalue(p *InitPlan, xoffset int64, key *Node, n *Node) {
	// special case: zero can be dropped entirely
	if iszero(n) {
		p.Zero += n.Type.Width
		return
	}

	// special case: inline struct and array (not slice) literals
	if isvaluelit(n) {
		initplan(n)
		q := initplans[n]
		for _, qe := range q.E {
			e := entry(p)
			*e = qe
			e.Xoffset += xoffset
		}
		return
	}

	// add to plan
	if n.Op == OLITERAL {
		p.Lit += n.Type.Width
	} else {
		p.Expr += n.Type.Width
	}

	e := entry(p)
	e.Xoffset = xoffset
	e.Expr = n
}

func iszero(n *Node) bool {
	switch n.Op {
	case OLITERAL:
		switch n.Val().Ctype() {
		default:
			Dump("unexpected literal", n)
			Fatalf("iszero")

		case CTNIL:
			return true

		case CTSTR:
			return n.Val().U.(string) == ""

		case CTBOOL:
			return !n.Val().U.(bool)

		case CTINT, CTRUNE:
			return mpcmpfixc(n.Val().U.(*Mpint), 0) == 0

		case CTFLT:
			return mpcmpfltc(n.Val().U.(*Mpflt), 0) == 0

		case CTCPLX:
			return mpcmpfltc(&n.Val().U.(*Mpcplx).Real, 0) == 0 && mpcmpfltc(&n.Val().U.(*Mpcplx).Imag, 0) == 0
		}

	case OARRAYLIT:
		if Isslice(n.Type) {
			break
		}
		fallthrough

		// fall through
	case OSTRUCTLIT:
		for l := n.List; l != nil; l = l.Next {
			if !iszero(l.N.Right) {
				return false
			}
		}
		return true
	}

	return false
}

func isvaluelit(n *Node) bool {
	return (n.Op == OARRAYLIT && Isfixedarray(n.Type)) || n.Op == OSTRUCTLIT
}

func entry(p *InitPlan) *InitEntry {
	p.E = append(p.E, InitEntry{})
	return &p.E[len(p.E)-1]
}

func gen_as_init(n *Node) bool {
	var nr *Node
	var nl *Node
	var nam Node

	if n.Dodata == 0 {
		goto no
	}

	nr = n.Right
	nl = n.Left
	if nr == nil {
		var nam Node
		if !stataddr(&nam, nl) {
			goto no
		}
		if nam.Class != PEXTERN {
			goto no
		}
		return true
	}

	if nr.Type == nil || !Eqtype(nl.Type, nr.Type) {
		goto no
	}

	if !stataddr(&nam, nl) {
		goto no
	}

	if nam.Class != PEXTERN {
		goto no
	}

	switch nr.Op {
	default:
		goto no

	case OCONVNOP:
		nr = nr.Left
		if nr == nil || nr.Op != OSLICEARR {
			goto no
		}
		fallthrough

		// fall through
	case OSLICEARR:
		if nr.Right.Op == OKEY && nr.Right.Left == nil && nr.Right.Right == nil {
			nr = nr.Left
			gused(nil) // in case the data is the dest of a goto
			nl := nr
			if nr == nil || nr.Op != OADDR {
				goto no
			}
			nr = nr.Left
			if nr == nil || nr.Op != ONAME {
				goto no
			}

			// nr is the array being converted to a slice
			if nr.Type == nil || nr.Type.Etype != TARRAY || nr.Type.Bound < 0 {
				goto no
			}

			nam.Xoffset += int64(Array_array)
			gdata(&nam, nl, int(Types[Tptr].Width))

			nam.Xoffset += int64(Array_nel) - int64(Array_array)
			var nod1 Node
			Nodconst(&nod1, Types[TINT], nr.Type.Bound)
			gdata(&nam, &nod1, Widthint)

			nam.Xoffset += int64(Array_cap) - int64(Array_nel)
			gdata(&nam, &nod1, Widthint)

			return true
		}

		goto no

	case OLITERAL:
		break
	}

	switch nr.Type.Etype {
	default:
		goto no

	case TBOOL,
		TINT8,
		TUINT8,
		TINT16,
		TUINT16,
		TINT32,
		TUINT32,
		TINT64,
		TUINT64,
		TINT,
		TUINT,
		TUINTPTR,
		TPTR32,
		TPTR64,
		TFLOAT32,
		TFLOAT64:
		gdata(&nam, nr, int(nr.Type.Width))

	case TCOMPLEX64, TCOMPLEX128:
		gdatacomplex(&nam, nr.Val().U.(*Mpcplx))

	case TSTRING:
		gdatastring(&nam, nr.Val().U.(string))
	}

	return true

no:
	if n.Dodata == 2 {
		Dump("\ngen_as_init", n)
		Fatalf("gen_as_init couldnt make data statement")
	}

	return false
}
