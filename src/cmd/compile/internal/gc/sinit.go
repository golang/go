// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import "fmt"

// static initialization
const (
	InitNotStarted = 0
	InitDone       = 1
	InitPending    = 2
)

type InitEntry struct {
	Xoffset int64 // struct, array only
	Expr    *Node // bytes of run-time computed expressions
}

type InitPlan struct {
	E []InitEntry
}

var (
	initlist  []*Node
	initplans map[*Node]*InitPlan
	inittemps = make(map[*Node]*Node)
)

// init1 walks the AST starting at n, and accumulates in out
// the list of definitions needing init code in dependency order.
func init1(n *Node, out *[]*Node) {
	if n == nil {
		return
	}
	init1(n.Left, out)
	init1(n.Right, out)
	for _, n1 := range n.List.Slice() {
		init1(n1, out)
	}

	if n.Left != nil && n.Type != nil && n.Left.Op == OTYPE && n.Class == PFUNC {
		// Methods called as Type.Method(receiver, ...).
		// Definitions for method expressions are stored in type->nname.
		init1(n.Type.Nname(), out)
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
				*out = append(*out, defn)
			}

		case OAS2FUNC, OAS2MAPR, OAS2DOTTYPE, OAS2RECV:
			if defn.Initorder == InitDone {
				break
			}
			defn.Initorder = InitPending
			for _, n2 := range defn.Rlist.Slice() {
				init1(n2, out)
			}
			if Debug['%'] != 0 {
				Dump("nonstatic", defn)
			}
			*out = append(*out, defn)
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
	flusherrors()
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
func init2(n *Node, out *[]*Node) {
	if n == nil || n.Initorder == InitDone {
		return
	}

	if n.Op == ONAME && n.Ninit.Len() != 0 {
		Fatalf("name %v with ninit: %+v\n", n.Sym, n)
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
		init2(n.Type.Nname(), out)
	}
}

func init2list(l Nodes, out *[]*Node) {
	for _, n := range l.Slice() {
		init2(n, out)
	}
}

func initreorder(l []*Node, out *[]*Node) {
	var n *Node
	for _, n = range l {
		switch n.Op {
		case ODCLFUNC, ODCLCONST, ODCLTYPE:
			continue
		}

		initreorder(n.Ninit.Slice(), out)
		n.Ninit.Set(nil)
		init1(n, out)
	}
}

// initfix computes initialization order for a list l of top-level
// declarations and outputs the corresponding list of statements
// to include in the init() function body.
func initfix(l []*Node) []*Node {
	var lout []*Node
	initplans = make(map[*Node]*InitPlan)
	lno := lineno
	initreorder(l, &lout)
	lineno = lno
	initplans = nil
	return lout
}

// compilation of top-level (static) assignments
// into DATA statements if at all possible.
func staticinit(n *Node, out *[]*Node) bool {
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
func staticcopy(l *Node, r *Node, out *[]*Node) bool {
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

	for r.Op == OCONVNOP && !eqtype(r.Type, l.Type) {
		r = r.Left
	}

	switch r.Op {
	case ONAME:
		if staticcopy(l, r, out) {
			return true
		}
		// We may have skipped past one or more OCONVNOPs, so
		// use conv to ensure r is assignable to l (#13263).
		*out = append(*out, nod(OAS, l, conv(r, l.Type)))
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
		case OARRAYLIT, OSLICELIT, OSTRUCTLIT, OMAPLIT:
			// copy pointer
			gdata(l, nod(OADDR, inittemps[r], nil), int(l.Type.Width))
			return true
		}

	case OSLICELIT:
		// copy slice
		a := inittemps[r]

		n := *l
		n.Xoffset = l.Xoffset + int64(array_array)
		gdata(&n, nod(OADDR, a, nil), Widthptr)
		n.Xoffset = l.Xoffset + int64(array_nel)
		gdata(&n, r.Right, Widthint)
		n.Xoffset = l.Xoffset + int64(array_cap)
		gdata(&n, r.Right, Widthint)
		return true

	case OARRAYLIT, OSTRUCTLIT:
		p := initplans[r]

		n := *l
		for i := range p.E {
			e := &p.E[i]
			n.Xoffset = l.Xoffset + e.Xoffset
			n.Type = e.Expr.Type
			if e.Expr.Op == OLITERAL {
				gdata(&n, e.Expr, int(n.Type.Width))
			} else {
				ll := nod(OXXX, nil, nil)
				*ll = n
				ll.Orig = ll // completely separate copy
				if !staticassign(ll, e.Expr, out) {
					// Requires computation, but we're
					// copying someone else's computation.
					rr := nod(OXXX, nil, nil)

					*rr = *orig
					rr.Orig = rr // completely separate copy
					rr.Type = ll.Type
					rr.Xoffset += e.Xoffset
					setlineno(rr)
					*out = append(*out, nod(OAS, ll, rr))
				}
			}
		}

		return true
	}

	return false
}

func staticassign(l *Node, r *Node, out *[]*Node) bool {
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
		case OARRAYLIT, OSLICELIT, OMAPLIT, OSTRUCTLIT:
			// Init pointer.
			a := staticname(r.Left.Type)

			inittemps[r] = a
			gdata(l, nod(OADDR, a, nil), int(l.Type.Width))

			// Init underlying literal.
			if !staticassign(a, r.Left, out) {
				*out = append(*out, nod(OAS, a, r.Left))
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

	case OSLICELIT:
		initplan(r)
		// Init slice.
		bound := r.Right.Int64()
		ta := typArray(r.Type.Elem(), bound)
		a := staticname(ta)
		inittemps[r] = a
		n := *l
		n.Xoffset = l.Xoffset + int64(array_array)
		gdata(&n, nod(OADDR, a, nil), Widthptr)
		n.Xoffset = l.Xoffset + int64(array_nel)
		gdata(&n, r.Right, Widthint)
		n.Xoffset = l.Xoffset + int64(array_cap)
		gdata(&n, r.Right, Widthint)

		// Fall through to init underlying array.
		l = a
		fallthrough

	case OARRAYLIT, OSTRUCTLIT:
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
				a := nod(OXXX, nil, nil)
				*a = n
				a.Orig = a // completely separate copy
				if !staticassign(a, e.Expr, out) {
					*out = append(*out, nod(OAS, a, e.Expr))
				}
			}
		}

		return true

	case OMAPLIT:
		break

	case OCLOSURE:
		if hasemptycvars(r) {
			if Debug_closure > 0 {
				Warnl(r.Lineno, "closure converted to global")
			}
			// Closures with no captured variables are globals,
			// so the assignment can be done at link time.
			n := *l
			gdata(&n, r.Func.Closure.Func.Nname, Widthptr)
			return true
		} else {
			closuredebugruntimecheck(r)
		}

	case OCONVIFACE:
		// This logic is mirrored in isStaticCompositeLiteral.
		// If you change something here, change it there, and vice versa.

		// Determine the underlying concrete type and value we are converting from.
		val := r
		for val.Op == OCONVIFACE {
			val = val.Left
		}
		if val.Type.IsInterface() {
			// val is an interface type.
			// If val is nil, we can statically initialize l;
			// both words are zero and so there no work to do, so report success.
			// If val is non-nil, we have no concrete type to record,
			// and we won't be able to statically initialize its value, so report failure.
			return Isconst(val, CTNIL)
		}

		var itab *Node
		if l.Type.IsEmptyInterface() {
			itab = typename(val.Type)
		} else {
			itab = itabname(val.Type, l.Type)
		}

		// Create a copy of l to modify while we emit data.
		n := *l

		// Emit itab, advance offset.
		gdata(&n, itab, Widthptr)
		n.Xoffset += int64(Widthptr)

		// Emit data.
		if isdirectiface(val.Type) {
			if Isconst(val, CTNIL) {
				// Nil is zero, nothing to do.
				return true
			}
			// Copy val directly into n.
			n.Type = val.Type
			setlineno(val)
			a := nod(OXXX, nil, nil)
			*a = n
			a.Orig = a
			if !staticassign(a, val, out) {
				*out = append(*out, nod(OAS, a, val))
			}
		} else {
			// Construct temp to hold val, write pointer to temp into n.
			a := staticname(val.Type)
			inittemps[val] = a
			if !staticassign(a, val, out) {
				*out = append(*out, nod(OAS, a, val))
			}
			ptr := nod(OADDR, a, nil)
			n.Type = ptrto(val.Type)
			gdata(&n, ptr, Widthptr)
		}

		return true
	}

	//dump("not static", r);
	return false
}

// initContext is the context in which static data is populated.
// It is either in an init function or in any other function.
// Static data populated in an init function will be written either
// zero times (as a readonly, static data symbol) or
// one time (during init function execution).
// Either way, there is no opportunity for races or further modification,
// so the data can be written to a (possibly readonly) data symbol.
// Static data populated in any other function needs to be local to
// that function to allow multiple instances of that function
// to execute concurrently without clobbering each others' data.
type initContext uint8

const (
	inInitFunction initContext = iota
	inNonInitFunction
)

// from here down is the walk analysis
// of composite literals.
// most of the work is to generate
// data statements for the constant
// part of the composite literal.

// staticname returns a name backed by a static data symbol.
// Callers should set n.Name.Readonly = true on the
// returned node for readonly nodes.
func staticname(t *Type) *Node {
	n := newname(lookupN("statictmp_", statuniqgen))
	statuniqgen++
	addvar(n, t, PEXTERN)
	return n
}

func isliteral(n *Node) bool {
	// Treat nils as zeros rather than literals.
	return n.Op == OLITERAL && n.Val().Ctype() != CTNIL
}

func (n *Node) isSimpleName() bool {
	return n.Op == ONAME && n.Addable && n.Class != PAUTOHEAP && n.Class != PEXTERN
}

func litas(l *Node, r *Node, init *Nodes) {
	a := nod(OAS, l, r)
	a = typecheck(a, Etop)
	a = walkexpr(a, init)
	init.Append(a)
}

// initGenType is a bitmap indicating the types of generation that will occur for a static value.
type initGenType uint8

const (
	initDynamic initGenType = 1 << iota // contains some dynamic values, for which init code will be generated
	initConst                           // contains some constant values, which may be written into data symbols
)

// getdyn calculates the initGenType for n.
// If top is false, getdyn is recursing.
func getdyn(n *Node, top bool) initGenType {
	switch n.Op {
	default:
		if isliteral(n) {
			return initConst
		}
		return initDynamic

	case OSLICELIT:
		if !top {
			return initDynamic
		}

	case OARRAYLIT, OSTRUCTLIT:
	}

	var mode initGenType
	for _, n1 := range n.List.Slice() {
		switch n1.Op {
		case OKEY:
			n1 = n1.Right
		case OSTRUCTKEY:
			n1 = n1.Left
		}
		mode |= getdyn(n1, false)
		if mode == initDynamic|initConst {
			break
		}
	}
	return mode
}

// isStaticCompositeLiteral reports whether n is a compile-time constant.
func isStaticCompositeLiteral(n *Node) bool {
	switch n.Op {
	case OSLICELIT:
		return false
	case OARRAYLIT:
		for _, r := range n.List.Slice() {
			if r.Op == OKEY {
				r = r.Right
			}
			if !isStaticCompositeLiteral(r) {
				return false
			}
		}
		return true
	case OSTRUCTLIT:
		for _, r := range n.List.Slice() {
			if r.Op != OSTRUCTKEY {
				Fatalf("isStaticCompositeLiteral: rhs not OSTRUCTKEY: %v", r)
			}
			if !isStaticCompositeLiteral(r.Left) {
				return false
			}
		}
		return true
	case OLITERAL:
		return true
	case OCONVIFACE:
		// See staticassign's OCONVIFACE case for comments.
		val := n
		for val.Op == OCONVIFACE {
			val = val.Left
		}
		if val.Type.IsInterface() {
			return Isconst(val, CTNIL)
		}
		if isdirectiface(val.Type) && Isconst(val, CTNIL) {
			return true
		}
		return isStaticCompositeLiteral(val)
	}
	return false
}

// initKind is a kind of static initialization: static, dynamic, or local.
// Static initialization represents literals and
// literal components of composite literals.
// Dynamic initialization represents non-literals and
// non-literal components of composite literals.
// LocalCode initializion represents initialization
// that occurs purely in generated code local to the function of use.
// Initialization code is sometimes generated in passes,
// first static then dynamic.
type initKind uint8

const (
	initKindStatic initKind = iota + 1
	initKindDynamic
	initKindLocalCode
)

// fixedlit handles struct, array, and slice literals.
// TODO: expand documentation.
func fixedlit(ctxt initContext, kind initKind, n *Node, var_ *Node, init *Nodes) {
	var splitnode func(*Node) (a *Node, value *Node)
	switch n.Op {
	case OARRAYLIT, OSLICELIT:
		var k int64
		splitnode = func(r *Node) (*Node, *Node) {
			if r.Op == OKEY {
				k = nonnegintconst(r.Left)
				r = r.Right
			}
			a := nod(OINDEX, var_, nodintconst(k))
			k++
			return a, r
		}
	case OSTRUCTLIT:
		splitnode = func(r *Node) (*Node, *Node) {
			if r.Op != OSTRUCTKEY {
				Fatalf("fixedlit: rhs not OSTRUCTKEY: %v", r)
			}
			return nodSym(ODOT, var_, r.Sym), r.Left
		}
	default:
		Fatalf("fixedlit bad op: %v", n.Op)
	}

	for _, r := range n.List.Slice() {
		a, value := splitnode(r)

		switch value.Op {
		case OSLICELIT:
			if (kind == initKindStatic && ctxt == inNonInitFunction) || (kind == initKindDynamic && ctxt == inInitFunction) {
				slicelit(ctxt, value, a, init)
				continue
			}

		case OARRAYLIT, OSTRUCTLIT:
			fixedlit(ctxt, kind, value, a, init)
			continue
		}

		islit := isliteral(value)
		if (kind == initKindStatic && !islit) || (kind == initKindDynamic && islit) {
			continue
		}

		// build list of assignments: var[index] = expr
		setlineno(value)
		a = nod(OAS, a, value)
		a = typecheck(a, Etop)
		switch kind {
		case initKindStatic:
			a = walkexpr(a, init) // add any assignments in r to top
			if a.Op == OASWB {
				// Static initialization never needs
				// write barriers.
				a.Op = OAS
			}
			if a.Op != OAS {
				Fatalf("fixedlit: not as, is %v", a)
			}
			a.IsStatic = true
		case initKindDynamic, initKindLocalCode:
			a = orderstmtinplace(a)
			a = walkstmt(a)
		default:
			Fatalf("fixedlit: bad kind %d", kind)
		}

		init.Append(a)
	}
}

func slicelit(ctxt initContext, n *Node, var_ *Node, init *Nodes) {
	// make an array type corresponding the number of elements we have
	t := typArray(n.Type.Elem(), n.Right.Int64())
	dowidth(t)

	if ctxt == inNonInitFunction {
		// put everything into static array
		vstat := staticname(t)

		fixedlit(ctxt, initKindStatic, n, vstat, init)
		fixedlit(ctxt, initKindDynamic, n, vstat, init)

		// copy static to slice
		a := nod(OSLICE, vstat, nil)

		a = nod(OAS, var_, a)
		a = typecheck(a, Etop)
		a.IsStatic = true
		init.Append(a)
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
	// 5. for each dynamic part assign to the array
	//	vauto[i] = dynamic part
	// 6. assign slice of allocated heap to var
	//	var = vauto[:]
	//
	// an optimization is done if there is no constant part
	//	3. var vauto *[...]t = new([...]t)
	//	5. vauto[i] = dynamic part
	//	6. var = vauto[:]

	// if the literal contains constants,
	// make static initialized array (1),(2)
	var vstat *Node

	mode := getdyn(n, true)
	if mode&initConst != 0 {
		vstat = staticname(t)
		if ctxt == inInitFunction {
			vstat.Name.Readonly = true
		}
		fixedlit(ctxt, initKindStatic, n, vstat, init)
	}

	// make new auto *array (3 declare)
	vauto := temp(ptrto(t))

	// set auto to point at new temp or heap (3 assign)
	var a *Node
	if x := prealloc[n]; x != nil {
		// temp allocated during order.go for dddarg
		x.Type = t

		if vstat == nil {
			a = nod(OAS, x, nil)
			a = typecheck(a, Etop)
			init.Append(a) // zero new temp
		}

		a = nod(OADDR, x, nil)
	} else if n.Esc == EscNone {
		a = temp(t)
		if vstat == nil {
			a = nod(OAS, temp(t), nil)
			a = typecheck(a, Etop)
			init.Append(a) // zero new temp
			a = a.Left
		}

		a = nod(OADDR, a, nil)
	} else {
		a = nod(ONEW, nil, nil)
		a.List.Set1(typenod(t))
	}

	a = nod(OAS, vauto, a)
	a = typecheck(a, Etop)
	a = walkexpr(a, init)
	init.Append(a)

	if vstat != nil {
		// copy static to heap (4)
		a = nod(OIND, vauto, nil)

		a = nod(OAS, a, vstat)
		a = typecheck(a, Etop)
		a = walkexpr(a, init)
		init.Append(a)
	}

	// put dynamics into array (5)
	var index int64
	for _, r := range n.List.Slice() {
		value := r
		if r.Op == OKEY {
			index = nonnegintconst(r.Left)
			value = r.Right
		}
		a := nod(OINDEX, vauto, nodintconst(index))
		a.Bounded = true
		index++

		// TODO need to check bounds?

		switch value.Op {
		case OSLICELIT:
			break

		case OARRAYLIT, OSTRUCTLIT:
			fixedlit(ctxt, initKindDynamic, value, a, init)
			continue
		}

		if isliteral(value) {
			continue
		}

		// build list of vauto[c] = expr
		setlineno(value)
		a = nod(OAS, a, value)

		a = typecheck(a, Etop)
		a = orderstmtinplace(a)
		a = walkstmt(a)
		init.Append(a)
	}

	// make slice out of heap (6)
	a = nod(OAS, var_, nod(OSLICE, vauto, nil))

	a = typecheck(a, Etop)
	a = orderstmtinplace(a)
	a = walkstmt(a)
	init.Append(a)
}

func maplit(n *Node, m *Node, init *Nodes) {
	// make the map var
	nerr := nerrors

	a := nod(OMAKE, nil, nil)
	a.List.Set2(typenod(n.Type), nodintconst(int64(len(n.List.Slice()))))
	litas(m, a, init)

	// count the initializers
	b := 0
	for _, r := range n.List.Slice() {
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
		// build types [count]Tindex and [count]Tvalue
		tk := typArray(n.Type.Key(), int64(b))
		tv := typArray(n.Type.Val(), int64(b))

		// TODO(josharian): suppress alg generation for these types?
		dowidth(tk)
		dowidth(tv)

		// make and initialize static arrays
		vstatk := staticname(tk)
		vstatk.Name.Readonly = true
		vstatv := staticname(tv)
		vstatv.Name.Readonly = true

		b := int64(0)
		for _, r := range n.List.Slice() {
			if r.Op != OKEY {
				Fatalf("maplit: rhs not OKEY: %v", r)
			}
			index := r.Left
			value := r.Right

			if isliteral(index) && isliteral(value) {
				// build vstatk[b] = index
				setlineno(index)
				lhs := nod(OINDEX, vstatk, nodintconst(b))
				as := nod(OAS, lhs, index)
				as = typecheck(as, Etop)
				as = walkexpr(as, init)
				as.IsStatic = true
				init.Append(as)

				// build vstatv[b] = value
				setlineno(value)
				lhs = nod(OINDEX, vstatv, nodintconst(b))
				as = nod(OAS, lhs, value)
				as = typecheck(as, Etop)
				as = walkexpr(as, init)
				as.IsStatic = true
				init.Append(as)

				b++
			}
		}

		// loop adding structure elements to map
		// for i = 0; i < len(vstatk); i++ {
		//	map[vstatk[i]] = vstatv[i]
		// }
		i := temp(Types[TINT])
		rhs := nod(OINDEX, vstatv, i)
		rhs.Bounded = true

		kidx := nod(OINDEX, vstatk, i)
		kidx.Bounded = true
		lhs := nod(OINDEX, m, kidx)

		zero := nod(OAS, i, nodintconst(0))
		cond := nod(OLT, i, nodintconst(tk.NumElem()))
		incr := nod(OAS, i, nod(OADD, i, nodintconst(1)))
		body := nod(OAS, lhs, rhs)

		loop := nod(OFOR, cond, incr)
		loop.Nbody.Set1(body)
		loop.Ninit.Set1(zero)

		loop = typecheck(loop, Etop)
		loop = walkstmt(loop)
		init.Append(loop)
	}

	// put in dynamic entries one-at-a-time
	var key, val *Node
	for _, r := range n.List.Slice() {
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
			key = temp(m.Type.Key())
			val = temp(m.Type.Val())
		}

		setlineno(index)
		a = nod(OAS, key, index)
		a = typecheck(a, Etop)
		a = walkstmt(a)
		init.Append(a)

		setlineno(value)
		a = nod(OAS, val, value)
		a = typecheck(a, Etop)
		a = walkstmt(a)
		init.Append(a)

		setlineno(val)
		a = nod(OAS, nod(OINDEX, m, key), val)
		a = typecheck(a, Etop)
		a = walkstmt(a)
		init.Append(a)

		if nerr != nerrors {
			break
		}
	}

	if key != nil {
		a = nod(OVARKILL, key, nil)
		a = typecheck(a, Etop)
		init.Append(a)
		a = nod(OVARKILL, val, nil)
		a = typecheck(a, Etop)
		init.Append(a)
	}
}

func anylit(n *Node, var_ *Node, init *Nodes) {
	t := n.Type
	switch n.Op {
	default:
		Fatalf("anylit: not lit, op=%v node=%v", n.Op, n)

	case OPTRLIT:
		if !t.IsPtr() {
			Fatalf("anylit: not ptr")
		}

		var r *Node
		if n.Right != nil {
			// n.Right is stack temporary used as backing store.
			init.Append(nod(OAS, n.Right, nil)) // zero backing store, just in case (#18410)
			r = nod(OADDR, n.Right, nil)
			r = typecheck(r, Erv)
		} else {
			r = nod(ONEW, nil, nil)
			r.Typecheck = 1
			r.Type = t
			r.Esc = n.Esc
		}

		r = walkexpr(r, init)
		a := nod(OAS, var_, r)

		a = typecheck(a, Etop)
		init.Append(a)

		var_ = nod(OIND, var_, nil)
		var_ = typecheck(var_, Erv|Easgn)
		anylit(n.Left, var_, init)

	case OSTRUCTLIT, OARRAYLIT:
		if !t.IsStruct() && !t.IsArray() {
			Fatalf("anylit: not struct/array")
		}

		if var_.isSimpleName() && n.List.Len() > 4 {
			// lay out static data
			vstat := staticname(t)
			vstat.Name.Readonly = true

			ctxt := inInitFunction
			if n.Op == OARRAYLIT {
				ctxt = inNonInitFunction
			}
			fixedlit(ctxt, initKindStatic, n, vstat, init)

			// copy static to var
			a := nod(OAS, var_, vstat)

			a = typecheck(a, Etop)
			a = walkexpr(a, init)
			init.Append(a)

			// add expressions to automatic
			fixedlit(inInitFunction, initKindDynamic, n, var_, init)
			break
		}

		var components int64
		if n.Op == OARRAYLIT {
			components = t.NumElem()
		} else {
			components = int64(t.NumFields())
		}
		// initialization of an array or struct with unspecified components (missing fields or arrays)
		if var_.isSimpleName() || int64(n.List.Len()) < components {
			a := nod(OAS, var_, nil)
			a = typecheck(a, Etop)
			a = walkexpr(a, init)
			init.Append(a)
		}

		fixedlit(inInitFunction, initKindLocalCode, n, var_, init)

	case OSLICELIT:
		slicelit(inInitFunction, n, var_, init)

	case OMAPLIT:
		if !t.IsMap() {
			Fatalf("anylit: not map")
		}
		maplit(n, var_, init)
	}
}

func oaslit(n *Node, init *Nodes) bool {
	if n.Left == nil || n.Right == nil {
		// not a special composite literal assignment
		return false
	}
	if n.Left.Type == nil || n.Right.Type == nil {
		// not a special composite literal assignment
		return false
	}
	if !n.Left.isSimpleName() {
		// not a special composite literal assignment
		return false
	}
	if !eqtype(n.Left.Type, n.Right.Type) {
		// not a special composite literal assignment
		return false
	}

	switch n.Right.Op {
	default:
		// not a special composite literal assignment
		return false

	case OSTRUCTLIT, OARRAYLIT, OSLICELIT, OMAPLIT:
		if vmatch1(n.Left, n.Right) {
			// not a special composite literal assignment
			return false
		}
		anylit(n.Right, n.Left, init)
	}

	n.Op = OEMPTY
	n.Right = nil
	return true
}

func getlit(lit *Node) int {
	if smallintconst(lit) {
		return int(lit.Int64())
	}
	return -1
}

// stataddr sets nam to the static address of n and reports whether it succeeded.
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
		if n.Left.Type.IsSlice() {
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

	case OARRAYLIT, OSLICELIT:
		var k int64
		for _, a := range n.List.Slice() {
			if a.Op == OKEY {
				k = nonnegintconst(a.Left)
				a = a.Right
			}
			addvalue(p, k*n.Type.Elem().Width, a)
			k++
		}

	case OSTRUCTLIT:
		for _, a := range n.List.Slice() {
			if a.Op != OSTRUCTKEY {
				Fatalf("initplan fixedlit")
			}
			addvalue(p, a.Xoffset, a.Left)
		}

	case OMAPLIT:
		for _, a := range n.List.Slice() {
			if a.Op != OKEY {
				Fatalf("initplan maplit")
			}
			addvalue(p, -1, a.Right)
		}
	}
}

func addvalue(p *InitPlan, xoffset int64, n *Node) {
	// special case: zero can be dropped entirely
	if iszero(n) {
		return
	}

	// special case: inline struct and array (not slice) literals
	if isvaluelit(n) {
		initplan(n)
		q := initplans[n]
		for _, qe := range q.E {
			// qe is a copy; we are not modifying entries in q.E
			qe.Xoffset += xoffset
			p.E = append(p.E, qe)
		}
		return
	}

	// add to plan
	p.E = append(p.E, InitEntry{Xoffset: xoffset, Expr: n})
}

func iszero(n *Node) bool {
	switch n.Op {
	case OLITERAL:
		switch u := n.Val().U.(type) {
		default:
			Dump("unexpected literal", n)
			Fatalf("iszero")
		case *NilVal:
			return true
		case string:
			return u == ""
		case bool:
			return !u
		case *Mpint:
			return u.CmpInt64(0) == 0
		case *Mpflt:
			return u.CmpFloat64(0) == 0
		case *Mpcplx:
			return u.Real.CmpFloat64(0) == 0 && u.Imag.CmpFloat64(0) == 0
		}

	case OARRAYLIT:
		for _, n1 := range n.List.Slice() {
			if n1.Op == OKEY {
				n1 = n1.Right
			}
			if !iszero(n1) {
				return false
			}
		}
		return true

	case OSTRUCTLIT:
		for _, n1 := range n.List.Slice() {
			if !iszero(n1.Left) {
				return false
			}
		}
		return true
	}

	return false
}

func isvaluelit(n *Node) bool {
	return n.Op == OARRAYLIT || n.Op == OSTRUCTLIT
}

// gen_as_init attempts to emit static data for n and reports whether it succeeded.
// If reportOnly is true, it does not emit static data and does not modify the AST.
func gen_as_init(n *Node, reportOnly bool) bool {
	success := genAsInitNoCheck(n, reportOnly)
	if !success && n.IsStatic {
		Dump("\ngen_as_init", n)
		Fatalf("gen_as_init couldn't generate static data")
	}
	return success
}

func genAsInitNoCheck(n *Node, reportOnly bool) bool {
	if !n.IsStatic {
		return false
	}

	nr := n.Right
	nl := n.Left
	if nr == nil {
		var nam Node
		return stataddr(&nam, nl) && nam.Class == PEXTERN
	}

	if nr.Type == nil || !eqtype(nl.Type, nr.Type) {
		return false
	}

	var nam Node
	if !stataddr(&nam, nl) || nam.Class != PEXTERN {
		return false
	}

	switch nr.Op {
	default:
		return false

	case OCONVNOP:
		nr = nr.Left
		if nr == nil || nr.Op != OSLICEARR {
			return false
		}
		fallthrough

	case OSLICEARR:
		low, high, _ := nr.SliceBounds()
		if low != nil || high != nil {
			return false
		}
		nr = nr.Left
		if nr == nil || nr.Op != OADDR {
			return false
		}
		ptr := nr
		nr = nr.Left
		if nr == nil || nr.Op != ONAME {
			return false
		}

		// nr is the array being converted to a slice
		if nr.Type == nil || !nr.Type.IsArray() {
			return false
		}

		if !reportOnly {
			nam.Xoffset += int64(array_array)
			gdata(&nam, ptr, Widthptr)

			nam.Xoffset += int64(array_nel) - int64(array_array)
			var nod1 Node
			Nodconst(&nod1, Types[TINT], nr.Type.NumElem())
			gdata(&nam, &nod1, Widthint)

			nam.Xoffset += int64(array_cap) - int64(array_nel)
			gdata(&nam, &nod1, Widthint)
		}

		return true

	case OLITERAL:
		if !reportOnly {
			gdata(&nam, nr, int(nr.Type.Width))
		}
		return true
	}
}
