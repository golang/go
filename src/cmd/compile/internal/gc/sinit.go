// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/types"
	"fmt"
)

type InitEntry struct {
	Xoffset int64 // struct, array only
	Expr    *Node // bytes of run-time computed expressions
}

type InitPlan struct {
	E []InitEntry
}

// An InitSchedule is used to decompose assignment statements into
// static and dynamic initialization parts. Static initializations are
// handled by populating variables' linker symbol data, while dynamic
// initializations are accumulated to be executed in order.
type InitSchedule struct {
	// out is the ordered list of dynamic initialization
	// statements.
	out []*Node

	initplans map[*Node]*InitPlan
	inittemps map[*Node]*Node
}

func (s *InitSchedule) append(n *Node) {
	s.out = append(s.out, n)
}

// staticInit adds an initialization statement n to the schedule.
func (s *InitSchedule) staticInit(n *Node) {
	if !s.tryStaticInit(n) {
		if Debug['%'] != 0 {
			Dump("nonstatic", n)
		}
		s.append(n)
	}
}

// tryStaticInit attempts to statically execute an initialization
// statement and reports whether it succeeded.
func (s *InitSchedule) tryStaticInit(n *Node) bool {
	// Only worry about simple "l = r" assignments. Multiple
	// variable/expression OAS2 assignments have already been
	// replaced by multiple simple OAS assignments, and the other
	// OAS2* assignments mostly necessitate dynamic execution
	// anyway.
	if n.Op != OAS {
		return false
	}
	if n.Left.isBlank() && candiscard(n.Right) {
		return true
	}
	lno := setlineno(n)
	defer func() { lineno = lno }()
	return s.staticassign(n.Left, n.Right)
}

// like staticassign but we are copying an already
// initialized value r.
func (s *InitSchedule) staticcopy(l *Node, r *Node) bool {
	if r.Op != ONAME {
		return false
	}
	if r.Class() == PFUNC {
		gdata(l, r, Widthptr)
		return true
	}
	if r.Class() != PEXTERN || r.Sym.Pkg != localpkg {
		return false
	}
	if r.Name.Defn == nil { // probably zeroed but perhaps supplied externally and of unknown value
		return false
	}
	if r.Name.Defn.Op != OAS {
		return false
	}
	if r.Type.IsString() { // perhaps overwritten by cmd/link -X (#34675)
		return false
	}
	orig := r
	r = r.Name.Defn.Right

	for r.Op == OCONVNOP && !types.Identical(r.Type, l.Type) {
		r = r.Left
	}

	switch r.Op {
	case ONAME:
		if s.staticcopy(l, r) {
			return true
		}
		// We may have skipped past one or more OCONVNOPs, so
		// use conv to ensure r is assignable to l (#13263).
		s.append(nod(OAS, l, conv(r, l.Type)))
		return true

	case OLITERAL:
		if isZero(r) {
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
			gdata(l, nod(OADDR, s.inittemps[r], nil), int(l.Type.Width))
			return true
		}

	case OSLICELIT:
		// copy slice
		a := s.inittemps[r]

		n := l.copy()
		n.Xoffset = l.Xoffset + int64(slice_array)
		gdata(n, nod(OADDR, a, nil), Widthptr)
		n.Xoffset = l.Xoffset + int64(slice_nel)
		gdata(n, r.Right, Widthptr)
		n.Xoffset = l.Xoffset + int64(slice_cap)
		gdata(n, r.Right, Widthptr)
		return true

	case OARRAYLIT, OSTRUCTLIT:
		p := s.initplans[r]

		n := l.copy()
		for i := range p.E {
			e := &p.E[i]
			n.Xoffset = l.Xoffset + e.Xoffset
			n.Type = e.Expr.Type
			if e.Expr.Op == OLITERAL {
				gdata(n, e.Expr, int(n.Type.Width))
				continue
			}
			ll := n.sepcopy()
			if s.staticcopy(ll, e.Expr) {
				continue
			}
			// Requires computation, but we're
			// copying someone else's computation.
			rr := orig.sepcopy()
			rr.Type = ll.Type
			rr.Xoffset += e.Xoffset
			setlineno(rr)
			s.append(nod(OAS, ll, rr))
		}

		return true
	}

	return false
}

func (s *InitSchedule) staticassign(l *Node, r *Node) bool {
	for r.Op == OCONVNOP {
		r = r.Left
	}

	switch r.Op {
	case ONAME:
		return s.staticcopy(l, r)

	case OLITERAL:
		if isZero(r) {
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

			s.inittemps[r] = a
			gdata(l, nod(OADDR, a, nil), int(l.Type.Width))

			// Init underlying literal.
			if !s.staticassign(a, r.Left) {
				s.append(nod(OAS, a, r.Left))
			}
			return true
		}
		//dump("not static ptrlit", r);

	case OSTR2BYTES:
		if l.Class() == PEXTERN && r.Left.Op == OLITERAL {
			sval := strlit(r.Left)
			slicebytes(l, sval, len(sval))
			return true
		}

	case OSLICELIT:
		s.initplan(r)
		// Init slice.
		bound := r.Right.Int64()
		ta := types.NewArray(r.Type.Elem(), bound)
		a := staticname(ta)
		s.inittemps[r] = a
		n := l.copy()
		n.Xoffset = l.Xoffset + int64(slice_array)
		gdata(n, nod(OADDR, a, nil), Widthptr)
		n.Xoffset = l.Xoffset + int64(slice_nel)
		gdata(n, r.Right, Widthptr)
		n.Xoffset = l.Xoffset + int64(slice_cap)
		gdata(n, r.Right, Widthptr)

		// Fall through to init underlying array.
		l = a
		fallthrough

	case OARRAYLIT, OSTRUCTLIT:
		s.initplan(r)

		p := s.initplans[r]
		n := l.copy()
		for i := range p.E {
			e := &p.E[i]
			n.Xoffset = l.Xoffset + e.Xoffset
			n.Type = e.Expr.Type
			if e.Expr.Op == OLITERAL {
				gdata(n, e.Expr, int(n.Type.Width))
				continue
			}
			setlineno(e.Expr)
			a := n.sepcopy()
			if !s.staticassign(a, e.Expr) {
				s.append(nod(OAS, a, e.Expr))
			}
		}

		return true

	case OMAPLIT:
		break

	case OCLOSURE:
		if hasemptycvars(r) {
			if Debug_closure > 0 {
				Warnl(r.Pos, "closure converted to global")
			}
			// Closures with no captured variables are globals,
			// so the assignment can be done at link time.
			gdata(l, r.Func.Closure.Func.Nname, Widthptr)
			return true
		}
		closuredebugruntimecheck(r)

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
		n := l.copy()

		// Emit itab, advance offset.
		gdata(n, itab, Widthptr)
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
			a := n.sepcopy()
			if !s.staticassign(a, val) {
				s.append(nod(OAS, a, val))
			}
		} else {
			// Construct temp to hold val, write pointer to temp into n.
			a := staticname(val.Type)
			s.inittemps[val] = a
			if !s.staticassign(a, val) {
				s.append(nod(OAS, a, val))
			}
			ptr := nod(OADDR, a, nil)
			n.Type = types.NewPtr(val.Type)
			gdata(n, ptr, Widthptr)
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

func (c initContext) String() string {
	if c == inInitFunction {
		return "inInitFunction"
	}
	return "inNonInitFunction"
}

// from here down is the walk analysis
// of composite literals.
// most of the work is to generate
// data statements for the constant
// part of the composite literal.

var statuniqgen int // name generator for static temps

// staticname returns a name backed by a static data symbol.
// Callers should call n.MarkReadonly on the
// returned node for readonly nodes.
func staticname(t *types.Type) *Node {
	// Don't use lookupN; it interns the resulting string, but these are all unique.
	n := newname(lookup(fmt.Sprintf(".stmp_%d", statuniqgen)))
	statuniqgen++
	addvar(n, t, PEXTERN)
	return n
}

func isLiteral(n *Node) bool {
	// Treat nils as zeros rather than literals.
	return n.Op == OLITERAL && n.Val().Ctype() != CTNIL
}

func (n *Node) isSimpleName() bool {
	return n.Op == ONAME && n.Class() != PAUTOHEAP && n.Class() != PEXTERN
}

func litas(l *Node, r *Node, init *Nodes) {
	a := nod(OAS, l, r)
	a = typecheck(a, ctxStmt)
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
		if isLiteral(n) {
			return initConst
		}
		return initDynamic

	case OSLICELIT:
		if !top {
			return initDynamic
		}
		if n.Right.Int64()/4 > int64(n.List.Len()) {
			// <25% of entries have explicit values.
			// Very rough estimation, it takes 4 bytes of instructions
			// to initialize 1 byte of result. So don't use a static
			// initializer if the dynamic initialization code would be
			// smaller than the static value.
			// See issue 23780.
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
// LocalCode initialization represents initialization
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
				k = indexconst(r.Left)
				if k < 0 {
					Fatalf("fixedlit: invalid index %v", r.Left)
				}
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
			if r.Sym.IsBlank() {
				return nblank, r.Left
			}
			setlineno(r)
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

		islit := isLiteral(value)
		if (kind == initKindStatic && !islit) || (kind == initKindDynamic && islit) {
			continue
		}

		// build list of assignments: var[index] = expr
		setlineno(a)
		a = nod(OAS, a, value)
		a = typecheck(a, ctxStmt)
		switch kind {
		case initKindStatic:
			genAsStatic(a)
		case initKindDynamic, initKindLocalCode:
			a = orderStmtInPlace(a, map[string][]*Node{})
			a = walkstmt(a)
			init.Append(a)
		default:
			Fatalf("fixedlit: bad kind %d", kind)
		}

	}
}

func isSmallSliceLit(n *Node) bool {
	if n.Op != OSLICELIT {
		return false
	}

	r := n.Right

	return smallintconst(r) && (n.Type.Elem().Width == 0 || r.Int64() <= smallArrayBytes/n.Type.Elem().Width)
}

func slicelit(ctxt initContext, n *Node, var_ *Node, init *Nodes) {
	// make an array type corresponding the number of elements we have
	t := types.NewArray(n.Type.Elem(), n.Right.Int64())
	dowidth(t)

	if ctxt == inNonInitFunction {
		// put everything into static array
		vstat := staticname(t)

		fixedlit(ctxt, initKindStatic, n, vstat, init)
		fixedlit(ctxt, initKindDynamic, n, vstat, init)

		// copy static to slice
		var_ = typecheck(var_, ctxExpr|ctxAssign)
		var nam Node
		if !stataddr(&nam, var_) || nam.Class() != PEXTERN {
			Fatalf("slicelit: %v", var_)
		}

		var v Node
		v.Type = types.Types[TINT]
		setintconst(&v, t.NumElem())

		nam.Xoffset += int64(slice_array)
		gdata(&nam, nod(OADDR, vstat, nil), Widthptr)
		nam.Xoffset += int64(slice_nel) - int64(slice_array)
		gdata(&nam, &v, Widthptr)
		nam.Xoffset += int64(slice_cap) - int64(slice_nel)
		gdata(&nam, &v, Widthptr)

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
	if mode&initConst != 0 && !isSmallSliceLit(n) {
		vstat = staticname(t)
		if ctxt == inInitFunction {
			vstat.MarkReadonly()
		}
		fixedlit(ctxt, initKindStatic, n, vstat, init)
	}

	// make new auto *array (3 declare)
	vauto := temp(types.NewPtr(t))

	// set auto to point at new temp or heap (3 assign)
	var a *Node
	if x := prealloc[n]; x != nil {
		// temp allocated during order.go for dddarg
		if !types.Identical(t, x.Type) {
			panic("dotdotdot base type does not match order's assigned type")
		}

		if vstat == nil {
			a = nod(OAS, x, nil)
			a = typecheck(a, ctxStmt)
			init.Append(a) // zero new temp
		} else {
			// Declare that we're about to initialize all of x.
			// (Which happens at the *vauto = vstat below.)
			init.Append(nod(OVARDEF, x, nil))
		}

		a = nod(OADDR, x, nil)
	} else if n.Esc == EscNone {
		a = temp(t)
		if vstat == nil {
			a = nod(OAS, temp(t), nil)
			a = typecheck(a, ctxStmt)
			init.Append(a) // zero new temp
			a = a.Left
		} else {
			init.Append(nod(OVARDEF, a, nil))
		}

		a = nod(OADDR, a, nil)
	} else {
		a = nod(ONEW, nil, nil)
		a.List.Set1(typenod(t))
	}

	a = nod(OAS, vauto, a)
	a = typecheck(a, ctxStmt)
	a = walkexpr(a, init)
	init.Append(a)

	if vstat != nil {
		// copy static to heap (4)
		a = nod(ODEREF, vauto, nil)

		a = nod(OAS, a, vstat)
		a = typecheck(a, ctxStmt)
		a = walkexpr(a, init)
		init.Append(a)
	}

	// put dynamics into array (5)
	var index int64
	for _, value := range n.List.Slice() {
		if value.Op == OKEY {
			index = indexconst(value.Left)
			if index < 0 {
				Fatalf("slicelit: invalid index %v", value.Left)
			}
			value = value.Right
		}
		a := nod(OINDEX, vauto, nodintconst(index))
		a.SetBounded(true)
		index++

		// TODO need to check bounds?

		switch value.Op {
		case OSLICELIT:
			break

		case OARRAYLIT, OSTRUCTLIT:
			k := initKindDynamic
			if vstat == nil {
				// Generate both static and dynamic initializations.
				// See issue #31987.
				k = initKindLocalCode
			}
			fixedlit(ctxt, k, value, a, init)
			continue
		}

		if vstat != nil && isLiteral(value) { // already set by copy from static value
			continue
		}

		// build list of vauto[c] = expr
		setlineno(value)
		a = nod(OAS, a, value)

		a = typecheck(a, ctxStmt)
		a = orderStmtInPlace(a, map[string][]*Node{})
		a = walkstmt(a)
		init.Append(a)
	}

	// make slice out of heap (6)
	a = nod(OAS, var_, nod(OSLICE, vauto, nil))

	a = typecheck(a, ctxStmt)
	a = orderStmtInPlace(a, map[string][]*Node{})
	a = walkstmt(a)
	init.Append(a)
}

func maplit(n *Node, m *Node, init *Nodes) {
	// make the map var
	a := nod(OMAKE, nil, nil)
	a.Esc = n.Esc
	a.List.Set2(typenod(n.Type), nodintconst(int64(n.List.Len())))
	litas(m, a, init)

	entries := n.List.Slice()

	// The order pass already removed any dynamic (runtime-computed) entries.
	// All remaining entries are static. Double-check that.
	for _, r := range entries {
		if !isStaticCompositeLiteral(r.Left) || !isStaticCompositeLiteral(r.Right) {
			Fatalf("maplit: entry is not a literal: %v", r)
		}
	}

	if len(entries) > 25 {
		// For a large number of entries, put them in an array and loop.

		// build types [count]Tindex and [count]Tvalue
		tk := types.NewArray(n.Type.Key(), int64(len(entries)))
		te := types.NewArray(n.Type.Elem(), int64(len(entries)))

		// TODO(josharian): suppress alg generation for these types?
		dowidth(tk)
		dowidth(te)

		// make and initialize static arrays
		vstatk := staticname(tk)
		vstatk.MarkReadonly()
		vstate := staticname(te)
		vstate.MarkReadonly()

		datak := nod(OARRAYLIT, nil, nil)
		datae := nod(OARRAYLIT, nil, nil)
		for _, r := range entries {
			datak.List.Append(r.Left)
			datae.List.Append(r.Right)
		}
		fixedlit(inInitFunction, initKindStatic, datak, vstatk, init)
		fixedlit(inInitFunction, initKindStatic, datae, vstate, init)

		// loop adding structure elements to map
		// for i = 0; i < len(vstatk); i++ {
		//	map[vstatk[i]] = vstate[i]
		// }
		i := temp(types.Types[TINT])
		rhs := nod(OINDEX, vstate, i)
		rhs.SetBounded(true)

		kidx := nod(OINDEX, vstatk, i)
		kidx.SetBounded(true)
		lhs := nod(OINDEX, m, kidx)

		zero := nod(OAS, i, nodintconst(0))
		cond := nod(OLT, i, nodintconst(tk.NumElem()))
		incr := nod(OAS, i, nod(OADD, i, nodintconst(1)))
		body := nod(OAS, lhs, rhs)

		loop := nod(OFOR, cond, incr)
		loop.Nbody.Set1(body)
		loop.Ninit.Set1(zero)

		loop = typecheck(loop, ctxStmt)
		loop = walkstmt(loop)
		init.Append(loop)
		return
	}
	// For a small number of entries, just add them directly.

	// Build list of var[c] = expr.
	// Use temporaries so that mapassign1 can have addressable key, elem.
	// TODO(josharian): avoid map key temporaries for mapfast_* assignments with literal keys.
	tmpkey := temp(m.Type.Key())
	tmpelem := temp(m.Type.Elem())

	for _, r := range entries {
		index, elem := r.Left, r.Right

		setlineno(index)
		a := nod(OAS, tmpkey, index)
		a = typecheck(a, ctxStmt)
		a = walkstmt(a)
		init.Append(a)

		setlineno(elem)
		a = nod(OAS, tmpelem, elem)
		a = typecheck(a, ctxStmt)
		a = walkstmt(a)
		init.Append(a)

		setlineno(tmpelem)
		a = nod(OAS, nod(OINDEX, m, tmpkey), tmpelem)
		a = typecheck(a, ctxStmt)
		a = walkstmt(a)
		init.Append(a)
	}

	a = nod(OVARKILL, tmpkey, nil)
	a = typecheck(a, ctxStmt)
	init.Append(a)
	a = nod(OVARKILL, tmpelem, nil)
	a = typecheck(a, ctxStmt)
	init.Append(a)
}

func anylit(n *Node, var_ *Node, init *Nodes) {
	t := n.Type
	switch n.Op {
	default:
		Fatalf("anylit: not lit, op=%v node=%v", n.Op, n)

	case ONAME:
		a := nod(OAS, var_, n)
		a = typecheck(a, ctxStmt)
		init.Append(a)

	case OPTRLIT:
		if !t.IsPtr() {
			Fatalf("anylit: not ptr")
		}

		var r *Node
		if n.Right != nil {
			// n.Right is stack temporary used as backing store.
			init.Append(nod(OAS, n.Right, nil)) // zero backing store, just in case (#18410)
			r = nod(OADDR, n.Right, nil)
			r = typecheck(r, ctxExpr)
		} else {
			r = nod(ONEW, nil, nil)
			r.SetTypecheck(1)
			r.Type = t
			r.Esc = n.Esc
		}

		r = walkexpr(r, init)
		a := nod(OAS, var_, r)

		a = typecheck(a, ctxStmt)
		init.Append(a)

		var_ = nod(ODEREF, var_, nil)
		var_ = typecheck(var_, ctxExpr|ctxAssign)
		anylit(n.Left, var_, init)

	case OSTRUCTLIT, OARRAYLIT:
		if !t.IsStruct() && !t.IsArray() {
			Fatalf("anylit: not struct/array")
		}

		if var_.isSimpleName() && n.List.Len() > 4 {
			// lay out static data
			vstat := staticname(t)
			vstat.MarkReadonly()

			ctxt := inInitFunction
			if n.Op == OARRAYLIT {
				ctxt = inNonInitFunction
			}
			fixedlit(ctxt, initKindStatic, n, vstat, init)

			// copy static to var
			a := nod(OAS, var_, vstat)

			a = typecheck(a, ctxStmt)
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
			a = typecheck(a, ctxStmt)
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
	if !types.Identical(n.Left.Type, n.Right.Type) {
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
		return true

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
		if n.Type.Width != 0 && thearch.MAXWIDTH/n.Type.Width <= int64(l) {
			break
		}
		nam.Xoffset += int64(l) * n.Type.Width
		nam.Type = n.Type
		return true
	}

	return false
}

func (s *InitSchedule) initplan(n *Node) {
	if s.initplans[n] != nil {
		return
	}
	p := new(InitPlan)
	s.initplans[n] = p
	switch n.Op {
	default:
		Fatalf("initplan")

	case OARRAYLIT, OSLICELIT:
		var k int64
		for _, a := range n.List.Slice() {
			if a.Op == OKEY {
				k = indexconst(a.Left)
				if k < 0 {
					Fatalf("initplan arraylit: invalid index %v", a.Left)
				}
				a = a.Right
			}
			s.addvalue(p, k*n.Type.Elem().Width, a)
			k++
		}

	case OSTRUCTLIT:
		for _, a := range n.List.Slice() {
			if a.Op != OSTRUCTKEY {
				Fatalf("initplan structlit")
			}
			if a.Sym.IsBlank() {
				continue
			}
			s.addvalue(p, a.Xoffset, a.Left)
		}

	case OMAPLIT:
		for _, a := range n.List.Slice() {
			if a.Op != OKEY {
				Fatalf("initplan maplit")
			}
			s.addvalue(p, -1, a.Right)
		}
	}
}

func (s *InitSchedule) addvalue(p *InitPlan, xoffset int64, n *Node) {
	// special case: zero can be dropped entirely
	if isZero(n) {
		return
	}

	// special case: inline struct and array (not slice) literals
	if isvaluelit(n) {
		s.initplan(n)
		q := s.initplans[n]
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

func isZero(n *Node) bool {
	switch n.Op {
	case OLITERAL:
		switch u := n.Val().U.(type) {
		default:
			Dump("unexpected literal", n)
			Fatalf("isZero")
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
			if !isZero(n1) {
				return false
			}
		}
		return true

	case OSTRUCTLIT:
		for _, n1 := range n.List.Slice() {
			if !isZero(n1.Left) {
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

func genAsStatic(as *Node) {
	if as.Left.Type == nil {
		Fatalf("genAsStatic as.Left not typechecked")
	}

	var nam Node
	if !stataddr(&nam, as.Left) || (nam.Class() != PEXTERN && as.Left != nblank) {
		Fatalf("genAsStatic: lhs %v", as.Left)
	}

	switch {
	case as.Right.Op == OLITERAL:
	case as.Right.Op == ONAME && as.Right.Class() == PFUNC:
	default:
		Fatalf("genAsStatic: rhs %v", as.Right)
	}

	gdata(&nam, as.Right, int(as.Right.Type.Width))
}
