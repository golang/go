// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"fmt"
	"go/constant"
)

type InitEntry struct {
	Xoffset int64   // struct, array only
	Expr    ir.Node // bytes of run-time computed expressions
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
	out []ir.Node

	initplans map[ir.Node]*InitPlan
	inittemps map[ir.Node]ir.Node
}

func (s *InitSchedule) append(n ir.Node) {
	s.out = append(s.out, n)
}

// staticInit adds an initialization statement n to the schedule.
func (s *InitSchedule) staticInit(n ir.Node) {
	if !s.tryStaticInit(n) {
		if base.Flag.Percent != 0 {
			ir.Dump("nonstatic", n)
		}
		s.append(n)
	}
}

// tryStaticInit attempts to statically execute an initialization
// statement and reports whether it succeeded.
func (s *InitSchedule) tryStaticInit(n ir.Node) bool {
	// Only worry about simple "l = r" assignments. Multiple
	// variable/expression OAS2 assignments have already been
	// replaced by multiple simple OAS assignments, and the other
	// OAS2* assignments mostly necessitate dynamic execution
	// anyway.
	if n.Op() != ir.OAS {
		return false
	}
	if ir.IsBlank(n.Left()) && candiscard(n.Right()) {
		return true
	}
	lno := setlineno(n)
	defer func() { base.Pos = lno }()
	return s.staticassign(n.Left(), n.Right())
}

// like staticassign but we are copying an already
// initialized value r.
func (s *InitSchedule) staticcopy(l ir.Node, r ir.Node) bool {
	if r.Op() != ir.ONAME && r.Op() != ir.OMETHEXPR {
		return false
	}
	if r.Class() == ir.PFUNC {
		pfuncsym(l, r)
		return true
	}
	if r.Class() != ir.PEXTERN || r.Sym().Pkg != ir.LocalPkg {
		return false
	}
	if r.Name().Defn == nil { // probably zeroed but perhaps supplied externally and of unknown value
		return false
	}
	if r.Name().Defn.Op() != ir.OAS {
		return false
	}
	if r.Type().IsString() { // perhaps overwritten by cmd/link -X (#34675)
		return false
	}
	orig := r
	r = r.Name().Defn.Right()

	for r.Op() == ir.OCONVNOP && !types.Identical(r.Type(), l.Type()) {
		r = r.Left()
	}

	switch r.Op() {
	case ir.ONAME, ir.OMETHEXPR:
		if s.staticcopy(l, r) {
			return true
		}
		// We may have skipped past one or more OCONVNOPs, so
		// use conv to ensure r is assignable to l (#13263).
		s.append(ir.Nod(ir.OAS, l, conv(r, l.Type())))
		return true

	case ir.ONIL:
		return true

	case ir.OLITERAL:
		if isZero(r) {
			return true
		}
		litsym(l, r, int(l.Type().Width))
		return true

	case ir.OADDR:
		if a := r.Left(); a.Op() == ir.ONAME {
			addrsym(l, a)
			return true
		}

	case ir.OPTRLIT:
		switch r.Left().Op() {
		case ir.OARRAYLIT, ir.OSLICELIT, ir.OSTRUCTLIT, ir.OMAPLIT:
			// copy pointer
			addrsym(l, s.inittemps[r])
			return true
		}

	case ir.OSLICELIT:
		// copy slice
		a := s.inittemps[r]
		slicesym(l, a, r.Right().Int64Val())
		return true

	case ir.OARRAYLIT, ir.OSTRUCTLIT:
		p := s.initplans[r]

		n := ir.Copy(l)
		for i := range p.E {
			e := &p.E[i]
			n.SetOffset(l.Offset() + e.Xoffset)
			n.SetType(e.Expr.Type())
			if e.Expr.Op() == ir.OLITERAL || e.Expr.Op() == ir.ONIL {
				litsym(n, e.Expr, int(n.Type().Width))
				continue
			}
			ll := ir.SepCopy(n)
			if s.staticcopy(ll, e.Expr) {
				continue
			}
			// Requires computation, but we're
			// copying someone else's computation.
			rr := ir.SepCopy(orig)
			rr.SetType(ll.Type())
			rr.SetOffset(rr.Offset() + e.Xoffset)
			setlineno(rr)
			s.append(ir.Nod(ir.OAS, ll, rr))
		}

		return true
	}

	return false
}

func (s *InitSchedule) staticassign(l ir.Node, r ir.Node) bool {
	for r.Op() == ir.OCONVNOP {
		r = r.Left()
	}

	switch r.Op() {
	case ir.ONAME, ir.OMETHEXPR:
		return s.staticcopy(l, r)

	case ir.ONIL:
		return true

	case ir.OLITERAL:
		if isZero(r) {
			return true
		}
		litsym(l, r, int(l.Type().Width))
		return true

	case ir.OADDR:
		if nam := stataddr(r.Left()); nam != nil {
			addrsym(l, nam)
			return true
		}
		fallthrough

	case ir.OPTRLIT:
		switch r.Left().Op() {
		case ir.OARRAYLIT, ir.OSLICELIT, ir.OMAPLIT, ir.OSTRUCTLIT:
			// Init pointer.
			a := staticname(r.Left().Type())

			s.inittemps[r] = a
			addrsym(l, a)

			// Init underlying literal.
			if !s.staticassign(a, r.Left()) {
				s.append(ir.Nod(ir.OAS, a, r.Left()))
			}
			return true
		}
		//dump("not static ptrlit", r);

	case ir.OSTR2BYTES:
		if l.Class() == ir.PEXTERN && r.Left().Op() == ir.OLITERAL {
			sval := r.Left().StringVal()
			slicebytes(l, sval)
			return true
		}

	case ir.OSLICELIT:
		s.initplan(r)
		// Init slice.
		bound := r.Right().Int64Val()
		ta := types.NewArray(r.Type().Elem(), bound)
		ta.SetNoalg(true)
		a := staticname(ta)
		s.inittemps[r] = a
		slicesym(l, a, bound)
		// Fall through to init underlying array.
		l = a
		fallthrough

	case ir.OARRAYLIT, ir.OSTRUCTLIT:
		s.initplan(r)

		p := s.initplans[r]
		n := ir.Copy(l)
		for i := range p.E {
			e := &p.E[i]
			n.SetOffset(l.Offset() + e.Xoffset)
			n.SetType(e.Expr.Type())
			if e.Expr.Op() == ir.OLITERAL || e.Expr.Op() == ir.ONIL {
				litsym(n, e.Expr, int(n.Type().Width))
				continue
			}
			setlineno(e.Expr)
			a := ir.SepCopy(n)
			if !s.staticassign(a, e.Expr) {
				s.append(ir.Nod(ir.OAS, a, e.Expr))
			}
		}

		return true

	case ir.OMAPLIT:
		break

	case ir.OCLOSURE:
		if hasemptycvars(r) {
			if base.Debug.Closure > 0 {
				base.WarnfAt(r.Pos(), "closure converted to global")
			}
			// Closures with no captured variables are globals,
			// so the assignment can be done at link time.
			pfuncsym(l, r.Func().Nname)
			return true
		}
		closuredebugruntimecheck(r)

	case ir.OCONVIFACE:
		// This logic is mirrored in isStaticCompositeLiteral.
		// If you change something here, change it there, and vice versa.

		// Determine the underlying concrete type and value we are converting from.
		val := r
		for val.Op() == ir.OCONVIFACE {
			val = val.Left()
		}

		if val.Type().IsInterface() {
			// val is an interface type.
			// If val is nil, we can statically initialize l;
			// both words are zero and so there no work to do, so report success.
			// If val is non-nil, we have no concrete type to record,
			// and we won't be able to statically initialize its value, so report failure.
			return val.Op() == ir.ONIL
		}

		markTypeUsedInInterface(val.Type(), l.Sym().Linksym())

		var itab ir.Node
		if l.Type().IsEmptyInterface() {
			itab = typename(val.Type())
		} else {
			itab = itabname(val.Type(), l.Type())
		}

		// Create a copy of l to modify while we emit data.
		n := ir.Copy(l)

		// Emit itab, advance offset.
		addrsym(n, itab.Left()) // itab is an OADDR node
		n.SetOffset(n.Offset() + int64(Widthptr))

		// Emit data.
		if isdirectiface(val.Type()) {
			if val.Op() == ir.ONIL {
				// Nil is zero, nothing to do.
				return true
			}
			// Copy val directly into n.
			n.SetType(val.Type())
			setlineno(val)
			a := ir.SepCopy(n)
			if !s.staticassign(a, val) {
				s.append(ir.Nod(ir.OAS, a, val))
			}
		} else {
			// Construct temp to hold val, write pointer to temp into n.
			a := staticname(val.Type())
			s.inittemps[val] = a
			if !s.staticassign(a, val) {
				s.append(ir.Nod(ir.OAS, a, val))
			}
			addrsym(n, a)
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

// staticname returns a name backed by a (writable) static data symbol.
// Use readonlystaticname for read-only node.
func staticname(t *types.Type) ir.Node {
	// Don't use lookupN; it interns the resulting string, but these are all unique.
	n := NewName(lookup(fmt.Sprintf("%s%d", obj.StaticNamePref, statuniqgen)))
	statuniqgen++
	declare(n, ir.PEXTERN)
	n.SetType(t)
	n.Sym().Linksym().Set(obj.AttrLocal, true)
	return n
}

// readonlystaticname returns a name backed by a (writable) static data symbol.
func readonlystaticname(t *types.Type) ir.Node {
	n := staticname(t)
	n.MarkReadonly()
	n.Sym().Linksym().Set(obj.AttrContentAddressable, true)
	return n
}

func isSimpleName(n ir.Node) bool {
	return (n.Op() == ir.ONAME || n.Op() == ir.OMETHEXPR) && n.Class() != ir.PAUTOHEAP && n.Class() != ir.PEXTERN
}

func litas(l ir.Node, r ir.Node, init *ir.Nodes) {
	a := ir.Nod(ir.OAS, l, r)
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
func getdyn(n ir.Node, top bool) initGenType {
	switch n.Op() {
	default:
		if isGoConst(n) {
			return initConst
		}
		return initDynamic

	case ir.OSLICELIT:
		if !top {
			return initDynamic
		}
		if n.Right().Int64Val()/4 > int64(n.List().Len()) {
			// <25% of entries have explicit values.
			// Very rough estimation, it takes 4 bytes of instructions
			// to initialize 1 byte of result. So don't use a static
			// initializer if the dynamic initialization code would be
			// smaller than the static value.
			// See issue 23780.
			return initDynamic
		}

	case ir.OARRAYLIT, ir.OSTRUCTLIT:
	}

	var mode initGenType
	for _, n1 := range n.List().Slice() {
		switch n1.Op() {
		case ir.OKEY:
			n1 = n1.Right()
		case ir.OSTRUCTKEY:
			n1 = n1.Left()
		}
		mode |= getdyn(n1, false)
		if mode == initDynamic|initConst {
			break
		}
	}
	return mode
}

// isStaticCompositeLiteral reports whether n is a compile-time constant.
func isStaticCompositeLiteral(n ir.Node) bool {
	switch n.Op() {
	case ir.OSLICELIT:
		return false
	case ir.OARRAYLIT:
		for _, r := range n.List().Slice() {
			if r.Op() == ir.OKEY {
				r = r.Right()
			}
			if !isStaticCompositeLiteral(r) {
				return false
			}
		}
		return true
	case ir.OSTRUCTLIT:
		for _, r := range n.List().Slice() {
			if r.Op() != ir.OSTRUCTKEY {
				base.Fatalf("isStaticCompositeLiteral: rhs not OSTRUCTKEY: %v", r)
			}
			if !isStaticCompositeLiteral(r.Left()) {
				return false
			}
		}
		return true
	case ir.OLITERAL, ir.ONIL:
		return true
	case ir.OCONVIFACE:
		// See staticassign's OCONVIFACE case for comments.
		val := n
		for val.Op() == ir.OCONVIFACE {
			val = val.Left()
		}
		if val.Type().IsInterface() {
			return val.Op() == ir.ONIL
		}
		if isdirectiface(val.Type()) && val.Op() == ir.ONIL {
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
func fixedlit(ctxt initContext, kind initKind, n ir.Node, var_ ir.Node, init *ir.Nodes) {
	isBlank := var_ == ir.BlankNode
	var splitnode func(ir.Node) (a ir.Node, value ir.Node)
	switch n.Op() {
	case ir.OARRAYLIT, ir.OSLICELIT:
		var k int64
		splitnode = func(r ir.Node) (ir.Node, ir.Node) {
			if r.Op() == ir.OKEY {
				k = indexconst(r.Left())
				if k < 0 {
					base.Fatalf("fixedlit: invalid index %v", r.Left())
				}
				r = r.Right()
			}
			a := ir.Nod(ir.OINDEX, var_, nodintconst(k))
			k++
			if isBlank {
				a = ir.BlankNode
			}
			return a, r
		}
	case ir.OSTRUCTLIT:
		splitnode = func(r ir.Node) (ir.Node, ir.Node) {
			if r.Op() != ir.OSTRUCTKEY {
				base.Fatalf("fixedlit: rhs not OSTRUCTKEY: %v", r)
			}
			if r.Sym().IsBlank() || isBlank {
				return ir.BlankNode, r.Left()
			}
			setlineno(r)
			return nodSym(ir.ODOT, var_, r.Sym()), r.Left()
		}
	default:
		base.Fatalf("fixedlit bad op: %v", n.Op())
	}

	for _, r := range n.List().Slice() {
		a, value := splitnode(r)
		if a == ir.BlankNode && candiscard(value) {
			continue
		}

		switch value.Op() {
		case ir.OSLICELIT:
			if (kind == initKindStatic && ctxt == inNonInitFunction) || (kind == initKindDynamic && ctxt == inInitFunction) {
				slicelit(ctxt, value, a, init)
				continue
			}

		case ir.OARRAYLIT, ir.OSTRUCTLIT:
			fixedlit(ctxt, kind, value, a, init)
			continue
		}

		islit := isGoConst(value)
		if (kind == initKindStatic && !islit) || (kind == initKindDynamic && islit) {
			continue
		}

		// build list of assignments: var[index] = expr
		setlineno(a)
		a = ir.Nod(ir.OAS, a, value)
		a = typecheck(a, ctxStmt)
		switch kind {
		case initKindStatic:
			genAsStatic(a)
		case initKindDynamic, initKindLocalCode:
			a = orderStmtInPlace(a, map[string][]ir.Node{})
			a = walkstmt(a)
			init.Append(a)
		default:
			base.Fatalf("fixedlit: bad kind %d", kind)
		}

	}
}

func isSmallSliceLit(n ir.Node) bool {
	if n.Op() != ir.OSLICELIT {
		return false
	}

	r := n.Right()

	return smallintconst(r) && (n.Type().Elem().Width == 0 || r.Int64Val() <= smallArrayBytes/n.Type().Elem().Width)
}

func slicelit(ctxt initContext, n ir.Node, var_ ir.Node, init *ir.Nodes) {
	// make an array type corresponding the number of elements we have
	t := types.NewArray(n.Type().Elem(), n.Right().Int64Val())
	dowidth(t)

	if ctxt == inNonInitFunction {
		// put everything into static array
		vstat := staticname(t)

		fixedlit(ctxt, initKindStatic, n, vstat, init)
		fixedlit(ctxt, initKindDynamic, n, vstat, init)

		// copy static to slice
		var_ = typecheck(var_, ctxExpr|ctxAssign)
		nam := stataddr(var_)
		if nam == nil || nam.Class() != ir.PEXTERN {
			base.Fatalf("slicelit: %v", var_)
		}
		slicesym(nam, vstat, t.NumElem())
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
	var vstat ir.Node

	mode := getdyn(n, true)
	if mode&initConst != 0 && !isSmallSliceLit(n) {
		if ctxt == inInitFunction {
			vstat = readonlystaticname(t)
		} else {
			vstat = staticname(t)
		}
		fixedlit(ctxt, initKindStatic, n, vstat, init)
	}

	// make new auto *array (3 declare)
	vauto := temp(types.NewPtr(t))

	// set auto to point at new temp or heap (3 assign)
	var a ir.Node
	if x := prealloc[n]; x != nil {
		// temp allocated during order.go for dddarg
		if !types.Identical(t, x.Type()) {
			panic("dotdotdot base type does not match order's assigned type")
		}

		if vstat == nil {
			a = ir.Nod(ir.OAS, x, nil)
			a = typecheck(a, ctxStmt)
			init.Append(a) // zero new temp
		} else {
			// Declare that we're about to initialize all of x.
			// (Which happens at the *vauto = vstat below.)
			init.Append(ir.Nod(ir.OVARDEF, x, nil))
		}

		a = ir.Nod(ir.OADDR, x, nil)
	} else if n.Esc() == EscNone {
		a = temp(t)
		if vstat == nil {
			a = ir.Nod(ir.OAS, temp(t), nil)
			a = typecheck(a, ctxStmt)
			init.Append(a) // zero new temp
			a = a.Left()
		} else {
			init.Append(ir.Nod(ir.OVARDEF, a, nil))
		}

		a = ir.Nod(ir.OADDR, a, nil)
	} else {
		a = ir.Nod(ir.ONEW, ir.TypeNode(t), nil)
	}

	a = ir.Nod(ir.OAS, vauto, a)
	a = typecheck(a, ctxStmt)
	a = walkexpr(a, init)
	init.Append(a)

	if vstat != nil {
		// copy static to heap (4)
		a = ir.Nod(ir.ODEREF, vauto, nil)

		a = ir.Nod(ir.OAS, a, vstat)
		a = typecheck(a, ctxStmt)
		a = walkexpr(a, init)
		init.Append(a)
	}

	// put dynamics into array (5)
	var index int64
	for _, value := range n.List().Slice() {
		if value.Op() == ir.OKEY {
			index = indexconst(value.Left())
			if index < 0 {
				base.Fatalf("slicelit: invalid index %v", value.Left())
			}
			value = value.Right()
		}
		a := ir.Nod(ir.OINDEX, vauto, nodintconst(index))
		a.SetBounded(true)
		index++

		// TODO need to check bounds?

		switch value.Op() {
		case ir.OSLICELIT:
			break

		case ir.OARRAYLIT, ir.OSTRUCTLIT:
			k := initKindDynamic
			if vstat == nil {
				// Generate both static and dynamic initializations.
				// See issue #31987.
				k = initKindLocalCode
			}
			fixedlit(ctxt, k, value, a, init)
			continue
		}

		if vstat != nil && isGoConst(value) { // already set by copy from static value
			continue
		}

		// build list of vauto[c] = expr
		setlineno(value)
		a = ir.Nod(ir.OAS, a, value)

		a = typecheck(a, ctxStmt)
		a = orderStmtInPlace(a, map[string][]ir.Node{})
		a = walkstmt(a)
		init.Append(a)
	}

	// make slice out of heap (6)
	a = ir.Nod(ir.OAS, var_, ir.Nod(ir.OSLICE, vauto, nil))

	a = typecheck(a, ctxStmt)
	a = orderStmtInPlace(a, map[string][]ir.Node{})
	a = walkstmt(a)
	init.Append(a)
}

func maplit(n ir.Node, m ir.Node, init *ir.Nodes) {
	// make the map var
	a := ir.Nod(ir.OMAKE, nil, nil)
	a.SetEsc(n.Esc())
	a.PtrList().Set2(ir.TypeNode(n.Type()), nodintconst(int64(n.List().Len())))
	litas(m, a, init)

	entries := n.List().Slice()

	// The order pass already removed any dynamic (runtime-computed) entries.
	// All remaining entries are static. Double-check that.
	for _, r := range entries {
		if !isStaticCompositeLiteral(r.Left()) || !isStaticCompositeLiteral(r.Right()) {
			base.Fatalf("maplit: entry is not a literal: %v", r)
		}
	}

	if len(entries) > 25 {
		// For a large number of entries, put them in an array and loop.

		// build types [count]Tindex and [count]Tvalue
		tk := types.NewArray(n.Type().Key(), int64(len(entries)))
		te := types.NewArray(n.Type().Elem(), int64(len(entries)))

		tk.SetNoalg(true)
		te.SetNoalg(true)

		dowidth(tk)
		dowidth(te)

		// make and initialize static arrays
		vstatk := readonlystaticname(tk)
		vstate := readonlystaticname(te)

		datak := ir.Nod(ir.OARRAYLIT, nil, nil)
		datae := ir.Nod(ir.OARRAYLIT, nil, nil)
		for _, r := range entries {
			datak.PtrList().Append(r.Left())
			datae.PtrList().Append(r.Right())
		}
		fixedlit(inInitFunction, initKindStatic, datak, vstatk, init)
		fixedlit(inInitFunction, initKindStatic, datae, vstate, init)

		// loop adding structure elements to map
		// for i = 0; i < len(vstatk); i++ {
		//	map[vstatk[i]] = vstate[i]
		// }
		i := temp(types.Types[types.TINT])
		rhs := ir.Nod(ir.OINDEX, vstate, i)
		rhs.SetBounded(true)

		kidx := ir.Nod(ir.OINDEX, vstatk, i)
		kidx.SetBounded(true)
		lhs := ir.Nod(ir.OINDEX, m, kidx)

		zero := ir.Nod(ir.OAS, i, nodintconst(0))
		cond := ir.Nod(ir.OLT, i, nodintconst(tk.NumElem()))
		incr := ir.Nod(ir.OAS, i, ir.Nod(ir.OADD, i, nodintconst(1)))
		body := ir.Nod(ir.OAS, lhs, rhs)

		loop := ir.Nod(ir.OFOR, cond, incr)
		loop.PtrBody().Set1(body)
		loop.PtrInit().Set1(zero)

		loop = typecheck(loop, ctxStmt)
		loop = walkstmt(loop)
		init.Append(loop)
		return
	}
	// For a small number of entries, just add them directly.

	// Build list of var[c] = expr.
	// Use temporaries so that mapassign1 can have addressable key, elem.
	// TODO(josharian): avoid map key temporaries for mapfast_* assignments with literal keys.
	tmpkey := temp(m.Type().Key())
	tmpelem := temp(m.Type().Elem())

	for _, r := range entries {
		index, elem := r.Left(), r.Right()

		setlineno(index)
		a := ir.Nod(ir.OAS, tmpkey, index)
		a = typecheck(a, ctxStmt)
		a = walkstmt(a)
		init.Append(a)

		setlineno(elem)
		a = ir.Nod(ir.OAS, tmpelem, elem)
		a = typecheck(a, ctxStmt)
		a = walkstmt(a)
		init.Append(a)

		setlineno(tmpelem)
		a = ir.Nod(ir.OAS, ir.Nod(ir.OINDEX, m, tmpkey), tmpelem)
		a = typecheck(a, ctxStmt)
		a = walkstmt(a)
		init.Append(a)
	}

	a = ir.Nod(ir.OVARKILL, tmpkey, nil)
	a = typecheck(a, ctxStmt)
	init.Append(a)
	a = ir.Nod(ir.OVARKILL, tmpelem, nil)
	a = typecheck(a, ctxStmt)
	init.Append(a)
}

func anylit(n ir.Node, var_ ir.Node, init *ir.Nodes) {
	t := n.Type()
	switch n.Op() {
	default:
		base.Fatalf("anylit: not lit, op=%v node=%v", n.Op(), n)

	case ir.ONAME, ir.OMETHEXPR:
		a := ir.Nod(ir.OAS, var_, n)
		a = typecheck(a, ctxStmt)
		init.Append(a)

	case ir.OPTRLIT:
		if !t.IsPtr() {
			base.Fatalf("anylit: not ptr")
		}

		var r ir.Node
		if n.Right() != nil {
			// n.Right is stack temporary used as backing store.
			init.Append(ir.Nod(ir.OAS, n.Right(), nil)) // zero backing store, just in case (#18410)
			r = ir.Nod(ir.OADDR, n.Right(), nil)
			r = typecheck(r, ctxExpr)
		} else {
			r = ir.Nod(ir.ONEW, ir.TypeNode(n.Left().Type()), nil)
			r = typecheck(r, ctxExpr)
			r.SetEsc(n.Esc())
		}

		r = walkexpr(r, init)
		a := ir.Nod(ir.OAS, var_, r)

		a = typecheck(a, ctxStmt)
		init.Append(a)

		var_ = ir.Nod(ir.ODEREF, var_, nil)
		var_ = typecheck(var_, ctxExpr|ctxAssign)
		anylit(n.Left(), var_, init)

	case ir.OSTRUCTLIT, ir.OARRAYLIT:
		if !t.IsStruct() && !t.IsArray() {
			base.Fatalf("anylit: not struct/array")
		}

		if isSimpleName(var_) && n.List().Len() > 4 {
			// lay out static data
			vstat := readonlystaticname(t)

			ctxt := inInitFunction
			if n.Op() == ir.OARRAYLIT {
				ctxt = inNonInitFunction
			}
			fixedlit(ctxt, initKindStatic, n, vstat, init)

			// copy static to var
			a := ir.Nod(ir.OAS, var_, vstat)

			a = typecheck(a, ctxStmt)
			a = walkexpr(a, init)
			init.Append(a)

			// add expressions to automatic
			fixedlit(inInitFunction, initKindDynamic, n, var_, init)
			break
		}

		var components int64
		if n.Op() == ir.OARRAYLIT {
			components = t.NumElem()
		} else {
			components = int64(t.NumFields())
		}
		// initialization of an array or struct with unspecified components (missing fields or arrays)
		if isSimpleName(var_) || int64(n.List().Len()) < components {
			a := ir.Nod(ir.OAS, var_, nil)
			a = typecheck(a, ctxStmt)
			a = walkexpr(a, init)
			init.Append(a)
		}

		fixedlit(inInitFunction, initKindLocalCode, n, var_, init)

	case ir.OSLICELIT:
		slicelit(inInitFunction, n, var_, init)

	case ir.OMAPLIT:
		if !t.IsMap() {
			base.Fatalf("anylit: not map")
		}
		maplit(n, var_, init)
	}
}

// oaslit handles special composite literal assignments.
// It returns true if n's effects have been added to init,
// in which case n should be dropped from the program by the caller.
func oaslit(n ir.Node, init *ir.Nodes) bool {
	if n.Left() == nil || n.Right() == nil {
		// not a special composite literal assignment
		return false
	}
	if n.Left().Type() == nil || n.Right().Type() == nil {
		// not a special composite literal assignment
		return false
	}
	if !isSimpleName(n.Left()) {
		// not a special composite literal assignment
		return false
	}
	if !types.Identical(n.Left().Type(), n.Right().Type()) {
		// not a special composite literal assignment
		return false
	}

	switch n.Right().Op() {
	default:
		// not a special composite literal assignment
		return false

	case ir.OSTRUCTLIT, ir.OARRAYLIT, ir.OSLICELIT, ir.OMAPLIT:
		if vmatch1(n.Left(), n.Right()) {
			// not a special composite literal assignment
			return false
		}
		anylit(n.Right(), n.Left(), init)
	}

	return true
}

func getlit(lit ir.Node) int {
	if smallintconst(lit) {
		return int(lit.Int64Val())
	}
	return -1
}

// stataddr returns the static address of n, if n has one, or else nil.
func stataddr(n ir.Node) ir.Node {
	if n == nil {
		return nil
	}

	switch n.Op() {
	case ir.ONAME, ir.OMETHEXPR:
		return ir.SepCopy(n)

	case ir.ODOT:
		nam := stataddr(n.Left())
		if nam == nil {
			break
		}
		nam.SetOffset(nam.Offset() + n.Offset())
		nam.SetType(n.Type())
		return nam

	case ir.OINDEX:
		if n.Left().Type().IsSlice() {
			break
		}
		nam := stataddr(n.Left())
		if nam == nil {
			break
		}
		l := getlit(n.Right())
		if l < 0 {
			break
		}

		// Check for overflow.
		if n.Type().Width != 0 && thearch.MAXWIDTH/n.Type().Width <= int64(l) {
			break
		}
		nam.SetOffset(nam.Offset() + int64(l)*n.Type().Width)
		nam.SetType(n.Type())
		return nam
	}

	return nil
}

func (s *InitSchedule) initplan(n ir.Node) {
	if s.initplans[n] != nil {
		return
	}
	p := new(InitPlan)
	s.initplans[n] = p
	switch n.Op() {
	default:
		base.Fatalf("initplan")

	case ir.OARRAYLIT, ir.OSLICELIT:
		var k int64
		for _, a := range n.List().Slice() {
			if a.Op() == ir.OKEY {
				k = indexconst(a.Left())
				if k < 0 {
					base.Fatalf("initplan arraylit: invalid index %v", a.Left())
				}
				a = a.Right()
			}
			s.addvalue(p, k*n.Type().Elem().Width, a)
			k++
		}

	case ir.OSTRUCTLIT:
		for _, a := range n.List().Slice() {
			if a.Op() != ir.OSTRUCTKEY {
				base.Fatalf("initplan structlit")
			}
			if a.Sym().IsBlank() {
				continue
			}
			s.addvalue(p, a.Offset(), a.Left())
		}

	case ir.OMAPLIT:
		for _, a := range n.List().Slice() {
			if a.Op() != ir.OKEY {
				base.Fatalf("initplan maplit")
			}
			s.addvalue(p, -1, a.Right())
		}
	}
}

func (s *InitSchedule) addvalue(p *InitPlan, xoffset int64, n ir.Node) {
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

func isZero(n ir.Node) bool {
	switch n.Op() {
	case ir.ONIL:
		return true

	case ir.OLITERAL:
		switch u := n.Val(); u.Kind() {
		case constant.String:
			return constant.StringVal(u) == ""
		case constant.Bool:
			return !constant.BoolVal(u)
		default:
			return constant.Sign(u) == 0
		}

	case ir.OARRAYLIT:
		for _, n1 := range n.List().Slice() {
			if n1.Op() == ir.OKEY {
				n1 = n1.Right()
			}
			if !isZero(n1) {
				return false
			}
		}
		return true

	case ir.OSTRUCTLIT:
		for _, n1 := range n.List().Slice() {
			if !isZero(n1.Left()) {
				return false
			}
		}
		return true
	}

	return false
}

func isvaluelit(n ir.Node) bool {
	return n.Op() == ir.OARRAYLIT || n.Op() == ir.OSTRUCTLIT
}

func genAsStatic(as ir.Node) {
	if as.Left().Type() == nil {
		base.Fatalf("genAsStatic as.Left not typechecked")
	}

	nam := stataddr(as.Left())
	if nam == nil || (nam.Class() != ir.PEXTERN && as.Left() != ir.BlankNode) {
		base.Fatalf("genAsStatic: lhs %v", as.Left())
	}

	switch {
	case as.Right().Op() == ir.OLITERAL:
		litsym(nam, as.Right(), int(as.Right().Type().Width))
	case (as.Right().Op() == ir.ONAME || as.Right().Op() == ir.OMETHEXPR) && as.Right().Class() == ir.PFUNC:
		pfuncsym(nam, as.Right())
	default:
		base.Fatalf("genAsStatic: rhs %v", as.Right())
	}
}
