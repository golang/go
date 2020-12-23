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
	inittemps map[ir.Node]*ir.Name
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
func (s *InitSchedule) tryStaticInit(nn ir.Node) bool {
	// Only worry about simple "l = r" assignments. Multiple
	// variable/expression OAS2 assignments have already been
	// replaced by multiple simple OAS assignments, and the other
	// OAS2* assignments mostly necessitate dynamic execution
	// anyway.
	if nn.Op() != ir.OAS {
		return false
	}
	n := nn.(*ir.AssignStmt)
	if ir.IsBlank(n.Left()) && !anySideEffects(n.Right()) {
		// Discard.
		return true
	}
	lno := setlineno(n)
	defer func() { base.Pos = lno }()
	nam := n.Left().(*ir.Name)
	return s.staticassign(nam, 0, n.Right(), nam.Type())
}

// like staticassign but we are copying an already
// initialized value r.
func (s *InitSchedule) staticcopy(l *ir.Name, loff int64, rn *ir.Name, typ *types.Type) bool {
	if rn.Class() == ir.PFUNC {
		// TODO if roff != 0 { panic }
		pfuncsym(l, loff, rn)
		return true
	}
	if rn.Class() != ir.PEXTERN || rn.Sym().Pkg != types.LocalPkg {
		return false
	}
	if rn.Defn == nil { // probably zeroed but perhaps supplied externally and of unknown value
		return false
	}
	if rn.Defn.Op() != ir.OAS {
		return false
	}
	if rn.Type().IsString() { // perhaps overwritten by cmd/link -X (#34675)
		return false
	}
	orig := rn
	r := rn.Defn.(*ir.AssignStmt).Right()

	for r.Op() == ir.OCONVNOP && !types.Identical(r.Type(), typ) {
		r = r.(*ir.ConvExpr).Left()
	}

	switch r.Op() {
	case ir.OMETHEXPR:
		r = r.(*ir.MethodExpr).FuncName()
		fallthrough
	case ir.ONAME:
		r := r.(*ir.Name)
		if s.staticcopy(l, loff, r, typ) {
			return true
		}
		// We may have skipped past one or more OCONVNOPs, so
		// use conv to ensure r is assignable to l (#13263).
		dst := ir.Node(l)
		if loff != 0 || !types.Identical(typ, l.Type()) {
			dst = ir.NewNameOffsetExpr(base.Pos, l, loff, typ)
		}
		s.append(ir.NewAssignStmt(base.Pos, dst, conv(r, typ)))
		return true

	case ir.ONIL:
		return true

	case ir.OLITERAL:
		if isZero(r) {
			return true
		}
		litsym(l, loff, r, int(typ.Width))
		return true

	case ir.OADDR:
		if a := r.Left(); a.Op() == ir.ONAME {
			a := a.(*ir.Name)
			addrsym(l, loff, a, 0)
			return true
		}

	case ir.OPTRLIT:
		switch r.Left().Op() {
		case ir.OARRAYLIT, ir.OSLICELIT, ir.OSTRUCTLIT, ir.OMAPLIT:
			// copy pointer
			addrsym(l, loff, s.inittemps[r], 0)
			return true
		}

	case ir.OSLICELIT:
		r := r.(*ir.CompLitExpr)
		// copy slice
		slicesym(l, loff, s.inittemps[r], r.Len)
		return true

	case ir.OARRAYLIT, ir.OSTRUCTLIT:
		p := s.initplans[r]
		for i := range p.E {
			e := &p.E[i]
			typ := e.Expr.Type()
			if e.Expr.Op() == ir.OLITERAL || e.Expr.Op() == ir.ONIL {
				litsym(l, loff+e.Xoffset, e.Expr, int(typ.Width))
				continue
			}
			x := e.Expr
			if x.Op() == ir.OMETHEXPR {
				x = x.(*ir.MethodExpr).FuncName()
			}
			if x.Op() == ir.ONAME && s.staticcopy(l, loff+e.Xoffset, x.(*ir.Name), typ) {
				continue
			}
			// Requires computation, but we're
			// copying someone else's computation.
			ll := ir.NewNameOffsetExpr(base.Pos, l, loff+e.Xoffset, typ)
			rr := ir.NewNameOffsetExpr(base.Pos, orig, e.Xoffset, typ)
			setlineno(rr)
			s.append(ir.NewAssignStmt(base.Pos, ll, rr))
		}

		return true
	}

	return false
}

func (s *InitSchedule) staticassign(l *ir.Name, loff int64, r ir.Node, typ *types.Type) bool {
	for r.Op() == ir.OCONVNOP {
		r = r.(*ir.ConvExpr).Left()
	}

	switch r.Op() {
	case ir.ONAME:
		r := r.(*ir.Name)
		return s.staticcopy(l, loff, r, typ)

	case ir.OMETHEXPR:
		r := r.(*ir.MethodExpr)
		return s.staticcopy(l, loff, r.FuncName(), typ)

	case ir.ONIL:
		return true

	case ir.OLITERAL:
		if isZero(r) {
			return true
		}
		litsym(l, loff, r, int(typ.Width))
		return true

	case ir.OADDR:
		if name, offset, ok := stataddr(r.Left()); ok {
			addrsym(l, loff, name, offset)
			return true
		}
		fallthrough

	case ir.OPTRLIT:
		switch r.Left().Op() {
		case ir.OARRAYLIT, ir.OSLICELIT, ir.OMAPLIT, ir.OSTRUCTLIT:
			// Init pointer.
			a := staticname(r.Left().Type())

			s.inittemps[r] = a
			addrsym(l, loff, a, 0)

			// Init underlying literal.
			if !s.staticassign(a, 0, r.Left(), a.Type()) {
				s.append(ir.NewAssignStmt(base.Pos, a, r.Left()))
			}
			return true
		}
		//dump("not static ptrlit", r);

	case ir.OSTR2BYTES:
		if l.Class() == ir.PEXTERN && r.Left().Op() == ir.OLITERAL {
			sval := ir.StringVal(r.Left())
			slicebytes(l, loff, sval)
			return true
		}

	case ir.OSLICELIT:
		r := r.(*ir.CompLitExpr)
		s.initplan(r)
		// Init slice.
		ta := types.NewArray(r.Type().Elem(), r.Len)
		ta.SetNoalg(true)
		a := staticname(ta)
		s.inittemps[r] = a
		slicesym(l, loff, a, r.Len)
		// Fall through to init underlying array.
		l = a
		loff = 0
		fallthrough

	case ir.OARRAYLIT, ir.OSTRUCTLIT:
		s.initplan(r)

		p := s.initplans[r]
		for i := range p.E {
			e := &p.E[i]
			if e.Expr.Op() == ir.OLITERAL || e.Expr.Op() == ir.ONIL {
				litsym(l, loff+e.Xoffset, e.Expr, int(e.Expr.Type().Width))
				continue
			}
			setlineno(e.Expr)
			if !s.staticassign(l, loff+e.Xoffset, e.Expr, e.Expr.Type()) {
				a := ir.NewNameOffsetExpr(base.Pos, l, loff+e.Xoffset, e.Expr.Type())
				s.append(ir.NewAssignStmt(base.Pos, a, e.Expr))
			}
		}

		return true

	case ir.OMAPLIT:
		break

	case ir.OCLOSURE:
		r := r.(*ir.ClosureExpr)
		if hasemptycvars(r) {
			if base.Debug.Closure > 0 {
				base.WarnfAt(r.Pos(), "closure converted to global")
			}
			// Closures with no captured variables are globals,
			// so the assignment can be done at link time.
			// TODO if roff != 0 { panic }
			pfuncsym(l, loff, r.Func().Nname)
			return true
		}
		closuredebugruntimecheck(r)

	case ir.OCONVIFACE:
		// This logic is mirrored in isStaticCompositeLiteral.
		// If you change something here, change it there, and vice versa.

		// Determine the underlying concrete type and value we are converting from.
		val := ir.Node(r)
		for val.Op() == ir.OCONVIFACE {
			val = val.(*ir.ConvExpr).Left()
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

		var itab *ir.AddrExpr
		if typ.IsEmptyInterface() {
			itab = typename(val.Type())
		} else {
			itab = itabname(val.Type(), typ)
		}

		// Create a copy of l to modify while we emit data.

		// Emit itab, advance offset.
		addrsym(l, loff, itab.Left().(*ir.Name), 0)

		// Emit data.
		if isdirectiface(val.Type()) {
			if val.Op() == ir.ONIL {
				// Nil is zero, nothing to do.
				return true
			}
			// Copy val directly into n.
			setlineno(val)
			if !s.staticassign(l, loff+int64(Widthptr), val, val.Type()) {
				a := ir.NewNameOffsetExpr(base.Pos, l, loff+int64(Widthptr), val.Type())
				s.append(ir.NewAssignStmt(base.Pos, a, val))
			}
		} else {
			// Construct temp to hold val, write pointer to temp into n.
			a := staticname(val.Type())
			s.inittemps[val] = a
			if !s.staticassign(a, 0, val, val.Type()) {
				s.append(ir.NewAssignStmt(base.Pos, a, val))
			}
			addrsym(l, loff+int64(Widthptr), a, 0)
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
func staticname(t *types.Type) *ir.Name {
	// Don't use lookupN; it interns the resulting string, but these are all unique.
	n := NewName(lookup(fmt.Sprintf("%s%d", obj.StaticNamePref, statuniqgen)))
	statuniqgen++
	declare(n, ir.PEXTERN)
	n.SetType(t)
	n.Sym().Linksym().Set(obj.AttrLocal, true)
	return n
}

// readonlystaticname returns a name backed by a (writable) static data symbol.
func readonlystaticname(t *types.Type) *ir.Name {
	n := staticname(t)
	n.MarkReadonly()
	n.Sym().Linksym().Set(obj.AttrContentAddressable, true)
	return n
}

func isSimpleName(nn ir.Node) bool {
	if nn.Op() != ir.ONAME {
		return false
	}
	n := nn.(*ir.Name)
	return n.Class() != ir.PAUTOHEAP && n.Class() != ir.PEXTERN
}

func litas(l ir.Node, r ir.Node, init *ir.Nodes) {
	appendWalkStmt(init, ir.NewAssignStmt(base.Pos, l, r))
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
		n := n.(*ir.CompLitExpr)
		if !top {
			return initDynamic
		}
		if n.Len/4 > int64(n.List().Len()) {
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
	lit := n.(*ir.CompLitExpr)

	var mode initGenType
	for _, n1 := range lit.List().Slice() {
		switch n1.Op() {
		case ir.OKEY:
			n1 = n1.(*ir.KeyExpr).Right()
		case ir.OSTRUCTKEY:
			n1 = n1.(*ir.StructKeyExpr).Left()
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
				r = r.(*ir.KeyExpr).Right()
			}
			if !isStaticCompositeLiteral(r) {
				return false
			}
		}
		return true
	case ir.OSTRUCTLIT:
		for _, r := range n.List().Slice() {
			r := r.(*ir.StructKeyExpr)
			if !isStaticCompositeLiteral(r.Left()) {
				return false
			}
		}
		return true
	case ir.OLITERAL, ir.ONIL:
		return true
	case ir.OCONVIFACE:
		// See staticassign's OCONVIFACE case for comments.
		val := ir.Node(n)
		for val.Op() == ir.OCONVIFACE {
			val = val.(*ir.ConvExpr).Left()
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
func fixedlit(ctxt initContext, kind initKind, n *ir.CompLitExpr, var_ ir.Node, init *ir.Nodes) {
	isBlank := var_ == ir.BlankNode
	var splitnode func(ir.Node) (a ir.Node, value ir.Node)
	switch n.Op() {
	case ir.OARRAYLIT, ir.OSLICELIT:
		var k int64
		splitnode = func(r ir.Node) (ir.Node, ir.Node) {
			if r.Op() == ir.OKEY {
				kv := r.(*ir.KeyExpr)
				k = indexconst(kv.Left())
				if k < 0 {
					base.Fatalf("fixedlit: invalid index %v", kv.Left())
				}
				r = kv.Right()
			}
			a := ir.NewIndexExpr(base.Pos, var_, nodintconst(k))
			k++
			if isBlank {
				return ir.BlankNode, r
			}
			return a, r
		}
	case ir.OSTRUCTLIT:
		splitnode = func(rn ir.Node) (ir.Node, ir.Node) {
			r := rn.(*ir.StructKeyExpr)
			if r.Sym().IsBlank() || isBlank {
				return ir.BlankNode, r.Left()
			}
			setlineno(r)
			return ir.NewSelectorExpr(base.Pos, ir.ODOT, var_, r.Sym()), r.Left()
		}
	default:
		base.Fatalf("fixedlit bad op: %v", n.Op())
	}

	for _, r := range n.List().Slice() {
		a, value := splitnode(r)
		if a == ir.BlankNode && !anySideEffects(value) {
			// Discard.
			continue
		}

		switch value.Op() {
		case ir.OSLICELIT:
			value := value.(*ir.CompLitExpr)
			if (kind == initKindStatic && ctxt == inNonInitFunction) || (kind == initKindDynamic && ctxt == inInitFunction) {
				slicelit(ctxt, value, a, init)
				continue
			}

		case ir.OARRAYLIT, ir.OSTRUCTLIT:
			value := value.(*ir.CompLitExpr)
			fixedlit(ctxt, kind, value, a, init)
			continue
		}

		islit := isGoConst(value)
		if (kind == initKindStatic && !islit) || (kind == initKindDynamic && islit) {
			continue
		}

		// build list of assignments: var[index] = expr
		setlineno(a)
		as := ir.NewAssignStmt(base.Pos, a, value)
		as = typecheck(as, ctxStmt).(*ir.AssignStmt)
		switch kind {
		case initKindStatic:
			genAsStatic(as)
		case initKindDynamic, initKindLocalCode:
			a = orderStmtInPlace(as, map[string][]*ir.Name{})
			a = walkstmt(a)
			init.Append(a)
		default:
			base.Fatalf("fixedlit: bad kind %d", kind)
		}

	}
}

func isSmallSliceLit(n *ir.CompLitExpr) bool {
	if n.Op() != ir.OSLICELIT {
		return false
	}

	return n.Type().Elem().Width == 0 || n.Len <= smallArrayBytes/n.Type().Elem().Width
}

func slicelit(ctxt initContext, n *ir.CompLitExpr, var_ ir.Node, init *ir.Nodes) {
	// make an array type corresponding the number of elements we have
	t := types.NewArray(n.Type().Elem(), n.Len)
	dowidth(t)

	if ctxt == inNonInitFunction {
		// put everything into static array
		vstat := staticname(t)

		fixedlit(ctxt, initKindStatic, n, vstat, init)
		fixedlit(ctxt, initKindDynamic, n, vstat, init)

		// copy static to slice
		var_ = typecheck(var_, ctxExpr|ctxAssign)
		name, offset, ok := stataddr(var_)
		if !ok || name.Class() != ir.PEXTERN {
			base.Fatalf("slicelit: %v", var_)
		}
		slicesym(name, offset, vstat, t.NumElem())
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
	if x := n.Prealloc; x != nil {
		// temp allocated during order.go for dddarg
		if !types.Identical(t, x.Type()) {
			panic("dotdotdot base type does not match order's assigned type")
		}

		if vstat == nil {
			a = ir.NewAssignStmt(base.Pos, x, nil)
			a = typecheck(a, ctxStmt)
			init.Append(a) // zero new temp
		} else {
			// Declare that we're about to initialize all of x.
			// (Which happens at the *vauto = vstat below.)
			init.Append(ir.NewUnaryExpr(base.Pos, ir.OVARDEF, x))
		}

		a = nodAddr(x)
	} else if n.Esc() == EscNone {
		a = temp(t)
		if vstat == nil {
			a = ir.NewAssignStmt(base.Pos, temp(t), nil)
			a = typecheck(a, ctxStmt)
			init.Append(a) // zero new temp
			a = a.(*ir.AssignStmt).Left()
		} else {
			init.Append(ir.NewUnaryExpr(base.Pos, ir.OVARDEF, a))
		}

		a = nodAddr(a)
	} else {
		a = ir.NewUnaryExpr(base.Pos, ir.ONEW, ir.TypeNode(t))
	}
	appendWalkStmt(init, ir.NewAssignStmt(base.Pos, vauto, a))

	if vstat != nil {
		// copy static to heap (4)
		a = ir.NewStarExpr(base.Pos, vauto)
		appendWalkStmt(init, ir.NewAssignStmt(base.Pos, a, vstat))
	}

	// put dynamics into array (5)
	var index int64
	for _, value := range n.List().Slice() {
		if value.Op() == ir.OKEY {
			kv := value.(*ir.KeyExpr)
			index = indexconst(kv.Left())
			if index < 0 {
				base.Fatalf("slicelit: invalid index %v", kv.Left())
			}
			value = kv.Right()
		}
		a := ir.NewIndexExpr(base.Pos, vauto, nodintconst(index))
		a.SetBounded(true)
		index++

		// TODO need to check bounds?

		switch value.Op() {
		case ir.OSLICELIT:
			break

		case ir.OARRAYLIT, ir.OSTRUCTLIT:
			value := value.(*ir.CompLitExpr)
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
		as := typecheck(ir.NewAssignStmt(base.Pos, a, value), ctxStmt)
		as = orderStmtInPlace(as, map[string][]*ir.Name{})
		as = walkstmt(as)
		init.Append(as)
	}

	// make slice out of heap (6)
	a = ir.NewAssignStmt(base.Pos, var_, ir.NewSliceExpr(base.Pos, ir.OSLICE, vauto))

	a = typecheck(a, ctxStmt)
	a = orderStmtInPlace(a, map[string][]*ir.Name{})
	a = walkstmt(a)
	init.Append(a)
}

func maplit(n *ir.CompLitExpr, m ir.Node, init *ir.Nodes) {
	// make the map var
	a := ir.NewCallExpr(base.Pos, ir.OMAKE, nil, nil)
	a.SetEsc(n.Esc())
	a.PtrList().Set2(ir.TypeNode(n.Type()), nodintconst(int64(n.List().Len())))
	litas(m, a, init)

	entries := n.List().Slice()

	// The order pass already removed any dynamic (runtime-computed) entries.
	// All remaining entries are static. Double-check that.
	for _, r := range entries {
		r := r.(*ir.KeyExpr)
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

		datak := ir.NewCompLitExpr(base.Pos, ir.OARRAYLIT, nil, nil)
		datae := ir.NewCompLitExpr(base.Pos, ir.OARRAYLIT, nil, nil)
		for _, r := range entries {
			r := r.(*ir.KeyExpr)
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
		rhs := ir.NewIndexExpr(base.Pos, vstate, i)
		rhs.SetBounded(true)

		kidx := ir.NewIndexExpr(base.Pos, vstatk, i)
		kidx.SetBounded(true)
		lhs := ir.NewIndexExpr(base.Pos, m, kidx)

		zero := ir.NewAssignStmt(base.Pos, i, nodintconst(0))
		cond := ir.NewBinaryExpr(base.Pos, ir.OLT, i, nodintconst(tk.NumElem()))
		incr := ir.NewAssignStmt(base.Pos, i, ir.NewBinaryExpr(base.Pos, ir.OADD, i, nodintconst(1)))
		body := ir.NewAssignStmt(base.Pos, lhs, rhs)

		loop := ir.NewForStmt(base.Pos, nil, cond, incr, nil)
		loop.PtrBody().Set1(body)
		loop.PtrInit().Set1(zero)

		appendWalkStmt(init, loop)
		return
	}
	// For a small number of entries, just add them directly.

	// Build list of var[c] = expr.
	// Use temporaries so that mapassign1 can have addressable key, elem.
	// TODO(josharian): avoid map key temporaries for mapfast_* assignments with literal keys.
	tmpkey := temp(m.Type().Key())
	tmpelem := temp(m.Type().Elem())

	for _, r := range entries {
		r := r.(*ir.KeyExpr)
		index, elem := r.Left(), r.Right()

		setlineno(index)
		appendWalkStmt(init, ir.NewAssignStmt(base.Pos, tmpkey, index))

		setlineno(elem)
		appendWalkStmt(init, ir.NewAssignStmt(base.Pos, tmpelem, elem))

		setlineno(tmpelem)
		appendWalkStmt(init, ir.NewAssignStmt(base.Pos, ir.NewIndexExpr(base.Pos, m, tmpkey), tmpelem))
	}

	appendWalkStmt(init, ir.NewUnaryExpr(base.Pos, ir.OVARKILL, tmpkey))
	appendWalkStmt(init, ir.NewUnaryExpr(base.Pos, ir.OVARKILL, tmpelem))
}

func anylit(n ir.Node, var_ ir.Node, init *ir.Nodes) {
	t := n.Type()
	switch n.Op() {
	default:
		base.Fatalf("anylit: not lit, op=%v node=%v", n.Op(), n)

	case ir.ONAME:
		appendWalkStmt(init, ir.NewAssignStmt(base.Pos, var_, n))

	case ir.OMETHEXPR:
		n := n.(*ir.MethodExpr)
		anylit(n.FuncName(), var_, init)

	case ir.OPTRLIT:
		if !t.IsPtr() {
			base.Fatalf("anylit: not ptr")
		}

		var r ir.Node
		if n.Right() != nil {
			// n.Right is stack temporary used as backing store.
			appendWalkStmt(init, ir.NewAssignStmt(base.Pos, n.Right(), nil)) // zero backing store, just in case (#18410)
			r = nodAddr(n.Right())
		} else {
			r = ir.NewUnaryExpr(base.Pos, ir.ONEW, ir.TypeNode(n.Left().Type()))
			r.SetEsc(n.Esc())
		}
		appendWalkStmt(init, ir.NewAssignStmt(base.Pos, var_, r))

		var_ = ir.NewStarExpr(base.Pos, var_)
		var_ = typecheck(var_, ctxExpr|ctxAssign)
		anylit(n.Left(), var_, init)

	case ir.OSTRUCTLIT, ir.OARRAYLIT:
		n := n.(*ir.CompLitExpr)
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
			appendWalkStmt(init, ir.NewAssignStmt(base.Pos, var_, vstat))

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
			appendWalkStmt(init, ir.NewAssignStmt(base.Pos, var_, nil))
		}

		fixedlit(inInitFunction, initKindLocalCode, n, var_, init)

	case ir.OSLICELIT:
		n := n.(*ir.CompLitExpr)
		slicelit(inInitFunction, n, var_, init)

	case ir.OMAPLIT:
		n := n.(*ir.CompLitExpr)
		if !t.IsMap() {
			base.Fatalf("anylit: not map")
		}
		maplit(n, var_, init)
	}
}

// oaslit handles special composite literal assignments.
// It returns true if n's effects have been added to init,
// in which case n should be dropped from the program by the caller.
func oaslit(n *ir.AssignStmt, init *ir.Nodes) bool {
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
		if refersToCommonName(n.Left(), n.Right()) {
			// not a special composite literal assignment
			return false
		}
		anylit(n.Right(), n.Left(), init)
	}

	return true
}

func getlit(lit ir.Node) int {
	if smallintconst(lit) {
		return int(ir.Int64Val(lit))
	}
	return -1
}

// stataddr returns the static address of n, if n has one, or else nil.
func stataddr(n ir.Node) (name *ir.Name, offset int64, ok bool) {
	if n == nil {
		return nil, 0, false
	}

	switch n.Op() {
	case ir.ONAME:
		n := n.(*ir.Name)
		return n, 0, true

	case ir.OMETHEXPR:
		n := n.(*ir.MethodExpr)
		return stataddr(n.FuncName())

	case ir.ODOT:
		if name, offset, ok = stataddr(n.Left()); !ok {
			break
		}
		offset += n.Offset()
		return name, offset, true

	case ir.OINDEX:
		if n.Left().Type().IsSlice() {
			break
		}
		if name, offset, ok = stataddr(n.Left()); !ok {
			break
		}
		l := getlit(n.Right())
		if l < 0 {
			break
		}

		// Check for overflow.
		if n.Type().Width != 0 && MaxWidth/n.Type().Width <= int64(l) {
			break
		}
		offset += int64(l) * n.Type().Width
		return name, offset, true
	}

	return nil, 0, false
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
				kv := a.(*ir.KeyExpr)
				k = indexconst(kv.Left())
				if k < 0 {
					base.Fatalf("initplan arraylit: invalid index %v", kv.Left())
				}
				a = kv.Right()
			}
			s.addvalue(p, k*n.Type().Elem().Width, a)
			k++
		}

	case ir.OSTRUCTLIT:
		for _, a := range n.List().Slice() {
			if a.Op() != ir.OSTRUCTKEY {
				base.Fatalf("initplan structlit")
			}
			a := a.(*ir.StructKeyExpr)
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
			a := a.(*ir.KeyExpr)
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
				n1 = n1.(*ir.KeyExpr).Right()
			}
			if !isZero(n1) {
				return false
			}
		}
		return true

	case ir.OSTRUCTLIT:
		for _, n1 := range n.List().Slice() {
			n1 := n1.(*ir.StructKeyExpr)
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

func genAsStatic(as *ir.AssignStmt) {
	if as.Left().Type() == nil {
		base.Fatalf("genAsStatic as.Left not typechecked")
	}

	name, offset, ok := stataddr(as.Left())
	if !ok || (name.Class() != ir.PEXTERN && as.Left() != ir.BlankNode) {
		base.Fatalf("genAsStatic: lhs %v", as.Left())
	}

	switch r := as.Right(); r.Op() {
	case ir.OLITERAL:
		litsym(name, offset, r, int(r.Type().Width))
		return
	case ir.OMETHEXPR:
		r := r.(*ir.MethodExpr)
		pfuncsym(name, offset, r.FuncName())
		return
	case ir.ONAME:
		r := r.(*ir.Name)
		if r.Offset() != 0 {
			base.Fatalf("genAsStatic %+v", as)
		}
		if r.Class() == ir.PFUNC {
			pfuncsym(name, offset, r)
			return
		}
	}
	base.Fatalf("genAsStatic: rhs %v", as.Right())
}
