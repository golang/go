// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package walk

import (
	"go/constant"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
)

// The result of walkCompare MUST be assigned back to n, e.g.
// 	n.Left = walkCompare(n.Left, init)
func walkCompare(n *ir.BinaryExpr, init *ir.Nodes) ir.Node {
	if n.X.Type().IsInterface() && n.Y.Type().IsInterface() && n.X.Op() != ir.ONIL && n.Y.Op() != ir.ONIL {
		return walkCompareInterface(n, init)
	}

	if n.X.Type().IsString() && n.Y.Type().IsString() {
		return walkCompareString(n, init)
	}

	n.X = walkExpr(n.X, init)
	n.Y = walkExpr(n.Y, init)

	// Given mixed interface/concrete comparison,
	// rewrite into types-equal && data-equal.
	// This is efficient, avoids allocations, and avoids runtime calls.
	if n.X.Type().IsInterface() != n.Y.Type().IsInterface() {
		// Preserve side-effects in case of short-circuiting; see #32187.
		l := cheapExpr(n.X, init)
		r := cheapExpr(n.Y, init)
		// Swap so that l is the interface value and r is the concrete value.
		if n.Y.Type().IsInterface() {
			l, r = r, l
		}

		// Handle both == and !=.
		eq := n.Op()
		andor := ir.OOROR
		if eq == ir.OEQ {
			andor = ir.OANDAND
		}
		// Check for types equal.
		// For empty interface, this is:
		//   l.tab == type(r)
		// For non-empty interface, this is:
		//   l.tab != nil && l.tab._type == type(r)
		var eqtype ir.Node
		tab := ir.NewUnaryExpr(base.Pos, ir.OITAB, l)
		rtyp := reflectdata.TypePtr(r.Type())
		if l.Type().IsEmptyInterface() {
			tab.SetType(types.NewPtr(types.Types[types.TUINT8]))
			tab.SetTypecheck(1)
			eqtype = ir.NewBinaryExpr(base.Pos, eq, tab, rtyp)
		} else {
			nonnil := ir.NewBinaryExpr(base.Pos, brcom(eq), typecheck.NodNil(), tab)
			match := ir.NewBinaryExpr(base.Pos, eq, itabType(tab), rtyp)
			eqtype = ir.NewLogicalExpr(base.Pos, andor, nonnil, match)
		}
		// Check for data equal.
		eqdata := ir.NewBinaryExpr(base.Pos, eq, ifaceData(n.Pos(), l, r.Type()), r)
		// Put it all together.
		expr := ir.NewLogicalExpr(base.Pos, andor, eqtype, eqdata)
		return finishCompare(n, expr, init)
	}

	// Must be comparison of array or struct.
	// Otherwise back end handles it.
	// While we're here, decide whether to
	// inline or call an eq alg.
	t := n.X.Type()
	var inline bool

	maxcmpsize := int64(4)
	unalignedLoad := ssagen.Arch.LinkArch.CanMergeLoads
	if unalignedLoad {
		// Keep this low enough to generate less code than a function call.
		maxcmpsize = 2 * int64(ssagen.Arch.LinkArch.RegSize)
	}

	switch t.Kind() {
	default:
		if base.Debug.Libfuzzer != 0 && t.IsInteger() {
			n.X = cheapExpr(n.X, init)
			n.Y = cheapExpr(n.Y, init)

			// If exactly one comparison operand is
			// constant, invoke the constcmp functions
			// instead, and arrange for the constant
			// operand to be the first argument.
			l, r := n.X, n.Y
			if r.Op() == ir.OLITERAL {
				l, r = r, l
			}
			constcmp := l.Op() == ir.OLITERAL && r.Op() != ir.OLITERAL

			var fn string
			var paramType *types.Type
			switch t.Size() {
			case 1:
				fn = "libfuzzerTraceCmp1"
				if constcmp {
					fn = "libfuzzerTraceConstCmp1"
				}
				paramType = types.Types[types.TUINT8]
			case 2:
				fn = "libfuzzerTraceCmp2"
				if constcmp {
					fn = "libfuzzerTraceConstCmp2"
				}
				paramType = types.Types[types.TUINT16]
			case 4:
				fn = "libfuzzerTraceCmp4"
				if constcmp {
					fn = "libfuzzerTraceConstCmp4"
				}
				paramType = types.Types[types.TUINT32]
			case 8:
				fn = "libfuzzerTraceCmp8"
				if constcmp {
					fn = "libfuzzerTraceConstCmp8"
				}
				paramType = types.Types[types.TUINT64]
			default:
				base.Fatalf("unexpected integer size %d for %v", t.Size(), t)
			}
			init.Append(mkcall(fn, nil, init, tracecmpArg(l, paramType, init), tracecmpArg(r, paramType, init)))
		}
		return n
	case types.TARRAY:
		// We can compare several elements at once with 2/4/8 byte integer compares
		inline = t.NumElem() <= 1 || (types.IsSimple[t.Elem().Kind()] && (t.NumElem() <= 4 || t.Elem().Size()*t.NumElem() <= maxcmpsize))
	case types.TSTRUCT:
		inline = t.NumComponents(types.IgnoreBlankFields) <= 4
	}

	cmpl := n.X
	for cmpl != nil && cmpl.Op() == ir.OCONVNOP {
		cmpl = cmpl.(*ir.ConvExpr).X
	}
	cmpr := n.Y
	for cmpr != nil && cmpr.Op() == ir.OCONVNOP {
		cmpr = cmpr.(*ir.ConvExpr).X
	}

	// Chose not to inline. Call equality function directly.
	if !inline {
		// eq algs take pointers; cmpl and cmpr must be addressable
		if !ir.IsAddressable(cmpl) || !ir.IsAddressable(cmpr) {
			base.Fatalf("arguments of comparison must be lvalues - %v %v", cmpl, cmpr)
		}

		fn, needsize := eqFor(t)
		call := ir.NewCallExpr(base.Pos, ir.OCALL, fn, nil)
		call.Args.Append(typecheck.NodAddr(cmpl))
		call.Args.Append(typecheck.NodAddr(cmpr))
		if needsize {
			call.Args.Append(ir.NewInt(t.Size()))
		}
		res := ir.Node(call)
		if n.Op() != ir.OEQ {
			res = ir.NewUnaryExpr(base.Pos, ir.ONOT, res)
		}
		return finishCompare(n, res, init)
	}

	// inline: build boolean expression comparing element by element
	andor := ir.OANDAND
	if n.Op() == ir.ONE {
		andor = ir.OOROR
	}
	var expr ir.Node
	compare := func(el, er ir.Node) {
		a := ir.NewBinaryExpr(base.Pos, n.Op(), el, er)
		if expr == nil {
			expr = a
		} else {
			expr = ir.NewLogicalExpr(base.Pos, andor, expr, a)
		}
	}
	cmpl = safeExpr(cmpl, init)
	cmpr = safeExpr(cmpr, init)
	if t.IsStruct() {
		for _, f := range t.Fields().Slice() {
			sym := f.Sym
			if sym.IsBlank() {
				continue
			}
			compare(
				ir.NewSelectorExpr(base.Pos, ir.OXDOT, cmpl, sym),
				ir.NewSelectorExpr(base.Pos, ir.OXDOT, cmpr, sym),
			)
		}
	} else {
		step := int64(1)
		remains := t.NumElem() * t.Elem().Size()
		combine64bit := unalignedLoad && types.RegSize == 8 && t.Elem().Size() <= 4 && t.Elem().IsInteger()
		combine32bit := unalignedLoad && t.Elem().Size() <= 2 && t.Elem().IsInteger()
		combine16bit := unalignedLoad && t.Elem().Size() == 1 && t.Elem().IsInteger()
		for i := int64(0); remains > 0; {
			var convType *types.Type
			switch {
			case remains >= 8 && combine64bit:
				convType = types.Types[types.TINT64]
				step = 8 / t.Elem().Size()
			case remains >= 4 && combine32bit:
				convType = types.Types[types.TUINT32]
				step = 4 / t.Elem().Size()
			case remains >= 2 && combine16bit:
				convType = types.Types[types.TUINT16]
				step = 2 / t.Elem().Size()
			default:
				step = 1
			}
			if step == 1 {
				compare(
					ir.NewIndexExpr(base.Pos, cmpl, ir.NewInt(i)),
					ir.NewIndexExpr(base.Pos, cmpr, ir.NewInt(i)),
				)
				i++
				remains -= t.Elem().Size()
			} else {
				elemType := t.Elem().ToUnsigned()
				cmplw := ir.Node(ir.NewIndexExpr(base.Pos, cmpl, ir.NewInt(i)))
				cmplw = typecheck.Conv(cmplw, elemType) // convert to unsigned
				cmplw = typecheck.Conv(cmplw, convType) // widen
				cmprw := ir.Node(ir.NewIndexExpr(base.Pos, cmpr, ir.NewInt(i)))
				cmprw = typecheck.Conv(cmprw, elemType)
				cmprw = typecheck.Conv(cmprw, convType)
				// For code like this:  uint32(s[0]) | uint32(s[1])<<8 | uint32(s[2])<<16 ...
				// ssa will generate a single large load.
				for offset := int64(1); offset < step; offset++ {
					lb := ir.Node(ir.NewIndexExpr(base.Pos, cmpl, ir.NewInt(i+offset)))
					lb = typecheck.Conv(lb, elemType)
					lb = typecheck.Conv(lb, convType)
					lb = ir.NewBinaryExpr(base.Pos, ir.OLSH, lb, ir.NewInt(8*t.Elem().Size()*offset))
					cmplw = ir.NewBinaryExpr(base.Pos, ir.OOR, cmplw, lb)
					rb := ir.Node(ir.NewIndexExpr(base.Pos, cmpr, ir.NewInt(i+offset)))
					rb = typecheck.Conv(rb, elemType)
					rb = typecheck.Conv(rb, convType)
					rb = ir.NewBinaryExpr(base.Pos, ir.OLSH, rb, ir.NewInt(8*t.Elem().Size()*offset))
					cmprw = ir.NewBinaryExpr(base.Pos, ir.OOR, cmprw, rb)
				}
				compare(cmplw, cmprw)
				i += step
				remains -= step * t.Elem().Size()
			}
		}
	}
	if expr == nil {
		expr = ir.NewBool(n.Op() == ir.OEQ)
		// We still need to use cmpl and cmpr, in case they contain
		// an expression which might panic. See issue 23837.
		t := typecheck.Temp(cmpl.Type())
		a1 := typecheck.Stmt(ir.NewAssignStmt(base.Pos, t, cmpl))
		a2 := typecheck.Stmt(ir.NewAssignStmt(base.Pos, t, cmpr))
		init.Append(a1, a2)
	}
	return finishCompare(n, expr, init)
}

func walkCompareInterface(n *ir.BinaryExpr, init *ir.Nodes) ir.Node {
	n.Y = cheapExpr(n.Y, init)
	n.X = cheapExpr(n.X, init)
	eqtab, eqdata := reflectdata.EqInterface(n.X, n.Y)
	var cmp ir.Node
	if n.Op() == ir.OEQ {
		cmp = ir.NewLogicalExpr(base.Pos, ir.OANDAND, eqtab, eqdata)
	} else {
		eqtab.SetOp(ir.ONE)
		cmp = ir.NewLogicalExpr(base.Pos, ir.OOROR, eqtab, ir.NewUnaryExpr(base.Pos, ir.ONOT, eqdata))
	}
	return finishCompare(n, cmp, init)
}

func walkCompareString(n *ir.BinaryExpr, init *ir.Nodes) ir.Node {
	// Rewrite comparisons to short constant strings as length+byte-wise comparisons.
	var cs, ncs ir.Node // const string, non-const string
	switch {
	case ir.IsConst(n.X, constant.String) && ir.IsConst(n.Y, constant.String):
		// ignore; will be constant evaluated
	case ir.IsConst(n.X, constant.String):
		cs = n.X
		ncs = n.Y
	case ir.IsConst(n.Y, constant.String):
		cs = n.Y
		ncs = n.X
	}
	if cs != nil {
		cmp := n.Op()
		// Our comparison below assumes that the non-constant string
		// is on the left hand side, so rewrite "" cmp x to x cmp "".
		// See issue 24817.
		if ir.IsConst(n.X, constant.String) {
			cmp = brrev(cmp)
		}

		// maxRewriteLen was chosen empirically.
		// It is the value that minimizes cmd/go file size
		// across most architectures.
		// See the commit description for CL 26758 for details.
		maxRewriteLen := 6
		// Some architectures can load unaligned byte sequence as 1 word.
		// So we can cover longer strings with the same amount of code.
		canCombineLoads := ssagen.Arch.LinkArch.CanMergeLoads
		combine64bit := false
		if canCombineLoads {
			// Keep this low enough to generate less code than a function call.
			maxRewriteLen = 2 * ssagen.Arch.LinkArch.RegSize
			combine64bit = ssagen.Arch.LinkArch.RegSize >= 8
		}

		var and ir.Op
		switch cmp {
		case ir.OEQ:
			and = ir.OANDAND
		case ir.ONE:
			and = ir.OOROR
		default:
			// Don't do byte-wise comparisons for <, <=, etc.
			// They're fairly complicated.
			// Length-only checks are ok, though.
			maxRewriteLen = 0
		}
		if s := ir.StringVal(cs); len(s) <= maxRewriteLen {
			if len(s) > 0 {
				ncs = safeExpr(ncs, init)
			}
			r := ir.Node(ir.NewBinaryExpr(base.Pos, cmp, ir.NewUnaryExpr(base.Pos, ir.OLEN, ncs), ir.NewInt(int64(len(s)))))
			remains := len(s)
			for i := 0; remains > 0; {
				if remains == 1 || !canCombineLoads {
					cb := ir.NewInt(int64(s[i]))
					ncb := ir.NewIndexExpr(base.Pos, ncs, ir.NewInt(int64(i)))
					r = ir.NewLogicalExpr(base.Pos, and, r, ir.NewBinaryExpr(base.Pos, cmp, ncb, cb))
					remains--
					i++
					continue
				}
				var step int
				var convType *types.Type
				switch {
				case remains >= 8 && combine64bit:
					convType = types.Types[types.TINT64]
					step = 8
				case remains >= 4:
					convType = types.Types[types.TUINT32]
					step = 4
				case remains >= 2:
					convType = types.Types[types.TUINT16]
					step = 2
				}
				ncsubstr := typecheck.Conv(ir.NewIndexExpr(base.Pos, ncs, ir.NewInt(int64(i))), convType)
				csubstr := int64(s[i])
				// Calculate large constant from bytes as sequence of shifts and ors.
				// Like this:  uint32(s[0]) | uint32(s[1])<<8 | uint32(s[2])<<16 ...
				// ssa will combine this into a single large load.
				for offset := 1; offset < step; offset++ {
					b := typecheck.Conv(ir.NewIndexExpr(base.Pos, ncs, ir.NewInt(int64(i+offset))), convType)
					b = ir.NewBinaryExpr(base.Pos, ir.OLSH, b, ir.NewInt(int64(8*offset)))
					ncsubstr = ir.NewBinaryExpr(base.Pos, ir.OOR, ncsubstr, b)
					csubstr |= int64(s[i+offset]) << uint8(8*offset)
				}
				csubstrPart := ir.NewInt(csubstr)
				// Compare "step" bytes as once
				r = ir.NewLogicalExpr(base.Pos, and, r, ir.NewBinaryExpr(base.Pos, cmp, csubstrPart, ncsubstr))
				remains -= step
				i += step
			}
			return finishCompare(n, r, init)
		}
	}

	var r ir.Node
	if n.Op() == ir.OEQ || n.Op() == ir.ONE {
		// prepare for rewrite below
		n.X = cheapExpr(n.X, init)
		n.Y = cheapExpr(n.Y, init)
		eqlen, eqmem := reflectdata.EqString(n.X, n.Y)
		// quick check of len before full compare for == or !=.
		// memequal then tests equality up to length len.
		if n.Op() == ir.OEQ {
			// len(left) == len(right) && memequal(left, right, len)
			r = ir.NewLogicalExpr(base.Pos, ir.OANDAND, eqlen, eqmem)
		} else {
			// len(left) != len(right) || !memequal(left, right, len)
			eqlen.SetOp(ir.ONE)
			r = ir.NewLogicalExpr(base.Pos, ir.OOROR, eqlen, ir.NewUnaryExpr(base.Pos, ir.ONOT, eqmem))
		}
	} else {
		// sys_cmpstring(s1, s2) :: 0
		r = mkcall("cmpstring", types.Types[types.TINT], init, typecheck.Conv(n.X, types.Types[types.TSTRING]), typecheck.Conv(n.Y, types.Types[types.TSTRING]))
		r = ir.NewBinaryExpr(base.Pos, n.Op(), r, ir.NewInt(0))
	}

	return finishCompare(n, r, init)
}

// The result of finishCompare MUST be assigned back to n, e.g.
// 	n.Left = finishCompare(n.Left, x, r, init)
func finishCompare(n *ir.BinaryExpr, r ir.Node, init *ir.Nodes) ir.Node {
	r = typecheck.Expr(r)
	r = typecheck.Conv(r, n.Type())
	r = walkExpr(r, init)
	return r
}

func eqFor(t *types.Type) (n ir.Node, needsize bool) {
	// Should only arrive here with large memory or
	// a struct/array containing a non-memory field/element.
	// Small memory is handled inline, and single non-memory
	// is handled by walkCompare.
	switch a, _ := types.AlgType(t); a {
	case types.AMEM:
		n := typecheck.LookupRuntime("memequal")
		n = typecheck.SubstArgTypes(n, t, t)
		return n, true
	case types.ASPECIAL:
		sym := reflectdata.TypeSymPrefix(".eq", t)
		// TODO(austin): This creates an ir.Name with a nil Func.
		n := typecheck.NewName(sym)
		ir.MarkFunc(n)
		n.SetType(types.NewSignature(types.NoPkg, nil, nil, []*types.Field{
			types.NewField(base.Pos, nil, types.NewPtr(t)),
			types.NewField(base.Pos, nil, types.NewPtr(t)),
		}, []*types.Field{
			types.NewField(base.Pos, nil, types.Types[types.TBOOL]),
		}))
		return n, false
	}
	base.Fatalf("eqFor %v", t)
	return nil, false
}

// brcom returns !(op).
// For example, brcom(==) is !=.
func brcom(op ir.Op) ir.Op {
	switch op {
	case ir.OEQ:
		return ir.ONE
	case ir.ONE:
		return ir.OEQ
	case ir.OLT:
		return ir.OGE
	case ir.OGT:
		return ir.OLE
	case ir.OLE:
		return ir.OGT
	case ir.OGE:
		return ir.OLT
	}
	base.Fatalf("brcom: no com for %v\n", op)
	return op
}

// brrev returns reverse(op).
// For example, Brrev(<) is >.
func brrev(op ir.Op) ir.Op {
	switch op {
	case ir.OEQ:
		return ir.OEQ
	case ir.ONE:
		return ir.ONE
	case ir.OLT:
		return ir.OGT
	case ir.OGT:
		return ir.OLT
	case ir.OLE:
		return ir.OGE
	case ir.OGE:
		return ir.OLE
	}
	base.Fatalf("brrev: no rev for %v\n", op)
	return op
}

func tracecmpArg(n ir.Node, t *types.Type, init *ir.Nodes) ir.Node {
	// Ugly hack to avoid "constant -1 overflows uintptr" errors, etc.
	if n.Op() == ir.OLITERAL && n.Type().IsSigned() && ir.Int64Val(n) < 0 {
		n = copyExpr(n, n.Type(), init)
	}

	return typecheck.Conv(n, t)
}
