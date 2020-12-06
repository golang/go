// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"fmt"
	"go/constant"
	"go/token"
	"strings"
)

// To enable tracing support (-t flag), set enableTrace to true.
const enableTrace = false

var traceIndent []byte
var skipDowidthForTracing bool

func tracePrint(title string, n ir.Node) func(np *ir.Node) {
	indent := traceIndent

	// guard against nil
	var pos, op string
	var tc uint8
	if n != nil {
		pos = base.FmtPos(n.Pos())
		op = n.Op().String()
		tc = n.Typecheck()
	}

	skipDowidthForTracing = true
	defer func() { skipDowidthForTracing = false }()
	fmt.Printf("%s: %s%s %p %s %v tc=%d\n", pos, indent, title, n, op, n, tc)
	traceIndent = append(traceIndent, ". "...)

	return func(np *ir.Node) {
		traceIndent = traceIndent[:len(traceIndent)-2]

		// if we have a result, use that
		if np != nil {
			n = *np
		}

		// guard against nil
		// use outer pos, op so we don't get empty pos/op if n == nil (nicer output)
		var tc uint8
		var typ *types.Type
		if n != nil {
			pos = base.FmtPos(n.Pos())
			op = n.Op().String()
			tc = n.Typecheck()
			typ = n.Type()
		}

		skipDowidthForTracing = true
		defer func() { skipDowidthForTracing = false }()
		fmt.Printf("%s: %s=> %p %s %v tc=%d type=%L\n", pos, indent, n, op, n, tc, typ)
	}
}

const (
	ctxStmt    = 1 << iota // evaluated at statement level
	ctxExpr                // evaluated in value context
	ctxType                // evaluated in type context
	ctxCallee              // call-only expressions are ok
	ctxMultiOK             // multivalue function returns are ok
	ctxAssign              // assigning to expression
)

// type checks the whole tree of an expression.
// calculates expression types.
// evaluates compile time constants.
// marks variables that escape the local frame.
// rewrites n.Op to be more specific in some cases.

var typecheckdefstack []ir.Node

// resolve ONONAME to definition, if any.
func resolve(n ir.Node) (res ir.Node) {
	if n == nil || n.Op() != ir.ONONAME {
		return n
	}

	// only trace if there's work to do
	if enableTrace && base.Flag.LowerT {
		defer tracePrint("resolve", n)(&res)
	}

	if n.Sym().Pkg != types.LocalPkg {
		if inimport {
			base.Fatalf("recursive inimport")
		}
		inimport = true
		expandDecl(n.(*ir.Name))
		inimport = false
		return n
	}

	r := ir.AsNode(n.Sym().Def)
	if r == nil {
		return n
	}

	if r.Op() == ir.OIOTA {
		if x := getIotaValue(); x >= 0 {
			return nodintconst(x)
		}
		return n
	}

	return r
}

func typecheckslice(l []ir.Node, top int) {
	for i := range l {
		l[i] = typecheck(l[i], top)
	}
}

var _typekind = []string{
	types.TINT:        "int",
	types.TUINT:       "uint",
	types.TINT8:       "int8",
	types.TUINT8:      "uint8",
	types.TINT16:      "int16",
	types.TUINT16:     "uint16",
	types.TINT32:      "int32",
	types.TUINT32:     "uint32",
	types.TINT64:      "int64",
	types.TUINT64:     "uint64",
	types.TUINTPTR:    "uintptr",
	types.TCOMPLEX64:  "complex64",
	types.TCOMPLEX128: "complex128",
	types.TFLOAT32:    "float32",
	types.TFLOAT64:    "float64",
	types.TBOOL:       "bool",
	types.TSTRING:     "string",
	types.TPTR:        "pointer",
	types.TUNSAFEPTR:  "unsafe.Pointer",
	types.TSTRUCT:     "struct",
	types.TINTER:      "interface",
	types.TCHAN:       "chan",
	types.TMAP:        "map",
	types.TARRAY:      "array",
	types.TSLICE:      "slice",
	types.TFUNC:       "func",
	types.TNIL:        "nil",
	types.TIDEAL:      "untyped number",
}

func typekind(t *types.Type) string {
	if t.IsUntyped() {
		return fmt.Sprintf("%v", t)
	}
	et := t.Kind()
	if int(et) < len(_typekind) {
		s := _typekind[et]
		if s != "" {
			return s
		}
	}
	return fmt.Sprintf("etype=%d", et)
}

func cycleFor(start ir.Node) []ir.Node {
	// Find the start node in typecheck_tcstack.
	// We know that it must exist because each time we mark
	// a node with n.SetTypecheck(2) we push it on the stack,
	// and each time we mark a node with n.SetTypecheck(2) we
	// pop it from the stack. We hit a cycle when we encounter
	// a node marked 2 in which case is must be on the stack.
	i := len(typecheck_tcstack) - 1
	for i > 0 && typecheck_tcstack[i] != start {
		i--
	}

	// collect all nodes with same Op
	var cycle []ir.Node
	for _, n := range typecheck_tcstack[i:] {
		if n.Op() == start.Op() {
			cycle = append(cycle, n)
		}
	}

	return cycle
}

func cycleTrace(cycle []ir.Node) string {
	var s string
	for i, n := range cycle {
		s += fmt.Sprintf("\n\t%v: %v uses %v", ir.Line(n), n, cycle[(i+1)%len(cycle)])
	}
	return s
}

var typecheck_tcstack []ir.Node

func typecheckFunc(fn *ir.Func) {
	new := typecheck(fn, ctxStmt)
	if new != fn {
		base.Fatalf("typecheck changed func")
	}
}

func typecheckNtype(n ir.Ntype) ir.Ntype {
	return typecheck(n, ctxType).(ir.Ntype)
}

// typecheck type checks node n.
// The result of typecheck MUST be assigned back to n, e.g.
// 	n.Left = typecheck(n.Left, top)
func typecheck(n ir.Node, top int) (res ir.Node) {
	// cannot type check until all the source has been parsed
	if !typecheckok {
		base.Fatalf("early typecheck")
	}

	if n == nil {
		return nil
	}

	// only trace if there's work to do
	if enableTrace && base.Flag.LowerT {
		defer tracePrint("typecheck", n)(&res)
	}

	lno := setlineno(n)

	// Skip over parens.
	for n.Op() == ir.OPAREN {
		n = n.Left()
	}

	// Resolve definition of name and value of iota lazily.
	n = resolve(n)

	// Skip typecheck if already done.
	// But re-typecheck ONAME/OTYPE/OLITERAL/OPACK node in case context has changed.
	if n.Typecheck() == 1 {
		switch n.Op() {
		case ir.ONAME, ir.OTYPE, ir.OLITERAL, ir.OPACK:
			break

		default:
			base.Pos = lno
			return n
		}
	}

	if n.Typecheck() == 2 {
		// Typechecking loop. Trying printing a meaningful message,
		// otherwise a stack trace of typechecking.
		switch n.Op() {
		// We can already diagnose variables used as types.
		case ir.ONAME:
			if top&(ctxExpr|ctxType) == ctxType {
				base.Errorf("%v is not a type", n)
			}

		case ir.OTYPE:
			// Only report a type cycle if we are expecting a type.
			// Otherwise let other code report an error.
			if top&ctxType == ctxType {
				// A cycle containing only alias types is an error
				// since it would expand indefinitely when aliases
				// are substituted.
				cycle := cycleFor(n)
				for _, n1 := range cycle {
					if n1.Name() != nil && !n1.Name().Alias() {
						// Cycle is ok. But if n is an alias type and doesn't
						// have a type yet, we have a recursive type declaration
						// with aliases that we can't handle properly yet.
						// Report an error rather than crashing later.
						if n.Name() != nil && n.Name().Alias() && n.Type() == nil {
							base.Pos = n.Pos()
							base.Fatalf("cannot handle alias type declaration (issue #25838): %v", n)
						}
						base.Pos = lno
						return n
					}
				}
				base.ErrorfAt(n.Pos(), "invalid recursive type alias %v%s", n, cycleTrace(cycle))
			}

		case ir.OLITERAL:
			if top&(ctxExpr|ctxType) == ctxType {
				base.Errorf("%v is not a type", n)
				break
			}
			base.ErrorfAt(n.Pos(), "constant definition loop%s", cycleTrace(cycleFor(n)))
		}

		if base.Errors() == 0 {
			var trace string
			for i := len(typecheck_tcstack) - 1; i >= 0; i-- {
				x := typecheck_tcstack[i]
				trace += fmt.Sprintf("\n\t%v %v", ir.Line(x), x)
			}
			base.Errorf("typechecking loop involving %v%s", n, trace)
		}

		base.Pos = lno
		return n
	}

	typecheck_tcstack = append(typecheck_tcstack, n)

	n.SetTypecheck(2)
	n = typecheck1(n, top)
	n.SetTypecheck(1)

	last := len(typecheck_tcstack) - 1
	typecheck_tcstack[last] = nil
	typecheck_tcstack = typecheck_tcstack[:last]

	_, isExpr := n.(ir.Expr)
	_, isStmt := n.(ir.Stmt)
	isMulti := false
	switch n.Op() {
	case ir.OCALLFUNC, ir.OCALLINTER, ir.OCALLMETH:
		if t := n.Left().Type(); t != nil && t.Kind() == types.TFUNC {
			nr := t.NumResults()
			isMulti = nr > 1
			if nr == 0 {
				isExpr = false
			}
		}
	case ir.OAPPEND:
		// Must be used (and not BinaryExpr/UnaryExpr).
		isStmt = false
	case ir.OCLOSE, ir.ODELETE, ir.OPANIC, ir.OPRINT, ir.OPRINTN, ir.OVARKILL, ir.OVARLIVE:
		// Must not be used.
		isExpr = false
		isStmt = true
	case ir.OCOPY, ir.ORECOVER, ir.ORECV:
		// Can be used or not.
		isStmt = true
	}

	t := n.Type()
	if t != nil && !t.IsFuncArgStruct() && n.Op() != ir.OTYPE {
		switch t.Kind() {
		case types.TFUNC, // might have TANY; wait until it's called
			types.TANY, types.TFORW, types.TIDEAL, types.TNIL, types.TBLANK:
			break

		default:
			checkwidth(t)
		}
	}
	if t != nil {
		n = evalConst(n)
		t = n.Type()
	}

	// TODO(rsc): Lots of the complexity here is because typecheck can
	// see OTYPE, ONAME, and OLITERAL nodes multiple times.
	// Once we make the IR a proper tree, we should be able to simplify
	// this code a bit, especially the final case.
	switch {
	case top&(ctxStmt|ctxExpr) == ctxExpr && !isExpr && n.Op() != ir.OTYPE && !isMulti:
		if !n.Diag() {
			base.Errorf("%v used as value", n)
			n.SetDiag(true)
		}
		if t != nil {
			n.SetType(nil)
		}

	case top&ctxType == 0 && n.Op() == ir.OTYPE && t != nil:
		if !n.Type().Broke() {
			base.Errorf("type %v is not an expression", n.Type())
		}
		n.SetType(nil)

	case top&(ctxStmt|ctxExpr) == ctxStmt && !isStmt && t != nil:
		if !n.Diag() {
			base.Errorf("%v evaluated but not used", n)
			n.SetDiag(true)
		}
		n.SetType(nil)

	case top&(ctxType|ctxExpr) == ctxType && n.Op() != ir.OTYPE && n.Op() != ir.ONONAME && (t != nil || n.Op() == ir.ONAME):
		base.Errorf("%v is not a type", n)
		if t != nil {
			n.SetType(nil)
		}

	}

	base.Pos = lno
	return n
}

// indexlit implements typechecking of untyped values as
// array/slice indexes. It is almost equivalent to defaultlit
// but also accepts untyped numeric values representable as
// value of type int (see also checkmake for comparison).
// The result of indexlit MUST be assigned back to n, e.g.
// 	n.Left = indexlit(n.Left)
func indexlit(n ir.Node) ir.Node {
	if n != nil && n.Type() != nil && n.Type().Kind() == types.TIDEAL {
		return defaultlit(n, types.Types[types.TINT])
	}
	return n
}

// typecheck1 should ONLY be called from typecheck.
func typecheck1(n ir.Node, top int) (res ir.Node) {
	if enableTrace && base.Flag.LowerT {
		defer tracePrint("typecheck1", n)(&res)
	}

	switch n.Op() {
	case ir.OLITERAL, ir.ONAME, ir.ONONAME, ir.OTYPE:
		if n.Sym() == nil {
			return n
		}

		if n.Op() == ir.ONAME && n.SubOp() != 0 && top&ctxCallee == 0 {
			base.Errorf("use of builtin %v not in function call", n.Sym())
			n.SetType(nil)
			return n
		}

		typecheckdef(n)
		if n.Op() == ir.ONONAME {
			n.SetType(nil)
			return n
		}
	}

	switch n.Op() {
	default:
		ir.Dump("typecheck", n)
		base.Fatalf("typecheck %v", n.Op())
		panic("unreachable")

	// names
	case ir.OLITERAL:
		if n.Type() == nil && n.Val().Kind() == constant.String {
			base.Fatalf("string literal missing type")
		}
		return n

	case ir.ONIL, ir.ONONAME:
		return n

	case ir.ONAME:
		if n.Name().Decldepth == 0 {
			n.Name().Decldepth = decldepth
		}
		if n.SubOp() != 0 {
			return n
		}
		if top&ctxAssign == 0 {
			// not a write to the variable
			if ir.IsBlank(n) {
				base.Errorf("cannot use _ as value")
				n.SetType(nil)
				return n
			}
			n.Name().SetUsed(true)
		}
		return n

	case ir.OPACK:
		base.Errorf("use of package %v without selector", n.Sym())
		n.SetType(nil)
		return n

	// types (ODEREF is with exprs)
	case ir.OTYPE:
		if n.Type() == nil {
			return n
		}
		return n

	case ir.OTSLICE:
		n := n.(*ir.SliceType)
		n.Elem = typecheck(n.Elem, ctxType)
		if n.Elem.Type() == nil {
			return n
		}
		t := types.NewSlice(n.Elem.Type())
		n.SetOTYPE(t)
		checkwidth(t)
		return n

	case ir.OTARRAY:
		n := n.(*ir.ArrayType)
		n.Elem = typecheck(n.Elem, ctxType)
		if n.Elem.Type() == nil {
			return n
		}
		if n.Len == nil { // [...]T
			if !n.Diag() {
				n.SetDiag(true)
				base.Errorf("use of [...] array outside of array literal")
			}
			return n
		}
		n.Len = indexlit(typecheck(n.Len, ctxExpr))
		size := n.Len
		if ir.ConstType(size) != constant.Int {
			switch {
			case size.Type() == nil:
				// Error already reported elsewhere.
			case size.Type().IsInteger() && size.Op() != ir.OLITERAL:
				base.Errorf("non-constant array bound %v", size)
			default:
				base.Errorf("invalid array bound %v", size)
			}
			return n
		}

		v := size.Val()
		if doesoverflow(v, types.Types[types.TINT]) {
			base.Errorf("array bound is too large")
			return n
		}

		if constant.Sign(v) < 0 {
			base.Errorf("array bound must be non-negative")
			return n
		}

		bound, _ := constant.Int64Val(v)
		t := types.NewArray(n.Elem.Type(), bound)
		n.SetOTYPE(t)
		checkwidth(t)
		return n

	case ir.OTMAP:
		n := n.(*ir.MapType)
		n.Key = typecheck(n.Key, ctxType)
		n.Elem = typecheck(n.Elem, ctxType)
		l := n.Key
		r := n.Elem
		if l.Type() == nil || r.Type() == nil {
			return n
		}
		if l.Type().NotInHeap() {
			base.Errorf("incomplete (or unallocatable) map key not allowed")
		}
		if r.Type().NotInHeap() {
			base.Errorf("incomplete (or unallocatable) map value not allowed")
		}
		n.SetOTYPE(types.NewMap(l.Type(), r.Type()))
		mapqueue = append(mapqueue, n) // check map keys when all types are settled
		return n

	case ir.OTCHAN:
		n := n.(*ir.ChanType)
		n.Elem = typecheck(n.Elem, ctxType)
		l := n.Elem
		if l.Type() == nil {
			return n
		}
		if l.Type().NotInHeap() {
			base.Errorf("chan of incomplete (or unallocatable) type not allowed")
		}
		n.SetOTYPE(types.NewChan(l.Type(), n.Dir))
		return n

	case ir.OTSTRUCT:
		n := n.(*ir.StructType)
		n.SetOTYPE(tostruct(n.Fields))
		return n

	case ir.OTINTER:
		n := n.(*ir.InterfaceType)
		n.SetOTYPE(tointerface(n.Methods))
		return n

	case ir.OTFUNC:
		n := n.(*ir.FuncType)
		n.SetOTYPE(functype(n.Recv, n.Params, n.Results))
		return n

	// type or expr
	case ir.ODEREF:
		n := n.(*ir.StarExpr)
		n.X = typecheck(n.X, ctxExpr|ctxType)
		l := n.X
		t := l.Type()
		if t == nil {
			n.SetType(nil)
			return n
		}
		if l.Op() == ir.OTYPE {
			n.SetOTYPE(types.NewPtr(l.Type()))
			// Ensure l.Type gets dowidth'd for the backend. Issue 20174.
			checkwidth(l.Type())
			return n
		}

		if !t.IsPtr() {
			if top&(ctxExpr|ctxStmt) != 0 {
				base.Errorf("invalid indirect of %L", n.Left())
				n.SetType(nil)
				return n
			}
			base.Errorf("%v is not a type", l)
			return n
		}

		n.SetType(t.Elem())
		return n

	// arithmetic exprs
	case ir.OASOP,
		ir.OADD,
		ir.OAND,
		ir.OANDAND,
		ir.OANDNOT,
		ir.ODIV,
		ir.OEQ,
		ir.OGE,
		ir.OGT,
		ir.OLE,
		ir.OLT,
		ir.OLSH,
		ir.ORSH,
		ir.OMOD,
		ir.OMUL,
		ir.ONE,
		ir.OOR,
		ir.OOROR,
		ir.OSUB,
		ir.OXOR:
		var l ir.Node
		var op ir.Op
		var r ir.Node
		if n.Op() == ir.OASOP {
			n.SetLeft(typecheck(n.Left(), ctxExpr))
			n.SetRight(typecheck(n.Right(), ctxExpr))
			l = n.Left()
			r = n.Right()
			checkassign(n, n.Left())
			if l.Type() == nil || r.Type() == nil {
				n.SetType(nil)
				return n
			}
			if n.Implicit() && !okforarith[l.Type().Kind()] {
				base.Errorf("invalid operation: %v (non-numeric type %v)", n, l.Type())
				n.SetType(nil)
				return n
			}
			// TODO(marvin): Fix Node.EType type union.
			op = n.SubOp()
		} else {
			n.SetLeft(typecheck(n.Left(), ctxExpr))
			n.SetRight(typecheck(n.Right(), ctxExpr))
			l = n.Left()
			r = n.Right()
			if l.Type() == nil || r.Type() == nil {
				n.SetType(nil)
				return n
			}
			op = n.Op()
		}
		if op == ir.OLSH || op == ir.ORSH {
			r = defaultlit(r, types.Types[types.TUINT])
			n.SetRight(r)
			t := r.Type()
			if !t.IsInteger() {
				base.Errorf("invalid operation: %v (shift count type %v, must be integer)", n, r.Type())
				n.SetType(nil)
				return n
			}
			if t.IsSigned() && !langSupported(1, 13, curpkg()) {
				base.ErrorfVers("go1.13", "invalid operation: %v (signed shift count type %v)", n, r.Type())
				n.SetType(nil)
				return n
			}
			t = l.Type()
			if t != nil && t.Kind() != types.TIDEAL && !t.IsInteger() {
				base.Errorf("invalid operation: %v (shift of type %v)", n, t)
				n.SetType(nil)
				return n
			}

			// no defaultlit for left
			// the outer context gives the type
			n.SetType(l.Type())
			if (l.Type() == types.UntypedFloat || l.Type() == types.UntypedComplex) && r.Op() == ir.OLITERAL {
				n.SetType(types.UntypedInt)
			}
			return n
		}

		// For "x == x && len(s)", it's better to report that "len(s)" (type int)
		// can't be used with "&&" than to report that "x == x" (type untyped bool)
		// can't be converted to int (see issue #41500).
		if n.Op() == ir.OANDAND || n.Op() == ir.OOROR {
			if !n.Left().Type().IsBoolean() {
				base.Errorf("invalid operation: %v (operator %v not defined on %s)", n, n.Op(), typekind(n.Left().Type()))
				n.SetType(nil)
				return n
			}
			if !n.Right().Type().IsBoolean() {
				base.Errorf("invalid operation: %v (operator %v not defined on %s)", n, n.Op(), typekind(n.Right().Type()))
				n.SetType(nil)
				return n
			}
		}

		// ideal mixed with non-ideal
		l, r = defaultlit2(l, r, false)

		n.SetLeft(l)
		n.SetRight(r)
		if l.Type() == nil || r.Type() == nil {
			n.SetType(nil)
			return n
		}
		t := l.Type()
		if t.Kind() == types.TIDEAL {
			t = r.Type()
		}
		et := t.Kind()
		if et == types.TIDEAL {
			et = types.TINT
		}
		aop := ir.OXXX
		if iscmp[n.Op()] && t.Kind() != types.TIDEAL && !types.Identical(l.Type(), r.Type()) {
			// comparison is okay as long as one side is
			// assignable to the other.  convert so they have
			// the same type.
			//
			// the only conversion that isn't a no-op is concrete == interface.
			// in that case, check comparability of the concrete type.
			// The conversion allocates, so only do it if the concrete type is huge.
			converted := false
			if r.Type().Kind() != types.TBLANK {
				aop, _ = assignop(l.Type(), r.Type())
				if aop != ir.OXXX {
					if r.Type().IsInterface() && !l.Type().IsInterface() && !IsComparable(l.Type()) {
						base.Errorf("invalid operation: %v (operator %v not defined on %s)", n, op, typekind(l.Type()))
						n.SetType(nil)
						return n
					}

					dowidth(l.Type())
					if r.Type().IsInterface() == l.Type().IsInterface() || l.Type().Width >= 1<<16 {
						l = ir.Nod(aop, l, nil)
						l.SetType(r.Type())
						l.SetTypecheck(1)
						n.SetLeft(l)
					}

					t = r.Type()
					converted = true
				}
			}

			if !converted && l.Type().Kind() != types.TBLANK {
				aop, _ = assignop(r.Type(), l.Type())
				if aop != ir.OXXX {
					if l.Type().IsInterface() && !r.Type().IsInterface() && !IsComparable(r.Type()) {
						base.Errorf("invalid operation: %v (operator %v not defined on %s)", n, op, typekind(r.Type()))
						n.SetType(nil)
						return n
					}

					dowidth(r.Type())
					if r.Type().IsInterface() == l.Type().IsInterface() || r.Type().Width >= 1<<16 {
						r = ir.Nod(aop, r, nil)
						r.SetType(l.Type())
						r.SetTypecheck(1)
						n.SetRight(r)
					}

					t = l.Type()
				}
			}

			et = t.Kind()
		}

		if t.Kind() != types.TIDEAL && !types.Identical(l.Type(), r.Type()) {
			l, r = defaultlit2(l, r, true)
			if l.Type() == nil || r.Type() == nil {
				n.SetType(nil)
				return n
			}
			if l.Type().IsInterface() == r.Type().IsInterface() || aop == 0 {
				base.Errorf("invalid operation: %v (mismatched types %v and %v)", n, l.Type(), r.Type())
				n.SetType(nil)
				return n
			}
		}

		if t.Kind() == types.TIDEAL {
			t = mixUntyped(l.Type(), r.Type())
		}
		if dt := defaultType(t); !okfor[op][dt.Kind()] {
			base.Errorf("invalid operation: %v (operator %v not defined on %s)", n, op, typekind(t))
			n.SetType(nil)
			return n
		}

		// okfor allows any array == array, map == map, func == func.
		// restrict to slice/map/func == nil and nil == slice/map/func.
		if l.Type().IsArray() && !IsComparable(l.Type()) {
			base.Errorf("invalid operation: %v (%v cannot be compared)", n, l.Type())
			n.SetType(nil)
			return n
		}

		if l.Type().IsSlice() && !ir.IsNil(l) && !ir.IsNil(r) {
			base.Errorf("invalid operation: %v (slice can only be compared to nil)", n)
			n.SetType(nil)
			return n
		}

		if l.Type().IsMap() && !ir.IsNil(l) && !ir.IsNil(r) {
			base.Errorf("invalid operation: %v (map can only be compared to nil)", n)
			n.SetType(nil)
			return n
		}

		if l.Type().Kind() == types.TFUNC && !ir.IsNil(l) && !ir.IsNil(r) {
			base.Errorf("invalid operation: %v (func can only be compared to nil)", n)
			n.SetType(nil)
			return n
		}

		if l.Type().IsStruct() {
			if f := IncomparableField(l.Type()); f != nil {
				base.Errorf("invalid operation: %v (struct containing %v cannot be compared)", n, f.Type)
				n.SetType(nil)
				return n
			}
		}

		if iscmp[n.Op()] {
			t = types.UntypedBool
			n.SetType(t)
			n = evalConst(n)
			if n.Op() != ir.OLITERAL {
				l, r = defaultlit2(l, r, true)
				n.SetLeft(l)
				n.SetRight(r)
			}
		}

		if et == types.TSTRING && n.Op() == ir.OADD {
			// create or update OADDSTR node with list of strings in x + y + z + (w + v) + ...
			if l.Op() == ir.OADDSTR {
				orig := n
				n = l
				n.SetPos(orig.Pos())
			} else {
				n = ir.NodAt(n.Pos(), ir.OADDSTR, nil, nil)
				n.PtrList().Set1(l)
			}
			if r.Op() == ir.OADDSTR {
				n.PtrList().AppendNodes(r.PtrList())
			} else {
				n.PtrList().Append(r)
			}
		}

		if (op == ir.ODIV || op == ir.OMOD) && ir.IsConst(r, constant.Int) {
			if constant.Sign(r.Val()) == 0 {
				base.Errorf("division by zero")
				n.SetType(nil)
				return n
			}
		}

		n.SetType(t)
		return n

	case ir.OBITNOT, ir.ONEG, ir.ONOT, ir.OPLUS:
		n.SetLeft(typecheck(n.Left(), ctxExpr))
		l := n.Left()
		t := l.Type()
		if t == nil {
			n.SetType(nil)
			return n
		}
		if !okfor[n.Op()][defaultType(t).Kind()] {
			base.Errorf("invalid operation: %v (operator %v not defined on %s)", n, n.Op(), typekind(t))
			n.SetType(nil)
			return n
		}

		n.SetType(t)
		return n

	// exprs
	case ir.OADDR:
		n.SetLeft(typecheck(n.Left(), ctxExpr))
		if n.Left().Type() == nil {
			n.SetType(nil)
			return n
		}

		switch n.Left().Op() {
		case ir.OARRAYLIT, ir.OMAPLIT, ir.OSLICELIT, ir.OSTRUCTLIT:
			n.SetOp(ir.OPTRLIT)

		default:
			checklvalue(n.Left(), "take the address of")
			r := outervalue(n.Left())
			if r.Op() == ir.ONAME {
				if ir.Orig(r) != r {
					base.Fatalf("found non-orig name node %v", r) // TODO(mdempsky): What does this mean?
				}
				r.Name().SetAddrtaken(true)
				if r.Name().IsClosureVar() && !capturevarscomplete {
					// Mark the original variable as Addrtaken so that capturevars
					// knows not to pass it by value.
					// But if the capturevars phase is complete, don't touch it,
					// in case l.Name's containing function has not yet been compiled.
					r.Name().Defn.Name().SetAddrtaken(true)
				}
			}
			n.SetLeft(defaultlit(n.Left(), nil))
			if n.Left().Type() == nil {
				n.SetType(nil)
				return n
			}
		}

		n.SetType(types.NewPtr(n.Left().Type()))
		return n

	case ir.OCOMPLIT:
		return typecheckcomplit(n)

	case ir.OXDOT, ir.ODOT:
		if n.Op() == ir.OXDOT {
			n = adddot(n)
			n.SetOp(ir.ODOT)
			if n.Left() == nil {
				n.SetType(nil)
				return n
			}
		}

		n.SetLeft(typecheck(n.Left(), ctxExpr|ctxType))

		n.SetLeft(defaultlit(n.Left(), nil))

		t := n.Left().Type()
		if t == nil {
			base.UpdateErrorDot(ir.Line(n), n.Left().String(), n.String())
			n.SetType(nil)
			return n
		}

		s := n.Sym()

		if n.Left().Op() == ir.OTYPE {
			return typecheckMethodExpr(n)
		}

		if t.IsPtr() && !t.Elem().IsInterface() {
			t = t.Elem()
			if t == nil {
				n.SetType(nil)
				return n
			}
			n.SetOp(ir.ODOTPTR)
			checkwidth(t)
		}

		if n.Sym().IsBlank() {
			base.Errorf("cannot refer to blank field or method")
			n.SetType(nil)
			return n
		}

		if lookdot(n, t, 0) == nil {
			// Legitimate field or method lookup failed, try to explain the error
			switch {
			case t.IsEmptyInterface():
				base.Errorf("%v undefined (type %v is interface with no methods)", n, n.Left().Type())

			case t.IsPtr() && t.Elem().IsInterface():
				// Pointer to interface is almost always a mistake.
				base.Errorf("%v undefined (type %v is pointer to interface, not interface)", n, n.Left().Type())

			case lookdot(n, t, 1) != nil:
				// Field or method matches by name, but it is not exported.
				base.Errorf("%v undefined (cannot refer to unexported field or method %v)", n, n.Sym())

			default:
				if mt := lookdot(n, t, 2); mt != nil && visible(mt.Sym) { // Case-insensitive lookup.
					base.Errorf("%v undefined (type %v has no field or method %v, but does have %v)", n, n.Left().Type(), n.Sym(), mt.Sym)
				} else {
					base.Errorf("%v undefined (type %v has no field or method %v)", n, n.Left().Type(), n.Sym())
				}
			}
			n.SetType(nil)
			return n
		}

		if (n.Op() == ir.ODOTINTER || n.Op() == ir.ODOTMETH) && top&ctxCallee == 0 {
			n = typecheckpartialcall(n, s)
		}
		return n

	case ir.ODOTTYPE:
		n.SetLeft(typecheck(n.Left(), ctxExpr))
		n.SetLeft(defaultlit(n.Left(), nil))
		l := n.Left()
		t := l.Type()
		if t == nil {
			n.SetType(nil)
			return n
		}
		if !t.IsInterface() {
			base.Errorf("invalid type assertion: %v (non-interface type %v on left)", n, t)
			n.SetType(nil)
			return n
		}

		if n.Right() != nil {
			n.SetRight(typecheck(n.Right(), ctxType))
			n.SetType(n.Right().Type())
			n.SetRight(nil)
			if n.Type() == nil {
				return n
			}
		}

		if n.Type() != nil && !n.Type().IsInterface() {
			var missing, have *types.Field
			var ptr int
			if !implements(n.Type(), t, &missing, &have, &ptr) {
				if have != nil && have.Sym == missing.Sym {
					base.Errorf("impossible type assertion:\n\t%v does not implement %v (wrong type for %v method)\n"+
						"\t\thave %v%S\n\t\twant %v%S", n.Type(), t, missing.Sym, have.Sym, have.Type, missing.Sym, missing.Type)
				} else if ptr != 0 {
					base.Errorf("impossible type assertion:\n\t%v does not implement %v (%v method has pointer receiver)", n.Type(), t, missing.Sym)
				} else if have != nil {
					base.Errorf("impossible type assertion:\n\t%v does not implement %v (missing %v method)\n"+
						"\t\thave %v%S\n\t\twant %v%S", n.Type(), t, missing.Sym, have.Sym, have.Type, missing.Sym, missing.Type)
				} else {
					base.Errorf("impossible type assertion:\n\t%v does not implement %v (missing %v method)", n.Type(), t, missing.Sym)
				}
				n.SetType(nil)
				return n
			}
		}
		return n

	case ir.OINDEX:
		n.SetLeft(typecheck(n.Left(), ctxExpr))
		n.SetLeft(defaultlit(n.Left(), nil))
		n.SetLeft(implicitstar(n.Left()))
		l := n.Left()
		n.SetRight(typecheck(n.Right(), ctxExpr))
		r := n.Right()
		t := l.Type()
		if t == nil || r.Type() == nil {
			n.SetType(nil)
			return n
		}
		switch t.Kind() {
		default:
			base.Errorf("invalid operation: %v (type %v does not support indexing)", n, t)
			n.SetType(nil)
			return n

		case types.TSTRING, types.TARRAY, types.TSLICE:
			n.SetRight(indexlit(n.Right()))
			if t.IsString() {
				n.SetType(types.ByteType)
			} else {
				n.SetType(t.Elem())
			}
			why := "string"
			if t.IsArray() {
				why = "array"
			} else if t.IsSlice() {
				why = "slice"
			}

			if n.Right().Type() != nil && !n.Right().Type().IsInteger() {
				base.Errorf("non-integer %s index %v", why, n.Right())
				return n
			}

			if !n.Bounded() && ir.IsConst(n.Right(), constant.Int) {
				x := n.Right().Val()
				if constant.Sign(x) < 0 {
					base.Errorf("invalid %s index %v (index must be non-negative)", why, n.Right())
				} else if t.IsArray() && constant.Compare(x, token.GEQ, constant.MakeInt64(t.NumElem())) {
					base.Errorf("invalid array index %v (out of bounds for %d-element array)", n.Right(), t.NumElem())
				} else if ir.IsConst(n.Left(), constant.String) && constant.Compare(x, token.GEQ, constant.MakeInt64(int64(len(ir.StringVal(n.Left()))))) {
					base.Errorf("invalid string index %v (out of bounds for %d-byte string)", n.Right(), len(ir.StringVal(n.Left())))
				} else if doesoverflow(x, types.Types[types.TINT]) {
					base.Errorf("invalid %s index %v (index too large)", why, n.Right())
				}
			}

		case types.TMAP:
			n.SetRight(assignconv(n.Right(), t.Key(), "map index"))
			n.SetType(t.Elem())
			n.SetOp(ir.OINDEXMAP)
			n.SetIndexMapLValue(false)
		}
		return n

	case ir.ORECV:
		n.SetLeft(typecheck(n.Left(), ctxExpr))
		n.SetLeft(defaultlit(n.Left(), nil))
		l := n.Left()
		t := l.Type()
		if t == nil {
			n.SetType(nil)
			return n
		}
		if !t.IsChan() {
			base.Errorf("invalid operation: %v (receive from non-chan type %v)", n, t)
			n.SetType(nil)
			return n
		}

		if !t.ChanDir().CanRecv() {
			base.Errorf("invalid operation: %v (receive from send-only type %v)", n, t)
			n.SetType(nil)
			return n
		}

		n.SetType(t.Elem())
		return n

	case ir.OSEND:
		n.SetLeft(typecheck(n.Left(), ctxExpr))
		n.SetRight(typecheck(n.Right(), ctxExpr))
		n.SetLeft(defaultlit(n.Left(), nil))
		t := n.Left().Type()
		if t == nil {
			return n
		}
		if !t.IsChan() {
			base.Errorf("invalid operation: %v (send to non-chan type %v)", n, t)
			return n
		}

		if !t.ChanDir().CanSend() {
			base.Errorf("invalid operation: %v (send to receive-only type %v)", n, t)
			return n
		}

		n.SetRight(assignconv(n.Right(), t.Elem(), "send"))
		if n.Right().Type() == nil {
			return n
		}
		return n

	case ir.OSLICEHEADER:
		// Errors here are Fatalf instead of Errorf because only the compiler
		// can construct an OSLICEHEADER node.
		// Components used in OSLICEHEADER that are supplied by parsed source code
		// have already been typechecked in e.g. OMAKESLICE earlier.
		t := n.Type()
		if t == nil {
			base.Fatalf("no type specified for OSLICEHEADER")
		}

		if !t.IsSlice() {
			base.Fatalf("invalid type %v for OSLICEHEADER", n.Type())
		}

		if n.Left() == nil || n.Left().Type() == nil || !n.Left().Type().IsUnsafePtr() {
			base.Fatalf("need unsafe.Pointer for OSLICEHEADER")
		}

		if x := n.List().Len(); x != 2 {
			base.Fatalf("expected 2 params (len, cap) for OSLICEHEADER, got %d", x)
		}

		n.SetLeft(typecheck(n.Left(), ctxExpr))
		l := typecheck(n.List().First(), ctxExpr)
		c := typecheck(n.List().Second(), ctxExpr)
		l = defaultlit(l, types.Types[types.TINT])
		c = defaultlit(c, types.Types[types.TINT])

		if ir.IsConst(l, constant.Int) && ir.Int64Val(l) < 0 {
			base.Fatalf("len for OSLICEHEADER must be non-negative")
		}

		if ir.IsConst(c, constant.Int) && ir.Int64Val(c) < 0 {
			base.Fatalf("cap for OSLICEHEADER must be non-negative")
		}

		if ir.IsConst(l, constant.Int) && ir.IsConst(c, constant.Int) && constant.Compare(l.Val(), token.GTR, c.Val()) {
			base.Fatalf("len larger than cap for OSLICEHEADER")
		}

		n.List().SetFirst(l)
		n.List().SetSecond(c)
		return n

	case ir.OMAKESLICECOPY:
		// Errors here are Fatalf instead of Errorf because only the compiler
		// can construct an OMAKESLICECOPY node.
		// Components used in OMAKESCLICECOPY that are supplied by parsed source code
		// have already been typechecked in OMAKE and OCOPY earlier.
		t := n.Type()

		if t == nil {
			base.Fatalf("no type specified for OMAKESLICECOPY")
		}

		if !t.IsSlice() {
			base.Fatalf("invalid type %v for OMAKESLICECOPY", n.Type())
		}

		if n.Left() == nil {
			base.Fatalf("missing len argument for OMAKESLICECOPY")
		}

		if n.Right() == nil {
			base.Fatalf("missing slice argument to copy for OMAKESLICECOPY")
		}

		n.SetLeft(typecheck(n.Left(), ctxExpr))
		n.SetRight(typecheck(n.Right(), ctxExpr))

		n.SetLeft(defaultlit(n.Left(), types.Types[types.TINT]))

		if !n.Left().Type().IsInteger() && n.Type().Kind() != types.TIDEAL {
			base.Errorf("non-integer len argument in OMAKESLICECOPY")
		}

		if ir.IsConst(n.Left(), constant.Int) {
			if doesoverflow(n.Left().Val(), types.Types[types.TINT]) {
				base.Fatalf("len for OMAKESLICECOPY too large")
			}
			if constant.Sign(n.Left().Val()) < 0 {
				base.Fatalf("len for OMAKESLICECOPY must be non-negative")
			}
		}
		return n

	case ir.OSLICE, ir.OSLICE3:
		n.SetLeft(typecheck(n.Left(), ctxExpr))
		low, high, max := n.SliceBounds()
		hasmax := n.Op().IsSlice3()
		low = typecheck(low, ctxExpr)
		high = typecheck(high, ctxExpr)
		max = typecheck(max, ctxExpr)
		n.SetLeft(defaultlit(n.Left(), nil))
		low = indexlit(low)
		high = indexlit(high)
		max = indexlit(max)
		n.SetSliceBounds(low, high, max)
		l := n.Left()
		if l.Type() == nil {
			n.SetType(nil)
			return n
		}
		if l.Type().IsArray() {
			if !islvalue(n.Left()) {
				base.Errorf("invalid operation %v (slice of unaddressable value)", n)
				n.SetType(nil)
				return n
			}

			n.SetLeft(ir.Nod(ir.OADDR, n.Left(), nil))
			n.Left().SetImplicit(true)
			n.SetLeft(typecheck(n.Left(), ctxExpr))
			l = n.Left()
		}
		t := l.Type()
		var tp *types.Type
		if t.IsString() {
			if hasmax {
				base.Errorf("invalid operation %v (3-index slice of string)", n)
				n.SetType(nil)
				return n
			}
			n.SetType(t)
			n.SetOp(ir.OSLICESTR)
		} else if t.IsPtr() && t.Elem().IsArray() {
			tp = t.Elem()
			n.SetType(types.NewSlice(tp.Elem()))
			dowidth(n.Type())
			if hasmax {
				n.SetOp(ir.OSLICE3ARR)
			} else {
				n.SetOp(ir.OSLICEARR)
			}
		} else if t.IsSlice() {
			n.SetType(t)
		} else {
			base.Errorf("cannot slice %v (type %v)", l, t)
			n.SetType(nil)
			return n
		}

		if low != nil && !checksliceindex(l, low, tp) {
			n.SetType(nil)
			return n
		}
		if high != nil && !checksliceindex(l, high, tp) {
			n.SetType(nil)
			return n
		}
		if max != nil && !checksliceindex(l, max, tp) {
			n.SetType(nil)
			return n
		}
		if !checksliceconst(low, high) || !checksliceconst(low, max) || !checksliceconst(high, max) {
			n.SetType(nil)
			return n
		}
		return n

	// call and call like
	case ir.OCALL:
		n.(*ir.CallExpr).Use = ir.CallUseExpr
		if top == ctxStmt {
			n.(*ir.CallExpr).Use = ir.CallUseStmt
		}
		typecheckslice(n.Init().Slice(), ctxStmt) // imported rewritten f(g()) calls (#30907)
		n.SetLeft(typecheck(n.Left(), ctxExpr|ctxType|ctxCallee))
		if n.Left().Diag() {
			n.SetDiag(true)
		}

		l := n.Left()

		if l.Op() == ir.ONAME && l.SubOp() != 0 {
			if n.IsDDD() && l.SubOp() != ir.OAPPEND {
				base.Errorf("invalid use of ... with builtin %v", l)
			}

			// builtin: OLEN, OCAP, etc.
			switch l.SubOp() {
			default:
				base.Fatalf("unknown builtin %v", l)
				return n

			case ir.OAPPEND, ir.ODELETE, ir.OMAKE, ir.OPRINT, ir.OPRINTN, ir.ORECOVER:
				n.SetOp(l.SubOp())
				n.SetLeft(nil)
				n.SetTypecheck(0) // re-typechecking new op is OK, not a loop

			case ir.OCAP, ir.OCLOSE, ir.OIMAG, ir.OLEN, ir.OPANIC, ir.OREAL:
				typecheckargs(n)
				fallthrough
			case ir.ONEW, ir.OALIGNOF, ir.OOFFSETOF, ir.OSIZEOF:
				arg, ok := needOneArg(n, "%v", n.Op())
				if !ok {
					n.SetType(nil)
					return n
				}
				old := n
				n = ir.NodAt(n.Pos(), l.SubOp(), arg, nil)
				n = initExpr(old.Init().Slice(), n) // typecheckargs can add to old.Init

			case ir.OCOMPLEX, ir.OCOPY:
				typecheckargs(n)
				arg1, arg2, ok := needTwoArgs(n)
				if !ok {
					n.SetType(nil)
					return n
				}
				old := n
				n = ir.NodAt(n.Pos(), l.SubOp(), arg1, arg2)
				n = initExpr(old.Init().Slice(), n) // typecheckargs can add to old.Init
			}
			return typecheck(n, top)
		}

		n.SetLeft(defaultlit(n.Left(), nil))
		l = n.Left()
		if l.Op() == ir.OTYPE {
			if n.IsDDD() {
				if !l.Type().Broke() {
					base.Errorf("invalid use of ... in type conversion to %v", l.Type())
				}
				n.SetDiag(true)
			}

			// pick off before type-checking arguments
			arg, ok := needOneArg(n, "conversion to %v", l.Type())
			if !ok {
				n.SetType(nil)
				return n
			}

			n = ir.NodAt(n.Pos(), ir.OCONV, arg, nil)
			n.SetType(l.Type())
			return typecheck1(n, top)
		}

		typecheckargs(n)
		t := l.Type()
		if t == nil {
			n.SetType(nil)
			return n
		}
		checkwidth(t)

		switch l.Op() {
		case ir.ODOTINTER:
			n.SetOp(ir.OCALLINTER)

		case ir.ODOTMETH:
			n.SetOp(ir.OCALLMETH)

			// typecheckaste was used here but there wasn't enough
			// information further down the call chain to know if we
			// were testing a method receiver for unexported fields.
			// It isn't necessary, so just do a sanity check.
			tp := t.Recv().Type

			if l.Left() == nil || !types.Identical(l.Left().Type(), tp) {
				base.Fatalf("method receiver")
			}

		default:
			n.SetOp(ir.OCALLFUNC)
			if t.Kind() != types.TFUNC {
				name := l.String()
				if isBuiltinFuncName(name) && l.Name().Defn != nil {
					// be more specific when the function
					// name matches a predeclared function
					base.Errorf("cannot call non-function %s (type %v), declared at %s",
						name, t, base.FmtPos(l.Name().Defn.Pos()))
				} else {
					base.Errorf("cannot call non-function %s (type %v)", name, t)
				}
				n.SetType(nil)
				return n
			}
		}

		typecheckaste(ir.OCALL, n.Left(), n.IsDDD(), t.Params(), n.List(), func() string { return fmt.Sprintf("argument to %v", n.Left()) })
		if t.NumResults() == 0 {
			return n
		}
		if t.NumResults() == 1 {
			n.SetType(l.Type().Results().Field(0).Type)

			if n.Op() == ir.OCALLFUNC && n.Left().Op() == ir.ONAME && isRuntimePkg(n.Left().Sym().Pkg) && n.Left().Sym().Name == "getg" {
				// Emit code for runtime.getg() directly instead of calling function.
				// Most such rewrites (for example the similar one for math.Sqrt) should be done in walk,
				// so that the ordering pass can make sure to preserve the semantics of the original code
				// (in particular, the exact time of the function call) by introducing temporaries.
				// In this case, we know getg() always returns the same result within a given function
				// and we want to avoid the temporaries, so we do the rewrite earlier than is typical.
				n.SetOp(ir.OGETG)
			}
			return n
		}

		// multiple return
		if top&(ctxMultiOK|ctxStmt) == 0 {
			base.Errorf("multiple-value %v() in single-value context", l)
			return n
		}

		n.SetType(l.Type().Results())
		return n

	case ir.OALIGNOF, ir.OOFFSETOF, ir.OSIZEOF:
		n.SetType(types.Types[types.TUINTPTR])
		return n

	case ir.OCAP, ir.OLEN:
		n.SetLeft(typecheck(n.Left(), ctxExpr))
		n.SetLeft(defaultlit(n.Left(), nil))
		n.SetLeft(implicitstar(n.Left()))
		l := n.Left()
		t := l.Type()
		if t == nil {
			n.SetType(nil)
			return n
		}

		var ok bool
		if n.Op() == ir.OLEN {
			ok = okforlen[t.Kind()]
		} else {
			ok = okforcap[t.Kind()]
		}
		if !ok {
			base.Errorf("invalid argument %L for %v", l, n.Op())
			n.SetType(nil)
			return n
		}

		n.SetType(types.Types[types.TINT])
		return n

	case ir.OREAL, ir.OIMAG:
		n.SetLeft(typecheck(n.Left(), ctxExpr))
		l := n.Left()
		t := l.Type()
		if t == nil {
			n.SetType(nil)
			return n
		}

		// Determine result type.
		switch t.Kind() {
		case types.TIDEAL:
			n.SetType(types.UntypedFloat)
		case types.TCOMPLEX64:
			n.SetType(types.Types[types.TFLOAT32])
		case types.TCOMPLEX128:
			n.SetType(types.Types[types.TFLOAT64])
		default:
			base.Errorf("invalid argument %L for %v", l, n.Op())
			n.SetType(nil)
			return n
		}
		return n

	case ir.OCOMPLEX:
		l := typecheck(n.Left(), ctxExpr)
		r := typecheck(n.Right(), ctxExpr)
		if l.Type() == nil || r.Type() == nil {
			n.SetType(nil)
			return n
		}
		l, r = defaultlit2(l, r, false)
		if l.Type() == nil || r.Type() == nil {
			n.SetType(nil)
			return n
		}
		n.SetLeft(l)
		n.SetRight(r)

		if !types.Identical(l.Type(), r.Type()) {
			base.Errorf("invalid operation: %v (mismatched types %v and %v)", n, l.Type(), r.Type())
			n.SetType(nil)
			return n
		}

		var t *types.Type
		switch l.Type().Kind() {
		default:
			base.Errorf("invalid operation: %v (arguments have type %v, expected floating-point)", n, l.Type())
			n.SetType(nil)
			return n

		case types.TIDEAL:
			t = types.UntypedComplex

		case types.TFLOAT32:
			t = types.Types[types.TCOMPLEX64]

		case types.TFLOAT64:
			t = types.Types[types.TCOMPLEX128]
		}
		n.SetType(t)
		return n

	case ir.OCLOSE:
		n.SetLeft(typecheck(n.Left(), ctxExpr))
		n.SetLeft(defaultlit(n.Left(), nil))
		l := n.Left()
		t := l.Type()
		if t == nil {
			n.SetType(nil)
			return n
		}
		if !t.IsChan() {
			base.Errorf("invalid operation: %v (non-chan type %v)", n, t)
			n.SetType(nil)
			return n
		}

		if !t.ChanDir().CanSend() {
			base.Errorf("invalid operation: %v (cannot close receive-only channel)", n)
			n.SetType(nil)
			return n
		}
		return n

	case ir.ODELETE:
		typecheckargs(n)
		args := n.List()
		if args.Len() == 0 {
			base.Errorf("missing arguments to delete")
			n.SetType(nil)
			return n
		}

		if args.Len() == 1 {
			base.Errorf("missing second (key) argument to delete")
			n.SetType(nil)
			return n
		}

		if args.Len() != 2 {
			base.Errorf("too many arguments to delete")
			n.SetType(nil)
			return n
		}

		l := args.First()
		r := args.Second()
		if l.Type() != nil && !l.Type().IsMap() {
			base.Errorf("first argument to delete must be map; have %L", l.Type())
			n.SetType(nil)
			return n
		}

		args.SetSecond(assignconv(r, l.Type().Key(), "delete"))
		return n

	case ir.OAPPEND:
		typecheckargs(n)
		args := n.List()
		if args.Len() == 0 {
			base.Errorf("missing arguments to append")
			n.SetType(nil)
			return n
		}

		t := args.First().Type()
		if t == nil {
			n.SetType(nil)
			return n
		}

		n.SetType(t)
		if !t.IsSlice() {
			if ir.IsNil(args.First()) {
				base.Errorf("first argument to append must be typed slice; have untyped nil")
				n.SetType(nil)
				return n
			}

			base.Errorf("first argument to append must be slice; have %L", t)
			n.SetType(nil)
			return n
		}

		if n.IsDDD() {
			if args.Len() == 1 {
				base.Errorf("cannot use ... on first argument to append")
				n.SetType(nil)
				return n
			}

			if args.Len() != 2 {
				base.Errorf("too many arguments to append")
				n.SetType(nil)
				return n
			}

			if t.Elem().IsKind(types.TUINT8) && args.Second().Type().IsString() {
				args.SetSecond(defaultlit(args.Second(), types.Types[types.TSTRING]))
				return n
			}

			args.SetSecond(assignconv(args.Second(), t.Underlying(), "append"))
			return n
		}

		as := args.Slice()[1:]
		for i, n := range as {
			if n.Type() == nil {
				continue
			}
			as[i] = assignconv(n, t.Elem(), "append")
			checkwidth(as[i].Type()) // ensure width is calculated for backend
		}
		return n

	case ir.OCOPY:
		n.SetType(types.Types[types.TINT])
		n.SetLeft(typecheck(n.Left(), ctxExpr))
		n.SetLeft(defaultlit(n.Left(), nil))
		n.SetRight(typecheck(n.Right(), ctxExpr))
		n.SetRight(defaultlit(n.Right(), nil))
		if n.Left().Type() == nil || n.Right().Type() == nil {
			n.SetType(nil)
			return n
		}

		// copy([]byte, string)
		if n.Left().Type().IsSlice() && n.Right().Type().IsString() {
			if types.Identical(n.Left().Type().Elem(), types.ByteType) {
				return n
			}
			base.Errorf("arguments to copy have different element types: %L and string", n.Left().Type())
			n.SetType(nil)
			return n
		}

		if !n.Left().Type().IsSlice() || !n.Right().Type().IsSlice() {
			if !n.Left().Type().IsSlice() && !n.Right().Type().IsSlice() {
				base.Errorf("arguments to copy must be slices; have %L, %L", n.Left().Type(), n.Right().Type())
			} else if !n.Left().Type().IsSlice() {
				base.Errorf("first argument to copy should be slice; have %L", n.Left().Type())
			} else {
				base.Errorf("second argument to copy should be slice or string; have %L", n.Right().Type())
			}
			n.SetType(nil)
			return n
		}

		if !types.Identical(n.Left().Type().Elem(), n.Right().Type().Elem()) {
			base.Errorf("arguments to copy have different element types: %L and %L", n.Left().Type(), n.Right().Type())
			n.SetType(nil)
			return n
		}
		return n

	case ir.OCONV:
		checkwidth(n.Type()) // ensure width is calculated for backend
		n.SetLeft(typecheck(n.Left(), ctxExpr))
		n.SetLeft(convlit1(n.Left(), n.Type(), true, nil))
		t := n.Left().Type()
		if t == nil || n.Type() == nil {
			n.SetType(nil)
			return n
		}
		op, why := convertop(n.Left().Op() == ir.OLITERAL, t, n.Type())
		if op == ir.OXXX {
			if !n.Diag() && !n.Type().Broke() && !n.Left().Diag() {
				base.Errorf("cannot convert %L to type %v%s", n.Left(), n.Type(), why)
				n.SetDiag(true)
			}
			n.SetOp(ir.OCONV)
			n.SetType(nil)
			return n
		}

		n.SetOp(op)
		switch n.Op() {
		case ir.OCONVNOP:
			if t.Kind() == n.Type().Kind() {
				switch t.Kind() {
				case types.TFLOAT32, types.TFLOAT64, types.TCOMPLEX64, types.TCOMPLEX128:
					// Floating point casts imply rounding and
					// so the conversion must be kept.
					n.SetOp(ir.OCONV)
				}
			}

		// do not convert to []byte literal. See CL 125796.
		// generated code and compiler memory footprint is better without it.
		case ir.OSTR2BYTES:
			// ok

		case ir.OSTR2RUNES:
			if n.Left().Op() == ir.OLITERAL {
				n = stringtoruneslit(n)
			}
		}
		return n

	case ir.OMAKE:
		args := n.List().Slice()
		if len(args) == 0 {
			base.Errorf("missing argument to make")
			n.SetType(nil)
			return n
		}

		n.PtrList().Set(nil)
		l := args[0]
		l = typecheck(l, ctxType)
		t := l.Type()
		if t == nil {
			n.SetType(nil)
			return n
		}

		i := 1
		var nn ir.Node
		switch t.Kind() {
		default:
			base.Errorf("cannot make type %v", t)
			n.SetType(nil)
			return n

		case types.TSLICE:
			if i >= len(args) {
				base.Errorf("missing len argument to make(%v)", t)
				n.SetType(nil)
				return n
			}

			l = args[i]
			i++
			l = typecheck(l, ctxExpr)
			var r ir.Node
			if i < len(args) {
				r = args[i]
				i++
				r = typecheck(r, ctxExpr)
			}

			if l.Type() == nil || (r != nil && r.Type() == nil) {
				n.SetType(nil)
				return n
			}
			if !checkmake(t, "len", &l) || r != nil && !checkmake(t, "cap", &r) {
				n.SetType(nil)
				return n
			}
			if ir.IsConst(l, constant.Int) && r != nil && ir.IsConst(r, constant.Int) && constant.Compare(l.Val(), token.GTR, r.Val()) {
				base.Errorf("len larger than cap in make(%v)", t)
				n.SetType(nil)
				return n
			}
			nn = ir.NodAt(n.Pos(), ir.OMAKESLICE, l, r)

		case types.TMAP:
			if i < len(args) {
				l = args[i]
				i++
				l = typecheck(l, ctxExpr)
				l = defaultlit(l, types.Types[types.TINT])
				if l.Type() == nil {
					n.SetType(nil)
					return n
				}
				if !checkmake(t, "size", &l) {
					n.SetType(nil)
					return n
				}
			} else {
				l = nodintconst(0)
			}
			nn = ir.NodAt(n.Pos(), ir.OMAKEMAP, l, nil)
			nn.SetEsc(n.Esc())

		case types.TCHAN:
			l = nil
			if i < len(args) {
				l = args[i]
				i++
				l = typecheck(l, ctxExpr)
				l = defaultlit(l, types.Types[types.TINT])
				if l.Type() == nil {
					n.SetType(nil)
					return n
				}
				if !checkmake(t, "buffer", &l) {
					n.SetType(nil)
					return n
				}
			} else {
				l = nodintconst(0)
			}
			nn = ir.NodAt(n.Pos(), ir.OMAKECHAN, l, nil)
		}

		if i < len(args) {
			base.Errorf("too many arguments to make(%v)", t)
			n.SetType(nil)
			return n
		}

		nn.SetType(t)
		n = nn
		return n

	case ir.ONEW:
		if n.Left() == nil {
			// Fatalf because the OCALL above checked for us,
			// so this must be an internally-generated mistake.
			base.Fatalf("missing argument to new")
		}
		l := n.Left()
		l = typecheck(l, ctxType)
		t := l.Type()
		if t == nil {
			n.SetType(nil)
			return n
		}
		n.SetLeft(l)
		n.SetType(types.NewPtr(t))
		return n

	case ir.OPRINT, ir.OPRINTN:
		typecheckargs(n)
		ls := n.List().Slice()
		for i1, n1 := range ls {
			// Special case for print: int constant is int64, not int.
			if ir.IsConst(n1, constant.Int) {
				ls[i1] = defaultlit(ls[i1], types.Types[types.TINT64])
			} else {
				ls[i1] = defaultlit(ls[i1], nil)
			}
		}
		return n

	case ir.OPANIC:
		n.SetLeft(typecheck(n.Left(), ctxExpr))
		n.SetLeft(defaultlit(n.Left(), types.Types[types.TINTER]))
		if n.Left().Type() == nil {
			n.SetType(nil)
			return n
		}
		return n

	case ir.ORECOVER:
		if n.List().Len() != 0 {
			base.Errorf("too many arguments to recover")
			n.SetType(nil)
			return n
		}

		n.SetType(types.Types[types.TINTER])
		return n

	case ir.OCLOSURE:
		typecheckclosure(n, top)
		if n.Type() == nil {
			return n
		}
		return n

	case ir.OITAB:
		n.SetLeft(typecheck(n.Left(), ctxExpr))
		t := n.Left().Type()
		if t == nil {
			n.SetType(nil)
			return n
		}
		if !t.IsInterface() {
			base.Fatalf("OITAB of %v", t)
		}
		n.SetType(types.NewPtr(types.Types[types.TUINTPTR]))
		return n

	case ir.OIDATA:
		// Whoever creates the OIDATA node must know a priori the concrete type at that moment,
		// usually by just having checked the OITAB.
		base.Fatalf("cannot typecheck interface data %v", n)
		panic("unreachable")

	case ir.OSPTR:
		n.SetLeft(typecheck(n.Left(), ctxExpr))
		t := n.Left().Type()
		if t == nil {
			n.SetType(nil)
			return n
		}
		if !t.IsSlice() && !t.IsString() {
			base.Fatalf("OSPTR of %v", t)
		}
		if t.IsString() {
			n.SetType(types.NewPtr(types.Types[types.TUINT8]))
		} else {
			n.SetType(types.NewPtr(t.Elem()))
		}
		return n

	case ir.OCLOSUREREAD:
		return n

	case ir.OCFUNC:
		n.SetLeft(typecheck(n.Left(), ctxExpr))
		n.SetType(types.Types[types.TUINTPTR])
		return n

	case ir.OCONVNOP:
		n.SetLeft(typecheck(n.Left(), ctxExpr))
		return n

	// statements
	case ir.OAS:
		typecheckas(n)

		// Code that creates temps does not bother to set defn, so do it here.
		if n.Left().Op() == ir.ONAME && ir.IsAutoTmp(n.Left()) {
			n.Left().Name().Defn = n
		}
		return n

	case ir.OAS2:
		typecheckas2(n)
		return n

	case ir.OBREAK,
		ir.OCONTINUE,
		ir.ODCL,
		ir.OGOTO,
		ir.OFALL,
		ir.OVARKILL,
		ir.OVARLIVE:
		return n

	case ir.OBLOCK:
		typecheckslice(n.List().Slice(), ctxStmt)
		return n

	case ir.OLABEL:
		decldepth++
		if n.Sym().IsBlank() {
			// Empty identifier is valid but useless.
			// Eliminate now to simplify life later.
			// See issues 7538, 11589, 11593.
			n = ir.NodAt(n.Pos(), ir.OBLOCK, nil, nil)
		}
		return n

	case ir.ODEFER:
		n.SetLeft(typecheck(n.Left(), ctxStmt|ctxExpr))
		if !n.Left().Diag() {
			checkdefergo(n)
		}
		return n

	case ir.OGO:
		n.SetLeft(typecheck(n.Left(), ctxStmt|ctxExpr))
		checkdefergo(n)
		return n

	case ir.OFOR, ir.OFORUNTIL:
		typecheckslice(n.Init().Slice(), ctxStmt)
		decldepth++
		n.SetLeft(typecheck(n.Left(), ctxExpr))
		n.SetLeft(defaultlit(n.Left(), nil))
		if n.Left() != nil {
			t := n.Left().Type()
			if t != nil && !t.IsBoolean() {
				base.Errorf("non-bool %L used as for condition", n.Left())
			}
		}
		n.SetRight(typecheck(n.Right(), ctxStmt))
		if n.Op() == ir.OFORUNTIL {
			typecheckslice(n.List().Slice(), ctxStmt)
		}
		typecheckslice(n.Body().Slice(), ctxStmt)
		decldepth--
		return n

	case ir.OIF:
		typecheckslice(n.Init().Slice(), ctxStmt)
		n.SetLeft(typecheck(n.Left(), ctxExpr))
		n.SetLeft(defaultlit(n.Left(), nil))
		if n.Left() != nil {
			t := n.Left().Type()
			if t != nil && !t.IsBoolean() {
				base.Errorf("non-bool %L used as if condition", n.Left())
			}
		}
		typecheckslice(n.Body().Slice(), ctxStmt)
		typecheckslice(n.Rlist().Slice(), ctxStmt)
		return n

	case ir.ORETURN:
		typecheckargs(n)
		if Curfn == nil {
			base.Errorf("return outside function")
			n.SetType(nil)
			return n
		}

		if hasNamedResults(Curfn) && n.List().Len() == 0 {
			return n
		}
		typecheckaste(ir.ORETURN, nil, false, Curfn.Type().Results(), n.List(), func() string { return "return argument" })
		return n

	case ir.ORETJMP:
		return n

	case ir.OSELECT:
		typecheckselect(n)
		return n

	case ir.OSWITCH:
		typecheckswitch(n)
		return n

	case ir.ORANGE:
		typecheckrange(n)
		return n

	case ir.OTYPESW:
		base.Errorf("use of .(type) outside type switch")
		n.SetType(nil)
		return n

	case ir.ODCLFUNC:
		typecheckfunc(n.(*ir.Func))
		return n

	case ir.ODCLCONST:
		n.SetLeft(typecheck(n.Left(), ctxExpr))
		return n

	case ir.ODCLTYPE:
		n.SetLeft(typecheck(n.Left(), ctxType))
		checkwidth(n.Left().Type())
		return n
	}

	// No return n here!
	// Individual cases can type-assert n, introducing a new one.
	// Each must execute its own return n.
}

func typecheckargs(n ir.Node) {
	if n.List().Len() != 1 || n.IsDDD() {
		typecheckslice(n.List().Slice(), ctxExpr)
		return
	}

	typecheckslice(n.List().Slice(), ctxExpr|ctxMultiOK)
	t := n.List().First().Type()
	if t == nil || !t.IsFuncArgStruct() {
		return
	}

	// Rewrite f(g()) into t1, t2, ... = g(); f(t1, t2, ...).

	// Save n as n.Orig for fmt.go.
	if ir.Orig(n) == n {
		n.(ir.OrigNode).SetOrig(ir.SepCopy(n))
	}

	as := ir.Nod(ir.OAS2, nil, nil)
	as.PtrRlist().AppendNodes(n.PtrList())

	// If we're outside of function context, then this call will
	// be executed during the generated init function. However,
	// init.go hasn't yet created it. Instead, associate the
	// temporary variables with initTodo for now, and init.go
	// will reassociate them later when it's appropriate.
	static := Curfn == nil
	if static {
		Curfn = initTodo
	}
	for _, f := range t.FieldSlice() {
		t := temp(f.Type)
		as.PtrInit().Append(ir.Nod(ir.ODCL, t, nil))
		as.PtrList().Append(t)
		n.PtrList().Append(t)
	}
	if static {
		Curfn = nil
	}

	as = typecheck(as, ctxStmt)
	n.PtrInit().Append(as)
}

func checksliceindex(l ir.Node, r ir.Node, tp *types.Type) bool {
	t := r.Type()
	if t == nil {
		return false
	}
	if !t.IsInteger() {
		base.Errorf("invalid slice index %v (type %v)", r, t)
		return false
	}

	if r.Op() == ir.OLITERAL {
		x := r.Val()
		if constant.Sign(x) < 0 {
			base.Errorf("invalid slice index %v (index must be non-negative)", r)
			return false
		} else if tp != nil && tp.NumElem() >= 0 && constant.Compare(x, token.GTR, constant.MakeInt64(tp.NumElem())) {
			base.Errorf("invalid slice index %v (out of bounds for %d-element array)", r, tp.NumElem())
			return false
		} else if ir.IsConst(l, constant.String) && constant.Compare(x, token.GTR, constant.MakeInt64(int64(len(ir.StringVal(l))))) {
			base.Errorf("invalid slice index %v (out of bounds for %d-byte string)", r, len(ir.StringVal(l)))
			return false
		} else if doesoverflow(x, types.Types[types.TINT]) {
			base.Errorf("invalid slice index %v (index too large)", r)
			return false
		}
	}

	return true
}

func checksliceconst(lo ir.Node, hi ir.Node) bool {
	if lo != nil && hi != nil && lo.Op() == ir.OLITERAL && hi.Op() == ir.OLITERAL && constant.Compare(lo.Val(), token.GTR, hi.Val()) {
		base.Errorf("invalid slice index: %v > %v", lo, hi)
		return false
	}

	return true
}

func checkdefergo(n ir.Node) {
	what := "defer"
	if n.Op() == ir.OGO {
		what = "go"
	}

	switch n.Left().Op() {
	// ok
	case ir.OCALLINTER,
		ir.OCALLMETH,
		ir.OCALLFUNC,
		ir.OCLOSE,
		ir.OCOPY,
		ir.ODELETE,
		ir.OPANIC,
		ir.OPRINT,
		ir.OPRINTN,
		ir.ORECOVER:
		return

	case ir.OAPPEND,
		ir.OCAP,
		ir.OCOMPLEX,
		ir.OIMAG,
		ir.OLEN,
		ir.OMAKE,
		ir.OMAKESLICE,
		ir.OMAKECHAN,
		ir.OMAKEMAP,
		ir.ONEW,
		ir.OREAL,
		ir.OLITERAL: // conversion or unsafe.Alignof, Offsetof, Sizeof
		if orig := ir.Orig(n.Left()); orig.Op() == ir.OCONV {
			break
		}
		base.ErrorfAt(n.Pos(), "%s discards result of %v", what, n.Left())
		return
	}

	// type is broken or missing, most likely a method call on a broken type
	// we will warn about the broken type elsewhere. no need to emit a potentially confusing error
	if n.Left().Type() == nil || n.Left().Type().Broke() {
		return
	}

	if !n.Diag() {
		// The syntax made sure it was a call, so this must be
		// a conversion.
		n.SetDiag(true)
		base.ErrorfAt(n.Pos(), "%s requires function call, not conversion", what)
	}
}

// The result of implicitstar MUST be assigned back to n, e.g.
// 	n.Left = implicitstar(n.Left)
func implicitstar(n ir.Node) ir.Node {
	// insert implicit * if needed for fixed array
	t := n.Type()
	if t == nil || !t.IsPtr() {
		return n
	}
	t = t.Elem()
	if t == nil {
		return n
	}
	if !t.IsArray() {
		return n
	}
	n = ir.Nod(ir.ODEREF, n, nil)
	n.SetImplicit(true)
	n = typecheck(n, ctxExpr)
	return n
}

func needOneArg(n ir.Node, f string, args ...interface{}) (ir.Node, bool) {
	if n.List().Len() == 0 {
		p := fmt.Sprintf(f, args...)
		base.Errorf("missing argument to %s: %v", p, n)
		return nil, false
	}

	if n.List().Len() > 1 {
		p := fmt.Sprintf(f, args...)
		base.Errorf("too many arguments to %s: %v", p, n)
		return n.List().First(), false
	}

	return n.List().First(), true
}

func needTwoArgs(n ir.Node) (ir.Node, ir.Node, bool) {
	if n.List().Len() != 2 {
		if n.List().Len() < 2 {
			base.Errorf("not enough arguments in call to %v", n)
		} else {
			base.Errorf("too many arguments in call to %v", n)
		}
		return nil, nil, false
	}
	return n.List().First(), n.List().Second(), true
}

func lookdot1(errnode ir.Node, s *types.Sym, t *types.Type, fs *types.Fields, dostrcmp int) *types.Field {
	var r *types.Field
	for _, f := range fs.Slice() {
		if dostrcmp != 0 && f.Sym.Name == s.Name {
			return f
		}
		if dostrcmp == 2 && strings.EqualFold(f.Sym.Name, s.Name) {
			return f
		}
		if f.Sym != s {
			continue
		}
		if r != nil {
			if errnode != nil {
				base.Errorf("ambiguous selector %v", errnode)
			} else if t.IsPtr() {
				base.Errorf("ambiguous selector (%v).%v", t, s)
			} else {
				base.Errorf("ambiguous selector %v.%v", t, s)
			}
			break
		}

		r = f
	}

	return r
}

// typecheckMethodExpr checks selector expressions (ODOT) where the
// base expression is a type expression (OTYPE).
func typecheckMethodExpr(n ir.Node) (res ir.Node) {
	if enableTrace && base.Flag.LowerT {
		defer tracePrint("typecheckMethodExpr", n)(&res)
	}

	t := n.Left().Type()

	// Compute the method set for t.
	var ms *types.Fields
	if t.IsInterface() {
		ms = t.Fields()
	} else {
		mt := methtype(t)
		if mt == nil {
			base.Errorf("%v undefined (type %v has no method %v)", n, t, n.Sym())
			n.SetType(nil)
			return n
		}
		expandmeth(mt)
		ms = mt.AllMethods()

		// The method expression T.m requires a wrapper when T
		// is different from m's declared receiver type. We
		// normally generate these wrappers while writing out
		// runtime type descriptors, which is always done for
		// types declared at package scope. However, we need
		// to make sure to generate wrappers for anonymous
		// receiver types too.
		if mt.Sym() == nil {
			addsignat(t)
		}
	}

	s := n.Sym()
	m := lookdot1(n, s, t, ms, 0)
	if m == nil {
		if lookdot1(n, s, t, ms, 1) != nil {
			base.Errorf("%v undefined (cannot refer to unexported method %v)", n, s)
		} else if _, ambig := dotpath(s, t, nil, false); ambig {
			base.Errorf("%v undefined (ambiguous selector)", n) // method or field
		} else {
			base.Errorf("%v undefined (type %v has no method %v)", n, t, s)
		}
		n.SetType(nil)
		return n
	}

	if !isMethodApplicable(t, m) {
		base.Errorf("invalid method expression %v (needs pointer receiver: (*%v).%S)", n, t, s)
		n.SetType(nil)
		return n
	}

	me := ir.NodAt(n.Pos(), ir.OMETHEXPR, n.Left(), NewName(n.Sym()))
	me.SetSym(methodSym(t, n.Sym()))
	me.SetType(methodfunc(m.Type, n.Left().Type()))
	me.SetOffset(0)
	me.SetClass(ir.PFUNC)
	me.(*ir.MethodExpr).Method = m

	// Issue 25065. Make sure that we emit the symbol for a local method.
	if base.Ctxt.Flag_dynlink && !inimport && (t.Sym() == nil || t.Sym().Pkg == types.LocalPkg) {
		makefuncsym(me.Sym())
	}

	return me
}

// isMethodApplicable reports whether method m can be called on a
// value of type t. This is necessary because we compute a single
// method set for both T and *T, but some *T methods are not
// applicable to T receivers.
func isMethodApplicable(t *types.Type, m *types.Field) bool {
	return t.IsPtr() || !m.Type.Recv().Type.IsPtr() || isifacemethod(m.Type) || m.Embedded == 2
}

func derefall(t *types.Type) *types.Type {
	for t != nil && t.IsPtr() {
		t = t.Elem()
	}
	return t
}

func lookdot(n ir.Node, t *types.Type, dostrcmp int) *types.Field {
	s := n.Sym()

	dowidth(t)
	var f1 *types.Field
	if t.IsStruct() || t.IsInterface() {
		f1 = lookdot1(n, s, t, t.Fields(), dostrcmp)
	}

	var f2 *types.Field
	if n.Left().Type() == t || n.Left().Type().Sym() == nil {
		mt := methtype(t)
		if mt != nil {
			f2 = lookdot1(n, s, mt, mt.Methods(), dostrcmp)
		}
	}

	if f1 != nil {
		if dostrcmp > 1 || f1.Broke() {
			// Already in the process of diagnosing an error.
			return f1
		}
		if f2 != nil {
			base.Errorf("%v is both field and method", n.Sym())
		}
		if f1.Offset == types.BADWIDTH {
			base.Fatalf("lookdot badwidth %v %p", f1, f1)
		}
		n.SetOffset(f1.Offset)
		n.SetType(f1.Type)
		if t.IsInterface() {
			if n.Left().Type().IsPtr() {
				n.SetLeft(ir.Nod(ir.ODEREF, n.Left(), nil)) // implicitstar
				n.Left().SetImplicit(true)
				n.SetLeft(typecheck(n.Left(), ctxExpr))
			}

			n.SetOp(ir.ODOTINTER)
		}
		n.(*ir.SelectorExpr).Selection = f1
		return f1
	}

	if f2 != nil {
		if dostrcmp > 1 {
			// Already in the process of diagnosing an error.
			return f2
		}
		tt := n.Left().Type()
		dowidth(tt)
		rcvr := f2.Type.Recv().Type
		if !types.Identical(rcvr, tt) {
			if rcvr.IsPtr() && types.Identical(rcvr.Elem(), tt) {
				checklvalue(n.Left(), "call pointer method on")
				n.SetLeft(ir.Nod(ir.OADDR, n.Left(), nil))
				n.Left().SetImplicit(true)
				n.SetLeft(typecheck(n.Left(), ctxType|ctxExpr))
			} else if tt.IsPtr() && (!rcvr.IsPtr() || rcvr.IsPtr() && rcvr.Elem().NotInHeap()) && types.Identical(tt.Elem(), rcvr) {
				n.SetLeft(ir.Nod(ir.ODEREF, n.Left(), nil))
				n.Left().SetImplicit(true)
				n.SetLeft(typecheck(n.Left(), ctxType|ctxExpr))
			} else if tt.IsPtr() && tt.Elem().IsPtr() && types.Identical(derefall(tt), derefall(rcvr)) {
				base.Errorf("calling method %v with receiver %L requires explicit dereference", n.Sym(), n.Left())
				for tt.IsPtr() {
					// Stop one level early for method with pointer receiver.
					if rcvr.IsPtr() && !tt.Elem().IsPtr() {
						break
					}
					n.SetLeft(ir.Nod(ir.ODEREF, n.Left(), nil))
					n.Left().SetImplicit(true)
					n.SetLeft(typecheck(n.Left(), ctxType|ctxExpr))
					tt = tt.Elem()
				}
			} else {
				base.Fatalf("method mismatch: %v for %v", rcvr, tt)
			}
		}

		pll := n
		ll := n.Left()
		for ll.Left() != nil && (ll.Op() == ir.ODOT || ll.Op() == ir.ODOTPTR || ll.Op() == ir.ODEREF) {
			pll = ll
			ll = ll.Left()
		}
		if pll.Implicit() && ll.Type().IsPtr() && ll.Type().Sym() != nil && ll.Type().Sym().Def != nil && ir.AsNode(ll.Type().Sym().Def).Op() == ir.OTYPE {
			// It is invalid to automatically dereference a named pointer type when selecting a method.
			// Make n.Left == ll to clarify error message.
			n.SetLeft(ll)
			return nil
		}

		n.SetSym(methodSym(n.Left().Type(), f2.Sym))
		n.SetOffset(f2.Offset)
		n.SetType(f2.Type)
		n.SetOp(ir.ODOTMETH)
		n.(*ir.SelectorExpr).Selection = f2

		return f2
	}

	return nil
}

func nokeys(l ir.Nodes) bool {
	for _, n := range l.Slice() {
		if n.Op() == ir.OKEY || n.Op() == ir.OSTRUCTKEY {
			return false
		}
	}
	return true
}

func hasddd(t *types.Type) bool {
	for _, tl := range t.Fields().Slice() {
		if tl.IsDDD() {
			return true
		}
	}

	return false
}

// typecheck assignment: type list = expression list
func typecheckaste(op ir.Op, call ir.Node, isddd bool, tstruct *types.Type, nl ir.Nodes, desc func() string) {
	var t *types.Type
	var i int

	lno := base.Pos
	defer func() { base.Pos = lno }()

	if tstruct.Broke() {
		return
	}

	var n ir.Node
	if nl.Len() == 1 {
		n = nl.First()
	}

	n1 := tstruct.NumFields()
	n2 := nl.Len()
	if !hasddd(tstruct) {
		if n2 > n1 {
			goto toomany
		}
		if n2 < n1 {
			goto notenough
		}
	} else {
		if !isddd {
			if n2 < n1-1 {
				goto notenough
			}
		} else {
			if n2 > n1 {
				goto toomany
			}
			if n2 < n1 {
				goto notenough
			}
		}
	}

	i = 0
	for _, tl := range tstruct.Fields().Slice() {
		t = tl.Type
		if tl.IsDDD() {
			if isddd {
				if i >= nl.Len() {
					goto notenough
				}
				if nl.Len()-i > 1 {
					goto toomany
				}
				n = nl.Index(i)
				setlineno(n)
				if n.Type() != nil {
					nl.SetIndex(i, assignconvfn(n, t, desc))
				}
				return
			}

			// TODO(mdempsky): Make into ... call with implicit slice.
			for ; i < nl.Len(); i++ {
				n = nl.Index(i)
				setlineno(n)
				if n.Type() != nil {
					nl.SetIndex(i, assignconvfn(n, t.Elem(), desc))
				}
			}
			return
		}

		if i >= nl.Len() {
			goto notenough
		}
		n = nl.Index(i)
		setlineno(n)
		if n.Type() != nil {
			nl.SetIndex(i, assignconvfn(n, t, desc))
		}
		i++
	}

	if i < nl.Len() {
		goto toomany
	}
	if isddd {
		if call != nil {
			base.Errorf("invalid use of ... in call to %v", call)
		} else {
			base.Errorf("invalid use of ... in %v", op)
		}
	}
	return

notenough:
	if n == nil || (!n.Diag() && n.Type() != nil) {
		details := errorDetails(nl, tstruct, isddd)
		if call != nil {
			// call is the expression being called, not the overall call.
			// Method expressions have the form T.M, and the compiler has
			// rewritten those to ONAME nodes but left T in Left.
			if call.Op() == ir.OMETHEXPR {
				base.Errorf("not enough arguments in call to method expression %v%s", call, details)
			} else {
				base.Errorf("not enough arguments in call to %v%s", call, details)
			}
		} else {
			base.Errorf("not enough arguments to %v%s", op, details)
		}
		if n != nil {
			n.SetDiag(true)
		}
	}
	return

toomany:
	details := errorDetails(nl, tstruct, isddd)
	if call != nil {
		base.Errorf("too many arguments in call to %v%s", call, details)
	} else {
		base.Errorf("too many arguments to %v%s", op, details)
	}
}

func errorDetails(nl ir.Nodes, tstruct *types.Type, isddd bool) string {
	// If we don't know any type at a call site, let's suppress any return
	// message signatures. See Issue https://golang.org/issues/19012.
	if tstruct == nil {
		return ""
	}
	// If any node has an unknown type, suppress it as well
	for _, n := range nl.Slice() {
		if n.Type() == nil {
			return ""
		}
	}
	return fmt.Sprintf("\n\thave %s\n\twant %v", fmtSignature(nl, isddd), tstruct)
}

// sigrepr is a type's representation to the outside world,
// in string representations of return signatures
// e.g in error messages about wrong arguments to return.
func sigrepr(t *types.Type, isddd bool) string {
	switch t {
	case types.UntypedString:
		return "string"
	case types.UntypedBool:
		return "bool"
	}

	if t.Kind() == types.TIDEAL {
		// "untyped number" is not commonly used
		// outside of the compiler, so let's use "number".
		// TODO(mdempsky): Revisit this.
		return "number"
	}

	// Turn []T... argument to ...T for clearer error message.
	if isddd {
		if !t.IsSlice() {
			base.Fatalf("bad type for ... argument: %v", t)
		}
		return "..." + t.Elem().String()
	}
	return t.String()
}

// sigerr returns the signature of the types at the call or return.
func fmtSignature(nl ir.Nodes, isddd bool) string {
	if nl.Len() < 1 {
		return "()"
	}

	var typeStrings []string
	for i, n := range nl.Slice() {
		isdddArg := isddd && i == nl.Len()-1
		typeStrings = append(typeStrings, sigrepr(n.Type(), isdddArg))
	}

	return fmt.Sprintf("(%s)", strings.Join(typeStrings, ", "))
}

// type check composite
func fielddup(name string, hash map[string]bool) {
	if hash[name] {
		base.Errorf("duplicate field name in struct literal: %s", name)
		return
	}
	hash[name] = true
}

// iscomptype reports whether type t is a composite literal type.
func iscomptype(t *types.Type) bool {
	switch t.Kind() {
	case types.TARRAY, types.TSLICE, types.TSTRUCT, types.TMAP:
		return true
	default:
		return false
	}
}

// pushtype adds elided type information for composite literals if
// appropriate, and returns the resulting expression.
func pushtype(n ir.Node, t *types.Type) ir.Node {
	if n == nil || n.Op() != ir.OCOMPLIT || n.Right() != nil {
		return n
	}

	switch {
	case iscomptype(t):
		// For T, return T{...}.
		n.SetRight(ir.TypeNode(t))

	case t.IsPtr() && iscomptype(t.Elem()):
		// For *T, return &T{...}.
		n.SetRight(ir.TypeNode(t.Elem()))

		n = ir.NodAt(n.Pos(), ir.OADDR, n, nil)
		n.SetImplicit(true)
	}

	return n
}

// The result of typecheckcomplit MUST be assigned back to n, e.g.
// 	n.Left = typecheckcomplit(n.Left)
func typecheckcomplit(n ir.Node) (res ir.Node) {
	if enableTrace && base.Flag.LowerT {
		defer tracePrint("typecheckcomplit", n)(&res)
	}

	lno := base.Pos
	defer func() {
		base.Pos = lno
	}()

	if n.Right() == nil {
		base.ErrorfAt(n.Pos(), "missing type in composite literal")
		n.SetType(nil)
		return n
	}

	// Save original node (including n.Right)
	n.(ir.OrigNode).SetOrig(ir.Copy(n))

	setlineno(n.Right())

	// Need to handle [...]T arrays specially.
	if array, ok := n.Right().(*ir.ArrayType); ok && array.Elem != nil && array.Len == nil {
		array.Elem = typecheck(array.Elem, ctxType)
		elemType := array.Elem.Type()
		if elemType == nil {
			n.SetType(nil)
			return n
		}
		length := typecheckarraylit(elemType, -1, n.List().Slice(), "array literal")
		n.SetOp(ir.OARRAYLIT)
		n.SetType(types.NewArray(elemType, length))
		n.SetRight(nil)
		return n
	}

	n.SetRight(typecheck(n.Right(), ctxType))
	t := n.Right().Type()
	if t == nil {
		n.SetType(nil)
		return n
	}
	n.SetType(t)

	switch t.Kind() {
	default:
		base.Errorf("invalid composite literal type %v", t)
		n.SetType(nil)

	case types.TARRAY:
		typecheckarraylit(t.Elem(), t.NumElem(), n.List().Slice(), "array literal")
		n.SetOp(ir.OARRAYLIT)
		n.SetRight(nil)

	case types.TSLICE:
		length := typecheckarraylit(t.Elem(), -1, n.List().Slice(), "slice literal")
		n.SetOp(ir.OSLICELIT)
		n.SetRight(nodintconst(length))

	case types.TMAP:
		var cs constSet
		for i3, l := range n.List().Slice() {
			setlineno(l)
			if l.Op() != ir.OKEY {
				n.List().SetIndex(i3, typecheck(l, ctxExpr))
				base.Errorf("missing key in map literal")
				continue
			}

			r := l.Left()
			r = pushtype(r, t.Key())
			r = typecheck(r, ctxExpr)
			l.SetLeft(assignconv(r, t.Key(), "map key"))
			cs.add(base.Pos, l.Left(), "key", "map literal")

			r = l.Right()
			r = pushtype(r, t.Elem())
			r = typecheck(r, ctxExpr)
			l.SetRight(assignconv(r, t.Elem(), "map value"))
		}

		n.SetOp(ir.OMAPLIT)
		n.SetRight(nil)

	case types.TSTRUCT:
		// Need valid field offsets for Xoffset below.
		dowidth(t)

		errored := false
		if n.List().Len() != 0 && nokeys(n.List()) {
			// simple list of variables
			ls := n.List().Slice()
			for i, n1 := range ls {
				setlineno(n1)
				n1 = typecheck(n1, ctxExpr)
				ls[i] = n1
				if i >= t.NumFields() {
					if !errored {
						base.Errorf("too many values in %v", n)
						errored = true
					}
					continue
				}

				f := t.Field(i)
				s := f.Sym
				if s != nil && !types.IsExported(s.Name) && s.Pkg != types.LocalPkg {
					base.Errorf("implicit assignment of unexported field '%s' in %v literal", s.Name, t)
				}
				// No pushtype allowed here. Must name fields for that.
				n1 = assignconv(n1, f.Type, "field value")
				n1 = nodSym(ir.OSTRUCTKEY, n1, f.Sym)
				n1.SetOffset(f.Offset)
				ls[i] = n1
			}
			if len(ls) < t.NumFields() {
				base.Errorf("too few values in %v", n)
			}
		} else {
			hash := make(map[string]bool)

			// keyed list
			ls := n.List().Slice()
			for i, l := range ls {
				setlineno(l)

				if l.Op() == ir.OKEY {
					key := l.Left()

					l.SetOp(ir.OSTRUCTKEY)
					l.SetLeft(l.Right())
					l.SetRight(nil)

					// An OXDOT uses the Sym field to hold
					// the field to the right of the dot,
					// so s will be non-nil, but an OXDOT
					// is never a valid struct literal key.
					if key.Sym() == nil || key.Op() == ir.OXDOT || key.Sym().IsBlank() {
						base.Errorf("invalid field name %v in struct initializer", key)
						l.SetLeft(typecheck(l.Left(), ctxExpr))
						continue
					}

					// Sym might have resolved to name in other top-level
					// package, because of import dot. Redirect to correct sym
					// before we do the lookup.
					s := key.Sym()
					if s.Pkg != types.LocalPkg && types.IsExported(s.Name) {
						s1 := lookup(s.Name)
						if s1.Origpkg == s.Pkg {
							s = s1
						}
					}
					l.SetSym(s)
				}

				if l.Op() != ir.OSTRUCTKEY {
					if !errored {
						base.Errorf("mixture of field:value and value initializers")
						errored = true
					}
					ls[i] = typecheck(ls[i], ctxExpr)
					continue
				}

				f := lookdot1(nil, l.Sym(), t, t.Fields(), 0)
				if f == nil {
					if ci := lookdot1(nil, l.Sym(), t, t.Fields(), 2); ci != nil { // Case-insensitive lookup.
						if visible(ci.Sym) {
							base.Errorf("unknown field '%v' in struct literal of type %v (but does have %v)", l.Sym(), t, ci.Sym)
						} else if nonexported(l.Sym()) && l.Sym().Name == ci.Sym.Name { // Ensure exactness before the suggestion.
							base.Errorf("cannot refer to unexported field '%v' in struct literal of type %v", l.Sym(), t)
						} else {
							base.Errorf("unknown field '%v' in struct literal of type %v", l.Sym(), t)
						}
						continue
					}
					var f *types.Field
					p, _ := dotpath(l.Sym(), t, &f, true)
					if p == nil || f.IsMethod() {
						base.Errorf("unknown field '%v' in struct literal of type %v", l.Sym(), t)
						continue
					}
					// dotpath returns the parent embedded types in reverse order.
					var ep []string
					for ei := len(p) - 1; ei >= 0; ei-- {
						ep = append(ep, p[ei].field.Sym.Name)
					}
					ep = append(ep, l.Sym().Name)
					base.Errorf("cannot use promoted field %v in struct literal of type %v", strings.Join(ep, "."), t)
					continue
				}
				fielddup(f.Sym.Name, hash)
				l.SetOffset(f.Offset)

				// No pushtype allowed here. Tried and rejected.
				l.SetLeft(typecheck(l.Left(), ctxExpr))
				l.SetLeft(assignconv(l.Left(), f.Type, "field value"))
			}
		}

		n.SetOp(ir.OSTRUCTLIT)
		n.SetRight(nil)
	}

	return n
}

// typecheckarraylit type-checks a sequence of slice/array literal elements.
func typecheckarraylit(elemType *types.Type, bound int64, elts []ir.Node, ctx string) int64 {
	// If there are key/value pairs, create a map to keep seen
	// keys so we can check for duplicate indices.
	var indices map[int64]bool
	for _, elt := range elts {
		if elt.Op() == ir.OKEY {
			indices = make(map[int64]bool)
			break
		}
	}

	var key, length int64
	for i, elt := range elts {
		setlineno(elt)
		r := elts[i]
		var kv ir.Node
		if elt.Op() == ir.OKEY {
			elt.SetLeft(typecheck(elt.Left(), ctxExpr))
			key = indexconst(elt.Left())
			if key < 0 {
				if !elt.Left().Diag() {
					if key == -2 {
						base.Errorf("index too large")
					} else {
						base.Errorf("index must be non-negative integer constant")
					}
					elt.Left().SetDiag(true)
				}
				key = -(1 << 30) // stay negative for a while
			}
			kv = elt
			r = elt.Right()
		}

		r = pushtype(r, elemType)
		r = typecheck(r, ctxExpr)
		r = assignconv(r, elemType, ctx)
		if kv != nil {
			kv.SetRight(r)
		} else {
			elts[i] = r
		}

		if key >= 0 {
			if indices != nil {
				if indices[key] {
					base.Errorf("duplicate index in %s: %d", ctx, key)
				} else {
					indices[key] = true
				}
			}

			if bound >= 0 && key >= bound {
				base.Errorf("array index %d out of bounds [0:%d]", key, bound)
				bound = -1
			}
		}

		key++
		if key > length {
			length = key
		}
	}

	return length
}

// visible reports whether sym is exported or locally defined.
func visible(sym *types.Sym) bool {
	return sym != nil && (types.IsExported(sym.Name) || sym.Pkg == types.LocalPkg)
}

// nonexported reports whether sym is an unexported field.
func nonexported(sym *types.Sym) bool {
	return sym != nil && !types.IsExported(sym.Name)
}

// lvalue etc
func islvalue(n ir.Node) bool {
	switch n.Op() {
	case ir.OINDEX:
		if n.Left().Type() != nil && n.Left().Type().IsArray() {
			return islvalue(n.Left())
		}
		if n.Left().Type() != nil && n.Left().Type().IsString() {
			return false
		}
		fallthrough
	case ir.ODEREF, ir.ODOTPTR, ir.OCLOSUREREAD:
		return true

	case ir.ODOT:
		return islvalue(n.Left())

	case ir.ONAME:
		if n.Class() == ir.PFUNC {
			return false
		}
		return true
	}

	return false
}

func checklvalue(n ir.Node, verb string) {
	if !islvalue(n) {
		base.Errorf("cannot %s %v", verb, n)
	}
}

func checkassign(stmt ir.Node, n ir.Node) {
	// Variables declared in ORANGE are assigned on every iteration.
	if !ir.DeclaredBy(n, stmt) || stmt.Op() == ir.ORANGE {
		r := outervalue(n)
		if r.Op() == ir.ONAME {
			r.Name().SetAssigned(true)
			if r.Name().IsClosureVar() {
				r.Name().Defn.Name().SetAssigned(true)
			}
		}
	}

	if islvalue(n) {
		return
	}
	if n.Op() == ir.OINDEXMAP {
		n.SetIndexMapLValue(true)
		return
	}

	// have already complained about n being invalid
	if n.Type() == nil {
		return
	}

	switch {
	case n.Op() == ir.ODOT && n.Left().Op() == ir.OINDEXMAP:
		base.Errorf("cannot assign to struct field %v in map", n)
	case (n.Op() == ir.OINDEX && n.Left().Type().IsString()) || n.Op() == ir.OSLICESTR:
		base.Errorf("cannot assign to %v (strings are immutable)", n)
	case n.Op() == ir.OLITERAL && n.Sym() != nil && isGoConst(n):
		base.Errorf("cannot assign to %v (declared const)", n)
	default:
		base.Errorf("cannot assign to %v", n)
	}
	n.SetType(nil)
}

func checkassignlist(stmt ir.Node, l ir.Nodes) {
	for _, n := range l.Slice() {
		checkassign(stmt, n)
	}
}

// samesafeexpr checks whether it is safe to reuse one of l and r
// instead of computing both. samesafeexpr assumes that l and r are
// used in the same statement or expression. In order for it to be
// safe to reuse l or r, they must:
// * be the same expression
// * not have side-effects (no function calls, no channel ops);
//   however, panics are ok
// * not cause inappropriate aliasing; e.g. two string to []byte
//   conversions, must result in two distinct slices
//
// The handling of OINDEXMAP is subtle. OINDEXMAP can occur both
// as an lvalue (map assignment) and an rvalue (map access). This is
// currently OK, since the only place samesafeexpr gets used on an
// lvalue expression is for OSLICE and OAPPEND optimizations, and it
// is correct in those settings.
func samesafeexpr(l ir.Node, r ir.Node) bool {
	if l.Op() != r.Op() || !types.Identical(l.Type(), r.Type()) {
		return false
	}

	switch l.Op() {
	case ir.ONAME, ir.OCLOSUREREAD:
		return l == r

	case ir.ODOT, ir.ODOTPTR:
		return l.Sym() != nil && r.Sym() != nil && l.Sym() == r.Sym() && samesafeexpr(l.Left(), r.Left())

	case ir.ODEREF, ir.OCONVNOP,
		ir.ONOT, ir.OBITNOT, ir.OPLUS, ir.ONEG:
		return samesafeexpr(l.Left(), r.Left())

	case ir.OCONV:
		// Some conversions can't be reused, such as []byte(str).
		// Allow only numeric-ish types. This is a bit conservative.
		return issimple[l.Type().Kind()] && samesafeexpr(l.Left(), r.Left())

	case ir.OINDEX, ir.OINDEXMAP,
		ir.OADD, ir.OSUB, ir.OOR, ir.OXOR, ir.OMUL, ir.OLSH, ir.ORSH, ir.OAND, ir.OANDNOT, ir.ODIV, ir.OMOD:
		return samesafeexpr(l.Left(), r.Left()) && samesafeexpr(l.Right(), r.Right())

	case ir.OLITERAL:
		return constant.Compare(l.Val(), token.EQL, r.Val())

	case ir.ONIL:
		return true
	}

	return false
}

// type check assignment.
// if this assignment is the definition of a var on the left side,
// fill in the var's type.
func typecheckas(n ir.Node) {
	if enableTrace && base.Flag.LowerT {
		defer tracePrint("typecheckas", n)(nil)
	}

	// delicate little dance.
	// the definition of n may refer to this assignment
	// as its definition, in which case it will call typecheckas.
	// in that case, do not call typecheck back, or it will cycle.
	// if the variable has a type (ntype) then typechecking
	// will not look at defn, so it is okay (and desirable,
	// so that the conversion below happens).
	n.SetLeft(resolve(n.Left()))

	if !ir.DeclaredBy(n.Left(), n) || n.Left().Name().Ntype != nil {
		n.SetLeft(typecheck(n.Left(), ctxExpr|ctxAssign))
	}

	// Use ctxMultiOK so we can emit an "N variables but M values" error
	// to be consistent with typecheckas2 (#26616).
	n.SetRight(typecheck(n.Right(), ctxExpr|ctxMultiOK))
	checkassign(n, n.Left())
	if n.Right() != nil && n.Right().Type() != nil {
		if n.Right().Type().IsFuncArgStruct() {
			base.Errorf("assignment mismatch: 1 variable but %v returns %d values", n.Right().Left(), n.Right().Type().NumFields())
			// Multi-value RHS isn't actually valid for OAS; nil out
			// to indicate failed typechecking.
			n.Right().SetType(nil)
		} else if n.Left().Type() != nil {
			n.SetRight(assignconv(n.Right(), n.Left().Type(), "assignment"))
		}
	}

	if ir.DeclaredBy(n.Left(), n) && n.Left().Name().Ntype == nil {
		n.SetRight(defaultlit(n.Right(), nil))
		n.Left().SetType(n.Right().Type())
	}

	// second half of dance.
	// now that right is done, typecheck the left
	// just to get it over with.  see dance above.
	n.SetTypecheck(1)

	if n.Left().Typecheck() == 0 {
		n.SetLeft(typecheck(n.Left(), ctxExpr|ctxAssign))
	}
	if !ir.IsBlank(n.Left()) {
		checkwidth(n.Left().Type()) // ensure width is calculated for backend
	}
}

func checkassignto(src *types.Type, dst ir.Node) {
	if op, why := assignop(src, dst.Type()); op == ir.OXXX {
		base.Errorf("cannot assign %v to %L in multiple assignment%s", src, dst, why)
		return
	}
}

func typecheckas2(n ir.Node) {
	if enableTrace && base.Flag.LowerT {
		defer tracePrint("typecheckas2", n)(nil)
	}

	ls := n.List().Slice()
	for i1, n1 := range ls {
		// delicate little dance.
		n1 = resolve(n1)
		ls[i1] = n1

		if !ir.DeclaredBy(n1, n) || n1.Name().Ntype != nil {
			ls[i1] = typecheck(ls[i1], ctxExpr|ctxAssign)
		}
	}

	cl := n.List().Len()
	cr := n.Rlist().Len()
	if cl > 1 && cr == 1 {
		n.Rlist().SetFirst(typecheck(n.Rlist().First(), ctxExpr|ctxMultiOK))
	} else {
		typecheckslice(n.Rlist().Slice(), ctxExpr)
	}
	checkassignlist(n, n.List())

	var l ir.Node
	var r ir.Node
	if cl == cr {
		// easy
		ls := n.List().Slice()
		rs := n.Rlist().Slice()
		for il, nl := range ls {
			nr := rs[il]
			if nl.Type() != nil && nr.Type() != nil {
				rs[il] = assignconv(nr, nl.Type(), "assignment")
			}
			if ir.DeclaredBy(nl, n) && nl.Name().Ntype == nil {
				rs[il] = defaultlit(rs[il], nil)
				nl.SetType(rs[il].Type())
			}
		}

		goto out
	}

	l = n.List().First()
	r = n.Rlist().First()

	// x,y,z = f()
	if cr == 1 {
		if r.Type() == nil {
			goto out
		}
		switch r.Op() {
		case ir.OCALLMETH, ir.OCALLINTER, ir.OCALLFUNC:
			if !r.Type().IsFuncArgStruct() {
				break
			}
			cr = r.Type().NumFields()
			if cr != cl {
				goto mismatch
			}
			r.(*ir.CallExpr).Use = ir.CallUseList
			n.SetOp(ir.OAS2FUNC)
			for i, l := range n.List().Slice() {
				f := r.Type().Field(i)
				if f.Type != nil && l.Type() != nil {
					checkassignto(f.Type, l)
				}
				if ir.DeclaredBy(l, n) && l.Name().Ntype == nil {
					l.SetType(f.Type)
				}
			}
			goto out
		}
	}

	// x, ok = y
	if cl == 2 && cr == 1 {
		if r.Type() == nil {
			goto out
		}
		switch r.Op() {
		case ir.OINDEXMAP, ir.ORECV, ir.ODOTTYPE:
			switch r.Op() {
			case ir.OINDEXMAP:
				n.SetOp(ir.OAS2MAPR)
			case ir.ORECV:
				n.SetOp(ir.OAS2RECV)
			case ir.ODOTTYPE:
				n.SetOp(ir.OAS2DOTTYPE)
				r.SetOp(ir.ODOTTYPE2)
			}
			if l.Type() != nil {
				checkassignto(r.Type(), l)
			}
			if ir.DeclaredBy(l, n) {
				l.SetType(r.Type())
			}
			l := n.List().Second()
			if l.Type() != nil && !l.Type().IsBoolean() {
				checkassignto(types.Types[types.TBOOL], l)
			}
			if ir.DeclaredBy(l, n) && l.Name().Ntype == nil {
				l.SetType(types.Types[types.TBOOL])
			}
			goto out
		}
	}

mismatch:
	switch r.Op() {
	default:
		base.Errorf("assignment mismatch: %d variables but %d values", cl, cr)
	case ir.OCALLFUNC, ir.OCALLMETH, ir.OCALLINTER:
		base.Errorf("assignment mismatch: %d variables but %v returns %d values", cl, r.Left(), cr)
	}

	// second half of dance
out:
	n.SetTypecheck(1)
	ls = n.List().Slice()
	for i1, n1 := range ls {
		if n1.Typecheck() == 0 {
			ls[i1] = typecheck(ls[i1], ctxExpr|ctxAssign)
		}
	}
}

// type check function definition
// To be called by typecheck, not directly.
// (Call typecheckFunc instead.)
func typecheckfunc(n *ir.Func) {
	if enableTrace && base.Flag.LowerT {
		defer tracePrint("typecheckfunc", n)(nil)
	}

	for _, ln := range n.Dcl {
		if ln.Op() == ir.ONAME && (ln.Class() == ir.PPARAM || ln.Class() == ir.PPARAMOUT) {
			ln.Decldepth = 1
		}
	}

	n.Nname = typecheck(n.Nname, ctxExpr|ctxAssign).(*ir.Name)
	t := n.Nname.Type()
	if t == nil {
		return
	}
	n.SetType(t)
	rcvr := t.Recv()
	if rcvr != nil && n.Shortname != nil {
		m := addmethod(n, n.Shortname, t, true, n.Pragma&ir.Nointerface != 0)
		if m == nil {
			return
		}

		n.Nname.SetSym(methodSym(rcvr.Type, n.Shortname))
		declare(n.Nname, ir.PFUNC)
	}

	if base.Ctxt.Flag_dynlink && !inimport && n.Nname != nil {
		makefuncsym(n.Sym())
	}
}

// The result of stringtoruneslit MUST be assigned back to n, e.g.
// 	n.Left = stringtoruneslit(n.Left)
func stringtoruneslit(n ir.Node) ir.Node {
	if n.Left().Op() != ir.OLITERAL || n.Left().Val().Kind() != constant.String {
		base.Fatalf("stringtoarraylit %v", n)
	}

	var l []ir.Node
	i := 0
	for _, r := range ir.StringVal(n.Left()) {
		l = append(l, ir.Nod(ir.OKEY, nodintconst(int64(i)), nodintconst(int64(r))))
		i++
	}

	nn := ir.Nod(ir.OCOMPLIT, nil, ir.TypeNode(n.Type()))
	nn.PtrList().Set(l)
	nn = typecheck(nn, ctxExpr)
	return nn
}

var mapqueue []*ir.MapType

func checkMapKeys() {
	for _, n := range mapqueue {
		k := n.Type().MapType().Key
		if !k.Broke() && !IsComparable(k) {
			base.ErrorfAt(n.Pos(), "invalid map key type %v", k)
		}
	}
	mapqueue = nil
}

func typecheckdeftype(n *ir.Name) {
	if enableTrace && base.Flag.LowerT {
		defer tracePrint("typecheckdeftype", n)(nil)
	}

	t := types.NewNamed(n)
	t.Vargen = n.Vargen
	if n.Pragma()&ir.NotInHeap != 0 {
		t.SetNotInHeap(true)
	}

	n.SetType(t)
	n.SetTypecheck(1)
	n.SetWalkdef(1)

	defercheckwidth()
	errorsBefore := base.Errors()
	n.Ntype = typecheckNtype(n.Ntype)
	if underlying := n.Ntype.Type(); underlying != nil {
		t.SetUnderlying(underlying)
	} else {
		n.SetDiag(true)
		n.SetType(nil)
	}
	if t.Kind() == types.TFORW && base.Errors() > errorsBefore {
		// Something went wrong during type-checking,
		// but it was reported. Silence future errors.
		t.SetBroke(true)
	}
	resumecheckwidth()
}

func typecheckdef(n ir.Node) {
	if enableTrace && base.Flag.LowerT {
		defer tracePrint("typecheckdef", n)(nil)
	}

	lno := setlineno(n)

	if n.Op() == ir.ONONAME {
		if !n.Diag() {
			n.SetDiag(true)

			// Note: adderrorname looks for this string and
			// adds context about the outer expression
			base.ErrorfAt(base.Pos, "undefined: %v", n.Sym())
		}
		base.Pos = lno
		return
	}

	if n.Walkdef() == 1 {
		base.Pos = lno
		return
	}

	typecheckdefstack = append(typecheckdefstack, n)
	if n.Walkdef() == 2 {
		base.FlushErrors()
		fmt.Printf("typecheckdef loop:")
		for i := len(typecheckdefstack) - 1; i >= 0; i-- {
			n := typecheckdefstack[i]
			fmt.Printf(" %v", n.Sym())
		}
		fmt.Printf("\n")
		base.Fatalf("typecheckdef loop")
	}

	n.SetWalkdef(2)

	if n.Type() != nil || n.Sym() == nil { // builtin or no name
		goto ret
	}

	switch n.Op() {
	default:
		base.Fatalf("typecheckdef %v", n.Op())

	case ir.OLITERAL:
		if n.Name().Ntype != nil {
			n.Name().Ntype = typecheckNtype(n.Name().Ntype)
			n.SetType(n.Name().Ntype.Type())
			n.Name().Ntype = nil
			if n.Type() == nil {
				n.SetDiag(true)
				goto ret
			}
		}

		e := n.Name().Defn
		n.Name().Defn = nil
		if e == nil {
			ir.Dump("typecheckdef nil defn", n)
			base.ErrorfAt(n.Pos(), "xxx")
		}

		e = typecheck(e, ctxExpr)
		if e.Type() == nil {
			goto ret
		}
		if !isGoConst(e) {
			if !e.Diag() {
				if e.Op() == ir.ONIL {
					base.ErrorfAt(n.Pos(), "const initializer cannot be nil")
				} else {
					base.ErrorfAt(n.Pos(), "const initializer %v is not a constant", e)
				}
				e.SetDiag(true)
			}
			goto ret
		}

		t := n.Type()
		if t != nil {
			if !ir.OKForConst[t.Kind()] {
				base.ErrorfAt(n.Pos(), "invalid constant type %v", t)
				goto ret
			}

			if !e.Type().IsUntyped() && !types.Identical(t, e.Type()) {
				base.ErrorfAt(n.Pos(), "cannot use %L as type %v in const initializer", e, t)
				goto ret
			}

			e = convlit(e, t)
		}

		n.SetType(e.Type())
		if n.Type() != nil {
			n.SetVal(e.Val())
		}

	case ir.ONAME:
		if n.Name().Ntype != nil {
			n.Name().Ntype = typecheckNtype(n.Name().Ntype)
			n.SetType(n.Name().Ntype.Type())
			if n.Type() == nil {
				n.SetDiag(true)
				goto ret
			}
		}

		if n.Type() != nil {
			break
		}
		if n.Name().Defn == nil {
			if n.SubOp() != 0 { // like OPRINTN
				break
			}
			if base.Errors() > 0 {
				// Can have undefined variables in x := foo
				// that make x have an n.name.Defn == nil.
				// If there are other errors anyway, don't
				// bother adding to the noise.
				break
			}

			base.Fatalf("var without type, init: %v", n.Sym())
		}

		if n.Name().Defn.Op() == ir.ONAME {
			n.Name().Defn = typecheck(n.Name().Defn, ctxExpr)
			n.SetType(n.Name().Defn.Type())
			break
		}

		n.Name().Defn = typecheck(n.Name().Defn, ctxStmt) // fills in n.Type

	case ir.OTYPE:
		n := n.(*ir.Name)
		if n.Alias() {
			// Type alias declaration: Simply use the rhs type - no need
			// to create a new type.
			// If we have a syntax error, name.Ntype may be nil.
			if n.Ntype != nil {
				n.Ntype = typecheckNtype(n.Ntype)
				n.SetType(n.Ntype.Type())
				if n.Type() == nil {
					n.SetDiag(true)
					goto ret
				}
				// For package-level type aliases, set n.Sym.Def so we can identify
				// it as a type alias during export. See also #31959.
				if n.Curfn == nil {
					n.Sym().Def = n.Ntype
				}
			}
			break
		}

		// regular type declaration
		typecheckdeftype(n)
	}

ret:
	if n.Op() != ir.OLITERAL && n.Type() != nil && n.Type().IsUntyped() {
		base.Fatalf("got %v for %v", n.Type(), n)
	}
	last := len(typecheckdefstack) - 1
	if typecheckdefstack[last] != n {
		base.Fatalf("typecheckdefstack mismatch")
	}
	typecheckdefstack[last] = nil
	typecheckdefstack = typecheckdefstack[:last]

	base.Pos = lno
	n.SetWalkdef(1)
}

func checkmake(t *types.Type, arg string, np *ir.Node) bool {
	n := *np
	if !n.Type().IsInteger() && n.Type().Kind() != types.TIDEAL {
		base.Errorf("non-integer %s argument in make(%v) - %v", arg, t, n.Type())
		return false
	}

	// Do range checks for constants before defaultlit
	// to avoid redundant "constant NNN overflows int" errors.
	if n.Op() == ir.OLITERAL {
		v := toint(n.Val())
		if constant.Sign(v) < 0 {
			base.Errorf("negative %s argument in make(%v)", arg, t)
			return false
		}
		if doesoverflow(v, types.Types[types.TINT]) {
			base.Errorf("%s argument too large in make(%v)", arg, t)
			return false
		}
	}

	// defaultlit is necessary for non-constants too: n might be 1.1<<k.
	// TODO(gri) The length argument requirements for (array/slice) make
	// are the same as for index expressions. Factor the code better;
	// for instance, indexlit might be called here and incorporate some
	// of the bounds checks done for make.
	n = defaultlit(n, types.Types[types.TINT])
	*np = n

	return true
}

// markBreak marks control statements containing break statements with SetHasBreak(true).
func markBreak(fn *ir.Func) {
	var labels map[*types.Sym]ir.Node
	var implicit ir.Node

	var mark func(ir.Node) error
	mark = func(n ir.Node) error {
		switch n.Op() {
		default:
			ir.DoChildren(n, mark)

		case ir.OBREAK:
			if n.Sym() == nil {
				if implicit != nil {
					implicit.SetHasBreak(true)
				}
			} else {
				if lab := labels[n.Sym()]; lab != nil {
					lab.SetHasBreak(true)
				}
			}

		case ir.OFOR, ir.OFORUNTIL, ir.OSWITCH, ir.OTYPESW, ir.OSELECT, ir.ORANGE:
			old := implicit
			implicit = n
			sym := n.Sym()
			if sym != nil {
				if labels == nil {
					// Map creation delayed until we need it - most functions don't.
					labels = make(map[*types.Sym]ir.Node)
				}
				labels[sym] = n
			}
			ir.DoChildren(n, mark)
			if sym != nil {
				delete(labels, sym)
			}
			implicit = old
		}
		return nil
	}

	mark(fn)
}

// isTermNodes reports whether the Nodes list ends with a terminating statement.
func isTermNodes(l ir.Nodes) bool {
	s := l.Slice()
	c := len(s)
	if c == 0 {
		return false
	}
	return isTermNode(s[c-1])
}

// isTermNode reports whether the node n, the last one in a
// statement list, is a terminating statement.
func isTermNode(n ir.Node) bool {
	switch n.Op() {
	// NOTE: OLABEL is treated as a separate statement,
	// not a separate prefix, so skipping to the last statement
	// in the block handles the labeled statement case by
	// skipping over the label. No case OLABEL here.

	case ir.OBLOCK:
		return isTermNodes(n.List())

	case ir.OGOTO, ir.ORETURN, ir.ORETJMP, ir.OPANIC, ir.OFALL:
		return true

	case ir.OFOR, ir.OFORUNTIL:
		if n.Left() != nil {
			return false
		}
		if n.HasBreak() {
			return false
		}
		return true

	case ir.OIF:
		return isTermNodes(n.Body()) && isTermNodes(n.Rlist())

	case ir.OSWITCH, ir.OTYPESW, ir.OSELECT:
		if n.HasBreak() {
			return false
		}
		def := false
		for _, n1 := range n.List().Slice() {
			if !isTermNodes(n1.Body()) {
				return false
			}
			if n1.List().Len() == 0 { // default
				def = true
			}
		}

		if n.Op() != ir.OSELECT && !def {
			return false
		}
		return true
	}

	return false
}

// checkreturn makes sure that fn terminates appropriately.
func checkreturn(fn *ir.Func) {
	if fn.Type().NumResults() != 0 && fn.Body().Len() != 0 {
		markBreak(fn)
		if !isTermNodes(fn.Body()) {
			base.ErrorfAt(fn.Endlineno, "missing return at end of function")
		}
	}
}

func deadcode(fn *ir.Func) {
	deadcodeslice(fn.PtrBody())

	if fn.Body().Len() == 0 {
		return
	}

	for _, n := range fn.Body().Slice() {
		if n.Init().Len() > 0 {
			return
		}
		switch n.Op() {
		case ir.OIF:
			if !ir.IsConst(n.Left(), constant.Bool) || n.Body().Len() > 0 || n.Rlist().Len() > 0 {
				return
			}
		case ir.OFOR:
			if !ir.IsConst(n.Left(), constant.Bool) || ir.BoolVal(n.Left()) {
				return
			}
		default:
			return
		}
	}

	fn.PtrBody().Set([]ir.Node{ir.Nod(ir.OBLOCK, nil, nil)})
}

func deadcodeslice(nn *ir.Nodes) {
	var lastLabel = -1
	for i, n := range nn.Slice() {
		if n != nil && n.Op() == ir.OLABEL {
			lastLabel = i
		}
	}
	for i, n := range nn.Slice() {
		// Cut is set to true when all nodes after i'th position
		// should be removed.
		// In other words, it marks whole slice "tail" as dead.
		cut := false
		if n == nil {
			continue
		}
		if n.Op() == ir.OIF {
			n.SetLeft(deadcodeexpr(n.Left()))
			if ir.IsConst(n.Left(), constant.Bool) {
				var body ir.Nodes
				if ir.BoolVal(n.Left()) {
					n.SetRlist(ir.Nodes{})
					body = n.Body()
				} else {
					n.SetBody(ir.Nodes{})
					body = n.Rlist()
				}
				// If "then" or "else" branch ends with panic or return statement,
				// it is safe to remove all statements after this node.
				// isterminating is not used to avoid goto-related complications.
				// We must be careful not to deadcode-remove labels, as they
				// might be the target of a goto. See issue 28616.
				if body := body.Slice(); len(body) != 0 {
					switch body[(len(body) - 1)].Op() {
					case ir.ORETURN, ir.ORETJMP, ir.OPANIC:
						if i > lastLabel {
							cut = true
						}
					}
				}
			}
		}

		deadcodeslice(n.PtrInit())
		switch n.Op() {
		case ir.OBLOCK:
			deadcodeslice(n.PtrList())
		case ir.OCASE:
			deadcodeslice(n.PtrBody())
		case ir.OFOR:
			deadcodeslice(n.PtrBody())
		case ir.OIF:
			deadcodeslice(n.PtrBody())
			deadcodeslice(n.PtrRlist())
		case ir.ORANGE:
			deadcodeslice(n.PtrBody())
		case ir.OSELECT:
			deadcodeslice(n.PtrList())
		case ir.OSWITCH:
			deadcodeslice(n.PtrList())
		}

		if cut {
			nn.Set(nn.Slice()[:i+1])
			break
		}
	}
}

func deadcodeexpr(n ir.Node) ir.Node {
	// Perform dead-code elimination on short-circuited boolean
	// expressions involving constants with the intent of
	// producing a constant 'if' condition.
	switch n.Op() {
	case ir.OANDAND:
		n.SetLeft(deadcodeexpr(n.Left()))
		n.SetRight(deadcodeexpr(n.Right()))
		if ir.IsConst(n.Left(), constant.Bool) {
			if ir.BoolVal(n.Left()) {
				return n.Right() // true && x => x
			} else {
				return n.Left() // false && x => false
			}
		}
	case ir.OOROR:
		n.SetLeft(deadcodeexpr(n.Left()))
		n.SetRight(deadcodeexpr(n.Right()))
		if ir.IsConst(n.Left(), constant.Bool) {
			if ir.BoolVal(n.Left()) {
				return n.Left() // true || x => true
			} else {
				return n.Right() // false || x => x
			}
		}
	}
	return n
}

// getIotaValue returns the current value for "iota",
// or -1 if not within a ConstSpec.
func getIotaValue() int64 {
	if i := len(typecheckdefstack); i > 0 {
		if x := typecheckdefstack[i-1]; x.Op() == ir.OLITERAL {
			return x.Iota()
		}
	}

	if Curfn != nil && Curfn.Iota() >= 0 {
		return Curfn.Iota()
	}

	return -1
}

// curpkg returns the current package, based on Curfn.
func curpkg() *types.Pkg {
	fn := Curfn
	if fn == nil {
		// Initialization expressions for package-scope variables.
		return types.LocalPkg
	}
	return fnpkg(fn.Nname)
}

// MethodName returns the ONAME representing the method
// referenced by expression n, which must be a method selector,
// method expression, or method value.
func methodExprName(n ir.Node) *ir.Name {
	name, _ := methodExprFunc(n).Nname.(*ir.Name)
	return name
}

// MethodFunc is like MethodName, but returns the types.Field instead.
func methodExprFunc(n ir.Node) *types.Field {
	switch n.Op() {
	case ir.ODOTMETH:
		return n.(*ir.SelectorExpr).Selection
	case ir.OMETHEXPR:
		return n.(*ir.MethodExpr).Method
	case ir.OCALLPART:
		return callpartMethod(n)
	}
	base.Fatalf("unexpected node: %v (%v)", n, n.Op())
	panic("unreachable")
}
