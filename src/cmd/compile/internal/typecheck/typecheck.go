// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typecheck

import (
	"fmt"
	"go/constant"
	"go/token"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

func AssignExpr(n ir.Node) ir.Node { return typecheck(n, ctxExpr|ctxAssign) }
func Expr(n ir.Node) ir.Node       { return typecheck(n, ctxExpr) }
func Stmt(n ir.Node) ir.Node       { return typecheck(n, ctxStmt) }

func Exprs(exprs []ir.Node) { typecheckslice(exprs, ctxExpr) }
func Stmts(stmts []ir.Node) { typecheckslice(stmts, ctxStmt) }

func Call(pos src.XPos, callee ir.Node, args []ir.Node, dots bool) ir.Node {
	call := ir.NewCallExpr(pos, ir.OCALL, callee, args)
	call.IsDDD = dots
	return typecheck(call, ctxStmt|ctxExpr)
}

func Callee(n ir.Node) ir.Node {
	return typecheck(n, ctxExpr|ctxCallee)
}

var traceIndent []byte

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

	types.SkipSizeForTracing = true
	defer func() { types.SkipSizeForTracing = false }()
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

		types.SkipSizeForTracing = true
		defer func() { types.SkipSizeForTracing = false }()
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

// typecheck type checks node n.
// The result of typecheck MUST be assigned back to n, e.g.
//
//	n.Left = typecheck(n.Left, top)
func typecheck(n ir.Node, top int) (res ir.Node) {
	if n == nil {
		return nil
	}

	// only trace if there's work to do
	if base.EnableTrace && base.Flag.LowerT {
		defer tracePrint("typecheck", n)(&res)
	}

	lno := ir.SetPos(n)
	defer func() { base.Pos = lno }()

	// Skip typecheck if already done.
	// But re-typecheck ONAME/OTYPE/OLITERAL/OPACK node in case context has changed.
	if n.Typecheck() == 1 || n.Typecheck() == 3 {
		switch n.Op() {
		case ir.ONAME:
			break

		default:
			return n
		}
	}

	if n.Typecheck() == 2 {
		base.FatalfAt(n.Pos(), "typechecking loop")
	}

	n.SetTypecheck(2)
	n = typecheck1(n, top)
	n.SetTypecheck(1)

	t := n.Type()
	if t != nil && !t.IsFuncArgStruct() && n.Op() != ir.OTYPE {
		switch t.Kind() {
		case types.TFUNC, // might have TANY; wait until it's called
			types.TANY, types.TFORW, types.TIDEAL, types.TNIL, types.TBLANK:
			break

		default:
			types.CheckSize(t)
		}
	}

	return n
}

// indexlit implements typechecking of untyped values as
// array/slice indexes. It is almost equivalent to DefaultLit
// but also accepts untyped numeric values representable as
// value of type int (see also checkmake for comparison).
// The result of indexlit MUST be assigned back to n, e.g.
//
//	n.Left = indexlit(n.Left)
func indexlit(n ir.Node) ir.Node {
	if n != nil && n.Type() != nil && n.Type().Kind() == types.TIDEAL {
		return DefaultLit(n, types.Types[types.TINT])
	}
	return n
}

// typecheck1 should ONLY be called from typecheck.
func typecheck1(n ir.Node, top int) ir.Node {
	// Skip over parens.
	for n.Op() == ir.OPAREN {
		n = n.(*ir.ParenExpr).X
	}

	switch n.Op() {
	default:
		ir.Dump("typecheck", n)
		base.Fatalf("typecheck %v", n.Op())
		panic("unreachable")

	case ir.ONAME:
		n := n.(*ir.Name)
		if n.BuiltinOp != 0 {
			if top&ctxCallee == 0 {
				base.Errorf("use of builtin %v not in function call", n.Sym())
				n.SetType(nil)
				return n
			}
			return n
		}
		if top&ctxAssign == 0 {
			// not a write to the variable
			if ir.IsBlank(n) {
				base.Errorf("cannot use _ as value")
				n.SetType(nil)
				return n
			}
			n.SetUsed(true)
		}
		return n

	// type or expr
	case ir.ODEREF:
		n := n.(*ir.StarExpr)
		return tcStar(n, top)

	// x op= y
	case ir.OASOP:
		n := n.(*ir.AssignOpStmt)
		n.X, n.Y = Expr(n.X), Expr(n.Y)
		checkassign(n.X)
		if n.IncDec && !okforarith[n.X.Type().Kind()] {
			base.Errorf("invalid operation: %v (non-numeric type %v)", n, n.X.Type())
			return n
		}
		switch n.AsOp {
		case ir.OLSH, ir.ORSH:
			n.X, n.Y, _ = tcShift(n, n.X, n.Y)
		case ir.OADD, ir.OAND, ir.OANDNOT, ir.ODIV, ir.OMOD, ir.OMUL, ir.OOR, ir.OSUB, ir.OXOR:
			n.X, n.Y, _ = tcArith(n, n.AsOp, n.X, n.Y)
		default:
			base.Fatalf("invalid assign op: %v", n.AsOp)
		}
		return n

	// logical operators
	case ir.OANDAND, ir.OOROR:
		n := n.(*ir.LogicalExpr)
		n.X, n.Y = Expr(n.X), Expr(n.Y)
		if n.X.Type() == nil || n.Y.Type() == nil {
			n.SetType(nil)
			return n
		}
		// For "x == x && len(s)", it's better to report that "len(s)" (type int)
		// can't be used with "&&" than to report that "x == x" (type untyped bool)
		// can't be converted to int (see issue #41500).
		if !n.X.Type().IsBoolean() {
			base.Errorf("invalid operation: %v (operator %v not defined on %s)", n, n.Op(), typekind(n.X.Type()))
			n.SetType(nil)
			return n
		}
		if !n.Y.Type().IsBoolean() {
			base.Errorf("invalid operation: %v (operator %v not defined on %s)", n, n.Op(), typekind(n.Y.Type()))
			n.SetType(nil)
			return n
		}
		l, r, t := tcArith(n, n.Op(), n.X, n.Y)
		n.X, n.Y = l, r
		n.SetType(t)
		return n

	// shift operators
	case ir.OLSH, ir.ORSH:
		n := n.(*ir.BinaryExpr)
		n.X, n.Y = Expr(n.X), Expr(n.Y)
		l, r, t := tcShift(n, n.X, n.Y)
		n.X, n.Y = l, r
		n.SetType(t)
		return n

	// comparison operators
	case ir.OEQ, ir.OGE, ir.OGT, ir.OLE, ir.OLT, ir.ONE:
		n := n.(*ir.BinaryExpr)
		n.X, n.Y = Expr(n.X), Expr(n.Y)
		l, r, t := tcArith(n, n.Op(), n.X, n.Y)
		if t != nil {
			n.X, n.Y = l, r
			n.SetType(types.UntypedBool)
			n.X, n.Y = defaultlit2(l, r, true)
		}
		return n

	// binary operators
	case ir.OADD, ir.OAND, ir.OANDNOT, ir.ODIV, ir.OMOD, ir.OMUL, ir.OOR, ir.OSUB, ir.OXOR:
		n := n.(*ir.BinaryExpr)
		n.X, n.Y = Expr(n.X), Expr(n.Y)
		l, r, t := tcArith(n, n.Op(), n.X, n.Y)
		if t != nil && t.Kind() == types.TSTRING && n.Op() == ir.OADD {
			// create or update OADDSTR node with list of strings in x + y + z + (w + v) + ...
			var add *ir.AddStringExpr
			if l.Op() == ir.OADDSTR {
				add = l.(*ir.AddStringExpr)
				add.SetPos(n.Pos())
			} else {
				add = ir.NewAddStringExpr(n.Pos(), []ir.Node{l})
			}
			if r.Op() == ir.OADDSTR {
				r := r.(*ir.AddStringExpr)
				add.List.Append(r.List.Take()...)
			} else {
				add.List.Append(r)
			}
			add.SetType(t)
			return add
		}
		n.X, n.Y = l, r
		n.SetType(t)
		return n

	case ir.OBITNOT, ir.ONEG, ir.ONOT, ir.OPLUS:
		n := n.(*ir.UnaryExpr)
		return tcUnaryArith(n)

	// exprs
	case ir.OCOMPLIT:
		return tcCompLit(n.(*ir.CompLitExpr))

	case ir.OXDOT, ir.ODOT:
		n := n.(*ir.SelectorExpr)
		return tcDot(n, top)

	case ir.ODOTTYPE:
		n := n.(*ir.TypeAssertExpr)
		return tcDotType(n)

	case ir.OINDEX:
		n := n.(*ir.IndexExpr)
		return tcIndex(n)

	case ir.ORECV:
		n := n.(*ir.UnaryExpr)
		return tcRecv(n)

	case ir.OSEND:
		n := n.(*ir.SendStmt)
		return tcSend(n)

	case ir.OSLICEHEADER:
		n := n.(*ir.SliceHeaderExpr)
		return tcSliceHeader(n)

	case ir.OSTRINGHEADER:
		n := n.(*ir.StringHeaderExpr)
		return tcStringHeader(n)

	case ir.OMAKESLICECOPY:
		n := n.(*ir.MakeExpr)
		return tcMakeSliceCopy(n)

	case ir.OSLICE, ir.OSLICE3:
		n := n.(*ir.SliceExpr)
		return tcSlice(n)

	// call and call like
	case ir.OCALL:
		n := n.(*ir.CallExpr)
		return tcCall(n, top)

	case ir.OCAP, ir.OLEN:
		n := n.(*ir.UnaryExpr)
		return tcLenCap(n)

	case ir.OMIN, ir.OMAX:
		n := n.(*ir.CallExpr)
		return tcMinMax(n)

	case ir.OREAL, ir.OIMAG:
		n := n.(*ir.UnaryExpr)
		return tcRealImag(n)

	case ir.OCOMPLEX:
		n := n.(*ir.BinaryExpr)
		return tcComplex(n)

	case ir.OCLEAR:
		n := n.(*ir.UnaryExpr)
		return tcClear(n)

	case ir.OCLOSE:
		n := n.(*ir.UnaryExpr)
		return tcClose(n)

	case ir.ODELETE:
		n := n.(*ir.CallExpr)
		return tcDelete(n)

	case ir.OAPPEND:
		n := n.(*ir.CallExpr)
		return tcAppend(n)

	case ir.OCOPY:
		n := n.(*ir.BinaryExpr)
		return tcCopy(n)

	case ir.OCONV:
		n := n.(*ir.ConvExpr)
		return tcConv(n)

	case ir.OMAKE:
		n := n.(*ir.CallExpr)
		return tcMake(n)

	case ir.ONEW:
		n := n.(*ir.UnaryExpr)
		return tcNew(n)

	case ir.OPRINT, ir.OPRINTLN:
		n := n.(*ir.CallExpr)
		return tcPrint(n)

	case ir.OPANIC:
		n := n.(*ir.UnaryExpr)
		return tcPanic(n)

	case ir.ORECOVER:
		n := n.(*ir.CallExpr)
		return tcRecover(n)

	case ir.OUNSAFEADD:
		n := n.(*ir.BinaryExpr)
		return tcUnsafeAdd(n)

	case ir.OUNSAFESLICE:
		n := n.(*ir.BinaryExpr)
		return tcUnsafeSlice(n)

	case ir.OUNSAFESLICEDATA:
		n := n.(*ir.UnaryExpr)
		return tcUnsafeData(n)

	case ir.OUNSAFESTRING:
		n := n.(*ir.BinaryExpr)
		return tcUnsafeString(n)

	case ir.OUNSAFESTRINGDATA:
		n := n.(*ir.UnaryExpr)
		return tcUnsafeData(n)

	case ir.OITAB:
		n := n.(*ir.UnaryExpr)
		return tcITab(n)

	case ir.OIDATA:
		// Whoever creates the OIDATA node must know a priori the concrete type at that moment,
		// usually by just having checked the OITAB.
		n := n.(*ir.UnaryExpr)
		base.Fatalf("cannot typecheck interface data %v", n)
		panic("unreachable")

	case ir.OSPTR:
		n := n.(*ir.UnaryExpr)
		return tcSPtr(n)

	case ir.OCFUNC:
		n := n.(*ir.UnaryExpr)
		n.X = Expr(n.X)
		n.SetType(types.Types[types.TUINTPTR])
		return n

	case ir.OGETCALLERPC, ir.OGETCALLERSP:
		n := n.(*ir.CallExpr)
		if len(n.Args) != 0 {
			base.FatalfAt(n.Pos(), "unexpected arguments: %v", n)
		}
		n.SetType(types.Types[types.TUINTPTR])
		return n

	case ir.OCONVNOP:
		n := n.(*ir.ConvExpr)
		n.X = Expr(n.X)
		return n

	// statements
	case ir.OAS:
		n := n.(*ir.AssignStmt)
		tcAssign(n)

		// Code that creates temps does not bother to set defn, so do it here.
		if n.X.Op() == ir.ONAME && ir.IsAutoTmp(n.X) {
			n.X.Name().Defn = n
		}
		return n

	case ir.OAS2:
		tcAssignList(n.(*ir.AssignListStmt))
		return n

	case ir.OBREAK,
		ir.OCONTINUE,
		ir.ODCL,
		ir.OGOTO,
		ir.OFALL:
		return n

	case ir.OBLOCK:
		n := n.(*ir.BlockStmt)
		Stmts(n.List)
		return n

	case ir.OLABEL:
		if n.Sym().IsBlank() {
			// Empty identifier is valid but useless.
			// Eliminate now to simplify life later.
			// See issues 7538, 11589, 11593.
			n = ir.NewBlockStmt(n.Pos(), nil)
		}
		return n

	case ir.ODEFER, ir.OGO:
		n := n.(*ir.GoDeferStmt)
		n.Call = typecheck(n.Call, ctxStmt|ctxExpr)
		tcGoDefer(n)
		return n

	case ir.OFOR:
		n := n.(*ir.ForStmt)
		return tcFor(n)

	case ir.OIF:
		n := n.(*ir.IfStmt)
		return tcIf(n)

	case ir.ORETURN:
		n := n.(*ir.ReturnStmt)
		return tcReturn(n)

	case ir.OTAILCALL:
		n := n.(*ir.TailCallStmt)
		n.Call = typecheck(n.Call, ctxStmt|ctxExpr).(*ir.CallExpr)
		return n

	case ir.OCHECKNIL:
		n := n.(*ir.UnaryExpr)
		return tcCheckNil(n)

	case ir.OSELECT:
		tcSelect(n.(*ir.SelectStmt))
		return n

	case ir.OSWITCH:
		tcSwitch(n.(*ir.SwitchStmt))
		return n

	case ir.ORANGE:
		tcRange(n.(*ir.RangeStmt))
		return n

	case ir.OTYPESW:
		n := n.(*ir.TypeSwitchGuard)
		base.Fatalf("use of .(type) outside type switch")
		return n

	case ir.ODCLFUNC:
		tcFunc(n.(*ir.Func))
		return n
	}

	// No return n here!
	// Individual cases can type-assert n, introducing a new one.
	// Each must execute its own return n.
}

func typecheckargs(n ir.InitNode) {
	var list []ir.Node
	switch n := n.(type) {
	default:
		base.Fatalf("typecheckargs %+v", n.Op())
	case *ir.CallExpr:
		list = n.Args
		if n.IsDDD {
			Exprs(list)
			return
		}
	case *ir.ReturnStmt:
		list = n.Results
	}
	if len(list) != 1 {
		Exprs(list)
		return
	}

	typecheckslice(list, ctxExpr|ctxMultiOK)
	t := list[0].Type()
	if t == nil || !t.IsFuncArgStruct() {
		return
	}

	// Rewrite f(g()) into t1, t2, ... = g(); f(t1, t2, ...).
	RewriteMultiValueCall(n, list[0])
}

// RewriteNonNameCall replaces non-Name call expressions with temps,
// rewriting f()(...) to t0 := f(); t0(...).
func RewriteNonNameCall(n *ir.CallExpr) {
	np := &n.Fun
	if dot, ok := (*np).(*ir.SelectorExpr); ok && (dot.Op() == ir.ODOTMETH || dot.Op() == ir.ODOTINTER || dot.Op() == ir.OMETHVALUE) {
		np = &dot.X // peel away method selector
	}

	// Check for side effects in the callee expression.
	// We explicitly special case new(T) though, because it doesn't have
	// observable side effects, and keeping it in place allows better escape analysis.
	if !ir.Any(*np, func { n -> n.Op() != ir.ONEW && callOrChan(n) }) {
		return
	}

	tmp := TempAt(base.Pos, ir.CurFunc, (*np).Type())
	as := ir.NewAssignStmt(base.Pos, tmp, *np)
	as.PtrInit().Append(Stmt(ir.NewDecl(n.Pos(), ir.ODCL, tmp)))
	*np = tmp

	n.PtrInit().Append(Stmt(as))
}

// RewriteMultiValueCall rewrites multi-valued f() to use temporaries,
// so the backend wouldn't need to worry about tuple-valued expressions.
func RewriteMultiValueCall(n ir.InitNode, call ir.Node) {
	as := ir.NewAssignListStmt(base.Pos, ir.OAS2, nil, []ir.Node{call})
	results := call.Type().Fields()
	list := make([]ir.Node, len(results))
	for i, result := range results {
		tmp := TempAt(base.Pos, ir.CurFunc, result.Type)
		as.PtrInit().Append(ir.NewDecl(base.Pos, ir.ODCL, tmp))
		as.Lhs.Append(tmp)
		list[i] = tmp
	}

	n.PtrInit().Append(Stmt(as))

	switch n := n.(type) {
	default:
		base.Fatalf("RewriteMultiValueCall %+v", n.Op())
	case *ir.CallExpr:
		n.Args = list
	case *ir.ReturnStmt:
		n.Results = list
	case *ir.AssignListStmt:
		if n.Op() != ir.OAS2FUNC {
			base.Fatalf("RewriteMultiValueCall: invalid op %v", n.Op())
		}
		as.SetOp(ir.OAS2FUNC)
		n.SetOp(ir.OAS2)
		n.Rhs = make([]ir.Node, len(list))
		for i, tmp := range list {
			n.Rhs[i] = AssignConv(tmp, n.Lhs[i].Type(), "assignment")
		}
	}
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
		} else if ir.ConstOverflow(x, types.Types[types.TINT]) {
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

// The result of implicitstar MUST be assigned back to n, e.g.
//
//	n.Left = implicitstar(n.Left)
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
	star := ir.NewStarExpr(base.Pos, n)
	star.SetImplicit(true)
	return Expr(star)
}

func needOneArg(n *ir.CallExpr, f string, args ...interface{}) (ir.Node, bool) {
	if len(n.Args) == 0 {
		p := fmt.Sprintf(f, args...)
		base.Errorf("missing argument to %s: %v", p, n)
		return nil, false
	}

	if len(n.Args) > 1 {
		p := fmt.Sprintf(f, args...)
		base.Errorf("too many arguments to %s: %v", p, n)
		return n.Args[0], false
	}

	return n.Args[0], true
}

func needTwoArgs(n *ir.CallExpr) (ir.Node, ir.Node, bool) {
	if len(n.Args) != 2 {
		if len(n.Args) < 2 {
			base.Errorf("not enough arguments in call to %v", n)
		} else {
			base.Errorf("too many arguments in call to %v", n)
		}
		return nil, nil, false
	}
	return n.Args[0], n.Args[1], true
}

// Lookdot1 looks up the specified method s in the list fs of methods, returning
// the matching field or nil. If dostrcmp is 0, it matches the symbols. If
// dostrcmp is 1, it matches by name exactly. If dostrcmp is 2, it matches names
// with case folding.
func Lookdot1(errnode ir.Node, s *types.Sym, t *types.Type, fs []*types.Field, dostrcmp int) *types.Field {
	var r *types.Field
	for _, f := range fs {
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

// NewMethodExpr returns an OMETHEXPR node representing method
// expression "recv.sym".
func NewMethodExpr(pos src.XPos, recv *types.Type, sym *types.Sym) *ir.SelectorExpr {
	// Compute the method set for recv.
	var ms []*types.Field
	if recv.IsInterface() {
		ms = recv.AllMethods()
	} else {
		mt := types.ReceiverBaseType(recv)
		if mt == nil {
			base.FatalfAt(pos, "type %v has no receiver base type", recv)
		}
		CalcMethods(mt)
		ms = mt.AllMethods()
	}

	m := Lookdot1(nil, sym, recv, ms, 0)
	if m == nil {
		base.FatalfAt(pos, "type %v has no method %v", recv, sym)
	}

	if !types.IsMethodApplicable(recv, m) {
		base.FatalfAt(pos, "invalid method expression %v.%v (needs pointer receiver)", recv, sym)
	}

	n := ir.NewSelectorExpr(pos, ir.OMETHEXPR, ir.TypeNode(recv), sym)
	n.Selection = m
	n.SetType(NewMethodType(m.Type, recv))
	n.SetTypecheck(1)
	return n
}

func derefall(t *types.Type) *types.Type {
	for t != nil && t.IsPtr() {
		t = t.Elem()
	}
	return t
}

// Lookdot looks up field or method n.Sel in the type t and returns the matching
// field. It transforms the op of node n to ODOTINTER or ODOTMETH, if appropriate.
// It also may add a StarExpr node to n.X as needed for access to non-pointer
// methods. If dostrcmp is 0, it matches the field/method with the exact symbol
// as n.Sel (appropriate for exported fields). If dostrcmp is 1, it matches by name
// exactly. If dostrcmp is 2, it matches names with case folding.
func Lookdot(n *ir.SelectorExpr, t *types.Type, dostrcmp int) *types.Field {
	s := n.Sel

	types.CalcSize(t)
	var f1 *types.Field
	if t.IsStruct() {
		f1 = Lookdot1(n, s, t, t.Fields(), dostrcmp)
	} else if t.IsInterface() {
		f1 = Lookdot1(n, s, t, t.AllMethods(), dostrcmp)
	}

	var f2 *types.Field
	if n.X.Type() == t || n.X.Type().Sym() == nil {
		mt := types.ReceiverBaseType(t)
		if mt != nil {
			f2 = Lookdot1(n, s, mt, mt.Methods(), dostrcmp)
		}
	}

	if f1 != nil {
		if dostrcmp > 1 {
			// Already in the process of diagnosing an error.
			return f1
		}
		if f2 != nil {
			base.Errorf("%v is both field and method", n.Sel)
		}
		if f1.Offset == types.BADWIDTH {
			base.Fatalf("Lookdot badwidth t=%v, f1=%v@%p", t, f1, f1)
		}
		n.Selection = f1
		n.SetType(f1.Type)
		if t.IsInterface() {
			if n.X.Type().IsPtr() {
				star := ir.NewStarExpr(base.Pos, n.X)
				star.SetImplicit(true)
				n.X = Expr(star)
			}

			n.SetOp(ir.ODOTINTER)
		}
		return f1
	}

	if f2 != nil {
		if dostrcmp > 1 {
			// Already in the process of diagnosing an error.
			return f2
		}
		orig := n.X
		tt := n.X.Type()
		types.CalcSize(tt)
		rcvr := f2.Type.Recv().Type
		if !types.Identical(rcvr, tt) {
			if rcvr.IsPtr() && types.Identical(rcvr.Elem(), tt) {
				checklvalue(n.X, "call pointer method on")
				addr := NodAddr(n.X)
				addr.SetImplicit(true)
				n.X = typecheck(addr, ctxType|ctxExpr)
			} else if tt.IsPtr() && (!rcvr.IsPtr() || rcvr.IsPtr() && rcvr.Elem().NotInHeap()) && types.Identical(tt.Elem(), rcvr) {
				star := ir.NewStarExpr(base.Pos, n.X)
				star.SetImplicit(true)
				n.X = typecheck(star, ctxType|ctxExpr)
			} else if tt.IsPtr() && tt.Elem().IsPtr() && types.Identical(derefall(tt), derefall(rcvr)) {
				base.Errorf("calling method %v with receiver %L requires explicit dereference", n.Sel, n.X)
				for tt.IsPtr() {
					// Stop one level early for method with pointer receiver.
					if rcvr.IsPtr() && !tt.Elem().IsPtr() {
						break
					}
					star := ir.NewStarExpr(base.Pos, n.X)
					star.SetImplicit(true)
					n.X = typecheck(star, ctxType|ctxExpr)
					tt = tt.Elem()
				}
			} else {
				base.Fatalf("method mismatch: %v for %v", rcvr, tt)
			}
		}

		// Check that we haven't implicitly dereferenced any defined pointer types.
		for x := n.X; ; {
			var inner ir.Node
			implicit := false
			switch x := x.(type) {
			case *ir.AddrExpr:
				inner, implicit = x.X, x.Implicit()
			case *ir.SelectorExpr:
				inner, implicit = x.X, x.Implicit()
			case *ir.StarExpr:
				inner, implicit = x.X, x.Implicit()
			}
			if !implicit {
				break
			}
			if inner.Type().Sym() != nil && (x.Op() == ir.ODEREF || x.Op() == ir.ODOTPTR) {
				// Found an implicit dereference of a defined pointer type.
				// Restore n.X for better error message.
				n.X = orig
				return nil
			}
			x = inner
		}

		n.Selection = f2
		n.SetType(f2.Type)
		n.SetOp(ir.ODOTMETH)

		return f2
	}

	return nil
}

func nokeys(l ir.Nodes) bool {
	for _, n := range l {
		if n.Op() == ir.OKEY || n.Op() == ir.OSTRUCTKEY {
			return false
		}
	}
	return true
}

func hasddd(params []*types.Field) bool {
	// TODO(mdempsky): Simply check the last param.
	for _, tl := range params {
		if tl.IsDDD() {
			return true
		}
	}

	return false
}

// typecheck assignment: type list = expression list
func typecheckaste(op ir.Op, call ir.Node, isddd bool, params []*types.Field, nl ir.Nodes, desc func() string) {
	var t *types.Type
	var i int

	lno := base.Pos
	defer func() { base.Pos = lno }()

	var n ir.Node
	if len(nl) == 1 {
		n = nl[0]
	}

	n1 := len(params)
	n2 := len(nl)
	if !hasddd(params) {
		if isddd {
			goto invalidddd
		}
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
	for _, tl := range params {
		t = tl.Type
		if tl.IsDDD() {
			if isddd {
				if i >= len(nl) {
					goto notenough
				}
				if len(nl)-i > 1 {
					goto toomany
				}
				n = nl[i]
				ir.SetPos(n)
				if n.Type() != nil {
					nl[i] = assignconvfn(n, t, desc)
				}
				return
			}

			// TODO(mdempsky): Make into ... call with implicit slice.
			for ; i < len(nl); i++ {
				n = nl[i]
				ir.SetPos(n)
				if n.Type() != nil {
					nl[i] = assignconvfn(n, t.Elem(), desc)
				}
			}
			return
		}

		if i >= len(nl) {
			goto notenough
		}
		n = nl[i]
		ir.SetPos(n)
		if n.Type() != nil {
			nl[i] = assignconvfn(n, t, desc)
		}
		i++
	}

	if i < len(nl) {
		goto toomany
	}

invalidddd:
	if isddd {
		if call != nil {
			base.Errorf("invalid use of ... in call to %v", call)
		} else {
			base.Errorf("invalid use of ... in %v", op)
		}
	}
	return

notenough:
	if n == nil || n.Type() != nil {
		base.Fatalf("not enough arguments to %v", op)
	}
	return

toomany:
	base.Fatalf("too many arguments to %v", op)
}

// type check composite.
func fielddup(name string, hash map[string]bool) {
	if hash[name] {
		base.Errorf("duplicate field name in struct literal: %s", name)
		return
	}
	hash[name] = true
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
		ir.SetPos(elt)
		r := elts[i]
		var kv *ir.KeyExpr
		if elt.Op() == ir.OKEY {
			elt := elt.(*ir.KeyExpr)
			elt.Key = Expr(elt.Key)
			key = IndexConst(elt.Key)
			if key < 0 {
				base.Fatalf("invalid index: %v", elt.Key)
			}
			kv = elt
			r = elt.Value
		}

		r = Expr(r)
		r = AssignConv(r, elemType, ctx)
		if kv != nil {
			kv.Value = r
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

func checklvalue(n ir.Node, verb string) {
	if !ir.IsAddressable(n) {
		base.Errorf("cannot %s %v", verb, n)
	}
}

func checkassign(n ir.Node) {
	// have already complained about n being invalid
	if n.Type() == nil {
		if base.Errors() == 0 {
			base.Fatalf("expected an error about %v", n)
		}
		return
	}

	if ir.IsAddressable(n) {
		return
	}
	if n.Op() == ir.OINDEXMAP {
		n := n.(*ir.IndexExpr)
		n.Assigned = true
		return
	}

	defer n.SetType(nil)

	switch {
	case n.Op() == ir.ODOT && n.(*ir.SelectorExpr).X.Op() == ir.OINDEXMAP:
		base.Errorf("cannot assign to struct field %v in map", n)
	case (n.Op() == ir.OINDEX && n.(*ir.IndexExpr).X.Type().IsString()) || n.Op() == ir.OSLICESTR:
		base.Errorf("cannot assign to %v (strings are immutable)", n)
	case n.Op() == ir.OLITERAL && n.Sym() != nil && ir.IsConstNode(n):
		base.Errorf("cannot assign to %v (declared const)", n)
	default:
		base.Errorf("cannot assign to %v", n)
	}
}

func checkassignto(src *types.Type, dst ir.Node) {
	// TODO(mdempsky): Handle all untyped types correctly.
	if src == types.UntypedBool && dst.Type().IsBoolean() {
		return
	}

	if op, why := assignOp(src, dst.Type()); op == ir.OXXX {
		base.Errorf("cannot assign %v to %L in multiple assignment%s", src, dst, why)
		return
	}
}

// The result of stringtoruneslit MUST be assigned back to n, e.g.
//
//	n.Left = stringtoruneslit(n.Left)
func stringtoruneslit(n *ir.ConvExpr) ir.Node {
	if n.X.Op() != ir.OLITERAL || n.X.Val().Kind() != constant.String {
		base.Fatalf("stringtoarraylit %v", n)
	}

	var l []ir.Node
	i := 0
	for _, r := range ir.StringVal(n.X) {
		l = append(l, ir.NewKeyExpr(base.Pos, ir.NewInt(base.Pos, int64(i)), ir.NewInt(base.Pos, int64(r))))
		i++
	}

	return Expr(ir.NewCompLitExpr(base.Pos, ir.OCOMPLIT, n.Type(), l))
}

func checkmake(t *types.Type, arg string, np *ir.Node) bool {
	n := *np
	if !n.Type().IsInteger() && n.Type().Kind() != types.TIDEAL {
		base.Errorf("non-integer %s argument in make(%v) - %v", arg, t, n.Type())
		return false
	}

	// Do range checks for constants before DefaultLit
	// to avoid redundant "constant NNN overflows int" errors.
	if n.Op() == ir.OLITERAL {
		v := toint(n.Val())
		if constant.Sign(v) < 0 {
			base.Errorf("negative %s argument in make(%v)", arg, t)
			return false
		}
		if ir.ConstOverflow(v, types.Types[types.TINT]) {
			base.Errorf("%s argument too large in make(%v)", arg, t)
			return false
		}
	}

	// DefaultLit is necessary for non-constants too: n might be 1.1<<k.
	// TODO(gri) The length argument requirements for (array/slice) make
	// are the same as for index expressions. Factor the code better;
	// for instance, indexlit might be called here and incorporate some
	// of the bounds checks done for make.
	n = DefaultLit(n, types.Types[types.TINT])
	*np = n

	return true
}

// checkunsafesliceorstring is like checkmake but for unsafe.{Slice,String}.
func checkunsafesliceorstring(op ir.Op, np *ir.Node) bool {
	n := *np
	if !n.Type().IsInteger() && n.Type().Kind() != types.TIDEAL {
		base.Errorf("non-integer len argument in %v - %v", op, n.Type())
		return false
	}

	// Do range checks for constants before DefaultLit
	// to avoid redundant "constant NNN overflows int" errors.
	if n.Op() == ir.OLITERAL {
		v := toint(n.Val())
		if constant.Sign(v) < 0 {
			base.Errorf("negative len argument in %v", op)
			return false
		}
		if ir.ConstOverflow(v, types.Types[types.TINT]) {
			base.Errorf("len argument too large in %v", op)
			return false
		}
	}

	// DefaultLit is necessary for non-constants too: n might be 1.1<<k.
	n = DefaultLit(n, types.Types[types.TINT])
	*np = n

	return true
}

func Conv(n ir.Node, t *types.Type) ir.Node {
	if types.IdenticalStrict(n.Type(), t) {
		return n
	}
	n = ir.NewConvExpr(base.Pos, ir.OCONV, nil, n)
	n.SetType(t)
	n = Expr(n)
	return n
}

// ConvNop converts node n to type t using the OCONVNOP op
// and typechecks the result with ctxExpr.
func ConvNop(n ir.Node, t *types.Type) ir.Node {
	if types.IdenticalStrict(n.Type(), t) {
		return n
	}
	n = ir.NewConvExpr(base.Pos, ir.OCONVNOP, nil, n)
	n.SetType(t)
	n = Expr(n)
	return n
}
