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
)

// Function collecting autotmps generated during typechecking,
// to be included in the package-level init function.
var InitTodoFunc = ir.NewFunc(base.Pos)

var inimport bool // set during import

var decldepth int32

var TypecheckAllowed bool

var (
	NeedFuncSym     = func(*types.Sym) {}
	NeedITab        = func(t, itype *types.Type) {}
	NeedRuntimeType = func(*types.Type) {}
)

func Init() {
	initUniverse()
	DeclContext = ir.PEXTERN
	base.Timer.Start("fe", "loadsys")
	loadsys()
}

func Package() {
	declareUniverse()

	TypecheckAllowed = true

	// Process top-level declarations in phases.

	// Phase 1: const, type, and names and types of funcs.
	//   This will gather all the information about types
	//   and methods but doesn't depend on any of it.
	//
	//   We also defer type alias declarations until phase 2
	//   to avoid cycles like #18640.
	//   TODO(gri) Remove this again once we have a fix for #25838.

	// Don't use range--typecheck can add closures to Target.Decls.
	base.Timer.Start("fe", "typecheck", "top1")
	for i := 0; i < len(Target.Decls); i++ {
		n := Target.Decls[i]
		if op := n.Op(); op != ir.ODCL && op != ir.OAS && op != ir.OAS2 && (op != ir.ODCLTYPE || !n.(*ir.Decl).X.Name().Alias()) {
			Target.Decls[i] = Stmt(n)
		}
	}

	// Phase 2: Variable assignments.
	//   To check interface assignments, depends on phase 1.

	// Don't use range--typecheck can add closures to Target.Decls.
	base.Timer.Start("fe", "typecheck", "top2")
	for i := 0; i < len(Target.Decls); i++ {
		n := Target.Decls[i]
		if op := n.Op(); op == ir.ODCL || op == ir.OAS || op == ir.OAS2 || op == ir.ODCLTYPE && n.(*ir.Decl).X.Name().Alias() {
			Target.Decls[i] = Stmt(n)
		}
	}

	// Phase 3: Type check function bodies.
	// Don't use range--typecheck can add closures to Target.Decls.
	base.Timer.Start("fe", "typecheck", "func")
	var fcount int64
	for i := 0; i < len(Target.Decls); i++ {
		n := Target.Decls[i]
		if n.Op() == ir.ODCLFUNC {
			FuncBody(n.(*ir.Func))
			fcount++
		}
	}

	// Phase 4: Check external declarations.
	// TODO(mdempsky): This should be handled when type checking their
	// corresponding ODCL nodes.
	base.Timer.Start("fe", "typecheck", "externdcls")
	for i, n := range Target.Externs {
		if n.Op() == ir.ONAME {
			Target.Externs[i] = Expr(Target.Externs[i])
		}
	}

	// Phase 5: With all user code type-checked, it's now safe to verify map keys.
	CheckMapKeys()

	// Phase 6: Decide how to capture closed variables.
	// This needs to run before escape analysis,
	// because variables captured by value do not escape.
	base.Timer.Start("fe", "capturevars")
	for _, n := range Target.Decls {
		if n.Op() == ir.ODCLFUNC {
			n := n.(*ir.Func)
			if n.OClosure != nil {
				ir.CurFunc = n
				CaptureVars(n)
			}
		}
	}
	CaptureVarsComplete = true
	ir.CurFunc = nil

	if base.Debug.TypecheckInl != 0 {
		// Typecheck imported function bodies if Debug.l > 1,
		// otherwise lazily when used or re-exported.
		AllImportedBodies()
	}
}

func AssignExpr(n ir.Node) ir.Node { return check(n, ctxExpr|ctxAssign) }
func Expr(n ir.Node) ir.Node       { return check(n, ctxExpr) }
func Stmt(n ir.Node) ir.Node       { return check(n, ctxStmt) }

func Exprs(exprs []ir.Node) { typecheckslice(exprs, ctxExpr) }
func Stmts(stmts []ir.Node) { typecheckslice(stmts, ctxStmt) }

func Call(call *ir.CallExpr) {
	t := call.X.Type()
	if t == nil {
		panic("misuse of Call")
	}
	ctx := ctxStmt
	if t.NumResults() > 0 {
		ctx = ctxExpr | ctxMultiOK
	}
	if check(call, ctx) != call {
		panic("bad typecheck")
	}
}

func Callee(n ir.Node) ir.Node {
	return check(n, ctxExpr|ctxCallee)
}

func FuncBody(n *ir.Func) {
	ir.CurFunc = n
	decldepth = 1
	errorsBefore := base.Errors()
	Stmts(n.Body)
	CheckReturn(n)
	if base.Errors() > errorsBefore {
		n.Body.Set(nil) // type errors; do not compile
	}
	// Now that we've checked whether n terminates,
	// we can eliminate some obviously dead code.
	deadcode(n)
}

var importlist []*ir.Func

func AllImportedBodies() {
	for _, n := range importlist {
		if n.Inl != nil {
			ImportedBody(n)
		}
	}
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

var typecheckdefstack []ir.Node

// Resolve ONONAME to definition, if any.
func Resolve(n ir.Node) (res ir.Node) {
	if n == nil || n.Op() != ir.ONONAME {
		return n
	}

	// only trace if there's work to do
	if base.EnableTrace && base.Flag.LowerT {
		defer tracePrint("resolve", n)(&res)
	}

	if sym := n.Sym(); sym.Pkg != types.LocalPkg {
		// We might have an ir.Ident from oldname or importDot.
		if id, ok := n.(*ir.Ident); ok {
			if pkgName := DotImportRefs[id]; pkgName != nil {
				pkgName.Used = true
			}
		}

		if inimport {
			base.Fatalf("recursive inimport")
		}
		inimport = true
		n = expandDecl(n)
		inimport = false
		return n
	}

	r := ir.AsNode(n.Sym().Def)
	if r == nil {
		return n
	}

	if r.Op() == ir.OIOTA {
		if x := getIotaValue(); x >= 0 {
			return ir.NewInt(x)
		}
		return n
	}

	return r
}

func typecheckslice(l []ir.Node, top int) {
	for i := range l {
		l[i] = check(l[i], top)
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

func Func(fn *ir.Func) {
	new := Stmt(fn)
	if new != fn {
		base.Fatalf("typecheck changed func")
	}
}

func typecheckNtype(n ir.Ntype) ir.Ntype {
	return check(n, ctxType).(ir.Ntype)
}

// check type checks node n.
// The result of check MUST be assigned back to n, e.g.
// 	n.Left = check(n.Left, top)
func check(n ir.Node, top int) (res ir.Node) {
	// cannot type check until all the source has been parsed
	if !TypecheckAllowed {
		base.Fatalf("early typecheck")
	}

	if n == nil {
		return nil
	}

	// only trace if there's work to do
	if base.EnableTrace && base.Flag.LowerT {
		defer tracePrint("typecheck", n)(&res)
	}

	lno := ir.SetPos(n)

	// Skip over parens.
	for n.Op() == ir.OPAREN {
		n = n.(*ir.ParenExpr).X
	}

	// Resolve definition of name and value of iota lazily.
	n = Resolve(n)

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
			n := n.(*ir.Name)
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
		n := n.(*ir.CallExpr)
		if t := n.X.Type(); t != nil && t.Kind() == types.TFUNC {
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
			types.CheckSize(t)
		}
	}
	if t != nil {
		n = EvalConst(n)
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
		return DefaultLit(n, types.Types[types.TINT])
	}
	return n
}

// typecheck1 should ONLY be called from typecheck.
func typecheck1(n ir.Node, top int) (res ir.Node) {
	if base.EnableTrace && base.Flag.LowerT {
		defer tracePrint("typecheck1", n)(&res)
	}

	switch n.Op() {
	case ir.OLITERAL, ir.ONAME, ir.ONONAME, ir.OTYPE:
		if n.Sym() == nil {
			return n
		}

		if n.Op() == ir.ONAME {
			n := n.(*ir.Name)
			if n.BuiltinOp != 0 && top&ctxCallee == 0 {
				base.Errorf("use of builtin %v not in function call", n.Sym())
				n.SetType(nil)
				return n
			}
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
		n := n.(*ir.Name)
		if n.Name().Decldepth == 0 {
			n.Name().Decldepth = decldepth
		}
		if n.BuiltinOp != 0 {
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

	case ir.ONAMEOFFSET:
		// type already set
		return n

	case ir.OPACK:
		n := n.(*ir.PkgName)
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
		n.Elem = check(n.Elem, ctxType)
		if n.Elem.Type() == nil {
			return n
		}
		t := types.NewSlice(n.Elem.Type())
		n.SetOTYPE(t)
		types.CheckSize(t)
		return n

	case ir.OTARRAY:
		n := n.(*ir.ArrayType)
		n.Elem = check(n.Elem, ctxType)
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
		n.Len = indexlit(Expr(n.Len))
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
		if ir.ConstOverflow(v, types.Types[types.TINT]) {
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
		types.CheckSize(t)
		return n

	case ir.OTMAP:
		n := n.(*ir.MapType)
		n.Key = check(n.Key, ctxType)
		n.Elem = check(n.Elem, ctxType)
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
		n.Elem = check(n.Elem, ctxType)
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
		n.SetOTYPE(NewStructType(n.Fields))
		return n

	case ir.OTINTER:
		n := n.(*ir.InterfaceType)
		n.SetOTYPE(tointerface(n.Methods))
		return n

	case ir.OTFUNC:
		n := n.(*ir.FuncType)
		n.SetOTYPE(NewFuncType(n.Recv, n.Params, n.Results))
		return n

	// type or expr
	case ir.ODEREF:
		n := n.(*ir.StarExpr)
		n.X = check(n.X, ctxExpr|ctxType)
		l := n.X
		t := l.Type()
		if t == nil {
			n.SetType(nil)
			return n
		}
		if l.Op() == ir.OTYPE {
			n.SetOTYPE(types.NewPtr(l.Type()))
			// Ensure l.Type gets dowidth'd for the backend. Issue 20174.
			types.CheckSize(l.Type())
			return n
		}

		if !t.IsPtr() {
			if top&(ctxExpr|ctxStmt) != 0 {
				base.Errorf("invalid indirect of %L", n.X)
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
		var l, r ir.Node
		var setLR func()
		switch n := n.(type) {
		case *ir.AssignOpStmt:
			l, r = n.X, n.Y
			setLR = func() { n.X = l; n.Y = r }
		case *ir.BinaryExpr:
			l, r = n.X, n.Y
			setLR = func() { n.X = l; n.Y = r }
		case *ir.LogicalExpr:
			l, r = n.X, n.Y
			setLR = func() { n.X = l; n.Y = r }
		}
		l = Expr(l)
		r = Expr(r)
		setLR()
		if l.Type() == nil || r.Type() == nil {
			n.SetType(nil)
			return n
		}
		op := n.Op()
		if n.Op() == ir.OASOP {
			n := n.(*ir.AssignOpStmt)
			checkassign(n, l)
			if n.IncDec && !okforarith[l.Type().Kind()] {
				base.Errorf("invalid operation: %v (non-numeric type %v)", n, l.Type())
				n.SetType(nil)
				return n
			}
			// TODO(marvin): Fix Node.EType type union.
			op = n.AsOp
		}
		if op == ir.OLSH || op == ir.ORSH {
			r = DefaultLit(r, types.Types[types.TUINT])
			setLR()
			t := r.Type()
			if !t.IsInteger() {
				base.Errorf("invalid operation: %v (shift count type %v, must be integer)", n, r.Type())
				n.SetType(nil)
				return n
			}
			if t.IsSigned() && !types.AllowsGoVersion(curpkg(), 1, 13) {
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
			n := n.(*ir.LogicalExpr)
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
		}

		// ideal mixed with non-ideal
		l, r = defaultlit2(l, r, false)
		setLR()

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
					if r.Type().IsInterface() && !l.Type().IsInterface() && !types.IsComparable(l.Type()) {
						base.Errorf("invalid operation: %v (operator %v not defined on %s)", n, op, typekind(l.Type()))
						n.SetType(nil)
						return n
					}

					types.CalcSize(l.Type())
					if r.Type().IsInterface() == l.Type().IsInterface() || l.Type().Width >= 1<<16 {
						l = ir.NewConvExpr(base.Pos, aop, r.Type(), l)
						l.SetTypecheck(1)
						setLR()
					}

					t = r.Type()
					converted = true
				}
			}

			if !converted && l.Type().Kind() != types.TBLANK {
				aop, _ = assignop(r.Type(), l.Type())
				if aop != ir.OXXX {
					if l.Type().IsInterface() && !r.Type().IsInterface() && !types.IsComparable(r.Type()) {
						base.Errorf("invalid operation: %v (operator %v not defined on %s)", n, op, typekind(r.Type()))
						n.SetType(nil)
						return n
					}

					types.CalcSize(r.Type())
					if r.Type().IsInterface() == l.Type().IsInterface() || r.Type().Width >= 1<<16 {
						r = ir.NewConvExpr(base.Pos, aop, l.Type(), r)
						r.SetTypecheck(1)
						setLR()
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
		if l.Type().IsArray() && !types.IsComparable(l.Type()) {
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
			if f := types.IncomparableField(l.Type()); f != nil {
				base.Errorf("invalid operation: %v (struct containing %v cannot be compared)", n, f.Type)
				n.SetType(nil)
				return n
			}
		}

		if iscmp[n.Op()] {
			t = types.UntypedBool
			n.SetType(t)
			if con := EvalConst(n); con.Op() == ir.OLITERAL {
				return con
			}
			l, r = defaultlit2(l, r, true)
			setLR()
			return n
		}

		if et == types.TSTRING && n.Op() == ir.OADD {
			// create or update OADDSTR node with list of strings in x + y + z + (w + v) + ...
			n := n.(*ir.BinaryExpr)
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
		n := n.(*ir.UnaryExpr)
		n.X = Expr(n.X)
		l := n.X
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
		n := n.(*ir.AddrExpr)
		n.X = Expr(n.X)
		if n.X.Type() == nil {
			n.SetType(nil)
			return n
		}

		switch n.X.Op() {
		case ir.OARRAYLIT, ir.OMAPLIT, ir.OSLICELIT, ir.OSTRUCTLIT:
			n.SetOp(ir.OPTRLIT)

		default:
			checklvalue(n.X, "take the address of")
			r := ir.OuterValue(n.X)
			if r.Op() == ir.ONAME {
				r := r.(*ir.Name)
				if ir.Orig(r) != r {
					base.Fatalf("found non-orig name node %v", r) // TODO(mdempsky): What does this mean?
				}
				r.Name().SetAddrtaken(true)
				if r.Name().IsClosureVar() && !CaptureVarsComplete {
					// Mark the original variable as Addrtaken so that capturevars
					// knows not to pass it by value.
					// But if the capturevars phase is complete, don't touch it,
					// in case l.Name's containing function has not yet been compiled.
					r.Name().Defn.Name().SetAddrtaken(true)
				}
			}
			n.X = DefaultLit(n.X, nil)
			if n.X.Type() == nil {
				n.SetType(nil)
				return n
			}
		}

		n.SetType(types.NewPtr(n.X.Type()))
		return n

	case ir.OCOMPLIT:
		return typecheckcomplit(n.(*ir.CompLitExpr))

	case ir.OXDOT, ir.ODOT:
		n := n.(*ir.SelectorExpr)
		if n.Op() == ir.OXDOT {
			n = AddImplicitDots(n)
			n.SetOp(ir.ODOT)
			if n.X == nil {
				n.SetType(nil)
				return n
			}
		}

		n.X = check(n.X, ctxExpr|ctxType)

		n.X = DefaultLit(n.X, nil)

		t := n.X.Type()
		if t == nil {
			base.UpdateErrorDot(ir.Line(n), fmt.Sprint(n.X), fmt.Sprint(n))
			n.SetType(nil)
			return n
		}

		s := n.Sel

		if n.X.Op() == ir.OTYPE {
			return typecheckMethodExpr(n)
		}

		if t.IsPtr() && !t.Elem().IsInterface() {
			t = t.Elem()
			if t == nil {
				n.SetType(nil)
				return n
			}
			n.SetOp(ir.ODOTPTR)
			types.CheckSize(t)
		}

		if n.Sel.IsBlank() {
			base.Errorf("cannot refer to blank field or method")
			n.SetType(nil)
			return n
		}

		if lookdot(n, t, 0) == nil {
			// Legitimate field or method lookup failed, try to explain the error
			switch {
			case t.IsEmptyInterface():
				base.Errorf("%v undefined (type %v is interface with no methods)", n, n.X.Type())

			case t.IsPtr() && t.Elem().IsInterface():
				// Pointer to interface is almost always a mistake.
				base.Errorf("%v undefined (type %v is pointer to interface, not interface)", n, n.X.Type())

			case lookdot(n, t, 1) != nil:
				// Field or method matches by name, but it is not exported.
				base.Errorf("%v undefined (cannot refer to unexported field or method %v)", n, n.Sel)

			default:
				if mt := lookdot(n, t, 2); mt != nil && visible(mt.Sym) { // Case-insensitive lookup.
					base.Errorf("%v undefined (type %v has no field or method %v, but does have %v)", n, n.X.Type(), n.Sel, mt.Sym)
				} else {
					base.Errorf("%v undefined (type %v has no field or method %v)", n, n.X.Type(), n.Sel)
				}
			}
			n.SetType(nil)
			return n
		}

		if (n.Op() == ir.ODOTINTER || n.Op() == ir.ODOTMETH) && top&ctxCallee == 0 {
			return typecheckpartialcall(n, s)
		}
		return n

	case ir.ODOTTYPE:
		n := n.(*ir.TypeAssertExpr)
		n.X = Expr(n.X)
		n.X = DefaultLit(n.X, nil)
		l := n.X
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

		if n.Ntype != nil {
			n.Ntype = check(n.Ntype, ctxType)
			n.SetType(n.Ntype.Type())
			n.Ntype = nil
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
		n := n.(*ir.IndexExpr)
		n.X = Expr(n.X)
		n.X = DefaultLit(n.X, nil)
		n.X = implicitstar(n.X)
		l := n.X
		n.Index = Expr(n.Index)
		r := n.Index
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
			n.Index = indexlit(n.Index)
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

			if n.Index.Type() != nil && !n.Index.Type().IsInteger() {
				base.Errorf("non-integer %s index %v", why, n.Index)
				return n
			}

			if !n.Bounded() && ir.IsConst(n.Index, constant.Int) {
				x := n.Index.Val()
				if constant.Sign(x) < 0 {
					base.Errorf("invalid %s index %v (index must be non-negative)", why, n.Index)
				} else if t.IsArray() && constant.Compare(x, token.GEQ, constant.MakeInt64(t.NumElem())) {
					base.Errorf("invalid array index %v (out of bounds for %d-element array)", n.Index, t.NumElem())
				} else if ir.IsConst(n.X, constant.String) && constant.Compare(x, token.GEQ, constant.MakeInt64(int64(len(ir.StringVal(n.X))))) {
					base.Errorf("invalid string index %v (out of bounds for %d-byte string)", n.Index, len(ir.StringVal(n.X)))
				} else if ir.ConstOverflow(x, types.Types[types.TINT]) {
					base.Errorf("invalid %s index %v (index too large)", why, n.Index)
				}
			}

		case types.TMAP:
			n.Index = AssignConv(n.Index, t.Key(), "map index")
			n.SetType(t.Elem())
			n.SetOp(ir.OINDEXMAP)
			n.Assigned = false
		}
		return n

	case ir.ORECV:
		n := n.(*ir.UnaryExpr)
		n.X = Expr(n.X)
		n.X = DefaultLit(n.X, nil)
		l := n.X
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
		n := n.(*ir.SendStmt)
		n.Chan = Expr(n.Chan)
		n.Value = Expr(n.Value)
		n.Chan = DefaultLit(n.Chan, nil)
		t := n.Chan.Type()
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

		n.Value = AssignConv(n.Value, t.Elem(), "send")
		if n.Value.Type() == nil {
			return n
		}
		return n

	case ir.OSLICEHEADER:
		// Errors here are Fatalf instead of Errorf because only the compiler
		// can construct an OSLICEHEADER node.
		// Components used in OSLICEHEADER that are supplied by parsed source code
		// have already been typechecked in e.g. OMAKESLICE earlier.
		n := n.(*ir.SliceHeaderExpr)
		t := n.Type()
		if t == nil {
			base.Fatalf("no type specified for OSLICEHEADER")
		}

		if !t.IsSlice() {
			base.Fatalf("invalid type %v for OSLICEHEADER", n.Type())
		}

		if n.Ptr == nil || n.Ptr.Type() == nil || !n.Ptr.Type().IsUnsafePtr() {
			base.Fatalf("need unsafe.Pointer for OSLICEHEADER")
		}

		if x := len(n.LenCap); x != 2 {
			base.Fatalf("expected 2 params (len, cap) for OSLICEHEADER, got %d", x)
		}

		n.Ptr = Expr(n.Ptr)
		l := Expr(n.LenCap[0])
		c := Expr(n.LenCap[1])
		l = DefaultLit(l, types.Types[types.TINT])
		c = DefaultLit(c, types.Types[types.TINT])

		if ir.IsConst(l, constant.Int) && ir.Int64Val(l) < 0 {
			base.Fatalf("len for OSLICEHEADER must be non-negative")
		}

		if ir.IsConst(c, constant.Int) && ir.Int64Val(c) < 0 {
			base.Fatalf("cap for OSLICEHEADER must be non-negative")
		}

		if ir.IsConst(l, constant.Int) && ir.IsConst(c, constant.Int) && constant.Compare(l.Val(), token.GTR, c.Val()) {
			base.Fatalf("len larger than cap for OSLICEHEADER")
		}

		n.LenCap[0] = l
		n.LenCap[1] = c
		return n

	case ir.OMAKESLICECOPY:
		// Errors here are Fatalf instead of Errorf because only the compiler
		// can construct an OMAKESLICECOPY node.
		// Components used in OMAKESCLICECOPY that are supplied by parsed source code
		// have already been typechecked in OMAKE and OCOPY earlier.
		n := n.(*ir.MakeExpr)
		t := n.Type()

		if t == nil {
			base.Fatalf("no type specified for OMAKESLICECOPY")
		}

		if !t.IsSlice() {
			base.Fatalf("invalid type %v for OMAKESLICECOPY", n.Type())
		}

		if n.Len == nil {
			base.Fatalf("missing len argument for OMAKESLICECOPY")
		}

		if n.Cap == nil {
			base.Fatalf("missing slice argument to copy for OMAKESLICECOPY")
		}

		n.Len = Expr(n.Len)
		n.Cap = Expr(n.Cap)

		n.Len = DefaultLit(n.Len, types.Types[types.TINT])

		if !n.Len.Type().IsInteger() && n.Type().Kind() != types.TIDEAL {
			base.Errorf("non-integer len argument in OMAKESLICECOPY")
		}

		if ir.IsConst(n.Len, constant.Int) {
			if ir.ConstOverflow(n.Len.Val(), types.Types[types.TINT]) {
				base.Fatalf("len for OMAKESLICECOPY too large")
			}
			if constant.Sign(n.Len.Val()) < 0 {
				base.Fatalf("len for OMAKESLICECOPY must be non-negative")
			}
		}
		return n

	case ir.OSLICE, ir.OSLICE3:
		n := n.(*ir.SliceExpr)
		n.X = Expr(n.X)
		low, high, max := n.SliceBounds()
		hasmax := n.Op().IsSlice3()
		low = Expr(low)
		high = Expr(high)
		max = Expr(max)
		n.X = DefaultLit(n.X, nil)
		low = indexlit(low)
		high = indexlit(high)
		max = indexlit(max)
		n.SetSliceBounds(low, high, max)
		l := n.X
		if l.Type() == nil {
			n.SetType(nil)
			return n
		}
		if l.Type().IsArray() {
			if !ir.IsAssignable(n.X) {
				base.Errorf("invalid operation %v (slice of unaddressable value)", n)
				n.SetType(nil)
				return n
			}

			addr := NodAddr(n.X)
			addr.SetImplicit(true)
			n.X = Expr(addr)
			l = n.X
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
			types.CalcSize(n.Type())
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
		n := n.(*ir.CallExpr)
		n.Use = ir.CallUseExpr
		if top == ctxStmt {
			n.Use = ir.CallUseStmt
		}
		Stmts(n.Init()) // imported rewritten f(g()) calls (#30907)
		n.X = check(n.X, ctxExpr|ctxType|ctxCallee)
		if n.X.Diag() {
			n.SetDiag(true)
		}

		l := n.X

		if l.Op() == ir.ONAME && l.(*ir.Name).BuiltinOp != 0 {
			l := l.(*ir.Name)
			if n.IsDDD && l.BuiltinOp != ir.OAPPEND {
				base.Errorf("invalid use of ... with builtin %v", l)
			}

			// builtin: OLEN, OCAP, etc.
			switch l.BuiltinOp {
			default:
				base.Fatalf("unknown builtin %v", l)

			case ir.OAPPEND, ir.ODELETE, ir.OMAKE, ir.OPRINT, ir.OPRINTN, ir.ORECOVER:
				n.SetOp(l.BuiltinOp)
				n.X = nil
				n.SetTypecheck(0) // re-typechecking new op is OK, not a loop
				return check(n, top)

			case ir.OCAP, ir.OCLOSE, ir.OIMAG, ir.OLEN, ir.OPANIC, ir.OREAL:
				typecheckargs(n)
				fallthrough
			case ir.ONEW, ir.OALIGNOF, ir.OOFFSETOF, ir.OSIZEOF:
				arg, ok := needOneArg(n, "%v", n.Op())
				if !ok {
					n.SetType(nil)
					return n
				}
				u := ir.NewUnaryExpr(n.Pos(), l.BuiltinOp, arg)
				return check(ir.InitExpr(n.Init(), u), top) // typecheckargs can add to old.Init

			case ir.OCOMPLEX, ir.OCOPY:
				typecheckargs(n)
				arg1, arg2, ok := needTwoArgs(n)
				if !ok {
					n.SetType(nil)
					return n
				}
				b := ir.NewBinaryExpr(n.Pos(), l.BuiltinOp, arg1, arg2)
				return check(ir.InitExpr(n.Init(), b), top) // typecheckargs can add to old.Init
			}
			panic("unreachable")
		}

		n.X = DefaultLit(n.X, nil)
		l = n.X
		if l.Op() == ir.OTYPE {
			if n.IsDDD {
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

			n := ir.NewConvExpr(n.Pos(), ir.OCONV, nil, arg)
			n.SetType(l.Type())
			return typecheck1(n, top)
		}

		typecheckargs(n)
		t := l.Type()
		if t == nil {
			n.SetType(nil)
			return n
		}
		types.CheckSize(t)

		switch l.Op() {
		case ir.ODOTINTER:
			n.SetOp(ir.OCALLINTER)

		case ir.ODOTMETH:
			l := l.(*ir.SelectorExpr)
			n.SetOp(ir.OCALLMETH)

			// typecheckaste was used here but there wasn't enough
			// information further down the call chain to know if we
			// were testing a method receiver for unexported fields.
			// It isn't necessary, so just do a sanity check.
			tp := t.Recv().Type

			if l.X == nil || !types.Identical(l.X.Type(), tp) {
				base.Fatalf("method receiver")
			}

		default:
			n.SetOp(ir.OCALLFUNC)
			if t.Kind() != types.TFUNC {
				// TODO(mdempsky): Remove "o.Sym() != nil" once we stop
				// using ir.Name for numeric literals.
				if o := ir.Orig(l); o.Name() != nil && o.Sym() != nil && types.BuiltinPkg.Lookup(o.Sym().Name).Def != nil {
					// be more specific when the non-function
					// name matches a predeclared function
					base.Errorf("cannot call non-function %L, declared at %s",
						l, base.FmtPos(o.Name().Pos()))
				} else {
					base.Errorf("cannot call non-function %L", l)
				}
				n.SetType(nil)
				return n
			}
		}

		typecheckaste(ir.OCALL, n.X, n.IsDDD, t.Params(), n.Args, func() string { return fmt.Sprintf("argument to %v", n.X) })
		if t.NumResults() == 0 {
			return n
		}
		if t.NumResults() == 1 {
			n.SetType(l.Type().Results().Field(0).Type)

			if n.Op() == ir.OCALLFUNC && n.X.Op() == ir.ONAME {
				if sym := n.X.(*ir.Name).Sym(); types.IsRuntimePkg(sym.Pkg) && sym.Name == "getg" {
					// Emit code for runtime.getg() directly instead of calling function.
					// Most such rewrites (for example the similar one for math.Sqrt) should be done in walk,
					// so that the ordering pass can make sure to preserve the semantics of the original code
					// (in particular, the exact time of the function call) by introducing temporaries.
					// In this case, we know getg() always returns the same result within a given function
					// and we want to avoid the temporaries, so we do the rewrite earlier than is typical.
					n.SetOp(ir.OGETG)
				}
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
		n := n.(*ir.UnaryExpr)
		n.SetType(types.Types[types.TUINTPTR])
		return n

	case ir.OCAP, ir.OLEN:
		n := n.(*ir.UnaryExpr)
		n.X = Expr(n.X)
		n.X = DefaultLit(n.X, nil)
		n.X = implicitstar(n.X)
		l := n.X
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
		n := n.(*ir.UnaryExpr)
		n.X = Expr(n.X)
		l := n.X
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
		n := n.(*ir.BinaryExpr)
		l := Expr(n.X)
		r := Expr(n.Y)
		if l.Type() == nil || r.Type() == nil {
			n.SetType(nil)
			return n
		}
		l, r = defaultlit2(l, r, false)
		if l.Type() == nil || r.Type() == nil {
			n.SetType(nil)
			return n
		}
		n.X = l
		n.Y = r

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
		n := n.(*ir.UnaryExpr)
		n.X = Expr(n.X)
		n.X = DefaultLit(n.X, nil)
		l := n.X
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
		n := n.(*ir.CallExpr)
		typecheckargs(n)
		args := n.Args
		if len(args) == 0 {
			base.Errorf("missing arguments to delete")
			n.SetType(nil)
			return n
		}

		if len(args) == 1 {
			base.Errorf("missing second (key) argument to delete")
			n.SetType(nil)
			return n
		}

		if len(args) != 2 {
			base.Errorf("too many arguments to delete")
			n.SetType(nil)
			return n
		}

		l := args[0]
		r := args[1]
		if l.Type() != nil && !l.Type().IsMap() {
			base.Errorf("first argument to delete must be map; have %L", l.Type())
			n.SetType(nil)
			return n
		}

		args[1] = AssignConv(r, l.Type().Key(), "delete")
		return n

	case ir.OAPPEND:
		n := n.(*ir.CallExpr)
		typecheckargs(n)
		args := n.Args
		if len(args) == 0 {
			base.Errorf("missing arguments to append")
			n.SetType(nil)
			return n
		}

		t := args[0].Type()
		if t == nil {
			n.SetType(nil)
			return n
		}

		n.SetType(t)
		if !t.IsSlice() {
			if ir.IsNil(args[0]) {
				base.Errorf("first argument to append must be typed slice; have untyped nil")
				n.SetType(nil)
				return n
			}

			base.Errorf("first argument to append must be slice; have %L", t)
			n.SetType(nil)
			return n
		}

		if n.IsDDD {
			if len(args) == 1 {
				base.Errorf("cannot use ... on first argument to append")
				n.SetType(nil)
				return n
			}

			if len(args) != 2 {
				base.Errorf("too many arguments to append")
				n.SetType(nil)
				return n
			}

			if t.Elem().IsKind(types.TUINT8) && args[1].Type().IsString() {
				args[1] = DefaultLit(args[1], types.Types[types.TSTRING])
				return n
			}

			args[1] = AssignConv(args[1], t.Underlying(), "append")
			return n
		}

		as := args[1:]
		for i, n := range as {
			if n.Type() == nil {
				continue
			}
			as[i] = AssignConv(n, t.Elem(), "append")
			types.CheckSize(as[i].Type()) // ensure width is calculated for backend
		}
		return n

	case ir.OCOPY:
		n := n.(*ir.BinaryExpr)
		n.SetType(types.Types[types.TINT])
		n.X = Expr(n.X)
		n.X = DefaultLit(n.X, nil)
		n.Y = Expr(n.Y)
		n.Y = DefaultLit(n.Y, nil)
		if n.X.Type() == nil || n.Y.Type() == nil {
			n.SetType(nil)
			return n
		}

		// copy([]byte, string)
		if n.X.Type().IsSlice() && n.Y.Type().IsString() {
			if types.Identical(n.X.Type().Elem(), types.ByteType) {
				return n
			}
			base.Errorf("arguments to copy have different element types: %L and string", n.X.Type())
			n.SetType(nil)
			return n
		}

		if !n.X.Type().IsSlice() || !n.Y.Type().IsSlice() {
			if !n.X.Type().IsSlice() && !n.Y.Type().IsSlice() {
				base.Errorf("arguments to copy must be slices; have %L, %L", n.X.Type(), n.Y.Type())
			} else if !n.X.Type().IsSlice() {
				base.Errorf("first argument to copy should be slice; have %L", n.X.Type())
			} else {
				base.Errorf("second argument to copy should be slice or string; have %L", n.Y.Type())
			}
			n.SetType(nil)
			return n
		}

		if !types.Identical(n.X.Type().Elem(), n.Y.Type().Elem()) {
			base.Errorf("arguments to copy have different element types: %L and %L", n.X.Type(), n.Y.Type())
			n.SetType(nil)
			return n
		}
		return n

	case ir.OCONV:
		n := n.(*ir.ConvExpr)
		types.CheckSize(n.Type()) // ensure width is calculated for backend
		n.X = Expr(n.X)
		n.X = convlit1(n.X, n.Type(), true, nil)
		t := n.X.Type()
		if t == nil || n.Type() == nil {
			n.SetType(nil)
			return n
		}
		op, why := convertop(n.X.Op() == ir.OLITERAL, t, n.Type())
		if op == ir.OXXX {
			if !n.Diag() && !n.Type().Broke() && !n.X.Diag() {
				base.Errorf("cannot convert %L to type %v%s", n.X, n.Type(), why)
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
			if n.X.Op() == ir.OLITERAL {
				return stringtoruneslit(n)
			}
		}
		return n

	case ir.OMAKE:
		n := n.(*ir.CallExpr)
		args := n.Args
		if len(args) == 0 {
			base.Errorf("missing argument to make")
			n.SetType(nil)
			return n
		}

		n.Args.Set(nil)
		l := args[0]
		l = check(l, ctxType)
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
			l = Expr(l)
			var r ir.Node
			if i < len(args) {
				r = args[i]
				i++
				r = Expr(r)
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
			nn = ir.NewMakeExpr(n.Pos(), ir.OMAKESLICE, l, r)

		case types.TMAP:
			if i < len(args) {
				l = args[i]
				i++
				l = Expr(l)
				l = DefaultLit(l, types.Types[types.TINT])
				if l.Type() == nil {
					n.SetType(nil)
					return n
				}
				if !checkmake(t, "size", &l) {
					n.SetType(nil)
					return n
				}
			} else {
				l = ir.NewInt(0)
			}
			nn = ir.NewMakeExpr(n.Pos(), ir.OMAKEMAP, l, nil)
			nn.SetEsc(n.Esc())

		case types.TCHAN:
			l = nil
			if i < len(args) {
				l = args[i]
				i++
				l = Expr(l)
				l = DefaultLit(l, types.Types[types.TINT])
				if l.Type() == nil {
					n.SetType(nil)
					return n
				}
				if !checkmake(t, "buffer", &l) {
					n.SetType(nil)
					return n
				}
			} else {
				l = ir.NewInt(0)
			}
			nn = ir.NewMakeExpr(n.Pos(), ir.OMAKECHAN, l, nil)
		}

		if i < len(args) {
			base.Errorf("too many arguments to make(%v)", t)
			n.SetType(nil)
			return n
		}

		nn.SetType(t)
		return nn

	case ir.ONEW:
		n := n.(*ir.UnaryExpr)
		if n.X == nil {
			// Fatalf because the OCALL above checked for us,
			// so this must be an internally-generated mistake.
			base.Fatalf("missing argument to new")
		}
		l := n.X
		l = check(l, ctxType)
		t := l.Type()
		if t == nil {
			n.SetType(nil)
			return n
		}
		n.X = l
		n.SetType(types.NewPtr(t))
		return n

	case ir.OPRINT, ir.OPRINTN:
		n := n.(*ir.CallExpr)
		typecheckargs(n)
		ls := n.Args
		for i1, n1 := range ls {
			// Special case for print: int constant is int64, not int.
			if ir.IsConst(n1, constant.Int) {
				ls[i1] = DefaultLit(ls[i1], types.Types[types.TINT64])
			} else {
				ls[i1] = DefaultLit(ls[i1], nil)
			}
		}
		return n

	case ir.OPANIC:
		n := n.(*ir.UnaryExpr)
		n.X = Expr(n.X)
		n.X = DefaultLit(n.X, types.Types[types.TINTER])
		if n.X.Type() == nil {
			n.SetType(nil)
			return n
		}
		return n

	case ir.ORECOVER:
		n := n.(*ir.CallExpr)
		if len(n.Args) != 0 {
			base.Errorf("too many arguments to recover")
			n.SetType(nil)
			return n
		}

		n.SetType(types.Types[types.TINTER])
		return n

	case ir.OCLOSURE:
		n := n.(*ir.ClosureExpr)
		typecheckclosure(n, top)
		if n.Type() == nil {
			return n
		}
		return n

	case ir.OITAB:
		n := n.(*ir.UnaryExpr)
		n.X = Expr(n.X)
		t := n.X.Type()
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
		n := n.(*ir.UnaryExpr)
		base.Fatalf("cannot typecheck interface data %v", n)
		panic("unreachable")

	case ir.OSPTR:
		n := n.(*ir.UnaryExpr)
		n.X = Expr(n.X)
		t := n.X.Type()
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
		n := n.(*ir.UnaryExpr)
		n.X = Expr(n.X)
		n.SetType(types.Types[types.TUINTPTR])
		return n

	case ir.OCONVNOP:
		n := n.(*ir.ConvExpr)
		n.X = Expr(n.X)
		return n

	// statements
	case ir.OAS:
		n := n.(*ir.AssignStmt)
		typecheckas(n)

		// Code that creates temps does not bother to set defn, so do it here.
		if n.X.Op() == ir.ONAME && ir.IsAutoTmp(n.X) {
			n.X.Name().Defn = n
		}
		return n

	case ir.OAS2:
		typecheckas2(n.(*ir.AssignListStmt))
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
		n := n.(*ir.BlockStmt)
		Stmts(n.List)
		return n

	case ir.OLABEL:
		decldepth++
		if n.Sym().IsBlank() {
			// Empty identifier is valid but useless.
			// Eliminate now to simplify life later.
			// See issues 7538, 11589, 11593.
			n = ir.NewBlockStmt(n.Pos(), nil)
		}
		return n

	case ir.ODEFER, ir.OGO:
		n := n.(*ir.GoDeferStmt)
		n.Call = check(n.Call, ctxStmt|ctxExpr)
		if !n.Call.Diag() {
			checkdefergo(n)
		}
		return n

	case ir.OFOR, ir.OFORUNTIL:
		n := n.(*ir.ForStmt)
		Stmts(n.Init())
		decldepth++
		n.Cond = Expr(n.Cond)
		n.Cond = DefaultLit(n.Cond, nil)
		if n.Cond != nil {
			t := n.Cond.Type()
			if t != nil && !t.IsBoolean() {
				base.Errorf("non-bool %L used as for condition", n.Cond)
			}
		}
		n.Post = Stmt(n.Post)
		if n.Op() == ir.OFORUNTIL {
			Stmts(n.Late)
		}
		Stmts(n.Body)
		decldepth--
		return n

	case ir.OIF:
		n := n.(*ir.IfStmt)
		Stmts(n.Init())
		n.Cond = Expr(n.Cond)
		n.Cond = DefaultLit(n.Cond, nil)
		if n.Cond != nil {
			t := n.Cond.Type()
			if t != nil && !t.IsBoolean() {
				base.Errorf("non-bool %L used as if condition", n.Cond)
			}
		}
		Stmts(n.Body)
		Stmts(n.Else)
		return n

	case ir.ORETURN:
		n := n.(*ir.ReturnStmt)
		typecheckargs(n)
		if ir.CurFunc == nil {
			base.Errorf("return outside function")
			n.SetType(nil)
			return n
		}

		if ir.HasNamedResults(ir.CurFunc) && len(n.Results) == 0 {
			return n
		}
		typecheckaste(ir.ORETURN, nil, false, ir.CurFunc.Type().Results(), n.Results, func() string { return "return argument" })
		return n

	case ir.ORETJMP:
		n := n.(*ir.BranchStmt)
		return n

	case ir.OSELECT:
		typecheckselect(n.(*ir.SelectStmt))
		return n

	case ir.OSWITCH:
		typecheckswitch(n.(*ir.SwitchStmt))
		return n

	case ir.ORANGE:
		typecheckrange(n.(*ir.RangeStmt))
		return n

	case ir.OTYPESW:
		n := n.(*ir.TypeSwitchGuard)
		base.Errorf("use of .(type) outside type switch")
		n.SetType(nil)
		return n

	case ir.ODCLFUNC:
		typecheckfunc(n.(*ir.Func))
		return n

	case ir.ODCLCONST:
		n := n.(*ir.Decl)
		n.X = Expr(n.X)
		return n

	case ir.ODCLTYPE:
		n := n.(*ir.Decl)
		n.X = check(n.X, ctxType)
		types.CheckSize(n.X.Type())
		return n
	}

	// No return n here!
	// Individual cases can type-assert n, introducing a new one.
	// Each must execute its own return n.
}

func typecheckargs(n ir.Node) {
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

	// Save n as n.Orig for fmt.go.
	if ir.Orig(n) == n {
		n.(ir.OrigNode).SetOrig(ir.SepCopy(n))
	}

	as := ir.NewAssignListStmt(base.Pos, ir.OAS2, nil, nil)
	as.Rhs.Append(list...)

	// If we're outside of function context, then this call will
	// be executed during the generated init function. However,
	// init.go hasn't yet created it. Instead, associate the
	// temporary variables with initTodo for now, and init.go
	// will reassociate them later when it's appropriate.
	static := ir.CurFunc == nil
	if static {
		ir.CurFunc = InitTodoFunc
	}
	list = nil
	for _, f := range t.FieldSlice() {
		t := Temp(f.Type)
		as.PtrInit().Append(ir.NewDecl(base.Pos, ir.ODCL, t))
		as.Lhs.Append(t)
		list = append(list, t)
	}
	if static {
		ir.CurFunc = nil
	}

	switch n := n.(type) {
	case *ir.CallExpr:
		n.Args.Set(list)
	case *ir.ReturnStmt:
		n.Results.Set(list)
	}

	n.PtrInit().Append(Stmt(as))
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

func checkdefergo(n *ir.GoDeferStmt) {
	what := "defer"
	if n.Op() == ir.OGO {
		what = "go"
	}

	switch n.Call.Op() {
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
		if orig := ir.Orig(n.Call); orig.Op() == ir.OCONV {
			break
		}
		base.ErrorfAt(n.Pos(), "%s discards result of %v", what, n.Call)
		return
	}

	// type is broken or missing, most likely a method call on a broken type
	// we will warn about the broken type elsewhere. no need to emit a potentially confusing error
	if n.Call.Type() == nil || n.Call.Type().Broke() {
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
func typecheckMethodExpr(n *ir.SelectorExpr) (res ir.Node) {
	if base.EnableTrace && base.Flag.LowerT {
		defer tracePrint("typecheckMethodExpr", n)(&res)
	}

	t := n.X.Type()

	// Compute the method set for t.
	var ms *types.Fields
	if t.IsInterface() {
		ms = t.Fields()
	} else {
		mt := types.ReceiverBaseType(t)
		if mt == nil {
			base.Errorf("%v undefined (type %v has no method %v)", n, t, n.Sel)
			n.SetType(nil)
			return n
		}
		CalcMethods(mt)
		ms = mt.AllMethods()

		// The method expression T.m requires a wrapper when T
		// is different from m's declared receiver type. We
		// normally generate these wrappers while writing out
		// runtime type descriptors, which is always done for
		// types declared at package scope. However, we need
		// to make sure to generate wrappers for anonymous
		// receiver types too.
		if mt.Sym() == nil {
			NeedRuntimeType(t)
		}
	}

	s := n.Sel
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

	if !types.IsMethodApplicable(t, m) {
		base.Errorf("invalid method expression %v (needs pointer receiver: (*%v).%S)", n, t, s)
		n.SetType(nil)
		return n
	}

	me := ir.NewMethodExpr(n.Pos(), n.X.Type(), m)
	me.SetType(NewMethodType(m.Type, n.X.Type()))
	f := NewName(ir.MethodSym(t, m.Sym))
	f.Class_ = ir.PFUNC
	f.SetType(me.Type())
	me.FuncName_ = f

	// Issue 25065. Make sure that we emit the symbol for a local method.
	if base.Ctxt.Flag_dynlink && !inimport && (t.Sym() == nil || t.Sym().Pkg == types.LocalPkg) {
		NeedFuncSym(me.FuncName_.Sym())
	}

	return me
}

func derefall(t *types.Type) *types.Type {
	for t != nil && t.IsPtr() {
		t = t.Elem()
	}
	return t
}

func lookdot(n *ir.SelectorExpr, t *types.Type, dostrcmp int) *types.Field {
	s := n.Sel

	types.CalcSize(t)
	var f1 *types.Field
	if t.IsStruct() || t.IsInterface() {
		f1 = lookdot1(n, s, t, t.Fields(), dostrcmp)
	}

	var f2 *types.Field
	if n.X.Type() == t || n.X.Type().Sym() == nil {
		mt := types.ReceiverBaseType(t)
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
			base.Errorf("%v is both field and method", n.Sel)
		}
		if f1.Offset == types.BADWIDTH {
			base.Fatalf("lookdot badwidth %v %p", f1, f1)
		}
		n.Offset = f1.Offset
		n.SetType(f1.Type)
		if t.IsInterface() {
			if n.X.Type().IsPtr() {
				star := ir.NewStarExpr(base.Pos, n.X)
				star.SetImplicit(true)
				n.X = Expr(star)
			}

			n.SetOp(ir.ODOTINTER)
		}
		n.Selection = f1
		return f1
	}

	if f2 != nil {
		if dostrcmp > 1 {
			// Already in the process of diagnosing an error.
			return f2
		}
		tt := n.X.Type()
		types.CalcSize(tt)
		rcvr := f2.Type.Recv().Type
		if !types.Identical(rcvr, tt) {
			if rcvr.IsPtr() && types.Identical(rcvr.Elem(), tt) {
				checklvalue(n.X, "call pointer method on")
				addr := NodAddr(n.X)
				addr.SetImplicit(true)
				n.X = check(addr, ctxType|ctxExpr)
			} else if tt.IsPtr() && (!rcvr.IsPtr() || rcvr.IsPtr() && rcvr.Elem().NotInHeap()) && types.Identical(tt.Elem(), rcvr) {
				star := ir.NewStarExpr(base.Pos, n.X)
				star.SetImplicit(true)
				n.X = check(star, ctxType|ctxExpr)
			} else if tt.IsPtr() && tt.Elem().IsPtr() && types.Identical(derefall(tt), derefall(rcvr)) {
				base.Errorf("calling method %v with receiver %L requires explicit dereference", n.Sel, n.X)
				for tt.IsPtr() {
					// Stop one level early for method with pointer receiver.
					if rcvr.IsPtr() && !tt.Elem().IsPtr() {
						break
					}
					star := ir.NewStarExpr(base.Pos, n.X)
					star.SetImplicit(true)
					n.X = check(star, ctxType|ctxExpr)
					tt = tt.Elem()
				}
			} else {
				base.Fatalf("method mismatch: %v for %v", rcvr, tt)
			}
		}

		implicit, ll := n.Implicit(), n.X
		for ll != nil && (ll.Op() == ir.ODOT || ll.Op() == ir.ODOTPTR || ll.Op() == ir.ODEREF) {
			switch l := ll.(type) {
			case *ir.SelectorExpr:
				implicit, ll = l.Implicit(), l.X
			case *ir.StarExpr:
				implicit, ll = l.Implicit(), l.X
			}
		}
		if implicit && ll.Type().IsPtr() && ll.Type().Sym() != nil && ll.Type().Sym().Def != nil && ir.AsNode(ll.Type().Sym().Def).Op() == ir.OTYPE {
			// It is invalid to automatically dereference a named pointer type when selecting a method.
			// Make n.Left == ll to clarify error message.
			n.X = ll
			return nil
		}

		n.Sel = ir.MethodSym(n.X.Type(), f2.Sym)
		n.Offset = f2.Offset
		n.SetType(f2.Type)
		n.SetOp(ir.ODOTMETH)
		n.Selection = f2

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
	if len(nl) == 1 {
		n = nl[0]
	}

	n1 := tstruct.NumFields()
	n2 := len(nl)
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
				call := call.(*ir.MethodExpr)
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
	for _, n := range nl {
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
	if len(nl) < 1 {
		return "()"
	}

	var typeStrings []string
	for i, n := range nl {
		isdddArg := isddd && i == len(nl)-1
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
func pushtype(nn ir.Node, t *types.Type) ir.Node {
	if nn == nil || nn.Op() != ir.OCOMPLIT {
		return nn
	}
	n := nn.(*ir.CompLitExpr)
	if n.Ntype != nil {
		return n
	}

	switch {
	case iscomptype(t):
		// For T, return T{...}.
		n.Ntype = ir.TypeNode(t)

	case t.IsPtr() && iscomptype(t.Elem()):
		// For *T, return &T{...}.
		n.Ntype = ir.TypeNode(t.Elem())

		addr := NodAddrAt(n.Pos(), n)
		addr.SetImplicit(true)
		return addr
	}
	return n
}

// The result of typecheckcomplit MUST be assigned back to n, e.g.
// 	n.Left = typecheckcomplit(n.Left)
func typecheckcomplit(n *ir.CompLitExpr) (res ir.Node) {
	if base.EnableTrace && base.Flag.LowerT {
		defer tracePrint("typecheckcomplit", n)(&res)
	}

	lno := base.Pos
	defer func() {
		base.Pos = lno
	}()

	if n.Ntype == nil {
		base.ErrorfAt(n.Pos(), "missing type in composite literal")
		n.SetType(nil)
		return n
	}

	// Save original node (including n.Right)
	n.SetOrig(ir.Copy(n))

	ir.SetPos(n.Ntype)

	// Need to handle [...]T arrays specially.
	if array, ok := n.Ntype.(*ir.ArrayType); ok && array.Elem != nil && array.Len == nil {
		array.Elem = check(array.Elem, ctxType)
		elemType := array.Elem.Type()
		if elemType == nil {
			n.SetType(nil)
			return n
		}
		length := typecheckarraylit(elemType, -1, n.List, "array literal")
		n.SetOp(ir.OARRAYLIT)
		n.SetType(types.NewArray(elemType, length))
		n.Ntype = nil
		return n
	}

	n.Ntype = ir.Node(check(n.Ntype, ctxType)).(ir.Ntype)
	t := n.Ntype.Type()
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
		typecheckarraylit(t.Elem(), t.NumElem(), n.List, "array literal")
		n.SetOp(ir.OARRAYLIT)
		n.Ntype = nil

	case types.TSLICE:
		length := typecheckarraylit(t.Elem(), -1, n.List, "slice literal")
		n.SetOp(ir.OSLICELIT)
		n.Ntype = nil
		n.Len = length

	case types.TMAP:
		var cs constSet
		for i3, l := range n.List {
			ir.SetPos(l)
			if l.Op() != ir.OKEY {
				n.List[i3] = Expr(l)
				base.Errorf("missing key in map literal")
				continue
			}
			l := l.(*ir.KeyExpr)

			r := l.Key
			r = pushtype(r, t.Key())
			r = Expr(r)
			l.Key = AssignConv(r, t.Key(), "map key")
			cs.add(base.Pos, l.Key, "key", "map literal")

			r = l.Value
			r = pushtype(r, t.Elem())
			r = Expr(r)
			l.Value = AssignConv(r, t.Elem(), "map value")
		}

		n.SetOp(ir.OMAPLIT)
		n.Ntype = nil

	case types.TSTRUCT:
		// Need valid field offsets for Xoffset below.
		types.CalcSize(t)

		errored := false
		if len(n.List) != 0 && nokeys(n.List) {
			// simple list of variables
			ls := n.List
			for i, n1 := range ls {
				ir.SetPos(n1)
				n1 = Expr(n1)
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
				n1 = AssignConv(n1, f.Type, "field value")
				sk := ir.NewStructKeyExpr(base.Pos, f.Sym, n1)
				sk.Offset = f.Offset
				ls[i] = sk
			}
			if len(ls) < t.NumFields() {
				base.Errorf("too few values in %v", n)
			}
		} else {
			hash := make(map[string]bool)

			// keyed list
			ls := n.List
			for i, l := range ls {
				ir.SetPos(l)

				if l.Op() == ir.OKEY {
					kv := l.(*ir.KeyExpr)
					key := kv.Key

					// Sym might have resolved to name in other top-level
					// package, because of import dot. Redirect to correct sym
					// before we do the lookup.
					s := key.Sym()
					if id, ok := key.(*ir.Ident); ok && DotImportRefs[id] != nil {
						s = Lookup(s.Name)
					}

					// An OXDOT uses the Sym field to hold
					// the field to the right of the dot,
					// so s will be non-nil, but an OXDOT
					// is never a valid struct literal key.
					if s == nil || s.Pkg != types.LocalPkg || key.Op() == ir.OXDOT || s.IsBlank() {
						base.Errorf("invalid field name %v in struct initializer", key)
						continue
					}

					l = ir.NewStructKeyExpr(l.Pos(), s, kv.Value)
					ls[i] = l
				}

				if l.Op() != ir.OSTRUCTKEY {
					if !errored {
						base.Errorf("mixture of field:value and value initializers")
						errored = true
					}
					ls[i] = Expr(ls[i])
					continue
				}
				l := l.(*ir.StructKeyExpr)

				f := lookdot1(nil, l.Field, t, t.Fields(), 0)
				if f == nil {
					if ci := lookdot1(nil, l.Field, t, t.Fields(), 2); ci != nil { // Case-insensitive lookup.
						if visible(ci.Sym) {
							base.Errorf("unknown field '%v' in struct literal of type %v (but does have %v)", l.Field, t, ci.Sym)
						} else if nonexported(l.Field) && l.Field.Name == ci.Sym.Name { // Ensure exactness before the suggestion.
							base.Errorf("cannot refer to unexported field '%v' in struct literal of type %v", l.Field, t)
						} else {
							base.Errorf("unknown field '%v' in struct literal of type %v", l.Field, t)
						}
						continue
					}
					var f *types.Field
					p, _ := dotpath(l.Field, t, &f, true)
					if p == nil || f.IsMethod() {
						base.Errorf("unknown field '%v' in struct literal of type %v", l.Field, t)
						continue
					}
					// dotpath returns the parent embedded types in reverse order.
					var ep []string
					for ei := len(p) - 1; ei >= 0; ei-- {
						ep = append(ep, p[ei].field.Sym.Name)
					}
					ep = append(ep, l.Field.Name)
					base.Errorf("cannot use promoted field %v in struct literal of type %v", strings.Join(ep, "."), t)
					continue
				}
				fielddup(f.Sym.Name, hash)
				l.Offset = f.Offset

				// No pushtype allowed here. Tried and rejected.
				l.Value = Expr(l.Value)
				l.Value = AssignConv(l.Value, f.Type, "field value")
			}
		}

		n.SetOp(ir.OSTRUCTLIT)
		n.Ntype = nil
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
		ir.SetPos(elt)
		r := elts[i]
		var kv *ir.KeyExpr
		if elt.Op() == ir.OKEY {
			elt := elt.(*ir.KeyExpr)
			elt.Key = Expr(elt.Key)
			key = IndexConst(elt.Key)
			if key < 0 {
				if !elt.Key.Diag() {
					if key == -2 {
						base.Errorf("index too large")
					} else {
						base.Errorf("index must be non-negative integer constant")
					}
					elt.Key.SetDiag(true)
				}
				key = -(1 << 30) // stay negative for a while
			}
			kv = elt
			r = elt.Value
		}

		r = pushtype(r, elemType)
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
	if !ir.IsAssignable(n) {
		base.Errorf("cannot %s %v", verb, n)
	}
}

func checkassign(stmt ir.Node, n ir.Node) {
	// Variables declared in ORANGE are assigned on every iteration.
	if !ir.DeclaredBy(n, stmt) || stmt.Op() == ir.ORANGE {
		r := ir.OuterValue(n)
		if r.Op() == ir.ONAME {
			r := r.(*ir.Name)
			r.Name().SetAssigned(true)
			if r.Name().IsClosureVar() {
				r.Name().Defn.Name().SetAssigned(true)
			}
		}
	}

	if ir.IsAssignable(n) {
		return
	}
	if n.Op() == ir.OINDEXMAP {
		n := n.(*ir.IndexExpr)
		n.Assigned = true
		return
	}

	// have already complained about n being invalid
	if n.Type() == nil {
		return
	}

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
	n.SetType(nil)
}

func checkassignlist(stmt ir.Node, l ir.Nodes) {
	for _, n := range l {
		checkassign(stmt, n)
	}
}

// type check assignment.
// if this assignment is the definition of a var on the left side,
// fill in the var's type.
func typecheckas(n *ir.AssignStmt) {
	if base.EnableTrace && base.Flag.LowerT {
		defer tracePrint("typecheckas", n)(nil)
	}

	// delicate little dance.
	// the definition of n may refer to this assignment
	// as its definition, in which case it will call typecheckas.
	// in that case, do not call typecheck back, or it will cycle.
	// if the variable has a type (ntype) then typechecking
	// will not look at defn, so it is okay (and desirable,
	// so that the conversion below happens).
	n.X = Resolve(n.X)

	if !ir.DeclaredBy(n.X, n) || n.X.Name().Ntype != nil {
		n.X = AssignExpr(n.X)
	}

	// Use ctxMultiOK so we can emit an "N variables but M values" error
	// to be consistent with typecheckas2 (#26616).
	n.Y = check(n.Y, ctxExpr|ctxMultiOK)
	checkassign(n, n.X)
	if n.Y != nil && n.Y.Type() != nil {
		if n.Y.Type().IsFuncArgStruct() {
			base.Errorf("assignment mismatch: 1 variable but %v returns %d values", n.Y.(*ir.CallExpr).X, n.Y.Type().NumFields())
			// Multi-value RHS isn't actually valid for OAS; nil out
			// to indicate failed typechecking.
			n.Y.SetType(nil)
		} else if n.X.Type() != nil {
			n.Y = AssignConv(n.Y, n.X.Type(), "assignment")
		}
	}

	if ir.DeclaredBy(n.X, n) && n.X.Name().Ntype == nil {
		n.Y = DefaultLit(n.Y, nil)
		n.X.SetType(n.Y.Type())
	}

	// second half of dance.
	// now that right is done, typecheck the left
	// just to get it over with.  see dance above.
	n.SetTypecheck(1)

	if n.X.Typecheck() == 0 {
		n.X = AssignExpr(n.X)
	}
	if !ir.IsBlank(n.X) {
		types.CheckSize(n.X.Type()) // ensure width is calculated for backend
	}
}

func checkassignto(src *types.Type, dst ir.Node) {
	if op, why := assignop(src, dst.Type()); op == ir.OXXX {
		base.Errorf("cannot assign %v to %L in multiple assignment%s", src, dst, why)
		return
	}
}

func typecheckas2(n *ir.AssignListStmt) {
	if base.EnableTrace && base.Flag.LowerT {
		defer tracePrint("typecheckas2", n)(nil)
	}

	ls := n.Lhs
	for i1, n1 := range ls {
		// delicate little dance.
		n1 = Resolve(n1)
		ls[i1] = n1

		if !ir.DeclaredBy(n1, n) || n1.Name().Ntype != nil {
			ls[i1] = AssignExpr(ls[i1])
		}
	}

	cl := len(n.Lhs)
	cr := len(n.Rhs)
	if cl > 1 && cr == 1 {
		n.Rhs[0] = check(n.Rhs[0], ctxExpr|ctxMultiOK)
	} else {
		Exprs(n.Rhs)
	}
	checkassignlist(n, n.Lhs)

	var l ir.Node
	var r ir.Node
	if cl == cr {
		// easy
		ls := n.Lhs
		rs := n.Rhs
		for il, nl := range ls {
			nr := rs[il]
			if nl.Type() != nil && nr.Type() != nil {
				rs[il] = AssignConv(nr, nl.Type(), "assignment")
			}
			if ir.DeclaredBy(nl, n) && nl.Name().Ntype == nil {
				rs[il] = DefaultLit(rs[il], nil)
				nl.SetType(rs[il].Type())
			}
		}

		goto out
	}

	l = n.Lhs[0]
	r = n.Rhs[0]

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
			for i, l := range n.Lhs {
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
				r := r.(*ir.TypeAssertExpr)
				n.SetOp(ir.OAS2DOTTYPE)
				r.SetOp(ir.ODOTTYPE2)
			}
			if l.Type() != nil {
				checkassignto(r.Type(), l)
			}
			if ir.DeclaredBy(l, n) {
				l.SetType(r.Type())
			}
			l := n.Lhs[1]
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
		r := r.(*ir.CallExpr)
		base.Errorf("assignment mismatch: %d variables but %v returns %d values", cl, r.X, cr)
	}

	// second half of dance
out:
	n.SetTypecheck(1)
	ls = n.Lhs
	for i1, n1 := range ls {
		if n1.Typecheck() == 0 {
			ls[i1] = AssignExpr(ls[i1])
		}
	}
}

// type check function definition
// To be called by typecheck, not directly.
// (Call typecheckFunc instead.)
func typecheckfunc(n *ir.Func) {
	if base.EnableTrace && base.Flag.LowerT {
		defer tracePrint("typecheckfunc", n)(nil)
	}

	for _, ln := range n.Dcl {
		if ln.Op() == ir.ONAME && (ln.Class_ == ir.PPARAM || ln.Class_ == ir.PPARAMOUT) {
			ln.Decldepth = 1
		}
	}

	n.Nname = AssignExpr(n.Nname).(*ir.Name)
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

		n.Nname.SetSym(ir.MethodSym(rcvr.Type, n.Shortname))
		Declare(n.Nname, ir.PFUNC)
	}

	if base.Ctxt.Flag_dynlink && !inimport && n.Nname != nil {
		NeedFuncSym(n.Sym())
	}
}

// The result of stringtoruneslit MUST be assigned back to n, e.g.
// 	n.Left = stringtoruneslit(n.Left)
func stringtoruneslit(n *ir.ConvExpr) ir.Node {
	if n.X.Op() != ir.OLITERAL || n.X.Val().Kind() != constant.String {
		base.Fatalf("stringtoarraylit %v", n)
	}

	var l []ir.Node
	i := 0
	for _, r := range ir.StringVal(n.X) {
		l = append(l, ir.NewKeyExpr(base.Pos, ir.NewInt(int64(i)), ir.NewInt(int64(r))))
		i++
	}

	nn := ir.NewCompLitExpr(base.Pos, ir.OCOMPLIT, ir.TypeNode(n.Type()).(ir.Ntype), nil)
	nn.List.Set(l)
	return Expr(nn)
}

var mapqueue []*ir.MapType

func CheckMapKeys() {
	for _, n := range mapqueue {
		k := n.Type().MapType().Key
		if !k.Broke() && !types.IsComparable(k) {
			base.ErrorfAt(n.Pos(), "invalid map key type %v", k)
		}
	}
	mapqueue = nil
}

func typecheckdeftype(n *ir.Name) {
	if base.EnableTrace && base.Flag.LowerT {
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

	types.DeferCheckSize()
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
	types.ResumeCheckSize()
}

func typecheckdef(n ir.Node) {
	if base.EnableTrace && base.Flag.LowerT {
		defer tracePrint("typecheckdef", n)(nil)
	}

	lno := ir.SetPos(n)

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

		e = Expr(e)
		if e.Type() == nil {
			goto ret
		}
		if !ir.IsConstNode(e) {
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
		n := n.(*ir.Name)
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
			if n.BuiltinOp != 0 { // like OPRINTN
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
			n.Name().Defn = Expr(n.Name().Defn)
			n.SetType(n.Name().Defn.Type())
			break
		}

		n.Name().Defn = Stmt(n.Name().Defn) // fills in n.Type

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
		if ir.ConstOverflow(v, types.Types[types.TINT]) {
			base.Errorf("%s argument too large in make(%v)", arg, t)
			return false
		}
	}

	// defaultlit is necessary for non-constants too: n might be 1.1<<k.
	// TODO(gri) The length argument requirements for (array/slice) make
	// are the same as for index expressions. Factor the code better;
	// for instance, indexlit might be called here and incorporate some
	// of the bounds checks done for make.
	n = DefaultLit(n, types.Types[types.TINT])
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
			n := n.(*ir.BranchStmt)
			if n.Label == nil {
				setHasBreak(implicit)
			} else {
				setHasBreak(labels[n.Label])
			}

		case ir.OFOR, ir.OFORUNTIL, ir.OSWITCH, ir.OSELECT, ir.ORANGE:
			old := implicit
			implicit = n
			var sym *types.Sym
			switch n := n.(type) {
			case *ir.ForStmt:
				sym = n.Label
			case *ir.RangeStmt:
				sym = n.Label
			case *ir.SelectStmt:
				sym = n.Label
			case *ir.SwitchStmt:
				sym = n.Label
			}
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

func controlLabel(n ir.Node) *types.Sym {
	switch n := n.(type) {
	default:
		base.Fatalf("controlLabel %+v", n.Op())
		return nil
	case *ir.ForStmt:
		return n.Label
	case *ir.RangeStmt:
		return n.Label
	case *ir.SelectStmt:
		return n.Label
	case *ir.SwitchStmt:
		return n.Label
	}
}

func setHasBreak(n ir.Node) {
	switch n := n.(type) {
	default:
		base.Fatalf("setHasBreak %+v", n.Op())
	case nil:
		// ignore
	case *ir.ForStmt:
		n.HasBreak = true
	case *ir.RangeStmt:
		n.HasBreak = true
	case *ir.SelectStmt:
		n.HasBreak = true
	case *ir.SwitchStmt:
		n.HasBreak = true
	}
}

// isTermNodes reports whether the Nodes list ends with a terminating statement.
func isTermNodes(l ir.Nodes) bool {
	s := l
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
		n := n.(*ir.BlockStmt)
		return isTermNodes(n.List)

	case ir.OGOTO, ir.ORETURN, ir.ORETJMP, ir.OPANIC, ir.OFALL:
		return true

	case ir.OFOR, ir.OFORUNTIL:
		n := n.(*ir.ForStmt)
		if n.Cond != nil {
			return false
		}
		if n.HasBreak {
			return false
		}
		return true

	case ir.OIF:
		n := n.(*ir.IfStmt)
		return isTermNodes(n.Body) && isTermNodes(n.Else)

	case ir.OSWITCH:
		n := n.(*ir.SwitchStmt)
		if n.HasBreak {
			return false
		}
		def := false
		for _, cas := range n.Cases {
			cas := cas.(*ir.CaseStmt)
			if !isTermNodes(cas.Body) {
				return false
			}
			if len(cas.List) == 0 { // default
				def = true
			}
		}
		return def

	case ir.OSELECT:
		n := n.(*ir.SelectStmt)
		if n.HasBreak {
			return false
		}
		for _, cas := range n.Cases {
			cas := cas.(*ir.CaseStmt)
			if !isTermNodes(cas.Body) {
				return false
			}
		}
		return true
	}

	return false
}

// CheckReturn makes sure that fn terminates appropriately.
func CheckReturn(fn *ir.Func) {
	if fn.Type().NumResults() != 0 && len(fn.Body) != 0 {
		markBreak(fn)
		if !isTermNodes(fn.Body) {
			base.ErrorfAt(fn.Endlineno, "missing return at end of function")
		}
	}
}

func deadcode(fn *ir.Func) {
	deadcodeslice(&fn.Body)

	if len(fn.Body) == 0 {
		return
	}

	for _, n := range fn.Body {
		if len(n.Init()) > 0 {
			return
		}
		switch n.Op() {
		case ir.OIF:
			n := n.(*ir.IfStmt)
			if !ir.IsConst(n.Cond, constant.Bool) || len(n.Body) > 0 || len(n.Else) > 0 {
				return
			}
		case ir.OFOR:
			n := n.(*ir.ForStmt)
			if !ir.IsConst(n.Cond, constant.Bool) || ir.BoolVal(n.Cond) {
				return
			}
		default:
			return
		}
	}

	fn.Body.Set([]ir.Node{ir.NewBlockStmt(base.Pos, nil)})
}

func deadcodeslice(nn *ir.Nodes) {
	var lastLabel = -1
	for i, n := range *nn {
		if n != nil && n.Op() == ir.OLABEL {
			lastLabel = i
		}
	}
	for i, n := range *nn {
		// Cut is set to true when all nodes after i'th position
		// should be removed.
		// In other words, it marks whole slice "tail" as dead.
		cut := false
		if n == nil {
			continue
		}
		if n.Op() == ir.OIF {
			n := n.(*ir.IfStmt)
			n.Cond = deadcodeexpr(n.Cond)
			if ir.IsConst(n.Cond, constant.Bool) {
				var body ir.Nodes
				if ir.BoolVal(n.Cond) {
					n.Else = ir.Nodes{}
					body = n.Body
				} else {
					n.Body = ir.Nodes{}
					body = n.Else
				}
				// If "then" or "else" branch ends with panic or return statement,
				// it is safe to remove all statements after this node.
				// isterminating is not used to avoid goto-related complications.
				// We must be careful not to deadcode-remove labels, as they
				// might be the target of a goto. See issue 28616.
				if body := body; len(body) != 0 {
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
			n := n.(*ir.BlockStmt)
			deadcodeslice(&n.List)
		case ir.OCASE:
			n := n.(*ir.CaseStmt)
			deadcodeslice(&n.Body)
		case ir.OFOR:
			n := n.(*ir.ForStmt)
			deadcodeslice(&n.Body)
		case ir.OIF:
			n := n.(*ir.IfStmt)
			deadcodeslice(&n.Body)
			deadcodeslice(&n.Else)
		case ir.ORANGE:
			n := n.(*ir.RangeStmt)
			deadcodeslice(&n.Body)
		case ir.OSELECT:
			n := n.(*ir.SelectStmt)
			deadcodeslice(&n.Cases)
		case ir.OSWITCH:
			n := n.(*ir.SwitchStmt)
			deadcodeslice(&n.Cases)
		}

		if cut {
			nn.Set((*nn)[:i+1])
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
		n := n.(*ir.LogicalExpr)
		n.X = deadcodeexpr(n.X)
		n.Y = deadcodeexpr(n.Y)
		if ir.IsConst(n.X, constant.Bool) {
			if ir.BoolVal(n.X) {
				return n.Y // true && x => x
			} else {
				return n.X // false && x => false
			}
		}
	case ir.OOROR:
		n := n.(*ir.LogicalExpr)
		n.X = deadcodeexpr(n.X)
		n.Y = deadcodeexpr(n.Y)
		if ir.IsConst(n.X, constant.Bool) {
			if ir.BoolVal(n.X) {
				return n.X // true || x => true
			} else {
				return n.Y // false || x => x
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
			return x.(*ir.Name).Iota()
		}
	}

	if ir.CurFunc != nil && ir.CurFunc.Iota >= 0 {
		return ir.CurFunc.Iota
	}

	return -1
}

// curpkg returns the current package, based on Curfn.
func curpkg() *types.Pkg {
	fn := ir.CurFunc
	if fn == nil {
		// Initialization expressions for package-scope variables.
		return types.LocalPkg
	}
	return fnpkg(fn.Nname)
}

func Conv(n ir.Node, t *types.Type) ir.Node {
	if types.Identical(n.Type(), t) {
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
	if types.Identical(n.Type(), t) {
		return n
	}
	n = ir.NewConvExpr(base.Pos, ir.OCONVNOP, nil, n)
	n.SetType(t)
	n = Expr(n)
	return n
}
