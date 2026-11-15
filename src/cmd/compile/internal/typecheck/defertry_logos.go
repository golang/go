// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typecheck

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"internal/logosfile"
	"strings"
)

func isLogosDeferTry(n *ir.GoDeferStmt) bool {
	if n.Op() != ir.ODEFER {
		return false
	}
	call, ok := n.Call.(*ir.CallExpr)
	if !ok || call.Op() != ir.OCALLFUNC {
		return false
	}
	callee := ir.StaticCalleeName(call.Fun)
	if callee == nil {
		return false
	}
	sym := callee.Sym()
	if sym == nil || sym.Pkg == nil || sym.Pkg.Path != "logos/rt" {
		return false
	}
	name := sym.Name
	if i := strings.IndexByte(name, '['); i >= 0 {
		name = name[:i]
	}
	if name != "DeferTry" {
		return false
	}
	if len(call.Args) != 2 {
		return false
	}
	return logosfile.Is(base.Ctxt.OutermostPos(n.Pos()).RelFilename())
}

func normalizeDeferTryCall(n *ir.GoDeferStmt, outer ir.Node, init *ir.Nodes) bool {
	init.Append(ir.TakeInit(outer)...)
	ocall, ok := outer.(*ir.CallExpr)
	if !ok || ocall.Op() != ir.OCALLFUNC || len(ocall.Args) != 2 {
		return false
	}
	errAddr, ok := ocall.Args[0].(*ir.AddrExpr)
	if !ok {
		return false
	}
	errName, ok := errAddr.X.(*ir.Name)
	if !ok {
		return false
	}
	inner, ok := ocall.Args[1].(*ir.CallExpr)
	if !ok {
		return false
	}
	if inner.Op() != ir.OCALLFUNC && inner.Op() != ir.OCALLINTER {
		return false
	}
	if inner.Type() == nil || inner.Type() != types.ErrorType {
		return false
	}
	pos := n.Pos()
	wrapperFn := ir.NewClosureFunc(pos, pos, ir.ODEFER, types.NewSignature(nil, nil, nil), ir.CurFunc, Target, 0)
	wrapperFn.DeclareParams(true)
	wrapperFn.SetWrapper(false)
	spillGoDeferOperands(wrapperFn, inner, init)
	cv := ir.NewClosureVar(pos, wrapperFn, errName)
	e := TempAt(pos, wrapperFn, inner.Type())
	as := ir.NewAssignListStmt(pos, ir.OAS2, []ir.Node{e}, []ir.Node{inner})
	e.Defn = as
	cond := ir.NewBinaryExpr(pos, ir.ONE, e, ir.NewNilExpr(pos, inner.Type()))
	join := Call(pos, ocall.Fun, []ir.Node{ir.NewAddrExpr(pos, cv), e}, false)
	wrapperFn.Body = []ir.Node{Stmt(ir.NewDecl(pos, ir.ODCL, e)), Stmt(as), Stmt(ir.NewIfStmt(pos, cond, []ir.Node{join}, nil))}
	wcall := Call(pos, wrapperFn.OClosure, nil, false).(*ir.CallExpr)
	wcall.GoDefer = true
	n.Call = wcall
	if errName.Class == ir.PPARAMOUT && errName.Curfn == ir.CurFunc {
		n.DeferTry = errName
	}
	return true
}

func spillGoDeferOperands(wrapperFn *ir.Func, call ir.Node, init *ir.Nodes) {
	var argps []*ir.Node

	var visit func(argp *ir.Node)
	visit = func(argp *ir.Node) {
		arg := *argp
		if arg == nil {
			return
		}
		switch arg.Op() {
		case ir.OLITERAL, ir.ONIL, ir.OMETHEXPR, ir.ONEW:
			return
		case ir.ONAME:
			arg := arg.(*ir.Name)
			if arg.Class == ir.PFUNC {
				return
			}
		case ir.OADDR:
			arg := arg.(*ir.AddrExpr)
			if arg.X.Op() == ir.OLINKSYMOFFSET {
				return
			}

		case ir.OCONVNOP:
			arg := arg.(*ir.ConvExpr)
			if arg.Type().IsUintptr() && arg.X.Type().IsUnsafePtr() {
				visit(&arg.X)
				return
			}

		case ir.OARRAYLIT, ir.OSLICELIT, ir.OSTRUCTLIT:
			list := arg.(*ir.CompLitExpr).List
			for i, el := range list {
				switch el := el.(type) {
				case *ir.KeyExpr:
					visit(&el.Value)
				case *ir.StructKeyExpr:
					visit(&el.Value)
				default:
					visit(&list[i])
				}
			}
			return
		}

		argps = append(argps, argp)
	}

	visitList := func(list []ir.Node) {
		for i := range list {
			visit(&list[i])
		}
	}

	switch call.Op() {
	default:
		base.Fatalf("unexpected call op: %v", call.Op())

	case ir.OCALLFUNC:
		call := call.(*ir.CallExpr)
		if wrapped := ir.StaticCalleeName(call.Fun); wrapped != nil {
			wrapperFn.WrappedFunc = wrapped.Func
		}

		visit(&call.Fun)
		visitList(call.Args)

	case ir.OCALLINTER:
		call := call.(*ir.CallExpr)
		argps = append(argps, &call.Fun.(*ir.SelectorExpr).X)
		visitList(call.Args)

	case ir.OAPPEND, ir.ODELETE, ir.OPRINT, ir.OPRINTLN, ir.ORECOVER:
		call := call.(*ir.CallExpr)
		visitList(call.Args)
		visit(&call.RType)

	case ir.OCOPY:
		call := call.(*ir.BinaryExpr)
		visit(&call.X)
		visit(&call.Y)
		visit(&call.RType)

	case ir.OCLEAR, ir.OCLOSE, ir.OPANIC:
		call := call.(*ir.UnaryExpr)
		visit(&call.X)
	}

	if len(argps) != 0 {
		stmtPos := base.Pos
		callPos := base.Pos

		as := ir.NewAssignListStmt(callPos, ir.OAS2, make([]ir.Node, len(argps)), make([]ir.Node, len(argps)))
		for i, argp := range argps {
			arg := *argp

			pos := callPos
			if ir.HasUniquePos(arg) {
				pos = arg.Pos()
			}

			tmp := TempAt(pos, ir.CurFunc, arg.Type())
			init.Append(Stmt(ir.NewDecl(pos, ir.ODCL, tmp)))
			tmp.Defn = as
			as.Lhs[i] = tmp
			as.Rhs[i] = arg

			*argp = ir.NewClosureVar(pos, wrapperFn, tmp)
		}
		init.Append(Stmt(as))

		if call.Op() == ir.OCALLINTER {
			iface := as.Lhs[0]
			init.Append(Stmt(ir.NewUnaryExpr(stmtPos, ir.OCHECKNIL, ir.NewUnaryExpr(iface.Pos(), ir.OITAB, iface))))
		}
	}
}
