// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package walk

import (
	"go/constant"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

func pooledTryDeferClosure(n *ir.GoDeferStmt) *ir.ClosureExpr {
	if n.DeferTry == nil || n.DeferAt != nil || n.Esc() == ir.EscNever {
		return nil
	}
	call, ok := n.Call.(*ir.CallExpr)
	if !ok {
		return nil
	}
	clo, ok := call.Fun.(*ir.ClosureExpr)
	if !ok || clo.DeferTryErr == nil {
		return nil
	}
	return clo
}

func pooledTryErrOffset(clo *ir.ClosureExpr) int64 {
	it := typecheck.NewClosureStructIter(clo.Func.ClosureVars)
	for {
		n, _, offset := it.Next()
		if n == nil {
			base.FatalfAt(clo.Pos(), "pooled try defer does not capture its error result %v", clo.DeferTryErr)
		}
		if n.Canonical() == clo.DeferTryErr {
			return offset
		}
	}
}

func walkPooledTryDefer(n *ir.GoDeferStmt, clo *ir.ClosureExpr) ir.Node {
	pos := n.Pos()
	ctype := typecheck.ClosureType(clo)
	roff := pooledTryErrOffset(clo)
	thunk := genPooledTryThunk(clo.Func, ctype, roff, pos)

	var init ir.Nodes
	init.Append(ir.TakeInit(n)...)
	init.Append(mkcallstmt("deferprocTry",
		ir.NewUnaryExpr(pos, ir.OCFUNC, thunk.Nname),
		reflectdata.TypePtrAt(pos, ctype),
		typecheck.ConvNop(clo, types.Types[types.TUNSAFEPTR]),
		offsetConst(pos, roff)))
	if base.Debug.Defer > 0 {
		base.WarnfAt(pos, "pooled try defer")
	}
	return ir.NewBlockStmt(pos, init)
}

func offsetConst(pos src.XPos, off int64) ir.Node {
	return ir.NewBasicLit(pos, types.Types[types.TUINTPTR], constant.MakeInt64(off))
}

func genPooledTryThunk(wrapped *ir.Func, ctype *types.Type, roff int64, pos src.XPos) *ir.Func {
	sym := wrapped.Sym().Pkg.Lookup(wrapped.Sym().Name + ".try")
	fn := ir.NewFunc(pos, pos, sym, types.NewSignature(nil, nil, nil))
	fn.SetDupok(wrapped.Dupok())
	fn.SetWrapper(true)
	fn.SetNeedctxt(true)
	fn.SetDeferTryThunk(true)
	fn.Pragma |= ir.Norace

	typecheck.DeclFunc(fn)
	rec := typecheck.TempAt(pos, fn, types.Types[types.TUNSAFEPTR])
	record := typecheck.TempAt(pos, fn, ctype)
	cloPtr := func() ir.Node {
		return typecheck.NodAddrAt(pos, record)
	}
	unpack := ir.NewCallExpr(pos, ir.OCALL, typecheck.LookupRuntime("deferTryUnpack"), []ir.Node{
		rec,
		reflectdata.TypePtrAt(pos, ctype),
		typecheck.ConvNop(cloPtr(), types.Types[types.TUNSAFEPTR]),
		offsetConst(pos, roff),
	})
	call := ir.NewCallExpr(pos, ir.OCALL, typecheck.ConvNop(cloPtr(), types.NewSignature(nil, nil, nil)), nil)
	fn.Body = []ir.Node{ir.NewAssignStmt(pos, record, nil), unpack, call}
	typecheck.FinishFuncBody()

	ir.WithFunc(fn, func() {
		typecheck.Stmts(fn.Body)
	})
	return fn
}
