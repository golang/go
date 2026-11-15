// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

func (check *Checker) logosGeneratorSig(sig *Signature) bool {
	_, ok := check.logosGeneratorArgs(sig)
	return ok
}

func (check *Checker) logosGeneratorArgs(sig *Signature) (targs []Type, ok bool) {
	if !check.conf.Logos || sig == nil {
		return nil, false
	}
	res := sig.results
	if res.Len() != 1 {
		return nil, false
	}
	named, isNamed := Unalias(res.vars[0].typ).(*Named)
	if !isNamed {
		return nil, false
	}
	obj := named.Obj()
	if obj == nil || obj.Pkg() == nil || obj.Pkg().Path() != "iter" {
		return nil, false
	}
	args := named.TypeArgs()
	switch obj.Name() {
	case "Seq":
		if args.Len() != 1 {
			return nil, false
		}
		return []Type{args.At(0)}, true
	case "Seq2":
		if args.Len() != 2 {
			return nil, false
		}
		return []Type{args.At(0), args.At(1)}, true
	}
	return nil, false
}

func (check *Checker) logosGeneratorReturn(s *syntax.ReturnStmt) bool {
	if !check.conf.Logos || !check.logosGeneratorSig(check.sig) {
		return false
	}
	for _, result := range syntax.UnpackListExpr(s.Results) {
		check.use(result)
	}
	return true
}

func (check *Checker) logosYield(s *syntax.YieldStmt) {
	if !check.conf.Logos {
		check.error(s, InvalidSyntaxTree, "yield requires logos mode")
		return
	}
	targs, ok := check.logosGeneratorArgs(check.sig)
	if !ok {
		check.error(s, WrongResultCount, "[correctness::yield_outside_generator] yield requires enclosing function to return iter.Seq[T] or iter.Seq2[K, V]")
		check.use(s.Value)
		return
	}
	vals := syntax.UnpackListExpr(s.Value)
	if len(vals) != len(targs) {
		if len(targs) == 1 {
			check.error(s, WrongArgCount, "yield requires 1 value for iter.Seq[T]")
		} else {
			check.error(s, WrongArgCount, "yield requires 2 values for iter.Seq2[K, V]")
		}
		for _, v := range vals {
			check.use(v)
		}
		return
	}
	for i, v := range vals {
		var x operand
		if !check.logosLambdaDefer(&x, v) {
			check.expr(nil, &x, v)
		}
		if !x.isValid() {
			continue
		}
		check.assignment(&x, targs[i], "yield")
	}
}

func (check *Checker) logosYieldForm(s syntax.Stmt) syntax.Stmt {
	if !check.conf.Logos || !check.logosGeneratorSig(check.sig) {
		return s
	}
	switch s := s.(type) {
	case *syntax.ExprStmt:
		if value, word, ok := syntax.YieldOperand(s.X); ok {
			return syntax.NewYieldStmt(word, value)
		}
		return check.logosYieldReserved(s, s.X, s.X)
	case *syntax.SendStmt:
		if value, word, ok := syntax.YieldSendOperand(s); ok {
			return syntax.NewYieldStmt(word, value)
		}
		return check.logosYieldReserved(s, s.Chan, s.Chan, s.Value)
	case *syntax.AssignStmt:
		return check.logosYieldReserved(s, syntax.UnpackListExpr(s.Lhs)[0], s.Lhs, s.Rhs)
	}
	return s
}

func (check *Checker) logosYieldReserved(s syntax.Stmt, lead syntax.Expr, operands ...syntax.Expr) syntax.Stmt {
	word, reserved := syntax.YieldReserved(lead)
	if !reserved {
		return s
	}
	check.error(word, InvalidSyntaxTree, "yield followed by [ or ( starts a yield statement in a generator, but its operand is not an expression")
	check.use(operands...)
	empty := new(syntax.EmptyStmt)
	empty.SetPos(s.Pos())
	return empty
}
