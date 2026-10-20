// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

// LoGo/s M6-S9 type rules for `yield` generators (T-104.8). stmt.go
// calls in through one hook line; the lowering lives in logos/yield.go.
//
// A `yield` statement is allowed only directly inside a function whose
// single result is `iter.Seq[T]` or `iter.Seq2[K, V]` (matched
// semantically, so spellings through aliases and dot imports count).
// The check runs on the innermost enclosing function, so a `yield`
// inside a nested function literal reports yield_outside_generator like
// any other non-generator. Values are checked assignable to the element
// types: one value for Seq, two for Seq2.

// logosGeneratorSig reports whether sig is a generator signature: one
// result of type `iter.Seq[T]` or `iter.Seq2[K, V]`.
func (check *Checker) logosGeneratorSig(sig *Signature) bool {
	_, ok := check.logosGeneratorArgs(sig)
	return ok
}

// logosGeneratorArgs reports the element types of a generator signature:
// one type for `iter.Seq[T]`, two for `iter.Seq2[K, V]`.
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

// logosYield checks a yield statement: generator position and value
// types. A use outside a generator reports yield_outside_generator once
// and still marks the value uses, so no follow-on errors surface.
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
		check.expr(nil, &x, v)
		if !x.isValid() {
			continue
		}
		check.assignment(&x, targs[i], "yield")
	}
}
