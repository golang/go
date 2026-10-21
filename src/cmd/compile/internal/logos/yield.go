// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// This file lowers M6-S9 `yield` generators (T-104.8): a function whose
// single result is `iter.Seq[T]` or `iter.Seq2[K, V]` becomes a function
// that returns an iterator closure over its body:
//
//	func Count(n int) iter.Seq[int] {
//		return func(_logos_yield func(int) bool) {
//			for i := range n {
//				if !_logos_yield(i) {
//					return
//				}
//			}
//		}
//	}
//
// Each `yield e` becomes `if !_logos_yield(e) { return }`, so an early
// `break` in the consumer (which makes the range body return false)
// stops the generator with no further body running. A `return` with
// values inside a generator becomes a bare `return`: the value is
// discarded (the function always returns the sequence), so explicit
// `return nil` and a bare `return` both mean stop.
//
// Element spellings come from the checked type arguments, so aliases of
// `iter.Seq` lower as well as the direct spelling. The pass runs before
// statement lowering; nested function literals are left for their own
// funcLower (a `yield` there is a types2 error and never reaches this
// pass from valid code).

// yieldParam is the name of the iterator closure's yield function. User
// code cannot collide with it: identifiers starting with `_logos_` are
// reserved in `.lgo` files (C6).
const yieldParam = "_logos_yield"

// generatorTarget reports the element types of fc's function when its
// single result is `iter.Seq[T]` or `iter.Seq2[K, V]`, matched
// semantically like the types2 generator check.
func (fc *funcLower) generatorTarget() ([]types2.Type, bool) {
	if len(fc.ft.ResultList) != 1 {
		return nil, false
	}
	typ := fc.fl.pkg.info.TypeOf(fc.ft.ResultList[0].Type)
	named, ok := types2.Unalias(typ).(*types2.Named)
	if !ok {
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
		return []types2.Type{args.At(0)}, true
	case "Seq2":
		if args.Len() != 2 {
			return nil, false
		}
		return []types2.Type{args.At(0), args.At(1)}, true
	}
	return nil, false
}

// yieldQualifier spells type arguments for the eject text: the current
// package needs no qualifier, every other package prints with the name
// that denotes it in the file (aliases and dot imports included).
func (fc *funcLower) yieldQualifier(p *types2.Package) string {
	if p == fc.fl.pkg.tpkg {
		return ""
	}
	if name, ok := fc.fl.importNames()[p]; ok {
		return name
	}
	return p.Name()
}

// lowerYield wraps a generator body in the iterator closure. It reports
// whether the function was a generator.
func (fc *funcLower) lowerYield() bool {
	targs, ok := fc.generatorTarget()
	if !ok {
		return false
	}
	fl := fc.fl
	pos := fc.body.Pos()
	spellings := make([]string, len(targs))
	natives := make([]syntax.Expr, len(targs))
	for i, targ := range targs {
		spellings[i] = types2.TypeString(targ, fc.yieldQualifier)
		decl, isType := reparseType(spellings[i])
		if !isType {
			base.Fatalf("logos: cannot respell yield type %q", spellings[i])
		}
		syntax.Inspect(decl, func(n syntax.Node) bool {
			if n != nil {
				n.SetPos(pos)
			}
			return true
		})
		natives[i] = decl
	}
	fc.rewriteYields(natives)
	params := make([]*syntax.Field, len(natives))
	for i, typ := range natives {
		params[i] = at(&syntax.Field{Type: typ}, pos)
	}
	yieldField := at(&syntax.Field{
		Name: nameOf(pos, yieldParam),
		Type: at(&syntax.FuncType{
			ParamList:  params,
			ResultList: []*syntax.Field{at(&syntax.Field{Type: nameOf(pos, "bool")}, pos)},
		}, pos),
	}, pos)
	lit := at(&syntax.FuncLit{
		Type: at(&syntax.FuncType{
			ParamList: []*syntax.Field{yieldField},
		}, pos),
		Body: blockOf(pos, fc.body.List),
	}, pos)
	fc.body.List = []syntax.Stmt{returnOf(pos, []syntax.Expr{lit})}
	braceOff := fl.off(pos)
	fl.add(braceOff+1, braceOff+1, " return func("+yieldParam+" func("+strings.Join(spellings, ", ")+") bool) {")
	rbraceOff := fl.off(fc.body.Rbrace)
	fl.add(rbraceOff, rbraceOff, " }")
	return true
}

// rewriteYields replaces every yield and valued return of the generator
// body with its lowered form, descending everywhere except nested
// function literals (which lower on their own).
func (fc *funcLower) rewriteYields(types []syntax.Expr) {
	fc.body.List = fc.rewriteYieldList(fc.body.List)
}

// rewriteYieldList rewrites one statement list.
func (fc *funcLower) rewriteYieldList(list []syntax.Stmt) []syntax.Stmt {
	for i, s := range list {
		list[i] = fc.rewriteYieldStmt(s)
	}
	return list
}

// rewriteYieldStmt rewrites one statement, returning its replacement.
func (fc *funcLower) rewriteYieldStmt(s syntax.Stmt) syntax.Stmt {
	switch s := s.(type) {
	case *syntax.YieldStmt:
		return fc.lowerYieldStmt(s)
	case *syntax.ReturnStmt:
		if len(syntax.UnpackListExpr(s.Results)) > 0 {
			fc.dropReturnValues(s)
			s.Results = nil
		}
		return s
	case *syntax.BlockStmt:
		s.List = fc.rewriteYieldList(s.List)
		return s
	case *syntax.IfStmt:
		s.Then.List = fc.rewriteYieldList(s.Then.List)
		switch els := s.Else.(type) {
		case *syntax.BlockStmt:
			els.List = fc.rewriteYieldList(els.List)
		case *syntax.IfStmt:
			s.Else = fc.rewriteYieldStmt(els)
		}
		return s
	case *syntax.ForStmt:
		s.Body.List = fc.rewriteYieldList(s.Body.List)
		return s
	case *syntax.SwitchStmt:
		for _, clause := range s.Body {
			clause.Body = fc.rewriteYieldList(clause.Body)
		}
		return s
	case *syntax.SelectStmt:
		for _, clause := range s.Body {
			clause.Body = fc.rewriteYieldList(clause.Body)
		}
		return s
	case *syntax.LabeledStmt:
		s.Stmt = fc.rewriteYieldStmt(s.Stmt)
		return s
	case *syntax.GroupStmt:
		return fc.yieldGroupStmt(s) // M8-S2 hook: yields stay synchronous in group bodies.
	default:
		return s
	}
}

// lowerYieldStmt rewrites `yield e` to `if !_logos_yield(e) { return }`.
func (fc *funcLower) lowerYieldStmt(s *syntax.YieldStmt) syntax.Stmt {
	fl := fc.fl
	pos := s.Pos()
	vals := syntax.UnpackListExpr(s.Value)
	args := make([]syntax.Expr, len(vals))
	copy(args, vals)
	call := callOf(pos, nameOf(pos, yieldParam), args)
	cond := at(&syntax.Operation{Op: syntax.Not, X: call}, pos)
	native := ifOf(pos, nil, cond, []syntax.Stmt{returnOf(pos, nil)})
	valStart := fl.off(syntax.StartPos(s.Value))
	valEnd := fl.off(syntax.EndPos(s.Value))
	fl.add(fl.off(pos), valEnd, "if !"+yieldParam+"("+fl.textOf(span{start: valStart, end: valEnd})+") { return }")
	return native
}

// dropReturnValues erases the values of a generator `return`, leaving a
// bare stop. The values are discarded: the function always returns the
// iterator closure built by lowerYield.
func (fc *funcLower) dropReturnValues(s *syntax.ReturnStmt) {
	fl := fc.fl
	vals := syntax.UnpackListExpr(s.Results)
	start := fl.off(syntax.StartPos(vals[0]))
	end := fl.off(syntax.EndPos(vals[len(vals)-1]))
	fl.add(start, end, "")
}
