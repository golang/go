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

const yieldParam = "_logos_yield"

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

func (fc *funcLower) yieldQualifier(p *types2.Package) string {
	if p == fc.fl.pkg.tpkg {
		return ""
	}
	if name, ok := fc.fl.importNames()[p]; ok {
		return name
	}
	return p.Name()
}

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
	fc.body.List = fc.rewriteYieldList(fc.body.List)
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

func (fc *funcLower) rewriteYieldList(list []syntax.Stmt) []syntax.Stmt {
	for i, s := range list {
		list[i] = fc.rewriteYieldStmt(s)
	}
	return list
}

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
		return fc.yieldGroupStmt(s)
	default:
		return s
	}
}

func (fc *funcLower) lowerYieldStmt(s *syntax.YieldStmt) syntax.Stmt {
	fl := fc.fl
	pos := s.Pos()
	vals := syntax.UnpackListExpr(s.Value)
	args := make([]syntax.Expr, len(vals))
	copy(args, vals)
	call := callOf(pos, nameOf(pos, yieldParam), args)
	cond := at(&syntax.Operation{Op: syntax.Not, X: call}, pos)
	native := ifOf(pos, nil, cond, []syntax.Stmt{returnOf(pos, nil)})
	valStart := fl.off(pos) + len("yield")
	valEnd := fl.exprSpan(vals[len(vals)-1]).end
	tail := fl.wrapperTail(vals[len(vals)-1])
	fl.addT(fl.off(pos), valEnd, cat(
		gen("if !"+yieldParam+"("),
		fl.lazyText(span{start: valStart, end: valEnd}),
		tail,
		gen(") { return }")))
	return native
}

func (fc *funcLower) dropReturnValues(s *syntax.ReturnStmt) {
	fl := fc.fl
	vals := syntax.UnpackListExpr(s.Results)
	end := fl.exprSpan(vals[len(vals)-1]).end
	_ = fl.wrapperTail(vals[len(vals)-1])
	fl.moveRegion(span{start: fl.off(s.Pos()) + len("return"), end: end}, "")
}
