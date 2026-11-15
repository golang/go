// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

func (w walker) walkLogos(n Node) {
	switch n := n.(type) {
	case *TryExpr:
		w.node(n.X)
	case *SpreadExpr:
		w.node(n.X)
	case *InterpExpr:
		for _, e := range n.Exprs {
			w.node(e)
		}
	case *OptionType:
		w.node(n.Elem)
	case *CoalesceExpr:
		w.node(n.X)
		w.node(n.Y)
	case *OptSelectExpr:
		w.node(n.X)
		w.node(n.Sel)
	case *ConstType:
		w.node(n.Elem)
	case *ConstFuncDecl:
		w.node(n.Func)
	case *ErrdeferStmt:
		w.node(n.Call)
	case *YieldStmt:
		w.node(n.Value)
	case *ScopeDeferStmt:
		w.node(n.Call)
	case *GroupStmt:
		w.node(n.Ctx)
		if n.Limit != nil {
			w.node(n.Limit)
		}
		w.node(n.Body)
	case *GroupChild:
		w.node(n.Body)
	case *SwitchExpr:
		if n.Init != nil {
			w.node(n.Init)
		}
		if n.Tag != nil {
			w.node(n.Tag)
		}
		for _, arm := range n.Arms {
			w.node(arm)
		}
	case *ArmClause:
		if n.Cases != nil {
			w.node(n.Cases)
		}
		w.node(n.Body)
	case *AssertedStruct:
		w.exprList(n.Asserts)
		w.node(n.Struct)
	case *PropertyType:
		w.node(n.Type)
	case *FieldDefault:
		w.node(n.Type)
		w.node(n.Default)
	case *TupleType:
		for _, f := range n.Fields {
			w.node(f)
		}
	case *ConstForStmt:
		if n.Index != nil {
			w.node(n.Index)
		}
		w.node(n.Field)
		w.node(n.Range)
		w.node(n.Body)
	case *FieldProjExpr:
		w.node(n.X)
		w.node(n.Field)
	case *TupleLit:
		w.exprList(n.Elems)
	case *LambdaExpr:
		for _, name := range n.Params {
			w.node(name)
		}
		w.node(n.Body)
	case *ConstIfStmt:
		w.node(n.Cond)
		w.node(n.Then)
		if n.Else != nil {
			w.node(n.Else)
		}
	case *ConstSwitchStmt:
		w.node(n.Targ)
		for _, c := range n.Cases {
			w.node(c.Type)
			w.stmtList(c.Body)
		}
	}
}
