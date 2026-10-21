// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s walk logic for M1-S2 and M2 (G3: a hook line
// only in walk.go; the logic lives here).

// walkLogos visits the children of loGo/s nodes in pre-order, matching
// the house order (parent before child): SpreadExpr before its source,
// TryExpr before its operand,
// ErrdeferStmt before its call, OptionType before its element,
// CoalesceExpr before its operands, OptSelectExpr before its receiver
// and name, ConstType before its element, SwitchExpr before its init,
// tag, and arms, ArmClause before its cases and body, AssertedStruct
// before its assertions and struct, PropertyType before its value type,
// InterpExpr before its operands.
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
	case *ConstFuncDecl: // M5-S2: walk the inner function.
		w.node(n.Func)
	case *ErrdeferStmt:
		w.node(n.Call)
	case *YieldStmt:
		w.node(n.Value)
	case *ScopeDeferStmt:
		w.node(n.Call)
	case *GroupStmt: // M8-S2: group context, limit, then body.
		w.node(n.Ctx)
		if n.Limit != nil {
			w.node(n.Limit)
		}
		w.node(n.Body)
	case *GroupChild: // M8-S2: child body.
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
			if f.Name != nil {
				w.node(f.Name)
			}
			w.node(f.Type)
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
