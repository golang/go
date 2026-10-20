// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s walk logic for M1-S2 and M2 (G3: a hook line
// only in walk.go; the logic lives here).

// walkLogos visits the children of loGo/s nodes in pre-order, matching
// the house order (parent before child): TryExpr before its operand,
// ErrdeferStmt before its call, OptionType before its element,
// CoalesceExpr before its operands, OptSelectExpr before its receiver
// and name, SwitchExpr before its init, tag, and arms, ArmClause before
// its cases and body, AssertedStruct before its assertions and struct,
// PropertyType before its value type.
func (w walker) walkLogos(n Node) {
	switch n := n.(type) {
	case *TryExpr:
		w.node(n.X)
	case *OptionType:
		w.node(n.Elem)
	case *CoalesceExpr:
		w.node(n.X)
		w.node(n.Y)
	case *OptSelectExpr:
		w.node(n.X)
		w.node(n.Sel)
	case *ErrdeferStmt:
		w.node(n.Call)
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
	}
}
