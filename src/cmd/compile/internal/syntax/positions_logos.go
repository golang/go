// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s position logic for M1-S2 (G3: hook lines
// only in positions.go; the logic lives here).

// startPosLogos reports the start position of loGo/s nodes: a TryExpr
// starts at its operand (so StartPos covers `g` in `g()?`), an
// OptionType starts at its element, a CoalesceExpr starts at its left
// operand, an OptSelectExpr starts at its receiver, and an ErrdeferStmt
// starts at `errdefer`. ok is false for any other node.
func startPosLogos(n Node) (pos Pos, ok bool) {
	switch n := n.(type) {
	case *TryExpr:
		return StartPos(n.X), true
	case *OptionType:
		return StartPos(n.Elem), true
	case *CoalesceExpr:
		return StartPos(n.X), true
	case *OptSelectExpr:
		return StartPos(n.X), true
	case *ErrdeferStmt:
		return n.Pos(), true
	}
	return Pos{}, false
}

// endPosLogos reports the approximate end position of loGo/s nodes: one
// past `?` for a TryExpr or an OptionType (mirroring the Name
// `p.Col()+len` precedent), the end of the right operand for a
// CoalesceExpr, the end of the name for an OptSelectExpr, the end of the
// call for an ErrdeferStmt, the closing brace of a SwitchExpr, the end
// of an arm's result for an ArmClause, the end of the value type for a
// PropertyType, and the opening brace of an AssertedStruct (the struct's
// closing brace is not recorded). ok is false otherwise.
func endPosLogos(n Node) (pos Pos, ok bool) {
	switch n := n.(type) {
	case *TryExpr:
		p := n.Pos()
		return MakePos(p.Base(), p.Line(), p.Col()+1), true
	case *OptionType:
		p := n.Pos()
		return MakePos(p.Base(), p.Line(), p.Col()+1), true
	case *CoalesceExpr:
		return EndPos(n.Y), true
	case *OptSelectExpr:
		return EndPos(n.Sel), true
	case *ErrdeferStmt:
		return EndPos(n.Call), true
	case *SwitchExpr:
		return n.Rbrace, true
	case *ArmClause:
		return EndPos(n.Body), true
	case *AssertedStruct:
		return n.Lbrace, true
	case *PropertyType:
		return EndPos(n.Type), true
	}
	return Pos{}, false
}
