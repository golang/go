// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import "strings"

// This file holds the loGo/s position logic for M1-S2 (G3: hook lines
// only in positions.go; the logic lives here).

// startPosLogos reports the start position of loGo/s nodes: a SpreadExpr
// starts at its first `.`, a TryExpr
// starts at its operand (so StartPos covers `g` in `g()?`), an
// OptionType starts at its element, a CoalesceExpr starts at its left
// operand, an OptSelectExpr starts at its receiver, a ConstType starts
// at `const`, an ErrdeferStmt starts at `errdefer`, and an InterpLit
// starts at `$`. ok is false for any other node.
func startPosLogos(n Node) (pos Pos, ok bool) {
	switch n := n.(type) {
	case *TryExpr:
		return StartPos(n.X), true
	case *SpreadExpr, *InterpExpr:
		return n.Pos(), true
	case *OptionType:
		return StartPos(n.Elem), true
	case *FieldProjExpr:
		return StartPos(n.X), true
	case *CoalesceExpr:
		return StartPos(n.X), true
	case *OptSelectExpr:
		return StartPos(n.X), true
	case *ConstType:
		return n.Pos(), true
	case *ConstFuncDecl: // M5-S2: a const func starts at `const`.
		return n.Pos(), true
	case *FieldDefault:
		return StartPos(n.Type), true
	case *TupleType, *TupleLit, *LambdaExpr:
		return n.Pos(), true
	case *ErrdeferStmt:
		return n.Pos(), true
	case *YieldStmt:
		return n.Pos(), true
	case *ScopeDeferStmt:
		return n.Pos(), true
	case *GroupStmt, *GroupChild: // M8-S2: groups and children start at `go`.
		return n.Pos(), true
	}
	return Pos{}, false
}

// endPosLogos reports the approximate end position of loGo/s nodes: the
// end of the source for a SpreadExpr, one
// past `?` for a TryExpr or an OptionType (mirroring the Name
// `p.Col()+len` precedent), the end of the right operand for a
// CoalesceExpr, the end of the name for an OptSelectExpr, the end of
// the element for a ConstType, the end of the call for an ErrdeferStmt,
// the closing brace of a SwitchExpr, the end of an arm's result for an
// ArmClause, the end of the value type for a PropertyType, the opening
// brace of an AssertedStruct (the struct's closing brace is not
// recorded), and one past the closing quote of an InterpLit (from its
// Raw spelling, so multiline literals land on the right line). ok is
// false otherwise.
func endPosLogos(n Node) (pos Pos, ok bool) {
	switch n := n.(type) {
	case *TryExpr:
		p := n.Pos()
		return MakePos(p.Base(), p.Line(), p.Col()+1), true
	case *ConstIfStmt:
		end := EndPos(constIfTail(n))
		return MakePos(end.Base(), end.Line(), end.Col()+1), true
	case *ConstSwitchStmt:
		p := n.Rbrace
		return MakePos(p.Base(), p.Line(), p.Col()+1), true
	case *SpreadExpr:
		return EndPos(n.X), true
	case *InterpExpr:
		p := n.Pos()
		if nl := strings.Count(n.Raw, "\n"); nl > 0 {
			last := n.Raw[strings.LastIndexByte(n.Raw, '\n')+1:]
			return MakePos(p.Base(), p.Line()+uint(nl), uint(len(last))+1), true
		}
		return MakePos(p.Base(), p.Line(), p.Col()+uint(len(n.Raw))), true
	case *OptionType:
		p := n.Pos()
		return MakePos(p.Base(), p.Line(), p.Col()+1), true
	case *CoalesceExpr:
		return EndPos(n.Y), true
	case *TupleType:
		return EndPos(n.Fields[len(n.Fields)-1].Type), true
	case *TupleLit:
		return EndPos(n.Elems[len(n.Elems)-1]), true
	case *LambdaExpr:
		return EndPos(n.Body), true
	case *ConstForStmt:
		return EndPos(n.Body), true
	case *FieldProjExpr:
		p := n.Rbrack
		return MakePos(p.Base(), p.Line(), p.Col()+1), true
	case *OptSelectExpr:
		return EndPos(n.Sel), true
	case *ConstType:
		return EndPos(n.Elem), true
	case *FieldDefault:
		return EndPos(n.Default), true
	case *ErrdeferStmt:
		return EndPos(n.Call), true
	case *YieldStmt:
		return EndPos(n.Value), true
	case *ScopeDeferStmt:
		return EndPos(n.Call), true
	case *GroupStmt: // M8-S2: a group ends with its body.
		return EndPos(n.Body), true
	case *GroupChild: // M8-S2: a child ends with its body.
		return EndPos(n.Body), true
	case *SwitchExpr:
		return n.Rbrace, true
	case *ArmClause:
		return EndPos(n.Body), true
	case *AssertedStruct:
		return n.Lbrace, true
	case *PropertyType:
		return EndPos(n.Type), true
	case *ConstFuncDecl: // M5-S2: a const func ends where its function ends.
		return EndPos(n.Func), true
	}
	return Pos{}, false
}

// constIfTail returns the last block of the `const if` chain rooted at s,
// the block that ends the statement.
func constIfTail(s *ConstIfStmt) *BlockStmt {
	switch els := s.Else.(type) {
	case *ConstIfStmt:
		return constIfTail(els)
	case *BlockStmt:
		return els
	}
	return s.Then
}
