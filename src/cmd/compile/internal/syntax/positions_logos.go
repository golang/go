// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s position logic for M1-S2 (G3: hook lines
// only in positions.go; the logic lives here).

// startPosLogos reports the start position of loGo/s nodes: a TryExpr
// starts at its operand (so StartPos covers `g` in `g()?`), and an
// ErrdeferStmt starts at `errdefer`. ok is false for any other node.
func startPosLogos(n Node) (pos Pos, ok bool) {
	switch n := n.(type) {
	case *TryExpr:
		return StartPos(n.X), true
	case *ErrdeferStmt:
		return n.Pos(), true
	}
	return Pos{}, false
}

// endPosLogos reports the approximate end position of loGo/s nodes: one
// past `?` for a TryExpr (mirroring the Name `p.Col()+len` precedent),
// and the end of the call for an ErrdeferStmt. ok is false otherwise.
func endPosLogos(n Node) (pos Pos, ok bool) {
	switch n := n.(type) {
	case *TryExpr:
		p := n.Pos()
		return MakePos(p.Base(), p.Line(), p.Col()+1), true
	case *ErrdeferStmt:
		return EndPos(n.Call), true
	}
	return Pos{}, false
}
