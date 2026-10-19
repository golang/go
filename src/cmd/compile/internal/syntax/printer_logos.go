// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s printing logic for M1-S2 (G3: hook lines
// only in printer.go; the logic lives here).

// printLogos prints loGo/s nodes in source form: `X?` for TryExpr and
// `errdefer <call>` for ErrdeferStmt. Both forms reparse to the same
// tree, so printing is idempotent.
func printLogos(p *printer, n Node) {
	switch n := n.(type) {
	case *TryExpr:
		p.print(n.X, _Question)
	case *ErrdeferStmt:
		p.print(_Name, "errdefer", blank, n.Call)
	}
}

// logosImpliesSemi reports whether a loGo/s token ends an operand like a
// name or `)`: `?` implies a semicolon, so the printer elides the
// redundant `;` after `?` between statements.
func logosImpliesSemi(tok token) bool {
	return tok == _Question
}
