// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s printing logic for M1-S2 (G3: a hook line
// only in printer.go; the logic lives here).

// printLogos prints loGo/s nodes in source form: `X?` for TryExpr and
// `errdefer <call>` for ErrdeferStmt. Both forms reparse to the same
// tree, so printing is idempotent. It reports whether n was printed.
func printLogos(p *printer, n Node) bool {
	switch n := n.(type) {
	case *TryExpr:
		p.print(n.X, _Question)
		return true
	case *ErrdeferStmt:
		p.print(_Name, "errdefer", blank, n.Call)
		return true
	}
	return false
}
