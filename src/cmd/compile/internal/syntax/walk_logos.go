// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// This file holds the loGo/s walk logic for M1-S2 (G3: a hook line only
// in walk.go; the logic lives here).

// walkLogos visits the children of loGo/s nodes in pre-order, matching
// the house order (parent before child): TryExpr before its operand,
// ErrdeferStmt before its call.
func (w walker) walkLogos(n Node) {
	switch n := n.(type) {
	case *TryExpr:
		w.node(n.X)
	case *ErrdeferStmt:
		w.node(n.Call)
	}
}
