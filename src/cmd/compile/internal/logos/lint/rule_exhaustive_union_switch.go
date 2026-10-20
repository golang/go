// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// exhaustiveUnionSwitch is pedantic::exhaustive_union_switch (SPEC Sec
// 1.2, #10): a default clause in a type switch over a closed union. The
// group defaults to allow, so the rule reports only under --pedantic or a
// logos.toml pedantic key. It applies to loGo/s packages only.
var exhaustiveUnionSwitch = Rule{
	ID:    "pedantic::exhaustive_union_switch",
	Group: Pedantic,
	Check: checkExhaustiveUnionSwitch,
}

// checkExhaustiveUnionSwitch is the Check of exhaustiveUnionSwitch.
func checkExhaustiveUnionSwitch(pass *Pass) {
	for _, f := range pass.Files {
		syntax.Inspect(f, func(n syntax.Node) bool {
			switch s := n.(type) {
			case *syntax.SwitchStmt:
				reportUnionGuard(pass, s.Tag, stmtDefaultPoss(s))
			case *syntax.SwitchExpr:
				reportUnionGuard(pass, s.Tag, exprDefaultPoss(s))
			}
			return true
		})
	}
}

// stmtDefaultPoss returns the positions of the default clauses of s.
func stmtDefaultPoss(s *syntax.SwitchStmt) []syntax.Pos {
	var poss []syntax.Pos
	for _, c := range s.Body {
		if c.Cases == nil {
			poss = append(poss, c.Pos())
		}
	}
	return poss
}

// exprDefaultPoss returns the positions of the default arms of s.
func exprDefaultPoss(s *syntax.SwitchExpr) []syntax.Pos {
	var poss []syntax.Pos
	for _, a := range s.Arms {
		if a.Cases == nil {
			poss = append(poss, a.Pos())
		}
	}
	return poss
}

// reportUnionGuard reports the default clauses at poss when tag switches on
// the type of a closed union value.
func reportUnionGuard(pass *Pass, tag syntax.Expr, poss []syntax.Pos) {
	guard, ok := tag.(*syntax.TypeSwitchGuard)
	if !ok || !isClosedUnion(TypeOf(guard.X)) {
		return
	}
	for _, pos := range poss {
		pass.Report(Diagnostic{
			Pos:     pos,
			Message: "default clause in a switch over closed union " + types2.TypeString(TypeOf(guard.X), nil) + "; list every member instead",
		})
	}
}

// isClosedUnion reports whether t is an interface whose type set is a
// finite union of non-~ terms.
func isClosedUnion(t types2.Type) bool {
	if t == nil {
		return false
	}
	iface, ok := types2.Unalias(t).Underlying().(*types2.Interface)
	if !ok {
		return false
	}
	closed := false
	for i := range iface.NumEmbeddeds() {
		u, ok := iface.EmbeddedType(i).(*types2.Union)
		if !ok {
			continue
		}
		for j := range u.Len() {
			if u.Term(j).Tilde() {
				return false
			}
		}
		closed = true
	}
	return closed
}
