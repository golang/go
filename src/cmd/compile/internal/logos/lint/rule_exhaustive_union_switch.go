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
			if s, ok := n.(*syntax.SwitchStmt); ok {
				reportUnionDefault(pass, s)
			}
			return true
		})
	}
}

// reportUnionDefault reports the default clause of s when s switches on the
// type of a closed union value.
func reportUnionDefault(pass *Pass, s *syntax.SwitchStmt) {
	guard, ok := s.Tag.(*syntax.TypeSwitchGuard)
	if !ok || !isClosedUnion(TypeOf(guard.X)) {
		return
	}
	for _, c := range s.Body {
		if c.Cases == nil {
			pass.Report(Diagnostic{
				Pos:     c.Pos(),
				Message: "default clause in a switch over closed union " + types2.TypeString(TypeOf(guard.X), nil) + "; list every member instead",
			})
		}
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
