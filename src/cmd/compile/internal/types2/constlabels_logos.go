// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import "cmd/compile/internal/syntax"

// logosConstBranches processes the blocks of a `const if` (M7-S1) or a
// `const switch` (M7-S4) statement for labels and gotos, and returns the
// forward jumps they leave unresolved. A `const if` walks like an `if`: its
// blocks share the enclosing block's labeled statement. A `const switch`
// walks like a `switch`: each case body is a block under one new block that
// carries the labeled statement. The result goes to the caller, which adds it
// to its own forward jumps.
func (check *Checker) logosConstBranches(all *Scope, b *block, lstmt *syntax.LabeledStmt, s syntax.Stmt) []*syntax.BranchStmt {
	switch s := s.(type) {
	case *syntax.ConstIfStmt:
		jumps := check.blockBranches(all, b, lstmt, s.Then.List)
		switch els := s.Else.(type) {
		case *syntax.BlockStmt:
			jumps = append(jumps, check.blockBranches(all, b, lstmt, els.List)...)
		case *syntax.ConstIfStmt:
			jumps = append(jumps, check.logosConstBranches(all, b, lstmt, els)...)
		}
		return jumps
	case *syntax.ConstSwitchStmt:
		inner := &block{b, lstmt, nil}
		var jumps []*syntax.BranchStmt
		for _, c := range s.Cases {
			jumps = append(jumps, check.blockBranches(all, inner, nil, c.Body)...)
		}
		return jumps
	}
	return nil
}
