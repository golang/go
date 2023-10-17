// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package shift

// Simplified dead code detector.
// Used for skipping shift checks on unreachable arch-specific code.

import (
	"go/ast"
	"go/constant"
	"go/types"
)

// updateDead puts unreachable "if" and "case" nodes into dead.
func updateDead(info *types.Info, dead map[ast.Node]bool, node ast.Node) {
	if dead[node] {
		// The node is already marked as dead.
		return
	}

	// setDead marks the node and all the children as dead.
	setDead := func(n ast.Node) {
		ast.Inspect(n, func(node ast.Node) bool {
			if node != nil {
				dead[node] = true
			}
			return true
		})
	}

	switch stmt := node.(type) {
	case *ast.IfStmt:
		// "if" branch is dead if its condition evaluates
		// to constant false.
		v := info.Types[stmt.Cond].Value
		if v == nil {
			return
		}
		if !constant.BoolVal(v) {
			setDead(stmt.Body)
			return
		}
		if stmt.Else != nil {
			setDead(stmt.Else)
		}
	case *ast.SwitchStmt:
		// Case clause with empty switch tag is dead if it evaluates
		// to constant false.
		if stmt.Tag == nil {
		BodyLoopBool:
			for _, stmt := range stmt.Body.List {
				cc := stmt.(*ast.CaseClause)
				if cc.List == nil {
					// Skip default case.
					continue
				}
				for _, expr := range cc.List {
					v := info.Types[expr].Value
					if v == nil || v.Kind() != constant.Bool || constant.BoolVal(v) {
						continue BodyLoopBool
					}
				}
				setDead(cc)
			}
			return
		}

		// Case clause is dead if its constant value doesn't match
		// the constant value from the switch tag.
		// TODO: This handles integer comparisons only.
		v := info.Types[stmt.Tag].Value
		if v == nil || v.Kind() != constant.Int {
			return
		}
		tagN, ok := constant.Uint64Val(v)
		if !ok {
			return
		}
	BodyLoopInt:
		for _, x := range stmt.Body.List {
			cc := x.(*ast.CaseClause)
			if cc.List == nil {
				// Skip default case.
				continue
			}
			for _, expr := range cc.List {
				v := info.Types[expr].Value
				if v == nil {
					continue BodyLoopInt
				}
				n, ok := constant.Uint64Val(v)
				if !ok || tagN == n {
					continue BodyLoopInt
				}
			}
			setDead(cc)
		}
	}
}
