// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Simplified dead code detector. Used for skipping certain checks
// on unreachable code (for instance, shift checks on arch-specific code).
//
package main

import (
	"go/ast"
	"go/constant"
)

// updateDead puts unreachable "if" and "case" nodes into f.dead.
func (f *File) updateDead(node ast.Node) {
	if f.dead[node] {
		// The node is already marked as dead.
		return
	}

	switch stmt := node.(type) {
	case *ast.IfStmt:
		// "if" branch is dead if its condition evaluates
		// to constant false.
		v := f.pkg.types[stmt.Cond].Value
		if v == nil {
			return
		}
		if !constant.BoolVal(v) {
			f.setDead(stmt.Body)
			return
		}
		f.setDead(stmt.Else)
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
					v := f.pkg.types[expr].Value
					if v == nil || constant.BoolVal(v) {
						continue BodyLoopBool
					}
				}
				f.setDead(cc)
			}
			return
		}

		// Case clause is dead if its constant value doesn't match
		// the constant value from the switch tag.
		// TODO: This handles integer comparisons only.
		v := f.pkg.types[stmt.Tag].Value
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
				v := f.pkg.types[expr].Value
				if v == nil {
					continue BodyLoopInt
				}
				n, ok := constant.Uint64Val(v)
				if !ok || tagN == n {
					continue BodyLoopInt
				}
			}
			f.setDead(cc)
		}
	}
}

// setDead marks the node and all the children as dead.
func (f *File) setDead(node ast.Node) {
	dv := deadVisitor{
		f: f,
	}
	ast.Walk(dv, node)
}

type deadVisitor struct {
	f *File
}

func (dv deadVisitor) Visit(node ast.Node) ast.Visitor {
	if node == nil {
		return nil
	}
	dv.f.dead[node] = true
	return dv
}
