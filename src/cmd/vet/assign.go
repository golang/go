// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
This file contains the code to check for useless assignments.
*/

package main

import (
	"go/ast"
	"go/token"
	"reflect"
)

func init() {
	register("assign",
		"check for useless assignments",
		checkAssignStmt,
		assignStmt)
}

// TODO: should also check for assignments to struct fields inside methods
// that are on T instead of *T.

// checkAssignStmt checks for assignments of the form "<expr> = <expr>".
// These are almost always useless, and even when they aren't they are usually a mistake.
func checkAssignStmt(f *File, node ast.Node) {
	stmt := node.(*ast.AssignStmt)
	if stmt.Tok != token.ASSIGN {
		return // ignore :=
	}
	if len(stmt.Lhs) != len(stmt.Rhs) {
		// If LHS and RHS have different cardinality, they can't be the same.
		return
	}
	for i, lhs := range stmt.Lhs {
		rhs := stmt.Rhs[i]
		if hasSideEffects(f, lhs) || hasSideEffects(f, rhs) {
			continue // expressions may not be equal
		}
		if reflect.TypeOf(lhs) != reflect.TypeOf(rhs) {
			continue // short-circuit the heavy-weight gofmt check
		}
		le := f.gofmt(lhs)
		re := f.gofmt(rhs)
		if le == re {
			f.Badf(stmt.Pos(), "self-assignment of %s to %s", re, le)
		}
	}
}
