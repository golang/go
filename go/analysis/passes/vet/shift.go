// +build ignore

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
This file contains the code to check for suspicious shifts.
*/

package main

import (
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
)

func init() {
	register("shift",
		"check for useless shifts",
		checkShift,
		binaryExpr, assignStmt)
}

func checkShift(f *File, node ast.Node) {
	if f.dead[node] {
		// Skip shift checks on unreachable nodes.
		return
	}

	switch node := node.(type) {
	case *ast.BinaryExpr:
		if node.Op == token.SHL || node.Op == token.SHR {
			checkLongShift(f, node, node.X, node.Y)
		}
	case *ast.AssignStmt:
		if len(node.Lhs) != 1 || len(node.Rhs) != 1 {
			return
		}
		if node.Tok == token.SHL_ASSIGN || node.Tok == token.SHR_ASSIGN {
			checkLongShift(f, node, node.Lhs[0], node.Rhs[0])
		}
	}
}

// checkLongShift checks if shift or shift-assign operations shift by more than
// the length of the underlying variable.
func checkLongShift(f *File, node ast.Node, x, y ast.Expr) {
	if f.pkg.types[x].Value != nil {
		// Ignore shifts of constants.
		// These are frequently used for bit-twiddling tricks
		// like ^uint(0) >> 63 for 32/64 bit detection and compatibility.
		return
	}

	v := f.pkg.types[y].Value
	if v == nil {
		return
	}
	amt, ok := constant.Int64Val(v)
	if !ok {
		return
	}
	t := f.pkg.types[x].Type
	if t == nil {
		return
	}
	b, ok := t.Underlying().(*types.Basic)
	if !ok {
		return
	}
	var size int64
	switch b.Kind() {
	case types.Uint8, types.Int8:
		size = 8
	case types.Uint16, types.Int16:
		size = 16
	case types.Uint32, types.Int32:
		size = 32
	case types.Uint64, types.Int64:
		size = 64
	case types.Int, types.Uint:
		size = uintBitSize
	case types.Uintptr:
		size = uintptrBitSize
	default:
		return
	}
	if amt >= size {
		ident := f.gofmt(x)
		f.Badf(node.Pos(), "%s (%d bits) too small for shift of %d", ident, size, amt)
	}
}

var (
	uintBitSize    = 8 * archSizes.Sizeof(types.Typ[types.Uint])
	uintptrBitSize = 8 * archSizes.Sizeof(types.Typ[types.Uintptr])
)
