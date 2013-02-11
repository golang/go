// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
	"go/token"
	"sync/atomic"
)

// checkAtomicAssignment walks the assignment statement checking for comomon
// mistaken usage of atomic package, such as: x = atomic.AddUint64(&x, 1)
func (f *File) checkAtomicAssignment(n *ast.AssignStmt) {
	if !vet("atomic") {
		return
	}

	if len(n.Lhs) != len(n.Rhs) {
		return
	}

	for i, right := range n.Rhs {
		call, ok := right.(*ast.CallExpr)
		if !ok {
			continue
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok {
			continue
		}
		pkg, ok := sel.X.(*ast.Ident)
		if !ok || pkg.Name != "atomic" {
			continue
		}

		switch sel.Sel.Name {
		case "AddInt32", "AddInt64", "AddUint32", "AddUint64", "AddUintptr":
			f.checkAtomicAddAssignment(n.Lhs[i], call)
		}
	}
}

// checkAtomicAddAssignment walks the atomic.Add* method calls checking for assigning the return value
// to the same variable being used in the operation
func (f *File) checkAtomicAddAssignment(left ast.Expr, call *ast.CallExpr) {
	arg := call.Args[0]
	broken := false

	if uarg, ok := arg.(*ast.UnaryExpr); ok && uarg.Op == token.AND {
		broken = f.gofmt(left) == f.gofmt(uarg.X)
	} else if star, ok := left.(*ast.StarExpr); ok {
		broken = f.gofmt(star.X) == f.gofmt(arg)
	}

	if broken {
		f.Warn(left.Pos(), "direct assignment to atomic value")
	}
}

type Counter uint64

func BadAtomicAssignmentUsedInTests() {
	x := uint64(1)
	x = atomic.AddUint64(&x, 1)        // ERROR "direct assignment to atomic value"
	_, x = 10, atomic.AddUint64(&x, 1) // ERROR "direct assignment to atomic value"
	x, _ = atomic.AddUint64(&x, 1), 10 // ERROR "direct assignment to atomic value"

	y := &x
	*y = atomic.AddUint64(y, 1) // ERROR "direct assignment to atomic value"

	var su struct{ Counter uint64 }
	su.Counter = atomic.AddUint64(&su.Counter, 1) // ERROR "direct assignment to atomic value"
	z1 := atomic.AddUint64(&su.Counter, 1)
	_ = z1 // Avoid err "z declared and not used"

	var sp struct{ Counter *uint64 }
	*sp.Counter = atomic.AddUint64(sp.Counter, 1) // ERROR "direct assignment to atomic value"
	z2 := atomic.AddUint64(sp.Counter, 1)
	_ = z2 // Avoid err "z declared and not used"

	au := []uint64{10, 20}
	au[0] = atomic.AddUint64(&au[0], 1) // ERROR "direct assignment to atomic value"
	au[1] = atomic.AddUint64(&au[0], 1)

	ap := []*uint64{&au[0], &au[1]}
	*ap[0] = atomic.AddUint64(ap[0], 1) // ERROR "direct assignment to atomic value"
	*ap[1] = atomic.AddUint64(ap[0], 1)
}
