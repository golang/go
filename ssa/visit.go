// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// This file defines utilities for visiting the SSA representation of
// a Program.
//
// TODO(adonovan): improve the API:
// - permit client to supply a callback for each function,
//   instruction, type with methods, etc?
// - return graph information about the traversal?
// - test coverage.

import "code.google.com/p/go.tools/go/types"

// AllFunctions returns the set of all functions (including anonymous
// functions and synthetic wrappers) in program prog.
//
// Precondition: all packages are built.
//
func AllFunctions(prog *Program) map[*Function]bool {
	visit := visitor{
		prog: prog,
		seen: make(map[*Function]bool),
	}
	visit.program()
	return visit.seen
}

type visitor struct {
	prog *Program
	seen map[*Function]bool
}

func (visit *visitor) program() {
	for _, pkg := range visit.prog.AllPackages() {
		for _, mem := range pkg.Members {
			switch mem := mem.(type) {
			case *Function:
				visit.function(mem)
			case *Type:
				visit.methodSet(mem.Type())
				visit.methodSet(types.NewPointer(mem.Type()))
			}
		}
	}
}

func (visit *visitor) methodSet(typ types.Type) {
	mset := typ.MethodSet()
	for i, n := 0, mset.Len(); i < n; i++ {
		// Side-effect: creates all wrapper methods.
		visit.function(visit.prog.Method(mset.At(i)))
	}
}

func (visit *visitor) function(fn *Function) {
	if !visit.seen[fn] {
		visit.seen[fn] = true
		for _, b := range fn.Blocks {
			for _, instr := range b.Instrs {
				switch instr := instr.(type) {
				case *MakeInterface:
					visit.methodSet(instr.X.Type())
				}
				var buf [10]*Value // avoid alloc in common case
				for _, op := range instr.Operands(buf[:0]) {
					if fn, ok := (*op).(*Function); ok {
						visit.function(fn)
					}
				}
			}
		}
	}
}
