// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// This file defines utilities for visiting the SSA representation of
// a Program.
//
// TODO(adonovan): test coverage.

// AllFunctions finds and returns the set of functions potentially
// needed by program prog, as determined by a simple linker-style
// reachability algorithm starting from the members and method-sets of
// each package.  The result may include anonymous functions and
// synthetic wrappers.
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
			if fn, ok := mem.(*Function); ok {
				visit.function(fn)
			}
		}
	}
	for _, T := range visit.prog.TypesWithMethodSets() {
		mset := T.MethodSet()
		for i, n := 0, mset.Len(); i < n; i++ {
			visit.function(visit.prog.Method(mset.At(i)))
		}
	}
}

func (visit *visitor) function(fn *Function) {
	if !visit.seen[fn] {
		visit.seen[fn] = true
		for _, b := range fn.Blocks {
			for _, instr := range b.Instrs {
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
