// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"cmd/internal/src"
	"fmt"
)

// Declaration stack & operations

var blockgen int32 = 1 // max block number
var Block int32        // current block number

// dclstack maintains a stack of shadowed symbol declarations so that
// popdcl can restore their declarations when a block scope ends.
// The stack is maintained as a linked list, using Sym's Link field.
//
// In practice, the "stack" actually ends up forming a tree: goto and label
// statements record the current state of dclstack so that checkgoto can
// validate that a goto statement does not jump over any declarations or
// into a new block scope.
//
// Finally, the Syms in this list are not "real" Syms as they don't actually
// represent object names. Sym is just a convenient type for saving shadowed
// Sym definitions, and only a subset of its fields are actually used.
var dclstack *Sym

func dcopy(a, b *Sym) {
	a.Pkg = b.Pkg
	a.Name = b.Name
	a.Def = b.Def
	a.Block = b.Block
	a.Lastlineno = b.Lastlineno
}

func push(pos src.XPos) *Sym {
	d := new(Sym)
	d.Lastlineno = pos
	d.Link = dclstack
	dclstack = d
	return d
}

// Pushdcl pushes the current declaration for symbol s (if any) so that
// it can be shadowed by a new declaration within a nested block scope.
func Pushdcl(s *Sym, pos src.XPos) {
	d := push(pos)
	dcopy(d, s)
}

// Popdcl pops the innermost block scope and restores all symbol declarations
// to their previous state.
func Popdcl() {
	d := dclstack
	for ; d != nil && d.Name != ""; d = d.Link {
		s := d.Pkg.Lookup(d.Name)
		lno := s.Lastlineno
		dcopy(s, d)
		d.Lastlineno = lno
	}

	if d == nil {
		Fatalf("popdcl: no mark")
	}

	dclstack = d.Link // pop mark
	Block = d.Block
}

// Markdcl records the start of a new block scope for declarations.
func Markdcl(lineno src.XPos) {
	d := push(lineno)
	d.Name = "" // used as a mark in fifo
	d.Block = Block

	blockgen++
	Block = blockgen
}

// keep around for debugging
func DumpDclstack() {
	i := 0
	for d := dclstack; d != nil; d = d.Link {
		fmt.Printf("%6d  %p", i, d)
		if d.Name != "" {
			fmt.Printf("  '%s'  %v\n", d.Name, d.Pkg.Lookup(d.Name))
		} else {
			fmt.Printf("  ---\n")
		}
		i++
	}
}

func IsDclstackValid() bool {
	for d := dclstack; d != nil; d = d.Link {
		if d.Name == "" {
			return false
		}
	}
	return true
}
