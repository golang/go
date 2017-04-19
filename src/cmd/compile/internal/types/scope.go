// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import "cmd/internal/src"

// Declaration stack & operations

var blockgen int32 = 1 // max block number
var Block int32        // current block number

// dclstack maintains a stack of shadowed symbol declarations so that
// popdcl can restore their declarations when a block scope ends.
//
// The Syms on this stack are not "real" Syms as they don't actually
// represent object names. Sym is just a convenient type for saving shadowed
// Sym definitions, and only a subset of its fields are actually used.
var dclstack []*Sym

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
	dclstack = append(dclstack, d)
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
	i := len(dclstack)
	for ; i > 0; i-- {
		d := dclstack[i-1]
		if d.Name == "" {
			break
		}
		s := d.Pkg.Lookup(d.Name)
		lno := s.Lastlineno
		dcopy(s, d)
		d.Lastlineno = lno
	}

	if i == 0 {
		Fatalf("popdcl: no mark")
	}

	Block = dclstack[i-1].Block
	dclstack = dclstack[:i-1] // pop mark
}

// Markdcl records the start of a new block scope for declarations.
func Markdcl(lineno src.XPos) {
	d := push(lineno)
	d.Name = "" // used as stack mark
	d.Block = Block

	blockgen++
	Block = blockgen
}

func IsDclstackValid() bool {
	for _, d := range dclstack {
		if d.Name == "" {
			return false
		}
	}
	return true
}
