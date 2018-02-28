// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import "cmd/internal/src"

// Declaration stack & operations

var blockgen int32 = 1 // max block number
var Block int32        // current block number

// A dsym stores a symbol's shadowed declaration so that it can be
// restored once the block scope ends.
type dsym struct {
	sym        *Sym // sym == nil indicates stack mark
	def        *Node
	block      int32
	lastlineno src.XPos // last declaration for diagnostic
}

// dclstack maintains a stack of shadowed symbol declarations so that
// Popdcl can restore their declarations when a block scope ends.
var dclstack []dsym

// Pushdcl pushes the current declaration for symbol s (if any) so that
// it can be shadowed by a new declaration within a nested block scope.
func Pushdcl(s *Sym) {
	dclstack = append(dclstack, dsym{
		sym:        s,
		def:        s.Def,
		block:      s.Block,
		lastlineno: s.Lastlineno,
	})
}

// Popdcl pops the innermost block scope and restores all symbol declarations
// to their previous state.
func Popdcl() {
	for i := len(dclstack); i > 0; i-- {
		d := &dclstack[i-1]
		s := d.sym
		if s == nil {
			// pop stack mark
			Block = d.block
			dclstack = dclstack[:i-1]
			return
		}

		s.Def = d.def
		s.Block = d.block
		s.Lastlineno = d.lastlineno

		// Clear dead pointer fields.
		d.sym = nil
		d.def = nil
	}
	Fatalf("popdcl: no stack mark")
}

// Markdcl records the start of a new block scope for declarations.
func Markdcl() {
	dclstack = append(dclstack, dsym{
		sym:   nil, // stack mark
		block: Block,
	})
	blockgen++
	Block = blockgen
}

func IsDclstackValid() bool {
	for _, d := range dclstack {
		if d.sym == nil {
			return false
		}
	}
	return true
}
