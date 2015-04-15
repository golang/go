// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

//go:generate go run rulegen/rulegen.go rulegen/lower_amd64.rules lowerAmd64 lowerAmd64.go

// convert to machine-dependent ops
func lower(f *Func) {
	// repeat rewrites until we find no more rewrites
	applyRewrite(f, f.Config.lower)

	// TODO: check for unlowered opcodes, fail if we find one

	// additional pass for 386/amd64, link condition codes directly to blocks
	// TODO: do generically somehow?  Special "block" rewrite rules?
	for _, b := range f.Blocks {
		switch b.Kind {
		case BlockIf:
			switch b.Control.Op {
			case OpSETL:
				b.Kind = BlockLT
				b.Control = b.Control.Args[0]
			case OpSETNE:
				b.Kind = BlockNE
				b.Control = b.Control.Args[0]
			case OpSETB:
				b.Kind = BlockULT
				b.Control = b.Control.Args[0]
				// TODO: others
			}
		case BlockLT:
			if b.Control.Op == OpInvertFlags {
				b.Kind = BlockGE
				b.Control = b.Control.Args[0]
			}
		case BlockULT:
			if b.Control.Op == OpInvertFlags {
				b.Kind = BlockUGE
				b.Control = b.Control.Args[0]
			}
		case BlockEQ:
			if b.Control.Op == OpInvertFlags {
				b.Kind = BlockNE
				b.Control = b.Control.Args[0]
			}
		case BlockNE:
			if b.Control.Op == OpInvertFlags {
				b.Kind = BlockEQ
				b.Control = b.Control.Args[0]
			}
			// TODO: others
		}
	}
}
