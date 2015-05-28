// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "log"

//go:generate go run rulegen/rulegen.go rulegen/lower_amd64.rules lowerAmd64 lowerAmd64.go

// convert to machine-dependent ops
func lower(f *Func) {
	// repeat rewrites until we find no more rewrites
	applyRewrite(f, f.Config.lower)

	// Check for unlowered opcodes, fail if we find one.
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op < OpGenericEnd && v.Op != OpFP && v.Op != OpSP && v.Op != OpArg && v.Op != OpCopy && v.Op != OpPhi {
				log.Panicf("%s not lowered", v.LongString())
			}
		}
	}

	// additional pass for 386/amd64, link condition codes directly to blocks
	// TODO: do generically somehow?  Special "block" rewrite rules?
	for _, b := range f.Blocks {
		for {
			switch b.Kind {
			case BlockIf:
				switch b.Control.Op {
				case OpSETL:
					b.Kind = BlockLT
					b.Control = b.Control.Args[0]
					continue
				case OpSETNE:
					b.Kind = BlockNE
					b.Control = b.Control.Args[0]
					continue
				case OpSETB:
					b.Kind = BlockULT
					b.Control = b.Control.Args[0]
					continue
				case OpMOVBload:
					b.Kind = BlockNE
					b.Control = b.NewValue2(OpTESTB, TypeFlags, nil, b.Control, b.Control)
					continue
					// TODO: others
				}
			case BlockLT:
				if b.Control.Op == OpInvertFlags {
					b.Kind = BlockGT
					b.Control = b.Control.Args[0]
					continue
				}
			case BlockGT:
				if b.Control.Op == OpInvertFlags {
					b.Kind = BlockLT
					b.Control = b.Control.Args[0]
					continue
				}
			case BlockLE:
				if b.Control.Op == OpInvertFlags {
					b.Kind = BlockGE
					b.Control = b.Control.Args[0]
					continue
				}
			case BlockGE:
				if b.Control.Op == OpInvertFlags {
					b.Kind = BlockLE
					b.Control = b.Control.Args[0]
					continue
				}
			case BlockULT:
				if b.Control.Op == OpInvertFlags {
					b.Kind = BlockUGT
					b.Control = b.Control.Args[0]
					continue
				}
			case BlockUGT:
				if b.Control.Op == OpInvertFlags {
					b.Kind = BlockULT
					b.Control = b.Control.Args[0]
					continue
				}
			case BlockULE:
				if b.Control.Op == OpInvertFlags {
					b.Kind = BlockUGE
					b.Control = b.Control.Args[0]
					continue
				}
			case BlockUGE:
				if b.Control.Op == OpInvertFlags {
					b.Kind = BlockULE
					b.Control = b.Control.Args[0]
					continue
				}
			case BlockEQ:
				if b.Control.Op == OpInvertFlags {
					b.Control = b.Control.Args[0]
					continue
				}
			case BlockNE:
				if b.Control.Op == OpInvertFlags {
					b.Control = b.Control.Args[0]
					continue
				}
			}
			break
		}
	}
}
