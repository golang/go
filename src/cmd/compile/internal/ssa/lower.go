// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// convert to machine-dependent ops.
func lower(f *Func) {
	// repeat rewrites until we find no more rewrites
	applyRewrite(f, f.Config.lowerBlock, f.Config.lowerValue, removeDeadValues)
}

// lateLower applies those rules that need to be run after the general lower rules.
func lateLower(f *Func) {
	// repeat rewrites until we find no more rewrites
	if f.Config.lateLowerValue != nil {
		applyRewrite(f, f.Config.lateLowerBlock, f.Config.lateLowerValue, removeDeadValues)
	}
}

// checkLower checks for unlowered opcodes and fails if we find one.
func checkLower(f *Func) {
	// Needs to be a separate phase because it must run after both
	// lowering and a subsequent dead code elimination (because lowering
	// rules may leave dead generic ops behind).
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if !opcodeTable[v.Op].generic {
				continue // lowered
			}
			switch v.Op {
			case OpSP, OpSPanchored, OpSB, OpInitMem, OpArg, OpArgIntReg, OpArgFloatReg, OpPhi, OpVarDef, OpVarLive, OpKeepAlive, OpSelect0, OpSelect1, OpSelectN, OpConvert, OpInlMark, OpWBend:
				continue // ok not to lower
			case OpMakeResult:
				if b.Controls[0] == v {
					continue
				}
			case OpGetG:
				if f.Config.hasGReg {
					// has hardware g register, regalloc takes care of it
					continue // ok not to lower
				}
			}
			s := "not lowered: " + v.String() + ", " + v.Op.String() + " " + v.Type.SimpleString()

			for _, a := range v.Args {
				s += " " + a.Type.SimpleString()
			}
			f.Fatalf("%s", s)
		}
	}
}
