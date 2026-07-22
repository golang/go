// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/ssa/ssacore"
	"cmd/compile/internal/ssa/ssaop"
)

// convert to machine-dependent ops.
func lower(f *ssacore.Func) {
	// repeat rewrites until we find no more rewrites
	applyRewrite(f, f.Config.LowerBlock, f.Config.LowerValue, RemoveDeadValues)
}

// lateLower applies those rules that need to be run after the general lower rules.
func lateLower(f *ssacore.Func) {
	// repeat rewrites until we find no more rewrites
	if f.Config.LateLowerValue != nil {
		applyRewrite(f, f.Config.LateLowerBlock, f.Config.LateLowerValue, RemoveDeadValues)
	}
}

// checkLower checks for unlowered opcodes and fails if we find one.
func checkLower(f *ssacore.Func) {
	// Needs to be a separate phase because it must run after both
	// lowering and a subsequent dead code elimination (because lowering
	// rules may leave dead generic ops behind).
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if !ssaop.OpcodeTable[v.Op].Generic {
				continue // lowered
			}
			switch v.Op {
			case ssaop.OpSP, ssaop.OpSPanchored, ssaop.OpSB, ssaop.OpInitMem, ssaop.OpArg, ssaop.OpArgIntReg, ssaop.OpArgFloatReg, ssaop.OpPhi, ssaop.OpVarDef, ssaop.OpVarLive, ssaop.OpKeepAlive, ssaop.OpSelect0, ssaop.OpSelect1, ssaop.OpSelectN, ssaop.OpConvert, ssaop.OpInlMark, ssaop.OpWBend:
				continue // ok not to lower
			case ssaop.OpMakeResult:
				if b.Controls[0] == v {
					continue
				}
			case ssaop.OpGetG:
				if f.Config.HasGReg {
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
