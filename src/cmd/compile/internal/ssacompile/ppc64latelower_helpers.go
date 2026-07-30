// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
)

// Convert a PPC64 opcode from the Op to OpCC form. This converts (op x y)
// to (Select0 (opCC x y)) without having to explicitly fixup every user
// of op.
//
// E.g consider the case:
// a = (ADD x y)
// b = (CMPconst [0] a)
// c = (OR a z)
//
// A rule like (CMPconst [0] (ADD x y)) => (CMPconst [0] (Select0 (ADDCC x y)))
// would produce:
// a  = (ADD x y)
// a' = (ADDCC x y)
// a” = (Select0 a')
// b  = (CMPconst [0] a”)
// c  = (OR a z)
//
// which makes it impossible to rewrite the second user. Instead the result
// of this conversion is:
// a' = (ADDCC x y)
// a  = (Select0 a')
// b  = (CMPconst [0] a)
// c  = (OR a z)
//
// Which makes it trivial to rewrite b using a lowering rule.
func convertPPC64OpToOpCC(op *ssa.Value) *ssa.Value {
	ccOpMap := map[ssaop.Op]ssaop.Op{
		ssaop.OpPPC64ADD:      ssaop.OpPPC64ADDCC,
		ssaop.OpPPC64ADDconst: ssaop.OpPPC64ADDCCconst,
		ssaop.OpPPC64AND:      ssaop.OpPPC64ANDCC,
		ssaop.OpPPC64ANDN:     ssaop.OpPPC64ANDNCC,
		ssaop.OpPPC64ANDconst: ssaop.OpPPC64ANDCCconst,
		ssaop.OpPPC64CNTLZD:   ssaop.OpPPC64CNTLZDCC,
		ssaop.OpPPC64MULHDU:   ssaop.OpPPC64MULHDUCC,
		ssaop.OpPPC64NEG:      ssaop.OpPPC64NEGCC,
		ssaop.OpPPC64NOR:      ssaop.OpPPC64NORCC,
		ssaop.OpPPC64OR:       ssaop.OpPPC64ORCC,
		ssaop.OpPPC64RLDICL:   ssaop.OpPPC64RLDICLCC,
		ssaop.OpPPC64SUB:      ssaop.OpPPC64SUBCC,
		ssaop.OpPPC64XOR:      ssaop.OpPPC64XORCC,
	}
	b := op.Block
	opCC := b.NewValue0I(op.Pos, ccOpMap[op.Op], types.NewTuple(op.Type, types.TypeFlags), op.AuxInt)
	opCC.AddArgs(op.Args...)
	op.Reset(ssaop.OpSelect0)
	op.AddArgs(opCC)
	return op
}

// Try converting a RLDICL to ANDCC. If successful, return the mask otherwise 0.
func convertPPC64RldiclAndccconst(sauxint int64) int64 {
	r, _, _, mask := ssa.DecodePPC64RotateMask(sauxint)
	if r != 0 || mask&0xFFFF != mask {
		return 0
	}
	return int64(mask)
}

// Merge (RLDICL [encoded] (SRDconst [s] x)) into (RLDICL [new_encoded] x)
// SRDconst on PPC64 is an extended mnemonic of RLDICL. If the input to an
// RLDICL is an SRDconst, and the RLDICL does not rotate its value, the two
// operations can be combined. This functions assumes the two opcodes can
// be merged, and returns an encoded rotate+mask value of the combined RLDICL.
func mergePPC64RLDICLandSRDconst(encoded, s int64) int64 {
	mb := s
	r := 64 - s
	// A larger mb is a smaller mask.
	if (encoded>>8)&0xFF < mb {
		encoded = (encoded &^ 0xFF00) | mb<<8
	}
	// The rotate is expected to be 0.
	if (encoded & 0xFF0000) != 0 {
		panic("non-zero rotate")
	}
	return encoded | r<<16
}
