// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file encapsulates some of the odd characteristics of the RISCV64
// instruction set, to minimize its interaction with the core of the
// assembler.

package arch

import (
	"cmd/internal/obj"
	"cmd/internal/obj/riscv"
)

// IsRISCV64AMO reports whether op is an AMO instruction that requires
// special handling.
func IsRISCV64AMO(op obj.As) bool {
	switch op {
	case riscv.ASCW, riscv.ASCD, riscv.AAMOSWAPW, riscv.AAMOSWAPD, riscv.AAMOADDW, riscv.AAMOADDD,
		riscv.AAMOANDW, riscv.AAMOANDD, riscv.AAMOORW, riscv.AAMOORD, riscv.AAMOXORW, riscv.AAMOXORD,
		riscv.AAMOMINW, riscv.AAMOMIND, riscv.AAMOMINUW, riscv.AAMOMINUD,
		riscv.AAMOMAXW, riscv.AAMOMAXD, riscv.AAMOMAXUW, riscv.AAMOMAXUD:
		return true
	}
	return false
}

// IsRISCV64VTypeI reports whether op is a vtype immediate instruction that
// requires special handling.
func IsRISCV64VTypeI(op obj.As) bool {
	return op == riscv.AVSETVLI || op == riscv.AVSETIVLI
}

var riscv64SpecialOperand map[string]riscv.SpecialOperand

// RISCV64SpecialOperand returns the internal representation of a special operand.
func RISCV64SpecialOperand(name string) riscv.SpecialOperand {
	if riscv64SpecialOperand == nil {
		// Generate mapping when function is first called.
		riscv64SpecialOperand = map[string]riscv.SpecialOperand{}
		for opd := riscv.SPOP_BEGIN; opd < riscv.SPOP_END; opd++ {
			riscv64SpecialOperand[opd.String()] = opd
		}
	}
	if opd, ok := riscv64SpecialOperand[name]; ok {
		return opd
	}
	return riscv.SPOP_END
}

// RISCV64ValidateVectorType reports whether the given configuration is a
// valid vector type.
func RISCV64ValidateVectorType(vsew, vlmul, vtail, vmask int64) error {
	_, err := riscv.EncodeVectorType(vsew, vlmul, vtail, vmask)
	return err
}
