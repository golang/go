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
	"fmt"
)

// IsRISCV64AMO reports whether op is an AMO instruction that requires
// special handling.
func IsRISCV64AMO(op obj.As) bool {
	switch op {
	case riscv.ASCW, riscv.ASCD, riscv.AAMOSWAPW, riscv.AAMOSWAPD, riscv.AAMOADDW, riscv.AAMOADDD,
		riscv.AAMOANDW, riscv.AAMOANDD, riscv.AAMOORW, riscv.AAMOORD, riscv.AAMOXORW, riscv.AAMOXORD,
		riscv.AAMOMINW, riscv.AAMOMIND, riscv.AAMOMINUW, riscv.AAMOMINUD,
		riscv.AAMOMAXW, riscv.AAMOMAXD, riscv.AAMOMAXUW, riscv.AAMOMAXUD,
		riscv.AAMOSWAPB, riscv.AAMOSWAPH, riscv.AAMOADDB, riscv.AAMOADDH,
		riscv.AAMOANDB, riscv.AAMOANDH, riscv.AAMOORB, riscv.AAMOORH,
		riscv.AAMOXORB, riscv.AAMOXORH, riscv.AAMOMINB, riscv.AAMOMINH,
		riscv.AAMOMINUB, riscv.AAMOMINUH, riscv.AAMOMAXB, riscv.AAMOMAXH,
		riscv.AAMOMAXUB, riscv.AAMOMAXUH, riscv.AAMOCASB, riscv.AAMOCASH,
		riscv.AAMOCASW, riscv.AAMOCASD, riscv.AAMOCASQ:
		return true
	}
	return false
}

// IsRISCV64VTypeI reports whether op is a vtype immediate instruction that
// requires special handling.
func IsRISCV64VTypeI(op obj.As) bool {
	return op == riscv.AVSETVLI || op == riscv.AVSETIVLI
}

// IsRISCV64CSRO reports whether the op is an instruction that uses
// CSR symbolic names and whether that instruction expects a register
// or an immediate source operand.
func IsRISCV64CSRO(op obj.As) (imm bool, ok bool) {
	switch op {
	case riscv.ACSRRCI, riscv.ACSRRSI, riscv.ACSRRWI:
		imm = true
		fallthrough
	case riscv.ACSRRC, riscv.ACSRRS, riscv.ACSRRW:
		ok = true
	}
	return
}

// IsRISCV64PseudoCSRO reports whether the op is a pseudo instruction
// that uses CSR symbolic names, whether that instruction expects a register
// or an immediate source operand and what the expected index of the operand
// containing the CSR name should be.
func IsRISCV64PseudoCSRO(op obj.As) (imm bool, index int, ok bool) {
	index = 1
	switch op {
	case riscv.ACSRCI, riscv.ACSRSI, riscv.ACSRWI:
		imm = true
		ok = true
	case riscv.ACSRC, riscv.ACSRR, riscv.ACSRS, riscv.ACSRW:
		ok = true
		if op == riscv.ACSRR {
			index = 0
		}
	}
	return
}

var riscv64SpecialOperand map[string]riscv.SpecialOperand

// RISCV64SpecialOperand returns the internal representation of a special operand.
func RISCV64SpecialOperand(name string) riscv.SpecialOperand {
	if riscv64SpecialOperand == nil {
		// Generate mapping when function is first called.
		riscv64SpecialOperand = map[string]riscv.SpecialOperand{}
		for opd := riscv.SPOP_RVV_BEGIN; opd < riscv.SPOP_RVV_END; opd++ {
			riscv64SpecialOperand[opd.String()] = opd
		}
		// Add the CSRs
		for csrCode, csrName := range riscv.CSRs {
			// The set of RVV special operand names and the set of CSR special operands
			// names are disjoint and so can safely share a single namespace. However,
			// it's possible that a future update to the CSRs in inst.go could introduce
			// a conflict. This check ensures that such a conflict does not go
			// unnoticed.
			if _, ok := riscv64SpecialOperand[csrName]; ok {
				panic(fmt.Sprintf("riscv64 special operand %q redefined", csrName))
			}
			riscv64SpecialOperand[csrName] = riscv.SpecialOperand(int(csrCode) + int(riscv.SPOP_CSR_BEGIN))
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
