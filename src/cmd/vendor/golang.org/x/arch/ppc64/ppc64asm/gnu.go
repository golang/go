// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ppc64asm

import (
	"bytes"
	"fmt"
	"strings"
)

// GNUSyntax returns the GNU assembler syntax for the instruction, as defined by GNU binutils.
// This form typically matches the syntax defined in the Power ISA Reference Manual.
func GNUSyntax(inst Inst) string {
	var buf bytes.Buffer
	if inst.Op == 0 {
		return "error: unkown instruction"
	}
	buf.WriteString(inst.Op.String())
	sep := " "
	for i, arg := range inst.Args[:] {
		if arg == nil {
			break
		}
		text := gnuArg(&inst, i, arg)
		if text == "" {
			continue
		}
		buf.WriteString(sep)
		sep = ","
		buf.WriteString(text)
	}
	return buf.String()
}

// gnuArg formats arg (which is the argIndex's arg in inst) according to GNU rules.
// NOTE: because GNUSyntax is the only caller of this func, and it receives a copy
//       of inst, it's ok to modify inst.Args here.
func gnuArg(inst *Inst, argIndex int, arg Arg) string {
	// special cases for load/store instructions
	if _, ok := arg.(Offset); ok {
		if argIndex+1 == len(inst.Args) || inst.Args[argIndex+1] == nil {
			panic(fmt.Errorf("wrong table: offset not followed by register"))
		}
	}
	switch arg := arg.(type) {
	case Reg:
		if isLoadStoreOp(inst.Op) && argIndex == 1 && arg == R0 {
			return "0"
		}
		return arg.String()
	case CondReg:
		if arg == CR0 && strings.HasPrefix(inst.Op.String(), "cmp") {
			return "" // don't show cr0 for cmp instructions
		} else if arg >= CR0 {
			return fmt.Sprintf("cr%d", int(arg-CR0))
		}
		bit := [4]string{"lt", "gt", "eq", "so"}[(arg-Cond0LT)%4]
		if arg <= Cond0SO {
			return bit
		}
		return fmt.Sprintf("4*cr%d+%s", int(arg-Cond0LT)/4, bit)
	case Imm:
		return fmt.Sprintf("%d", arg)
	case SpReg:
		return fmt.Sprintf("%d", int(arg))
	case PCRel:
		return fmt.Sprintf(".%+#x", int(arg))
	case Label:
		return fmt.Sprintf("%#x", uint32(arg))
	case Offset:
		reg := inst.Args[argIndex+1].(Reg)
		removeArg(inst, argIndex+1)
		if reg == R0 {
			return fmt.Sprintf("%d(0)", int(arg))
		}
		return fmt.Sprintf("%d(r%d)", int(arg), reg-R0)
	}
	return fmt.Sprintf("???(%v)", arg)
}

// removeArg removes the arg in inst.Args[index].
func removeArg(inst *Inst, index int) {
	for i := index; i < len(inst.Args); i++ {
		if i+1 < len(inst.Args) {
			inst.Args[i] = inst.Args[i+1]
		} else {
			inst.Args[i] = nil
		}
	}
}

// isLoadStoreOp returns true if op is a load or store instruction
func isLoadStoreOp(op Op) bool {
	switch op {
	case LBZ, LBZU, LBZX, LBZUX:
		return true
	case LHZ, LHZU, LHZX, LHZUX:
		return true
	case LHA, LHAU, LHAX, LHAUX:
		return true
	case LWZ, LWZU, LWZX, LWZUX:
		return true
	case LWA, LWAX, LWAUX:
		return true
	case LD, LDU, LDX, LDUX:
		return true
	case LQ:
		return true
	case STB, STBU, STBX, STBUX:
		return true
	case STH, STHU, STHX, STHUX:
		return true
	case STW, STWU, STWX, STWUX:
		return true
	case STD, STDU, STDX, STDUX:
		return true
	case STQ:
		return true
	case LHBRX, LWBRX, STHBRX, STWBRX:
		return true
	}
	return false
}
