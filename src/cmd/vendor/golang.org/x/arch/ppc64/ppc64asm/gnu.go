// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ppc64asm

import (
	"bytes"
	"fmt"
	"strings"
)

var (
	condBit    = [4]string{"lt", "gt", "eq", "so"}
	condBitNeg = [4]string{"ge", "le", "ne", "so"}
)

// GNUSyntax returns the GNU assembler syntax for the instruction, as defined by GNU binutils.
// This form typically matches the syntax defined in the Power ISA Reference Manual.
func GNUSyntax(inst Inst, pc uint64) string {
	var buf bytes.Buffer
	// When there are all 0s, identify them as the disassembler
	// in binutils would.
	if inst.Enc == 0 {
		return ".long 0x0"
	} else if inst.Op == 0 {
		return "error: unknown instruction"
	}

	PC := pc
	// Special handling for some ops
	startArg := 0
	sep := " "
	switch inst.Op.String() {
	case "bc":
		bo := gnuArg(&inst, 0, inst.Args[0], PC)
		bi := inst.Args[1]
		switch bi := bi.(type) {
		case CondReg:
			if bi >= CR0 {
				if bi == CR0 && bo == "16" {
					buf.WriteString("bdnz")
				}
				buf.WriteString(fmt.Sprintf("bc cr%d", bi-CR0))
			}
			cr := bi / 4
			switch bo {
			case "4":
				bit := condBitNeg[(bi-Cond0LT)%4]
				if cr == 0 {
					buf.WriteString(fmt.Sprintf("b%s", bit))
				} else {
					buf.WriteString(fmt.Sprintf("b%s cr%d,", bit, cr))
					sep = ""
				}
			case "12":
				bit := condBit[(bi-Cond0LT)%4]
				if cr == 0 {
					buf.WriteString(fmt.Sprintf("b%s", bit))
				} else {
					buf.WriteString(fmt.Sprintf("b%s cr%d,", bit, cr))
					sep = ""
				}
			case "8":
				bit := condBit[(bi-Cond0LT)%4]
				sep = ""
				if cr == 0 {
					buf.WriteString(fmt.Sprintf("bdnzt %s,", bit))
				} else {
					buf.WriteString(fmt.Sprintf("bdnzt cr%d,%s,", cr, bit))
				}
			case "16":
				if cr == 0 && bi == Cond0LT {
					buf.WriteString("bdnz")
				} else {
					buf.WriteString(fmt.Sprintf("bdnz cr%d,", cr))
					sep = ""
				}
			}
			startArg = 2
		default:
			fmt.Printf("Unexpected bi: %d for bc with bo: %s\n", bi, bo)
		}
		startArg = 2
	case "mtspr":
		opcode := inst.Op.String()
		buf.WriteString(opcode[0:2])
		switch spr := inst.Args[0].(type) {
		case SpReg:
			switch spr {
			case 1:
				buf.WriteString("xer")
				startArg = 1
			case 8:
				buf.WriteString("lr")
				startArg = 1
			case 9:
				buf.WriteString("ctr")
				startArg = 1
			default:
				buf.WriteString("spr")
			}
		default:
			buf.WriteString("spr")
		}

	case "mfspr":
		opcode := inst.Op.String()
		buf.WriteString(opcode[0:2])
		arg := inst.Args[0]
		switch spr := inst.Args[1].(type) {
		case SpReg:
			switch spr {
			case 1:
				buf.WriteString("xer ")
				buf.WriteString(gnuArg(&inst, 0, arg, PC))
				startArg = 2
			case 8:
				buf.WriteString("lr ")
				buf.WriteString(gnuArg(&inst, 0, arg, PC))
				startArg = 2
			case 9:
				buf.WriteString("ctr ")
				buf.WriteString(gnuArg(&inst, 0, arg, PC))
				startArg = 2
			case 268:
				buf.WriteString("tb ")
				buf.WriteString(gnuArg(&inst, 0, arg, PC))
				startArg = 2
			default:
				buf.WriteString("spr")
			}
		default:
			buf.WriteString("spr")
		}

	default:
		buf.WriteString(inst.Op.String())
	}
	for i, arg := range inst.Args[:] {
		if arg == nil {
			break
		}
		if i < startArg {
			continue
		}
		text := gnuArg(&inst, i, arg, PC)
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
func gnuArg(inst *Inst, argIndex int, arg Arg, pc uint64) string {
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
		// The CondReg can either be found in a CMP, where the
		// condition register field is being set, or in an instruction
		// like a branch or isel that is testing a bit in a condition
		// register field.
		if arg == CR0 && strings.HasPrefix(inst.Op.String(), "cmp") {
			return "" // don't show cr0 for cmp instructions
		} else if arg >= CR0 {
			return fmt.Sprintf("cr%d", int(arg-CR0))
		}
		bit := condBit[(arg-Cond0LT)%4]
		if arg <= Cond0SO {
			return bit
		}
		return fmt.Sprintf("%s cr%d", bit, int(arg-Cond0LT)/4)
	case Imm:
		return fmt.Sprintf("%d", arg)
	case SpReg:
		switch int(arg) {
		case 1:
			return "xer"
		case 8:
			return "lr"
		case 9:
			return "ctr"
		case 268:
			return "tb"
		default:
			return fmt.Sprintf("%d", int(arg))
		}
	case PCRel:
		// If the arg is 0, use the relative address format.
		// Otherwise the pc is meaningful, use absolute address.
		if int(arg) == 0 {
			return fmt.Sprintf(".%+#x", int(arg))
		}
		addr := pc + uint64(int64(arg))
		return fmt.Sprintf("%#x", addr)
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
