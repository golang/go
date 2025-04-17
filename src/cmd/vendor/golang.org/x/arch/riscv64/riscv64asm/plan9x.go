// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package riscv64asm

import (
	"fmt"
	"io"
	"strconv"
	"strings"
)

// GoSyntax returns the Go assembler syntax for the instruction.
// The syntax was originally defined by Plan 9.
// The pc is the program counter of the instruction, used for
// expanding PC-relative addresses into absolute ones.
// The symname function queries the symbol table for the program
// being disassembled. Given a target address it returns the name
// and base address of the symbol containing the target, if any;
// otherwise it returns "", 0.
// The reader text should read from the text segment using text addresses
// as offsets; it is used to display pc-relative loads as constant loads.
func GoSyntax(inst Inst, pc uint64, symname func(uint64) (string, uint64), text io.ReaderAt) string {
	if symname == nil {
		symname = func(uint64) (string, uint64) { return "", 0 }
	}

	var args []string
	for _, a := range inst.Args {
		if a == nil {
			break
		}
		args = append(args, plan9Arg(&inst, pc, symname, a))
	}

	op := inst.Op.String()

	switch inst.Op {

	case AMOADD_D, AMOADD_D_AQ, AMOADD_D_RL, AMOADD_D_AQRL, AMOADD_W, AMOADD_W_AQ,
		AMOADD_W_RL, AMOADD_W_AQRL, AMOAND_D, AMOAND_D_AQ, AMOAND_D_RL, AMOAND_D_AQRL,
		AMOAND_W, AMOAND_W_AQ, AMOAND_W_RL, AMOAND_W_AQRL, AMOMAXU_D, AMOMAXU_D_AQ,
		AMOMAXU_D_RL, AMOMAXU_D_AQRL, AMOMAXU_W, AMOMAXU_W_AQ, AMOMAXU_W_RL, AMOMAXU_W_AQRL,
		AMOMAX_D, AMOMAX_D_AQ, AMOMAX_D_RL, AMOMAX_D_AQRL, AMOMAX_W, AMOMAX_W_AQ, AMOMAX_W_RL,
		AMOMAX_W_AQRL, AMOMINU_D, AMOMINU_D_AQ, AMOMINU_D_RL, AMOMINU_D_AQRL, AMOMINU_W,
		AMOMINU_W_AQ, AMOMINU_W_RL, AMOMINU_W_AQRL, AMOMIN_D, AMOMIN_D_AQ, AMOMIN_D_RL,
		AMOMIN_D_AQRL, AMOMIN_W, AMOMIN_W_AQ, AMOMIN_W_RL, AMOMIN_W_AQRL, AMOOR_D, AMOOR_D_AQ,
		AMOOR_D_RL, AMOOR_D_AQRL, AMOOR_W, AMOOR_W_AQ, AMOOR_W_RL, AMOOR_W_AQRL, AMOSWAP_D,
		AMOSWAP_D_AQ, AMOSWAP_D_RL, AMOSWAP_D_AQRL, AMOSWAP_W, AMOSWAP_W_AQ, AMOSWAP_W_RL,
		AMOSWAP_W_AQRL, AMOXOR_D, AMOXOR_D_AQ, AMOXOR_D_RL, AMOXOR_D_AQRL, AMOXOR_W,
		AMOXOR_W_AQ, AMOXOR_W_RL, AMOXOR_W_AQRL, SC_D, SC_D_AQ, SC_D_RL, SC_D_AQRL,
		SC_W, SC_W_AQ, SC_W_RL, SC_W_AQRL:
		// Atomic instructions have special operand order.
		args[2], args[1] = args[1], args[2]

	case ADDI:
		if inst.Args[2].(Simm).Imm == 0 {
			op = "MOV"
			args = args[:len(args)-1]
		}

	case ADDIW:
		if inst.Args[2].(Simm).Imm == 0 {
			op = "MOVW"
			args = args[:len(args)-1]
		}

	case ANDI:
		if inst.Args[2].(Simm).Imm == 255 {
			op = "MOVBU"
			args = args[:len(args)-1]
		}

	case BEQ:
		if inst.Args[1].(Reg) == X0 {
			op = "BEQZ"
			args[1] = args[2]
			args = args[:len(args)-1]
		}
		for i, j := 0, len(args)-1; i < j; i, j = i+1, j-1 {
			args[i], args[j] = args[j], args[i]
		}

	case BGE:
		if inst.Args[1].(Reg) == X0 {
			op = "BGEZ"
			args[1] = args[2]
			args = args[:len(args)-1]
		}
		for i, j := 0, len(args)-1; i < j; i, j = i+1, j-1 {
			args[i], args[j] = args[j], args[i]
		}

	case BLT:
		if inst.Args[1].(Reg) == X0 {
			op = "BLTZ"
			args[1] = args[2]
			args = args[:len(args)-1]
		}
		for i, j := 0, len(args)-1; i < j; i, j = i+1, j-1 {
			args[i], args[j] = args[j], args[i]
		}

	case BNE:
		if inst.Args[1].(Reg) == X0 {
			op = "BNEZ"
			args[1] = args[2]
			args = args[:len(args)-1]
		}
		for i, j := 0, len(args)-1; i < j; i, j = i+1, j-1 {
			args[i], args[j] = args[j], args[i]
		}

	case BLTU, BGEU:
		for i, j := 0, len(args)-1; i < j; i, j = i+1, j-1 {
			args[i], args[j] = args[j], args[i]
		}

	case CSRRW:
		switch inst.Args[1].(CSR) {
		case FCSR:
			op = "FSCSR"
			args[1] = args[2]
			args = args[:len(args)-1]
		case FFLAGS:
			op = "FSFLAGS"
			args[1] = args[2]
			args = args[:len(args)-1]
		case FRM:
			op = "FSRM"
			args[1] = args[2]
			args = args[:len(args)-1]
		case CYCLE:
			if inst.Args[0].(Reg) == X0 && inst.Args[2].(Reg) == X0 {
				op = "UNIMP"
				args = nil
			}
		}

	case CSRRS:
		if inst.Args[2].(Reg) == X0 {
			switch inst.Args[1].(CSR) {
			case FCSR:
				op = "FRCSR"
				args = args[:len(args)-2]
			case FFLAGS:
				op = "FRFLAGS"
				args = args[:len(args)-2]
			case FRM:
				op = "FRRM"
				args = args[:len(args)-2]
			case CYCLE:
				op = "RDCYCLE"
				args = args[:len(args)-2]
			case CYCLEH:
				op = "RDCYCLEH"
				args = args[:len(args)-2]
			case INSTRET:
				op = "RDINSTRET"
				args = args[:len(args)-2]
			case INSTRETH:
				op = "RDINSTRETH"
				args = args[:len(args)-2]
			case TIME:
				op = "RDTIME"
				args = args[:len(args)-2]
			case TIMEH:
				op = "RDTIMEH"
				args = args[:len(args)-2]
			}
		}

	// Fence instruction in plan9 doesn't have any operands.
	case FENCE:
		args = nil

	case FMADD_D, FMADD_H, FMADD_Q, FMADD_S, FMSUB_D, FMSUB_H,
		FMSUB_Q, FMSUB_S, FNMADD_D, FNMADD_H, FNMADD_Q, FNMADD_S,
		FNMSUB_D, FNMSUB_H, FNMSUB_Q, FNMSUB_S:
		args[1], args[3] = args[3], args[1]

	case FSGNJ_S:
		if inst.Args[2] == inst.Args[1] {
			op = "MOVF"
			args = args[:len(args)-1]
		}

	case FSGNJ_D:
		if inst.Args[2] == inst.Args[1] {
			op = "MOVD"
			args = args[:len(args)-1]
		}

	case FSGNJX_S:
		if inst.Args[2] == inst.Args[1] {
			op = "FABSS"
			args = args[:len(args)-1]
		}

	case FSGNJX_D:
		if inst.Args[2] == inst.Args[1] {
			op = "FABSD"
			args = args[:len(args)-1]
		}

	case FSGNJN_S:
		if inst.Args[2] == inst.Args[1] {
			op = "FNEGS"
			args = args[:len(args)-1]
		}

	case FSGNJN_D:
		if inst.Args[2] == inst.Args[1] {
			op = "FNESD"
			args = args[:len(args)-1]
		}

	case LD, SD:
		op = "MOV"
		if inst.Op == SD {
			args[0], args[1] = args[1], args[0]
		}

	case LB, SB:
		op = "MOVB"
		if inst.Op == SB {
			args[0], args[1] = args[1], args[0]
		}

	case LH, SH:
		op = "MOVH"
		if inst.Op == SH {
			args[0], args[1] = args[1], args[0]
		}

	case LW, SW:
		op = "MOVW"
		if inst.Op == SW {
			args[0], args[1] = args[1], args[0]
		}

	case LBU:
		op = "MOVBU"

	case LHU:
		op = "MOVHU"

	case LWU:
		op = "MOVWU"

	case FLW, FSW:
		op = "MOVF"
		if inst.Op == FLW {
			args[0], args[1] = args[1], args[0]
		}

	case FLD, FSD:
		op = "MOVD"
		if inst.Op == FLD {
			args[0], args[1] = args[1], args[0]
		}

	case SUB:
		if inst.Args[1].(Reg) == X0 {
			op = "NEG"
			args[1] = args[2]
			args = args[:len(args)-1]
		}

	case XORI:
		if inst.Args[2].(Simm).String() == "-1" {
			op = "NOT"
			args = args[:len(args)-1]
		}

	case SLTIU:
		if inst.Args[2].(Simm).Imm == 1 {
			op = "SEQZ"
			args = args[:len(args)-1]
		}

	case SLTU:
		if inst.Args[1].(Reg) == X0 {
			op = "SNEZ"
			args[1] = args[2]
			args = args[:len(args)-1]
		}

	case JAL:
		if inst.Args[0].(Reg) == X0 {
			op = "JMP"
			args[0] = args[1]
			args = args[:len(args)-1]
		} else if inst.Args[0].(Reg) == X1 {
			op = "CALL"
			args[0] = args[1]
			args = args[:len(args)-1]
		} else {
			args[0], args[1] = args[1], args[0]
		}

	case JALR:
		if inst.Args[0].(Reg) == X0 {
			if inst.Args[1].(RegOffset).OfsReg == X1 && inst.Args[1].(RegOffset).Ofs.Imm == 0 {
				op = "RET"
				args = nil
				break
			}
			op = "JMP"
			args[0] = args[1]
			args = args[:len(args)-1]
		} else if inst.Args[0].(Reg) == X1 {
			op = "CALL"
			args[0] = args[1]
			args = args[:len(args)-1]
		} else {
			args[0], args[1] = args[1], args[0]
		}
	}

	// Reverse args, placing dest last.
	for i, j := 0, len(args)-1; i < j; i, j = i+1, j-1 {
		args[i], args[j] = args[j], args[i]
	}

	// Change to plan9 opcode format
	// Atomic instructions do not have reorder suffix, so remove them
	op = strings.Replace(op, ".AQRL", "", -1)
	op = strings.Replace(op, ".AQ", "", -1)
	op = strings.Replace(op, ".RL", "", -1)
	op = strings.Replace(op, ".", "", -1)

	if args != nil {
		op += " " + strings.Join(args, ", ")
	}

	return op
}

func plan9Arg(inst *Inst, pc uint64, symname func(uint64) (string, uint64), arg Arg) string {
	switch a := arg.(type) {
	case Uimm:
		return fmt.Sprintf("$%d", uint32(a.Imm))

	case Simm:
		imm, _ := strconv.Atoi(a.String())
		if a.Width == 13 || a.Width == 21 {
			addr := int64(pc) + int64(imm)
			if s, base := symname(uint64(addr)); s != "" && uint64(addr) == base {
				return fmt.Sprintf("%s(SB)", s)
			}
			return fmt.Sprintf("%d(PC)", imm/4)
		}
		return fmt.Sprintf("$%d", int32(imm))

	case Reg:
		if a <= 31 {
			return fmt.Sprintf("X%d", a)
		} else {
			return fmt.Sprintf("F%d", a-32)
		}

	case RegOffset:
		if a.Ofs.Imm == 0 {
			return fmt.Sprintf("(X%d)", a.OfsReg)
		} else {
			return fmt.Sprintf("%s(X%d)", a.Ofs.String(), a.OfsReg)
		}

	case AmoReg:
		return fmt.Sprintf("(X%d)", a.reg)

	default:
		return strings.ToUpper(arg.String())
	}
}
