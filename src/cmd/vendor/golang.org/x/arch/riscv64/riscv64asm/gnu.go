// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package riscv64asm

import (
	"strings"
)

// GNUSyntax returns the GNU assembler syntax for the instruction, as defined by GNU binutils.
// This form typically matches the syntax defined in the RISC-V Instruction Set Manual. See
// https://github.com/riscv/riscv-isa-manual/releases/download/Ratified-IMAFDQC/riscv-spec-20191213.pdf
func GNUSyntax(inst Inst) string {
	hasVectorArg := false
	var args []string
	for _, a := range inst.Args {
		if a == nil {
			break
		}
		args = append(args, strings.ToLower(a.String()))
		if r, ok := a.(Reg); ok {
			hasVectorArg = hasVectorArg || (r >= V0 && r <= V31)
		}
	}

	if hasVectorArg {
		return gnuVectorOp(inst, args)
	}

	op := strings.ToLower(inst.Op.String())
	switch inst.Op {
	case ADDI, ADDIW, ANDI, ORI, SLLI, SLLIW, SRAI, SRAIW, SRLI, SRLIW, XORI:
		if inst.Op == ADDI {
			if inst.Args[1].(Reg) == X0 && inst.Args[0].(Reg) != X0 {
				op = "li"
				args[1] = args[2]
				args = args[:len(args)-1]
				break
			}

			if inst.Args[2].(Simm).Imm == 0 {
				if inst.Args[0].(Reg) == X0 && inst.Args[1].(Reg) == X0 {
					op = "nop"
					args = nil
				} else {
					op = "mv"
					args = args[:len(args)-1]
				}
			}
		}

		if inst.Op == ANDI && inst.Args[2].(Simm).Imm == 255 {
			op = "zext.b"
			args = args[:len(args)-1]
		}

		if inst.Op == ADDIW && inst.Args[2].(Simm).Imm == 0 {
			op = "sext.w"
			args = args[:len(args)-1]
		}

		if inst.Op == XORI && inst.Args[2].(Simm).String() == "-1" {
			op = "not"
			args = args[:len(args)-1]
		}

	case ADD:
		if inst.Args[1].(Reg) == X0 {
			op = "mv"
			args[1] = args[2]
			args = args[:len(args)-1]
		}

	case BEQ:
		if inst.Args[1].(Reg) == X0 {
			op = "beqz"
			args[1] = args[2]
			args = args[:len(args)-1]
		}

	case BGE:
		if inst.Args[1].(Reg) == X0 {
			op = "bgez"
			args[1] = args[2]
			args = args[:len(args)-1]
		} else if inst.Args[0].(Reg) == X0 {
			op = "blez"
			args[0], args[1] = args[1], args[2]
			args = args[:len(args)-1]
		}

	case BLT:
		if inst.Args[1].(Reg) == X0 {
			op = "bltz"
			args[1] = args[2]
			args = args[:len(args)-1]
		} else if inst.Args[0].(Reg) == X0 {
			op = "bgtz"
			args[0], args[1] = args[1], args[2]
			args = args[:len(args)-1]
		}

	case BNE:
		if inst.Args[1].(Reg) == X0 {
			op = "bnez"
			args[1] = args[2]
			args = args[:len(args)-1]
		}

	case CSRRC:
		if inst.Args[0].(Reg) == X0 {
			op = "csrc"
			args[0], args[1] = args[1], args[2]
			args = args[:len(args)-1]
		}

	case CSRRCI:
		if inst.Args[0].(Reg) == X0 {
			op = "csrci"
			args[0], args[1] = args[1], args[2]
			args = args[:len(args)-1]
		}

	case CSRRS:
		if inst.Args[2].(Reg) == X0 {
			switch inst.Args[1].(CSR) {
			case FCSR:
				op = "frcsr"
				args = args[:len(args)-2]

			case FFLAGS:
				op = "frflags"
				args = args[:len(args)-2]

			case FRM:
				op = "frrm"
				args = args[:len(args)-2]

			// rdcycleh, rdinstreth and rdtimeh are RV-32 only instructions.
			// So not included there.
			case CYCLE:
				op = "rdcycle"
				args = args[:len(args)-2]

			case INSTRET:
				op = "rdinstret"
				args = args[:len(args)-2]

			case TIME:
				op = "rdtime"
				args = args[:len(args)-2]

			default:
				op = "csrr"
				args = args[:len(args)-1]
			}
		} else if inst.Args[0].(Reg) == X0 {
			op = "csrs"
			args[0], args[1] = args[1], args[2]
			args = args[:len(args)-1]
		}

	case CSRRSI:
		if inst.Args[0].(Reg) == X0 {
			op = "csrsi"
			args[0], args[1] = args[1], args[2]
			args = args[:len(args)-1]
		}

	case CSRRW:
		switch inst.Args[1].(CSR) {
		case FCSR:
			op = "fscsr"
			if inst.Args[0].(Reg) == X0 {
				args[0] = args[2]
				args = args[:len(args)-2]
			} else {
				args[1] = args[2]
				args = args[:len(args)-1]
			}

		case FFLAGS:
			op = "fsflags"
			if inst.Args[0].(Reg) == X0 {
				args[0] = args[2]
				args = args[:len(args)-2]
			} else {
				args[1] = args[2]
				args = args[:len(args)-1]
			}

		case FRM:
			op = "fsrm"
			if inst.Args[0].(Reg) == X0 {
				args[0] = args[2]
				args = args[:len(args)-2]
			} else {
				args[1] = args[2]
				args = args[:len(args)-1]
			}

		case CYCLE:
			if inst.Args[0].(Reg) == X0 && inst.Args[2].(Reg) == X0 {
				op = "unimp"
				args = nil
			}

		default:
			if inst.Args[0].(Reg) == X0 {
				op = "csrw"
				args[0], args[1] = args[1], args[2]
				args = args[:len(args)-1]
			}
		}

	case CSRRWI:
		if inst.Args[0].(Reg) == X0 {
			op = "csrwi"
			args[0], args[1] = args[1], args[2]
			args = args[:len(args)-1]
		}

	// When both pred and succ equals to iorw, the GNU objdump will omit them.
	case FENCE:
		if inst.Args[0].(MemOrder).String() == "iorw" &&
			inst.Args[1].(MemOrder).String() == "iorw" {
			args = nil
		}

	case FSGNJX_D:
		if inst.Args[1].(Reg) == inst.Args[2].(Reg) {
			op = "fabs.d"
			args = args[:len(args)-1]
		}

	case FSGNJX_S:
		if inst.Args[1].(Reg) == inst.Args[2].(Reg) {
			op = "fabs.s"
			args = args[:len(args)-1]
		}

	case FSGNJ_D:
		if inst.Args[1].(Reg) == inst.Args[2].(Reg) {
			op = "fmv.d"
			args = args[:len(args)-1]
		}

	case FSGNJ_S:
		if inst.Args[1].(Reg) == inst.Args[2].(Reg) {
			op = "fmv.s"
			args = args[:len(args)-1]
		}

	case FSGNJN_D:
		if inst.Args[1].(Reg) == inst.Args[2].(Reg) {
			op = "fneg.d"
			args = args[:len(args)-1]
		}

	case FSGNJN_S:
		if inst.Args[1].(Reg) == inst.Args[2].(Reg) {
			op = "fneg.s"
			args = args[:len(args)-1]
		}

	case JAL:
		if inst.Args[0].(Reg) == X0 {
			op = "j"
			args[0] = args[1]
			args = args[:len(args)-1]
		} else if inst.Args[0].(Reg) == X1 {
			op = "jal"
			args[0] = args[1]
			args = args[:len(args)-1]
		}

	case JALR:
		if inst.Args[0].(Reg) == X1 && inst.Args[1].(RegOffset).Ofs.Imm == 0 {
			args[0] = inst.Args[1].(RegOffset).OfsReg.String()
			args = args[:len(args)-1]
		}

		if inst.Args[0].(Reg) == X0 {
			if inst.Args[1].(RegOffset).OfsReg == X1 && inst.Args[1].(RegOffset).Ofs.Imm == 0 {
				op = "ret"
				args = nil
			} else if inst.Args[1].(RegOffset).Ofs.Imm == 0 {
				op = "jr"
				args[0] = inst.Args[1].(RegOffset).OfsReg.String()
				args = args[:len(args)-1]
			} else {
				op = "jr"
				args[0] = inst.Args[1].(RegOffset).String()
				args = args[:len(args)-1]
			}
		}

	case SLTIU:
		if inst.Args[2].(Simm).String() == "1" {
			op = "seqz"
			args = args[:len(args)-1]
		}

	case SLT:
		if inst.Args[1].(Reg) == X0 {
			op = "sgtz"
			args[1] = args[2]
			args = args[:len(args)-1]
		} else if inst.Args[2].(Reg) == X0 {
			op = "sltz"
			args = args[:len(args)-1]
		}

	case SLTU:
		if inst.Args[1].(Reg) == X0 {
			op = "snez"
			args[1] = args[2]
			args = args[:len(args)-1]
		}

	case SUB:
		if inst.Args[1].(Reg) == X0 {
			op = "neg"
			args[1] = args[2]
			args = args[:len(args)-1]
		}

	case SUBW:
		if inst.Args[1].(Reg) == X0 {
			op = "negw"
			args[1] = args[2]
			args = args[:len(args)-1]
		}

	case VSETVLI, VSETIVLI:
		args[0], args[2] = args[2], strings.ReplaceAll(args[0], " ", "")

	case VSETVL:
		args[0], args[2] = args[2], args[0]
	}

	if args != nil {
		op += " " + strings.Join(args, ",")
	}
	return op
}

func gnuVectorOp(inst Inst, args []string) string {
	// Instruction is either a vector load, store or an arithmetic
	// operation. We can use the inst.Enc to figure out which. Whatever
	// it is, it has at least one argument.

	rawArgs := inst.Args[:]

	var mask string
	var op string
	if inst.Enc&(1<<25) == 0 {
		if implicitMask(inst.Op) {
			mask = "v0"
		} else {
			mask = "v0.t"
			args = args[1:]
			rawArgs = rawArgs[1:]
		}
	}

	if len(args) > 1 {
		if inst.Enc&0x7f == 0x7 || inst.Enc&0x7f == 0x27 {
			// It's a load or a store
			if len(args) >= 2 {
				args[0], args[len(args)-1] = args[len(args)-1], args[0]
			}
			op = pseudoRVVLoad(inst.Op)
		} else {
			// It's an arithmetic instruction

			op, args = pseudoRVVArith(inst.Op, rawArgs, args)

			if len(args) == 3 {
				if imaOrFma(inst.Op) {
					args[0], args[2] = args[2], args[0]
				} else {
					args[0], args[1], args[2] = args[2], args[0], args[1]
				}
			} else if len(args) == 2 {
				args[0], args[1] = args[1], args[0]
			}
		}
	}

	// The mask is always the last argument

	if mask != "" {
		args = append(args, mask)
	}

	if op == "" {
		op = inst.Op.String()
	}
	op = strings.ToLower(op)

	return op + " " + strings.Join(args, ",")
}
