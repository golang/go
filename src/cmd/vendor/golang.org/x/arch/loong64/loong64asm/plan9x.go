// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loong64asm

import (
	"fmt"
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
func GoSyntax(inst Inst, pc uint64, symname func(uint64) (string, uint64)) string {
	if symname == nil {
		symname = func(uint64) (string, uint64) { return "", 0 }
	}
	if inst.Op == 0 && inst.Enc == 0 {
		return "WORD $0"
	} else if inst.Op == 0 {
		return "?"
	}

	var args []string
	for _, a := range inst.Args {
		if a == nil {
			break
		}
		args = append(args, plan9Arg(&inst, pc, symname, a))
	}

	var op string = plan9OpMap[inst.Op]
	if op == "" {
		op = "Unknown " + inst.Op.String()
	}

	switch inst.Op {
	case BSTRPICK_W, BSTRPICK_D, BSTRINS_W, BSTRINS_D:
		msbw, lsbw := inst.Args[2].(Uimm), inst.Args[3].(Uimm)
		if inst.Op == BSTRPICK_D && msbw.Imm == 15 && lsbw.Imm == 0 {
			op = "MOVHU"
			args = append(args[1:2], args[0:1]...)
		} else {
			args[0], args[2], args[3] = args[2], args[3], args[0]
		}

	case BCNEZ, BCEQZ:
		args = args[1:2]

	case BEQ, BNE:
		rj := inst.Args[0].(Reg)
		rd := inst.Args[1].(Reg)
		if rj == rd && inst.Op == BEQ {
			op = "JMP"
			args = args[2:]
		} else if rj == R0 {
			args = args[1:]
		} else if rd == R0 {
			args = append(args[:1], args[2:]...)
		}

	case BEQZ, BNEZ:
		if inst.Args[0].(Reg) == R0 && inst.Op == BEQ {
			op = "JMP"
			args = args[1:]
		}

	case BLT, BLTU, BGE, BGEU:
		rj := inst.Args[0].(Reg)
		rd := inst.Args[1].(Reg)
		if rj == rd && (inst.Op == BGE || inst.Op == BGEU) {
			op = "JMP"
			args = args[2:]
		} else if rj == R0 {
			switch inst.Op {
			case BGE:
				op = "BLEZ"
			case BLT:
				op = "BGTZ"
			}
			args = args[1:]
		} else if rd == R0 {
			if !strings.HasSuffix(op, "U") {
				op += "Z"
			}
			args = append(args[:1], args[2:]...)
		}

	case JIRL:
		rd := inst.Args[0].(Reg)
		rj := inst.Args[1].(Reg)
		regno := uint16(rj) & 31
		off := inst.Args[2].(OffsetSimm).Imm
		if rd == R0 && rj == R1 && off == 0 {
			return fmt.Sprintf("RET")
		} else if rd == R0 && off == 0 {
			return fmt.Sprintf("JMP (R%d)", regno)
		} else if rd == R0 {
			return fmt.Sprintf("JMP %d(R%d)", off, regno)
		}
		return fmt.Sprintf("CALL (R%d)", regno)

	case LD_B, LD_H, LD_W, LD_D, LD_BU, LD_HU, LD_WU, LL_W, LL_D,
		ST_B, ST_H, ST_W, ST_D, SC_W, SC_D, FLD_S, FLD_D, FST_S, FST_D:
		var off int32
		switch a := inst.Args[2].(type) {
		case Simm16:
			off = signumConvInt32(int32(a.Imm), a.Width)
		case Simm32:
			off = signumConvInt32(int32(a.Imm), a.Width) >> 2
		}
		Iop := strings.ToUpper(inst.Op.String())
		if strings.HasPrefix(Iop, "L") || strings.HasPrefix(Iop, "FL") {
			return fmt.Sprintf("%s %d(%s), %s", op, off, args[1], args[0])
		}
		return fmt.Sprintf("%s %s, %d(%s)", op, args[0], off, args[1])

	case LDX_B, LDX_H, LDX_W, LDX_D, LDX_BU, LDX_HU, LDX_WU, FLDX_S, FLDX_D,
		STX_B, STX_H, STX_W, STX_D, FSTX_S, FSTX_D:
		Iop := strings.ToUpper(inst.Op.String())
		if strings.HasPrefix(Iop, "L") || strings.HasPrefix(Iop, "FL") {
			return fmt.Sprintf("%s (%s)(%s), %s", op, args[1], args[2], args[0])
		}
		return fmt.Sprintf("%s %s, (%s)(%s)", op, args[0], args[1], args[2])

	case AMADD_B, AMADD_D, AMADD_DB_B, AMADD_DB_D, AMADD_DB_H, AMADD_DB_W, AMADD_H,
		AMADD_W, AMAND_D, AMAND_DB_D, AMAND_DB_W, AMAND_W, AMCAS_B, AMCAS_D, AMCAS_DB_B,
		AMCAS_DB_D, AMCAS_DB_H, AMCAS_DB_W, AMCAS_H, AMCAS_W, AMMAX_D, AMMAX_DB_D,
		AMMAX_DB_DU, AMMAX_DB_W, AMMAX_DB_WU, AMMAX_DU, AMMAX_W, AMMAX_WU, AMMIN_D,
		AMMIN_DB_D, AMMIN_DB_DU, AMMIN_DB_W, AMMIN_DB_WU, AMMIN_DU, AMMIN_W, AMMIN_WU,
		AMOR_D, AMOR_DB_D, AMOR_DB_W, AMOR_W, AMSWAP_B, AMSWAP_D, AMSWAP_DB_B, AMSWAP_DB_D,
		AMSWAP_DB_H, AMSWAP_DB_W, AMSWAP_H, AMSWAP_W, AMXOR_D, AMXOR_DB_D, AMXOR_DB_W, AMXOR_W:
		return fmt.Sprintf("%s %s, (%s), %s", op, args[1], args[2], args[0])

	default:
		// Reverse args, placing dest last
		for i, j := 0, len(args)-1; i < j; i, j = i+1, j-1 {
			args[i], args[j] = args[j], args[i]
		}
		switch len(args) { // Special use cases
		case 0, 1:
			if inst.Op != B && inst.Op != BL {
				return op
			}

		case 3:
			switch a0 := inst.Args[0].(type) {
			case Reg:
				rj := inst.Args[1].(Reg)
				if a0 == rj && a0 != R0 {
					args = args[0:2]
				}
			}
			switch inst.Op {
			case SUB_W, SUB_D, ADDI_W, ADDI_D, ORI:
				rj := inst.Args[1].(Reg)
				if rj == R0 {
					args = append(args[0:1], args[2:]...)
					if inst.Op == SUB_W {
						op = "NEGW"
					} else if inst.Op == SUB_D {
						op = "NEGV"
					} else {
						op = "MOVW"
					}
				}

			case ANDI:
				ui12 := inst.Args[2].(Uimm)
				if ui12.Imm == uint32(0xff) {
					op = "MOVBU"
					args = args[1:]
				} else if ui12.Imm == 0 && inst.Args[0].(Reg) == R0 && inst.Args[1].(Reg) == R0 {
					return "NOOP"
				}

			case SLL_W, OR:
				rk := inst.Args[2].(Reg)
				if rk == R0 {
					args = args[1:]
					if inst.Op == SLL_W {
						op = "MOVW"
					} else {
						op = "MOVV"
					}
				}
			}
		}
	}

	if args != nil {
		op += " " + strings.Join(args, ", ")
	}
	return op
}

func plan9Arg(inst *Inst, pc uint64, symname func(uint64) (string, uint64), arg Arg) string {
	// Reg:			gpr[0, 31] and fpr[0, 31]
	// Fcsr:		fcsr[0, 3]
	// Fcc:			fcc[0, 7]
	// Uimm:		unsigned integer constant
	// Simm16:		si16
	// Simm32:		si32
	// OffsetSimm:	si32
	switch a := arg.(type) {
	case Reg:
		regenum := uint16(a)
		regno := uint16(a) & 0x1f
		// General-purpose register
		if regenum >= uint16(R0) && regenum <= uint16(R31) {
			return fmt.Sprintf("R%d", regno)
		} else { // Float point register
			return fmt.Sprintf("F%d", regno)
		}

	case Fcsr:
		regno := uint8(a) & 0x1f
		return fmt.Sprintf("FCSR%d", regno)

	case Fcc:
		regno := uint8(a) & 0x1f
		return fmt.Sprintf("FCC%d", regno)

	case Uimm:
		return fmt.Sprintf("$%d", a.Imm)

	case Simm16:
		si16 := signumConvInt32(int32(a.Imm), a.Width)
		return fmt.Sprintf("$%d", si16)

	case Simm32:
		si32 := signumConvInt32(a.Imm, a.Width)
		return fmt.Sprintf("$%d", si32)

	case OffsetSimm:
		offs := offsConvInt32(a.Imm, a.Width)
		if inst.Op == B || inst.Op == BL {
			addr := int64(pc) + int64(a.Imm)
			if s, base := symname(uint64(addr)); s != "" && uint64(addr) == base {
				return fmt.Sprintf("%s(SB)", s)
			}
		}
		return fmt.Sprintf("%d(PC)", offs>>2)

	case SaSimm:
		return fmt.Sprintf("$%d", a)

	case CodeSimm:
		return fmt.Sprintf("$%d", a)

	}
	return strings.ToUpper(arg.String())
}

func signumConvInt32(imm int32, width uint8) int32 {
	active := uint32(1<<width) - 1
	signum := uint32(imm) & active
	if ((signum >> (width - 1)) & 0x1) == 1 {
		signum |= ^active
	}
	return int32(signum)
}

func offsConvInt32(imm int32, width uint8) int32 {
	relWidth := width + 2
	return signumConvInt32(imm, relWidth)
}

var plan9OpMap = map[Op]string{
	ADD_W:       "ADD",
	ADD_D:       "ADDV",
	SUB_W:       "SUB",
	SUB_D:       "SUBV",
	ADDI_W:      "ADD",
	ADDI_D:      "ADDV",
	LU12I_W:     "LU12IW",
	LU32I_D:     "LU32ID",
	LU52I_D:     "LU52ID",
	SLT:         "SGT",
	SLTU:        "SGTU",
	SLTI:        "SGT",
	SLTUI:       "SGTU",
	PCADDU12I:   "PCADDU12I",
	PCALAU12I:   "PCALAU12I",
	AND:         "AND",
	OR:          "OR",
	NOR:         "NOR",
	XOR:         "XOR",
	ANDI:        "AND",
	ORI:         "OR",
	XORI:        "XOR",
	MUL_W:       "MUL",
	MULH_W:      "MULH",
	MULH_WU:     "MULHU",
	MUL_D:       "MULV",
	MULH_D:      "MULHV",
	MULH_DU:     "MULHVU",
	DIV_W:       "DIV",
	DIV_WU:      "DIVU",
	DIV_D:       "DIVV",
	DIV_DU:      "DIVVU",
	MOD_W:       "REM",
	MOD_WU:      "REMU",
	MOD_D:       "REMV",
	MOD_DU:      "REMVU",
	SLL_W:       "SLL",
	SRL_W:       "SRL",
	SRA_W:       "SRA",
	ROTR_W:      "ROTR",
	SLL_D:       "SLLV",
	SRL_D:       "SRLV",
	SRA_D:       "SRAV",
	ROTR_D:      "ROTRV",
	SLLI_W:      "SLL",
	SRLI_W:      "SRL",
	SRAI_W:      "SRA",
	ROTRI_W:     "ROTR",
	SLLI_D:      "SLLV",
	SRLI_D:      "SRLV",
	SRAI_D:      "SRAV",
	ROTRI_D:     "ROTRV",
	EXT_W_B:     "?",
	EXT_W_H:     "?",
	BITREV_W:    "BITREVW",
	BITREV_D:    "BITREVV",
	CLO_W:       "CLOW",
	CLO_D:       "CLOV",
	CLZ_W:       "CLZW",
	CLZ_D:       "CLZV",
	CTO_W:       "CTOW",
	CTO_D:       "CTOV",
	CTZ_W:       "CTZW",
	CTZ_D:       "CTZV",
	REVB_2H:     "REVB2H",
	REVB_2W:     "REVB2W",
	REVB_4H:     "REVB4H",
	REVB_D:      "REVBV",
	BSTRPICK_W:  "BSTRPICKW",
	BSTRPICK_D:  "BSTRPICKV",
	BSTRINS_W:   "BSTRINSW",
	BSTRINS_D:   "BSTRINSV",
	MASKEQZ:     "MASKEQZ",
	MASKNEZ:     "MASKNEZ",
	BCNEZ:       "BFPT",
	BCEQZ:       "BFPF",
	BEQ:         "BEQ",
	BNE:         "BNE",
	BEQZ:        "BEQ",
	BNEZ:        "BNE",
	BLT:         "BLT",
	BLTU:        "BLTU",
	BGE:         "BGE",
	BGEU:        "BGEU",
	B:           "JMP",
	BL:          "CALL",
	LD_B:        "MOVB",
	LD_H:        "MOVH",
	LD_W:        "MOVW",
	LD_D:        "MOVV",
	LD_BU:       "MOVBU",
	LD_HU:       "MOVHU",
	LD_WU:       "MOVWU",
	ST_B:        "MOVB",
	ST_H:        "MOVH",
	ST_W:        "MOVW",
	ST_D:        "MOVV",
	LDX_B:       "MOVB",
	LDX_BU:      "MOVBU",
	LDX_D:       "MOVV",
	LDX_H:       "MOVH",
	LDX_HU:      "MOVHU",
	LDX_W:       "MOVW",
	LDX_WU:      "MOVWU",
	STX_B:       "MOVB",
	STX_D:       "MOVV",
	STX_H:       "MOVH",
	STX_W:       "MOVW",
	AMADD_B:     "AMADDB",
	AMADD_D:     "AMADDV",
	AMADD_DB_B:  "AMADDDBB",
	AMADD_DB_D:  "AMADDDBV",
	AMADD_DB_H:  "AMADDDBH",
	AMADD_DB_W:  "AMADDDBW",
	AMADD_H:     "AMADDH",
	AMADD_W:     "AMADDW",
	AMAND_D:     "AMANDV",
	AMAND_DB_D:  "AMANDDBV",
	AMAND_DB_W:  "AMANDDBW",
	AMAND_W:     "AMANDW",
	AMCAS_B:     "AMCASB",
	AMCAS_D:     "AMCASV",
	AMCAS_DB_B:  "AMCASDBB",
	AMCAS_DB_D:  "AMCASDBV",
	AMCAS_DB_H:  "AMCASDBH",
	AMCAS_DB_W:  "AMCASDBW",
	AMCAS_H:     "AMCASH",
	AMCAS_W:     "AMCASW",
	AMMAX_D:     "AMMAXV",
	AMMAX_DB_D:  "AMMAXDBV",
	AMMAX_DB_DU: "AMMAXDBVU",
	AMMAX_DB_W:  "AMMAXDBW",
	AMMAX_DB_WU: "AMMAXDBWU",
	AMMAX_DU:    "AMMAXVU",
	AMMAX_W:     "AMMAXW",
	AMMAX_WU:    "AMMAXWU",
	AMMIN_D:     "AMMINV",
	AMMIN_DB_D:  "AMMINDBV",
	AMMIN_DB_DU: "AMMINDBVU",
	AMMIN_DB_W:  "AMMINDBW",
	AMMIN_DB_WU: "AMMINDBWU",
	AMMIN_DU:    "AMMINVU",
	AMMIN_W:     "AMMINW",
	AMMIN_WU:    "AMMINWU",
	AMOR_D:      "AMORV",
	AMOR_DB_D:   "AMORDBV",
	AMOR_DB_W:   "AMORDBW",
	AMOR_W:      "AMORW",
	AMSWAP_B:    "AMSWAPB",
	AMSWAP_D:    "AMSWAPV",
	AMSWAP_DB_B: "AMSWAPDBB",
	AMSWAP_DB_D: "AMSWAPDBV",
	AMSWAP_DB_H: "AMSWAPDBH",
	AMSWAP_DB_W: "AMSWAPDBW",
	AMSWAP_H:    "AMSWAPH",
	AMSWAP_W:    "AMSWAPW",
	AMXOR_D:     "AMXORV",
	AMXOR_DB_D:  "AMXORDBV",
	AMXOR_DB_W:  "AMXORDBW",
	AMXOR_W:     "AMXORW",
	LL_W:        "LL",
	LL_D:        "LLV",
	SC_W:        "SC",
	SC_D:        "SCV",
	CRCC_W_B_W:  "CRCCWBW",
	CRCC_W_D_W:  "CRCCWVW",
	CRCC_W_H_W:  "CRCCWHW",
	CRCC_W_W_W:  "CRCCWWW",
	CRC_W_B_W:   "CRCWBW",
	CRC_W_D_W:   "CRCWVW",
	CRC_W_H_W:   "CRCWHW",
	CRC_W_W_W:   "CRCWWW",
	DBAR:        "DBAR",
	SYSCALL:     "SYSCALL",
	BREAK:       "BREAK",
	RDTIMEL_W:   "RDTIMELW",
	RDTIMEH_W:   "RDTIMEHW",
	RDTIME_D:    "RDTIMED",
	CPUCFG:      "CPUCFG",

	// Floating-point instructions
	FADD_S:       "ADDF",
	FADD_D:       "ADDD",
	FSUB_S:       "SUBF",
	FSUB_D:       "SUBD",
	FMUL_S:       "MULF",
	FMUL_D:       "MULD",
	FDIV_S:       "DIVF",
	FDIV_D:       "DIVD",
	FMSUB_S:      "FMSUBF",
	FMSUB_D:      "FMSUBD",
	FMADD_S:      "FMADDF",
	FMADD_D:      "FMADDD",
	FNMADD_S:     "FNMADDF",
	FNMADD_D:     "FNMADDD",
	FNMSUB_S:     "FNMSUBF",
	FNMSUB_D:     "FNMSUBD",
	FABS_S:       "ABSF",
	FABS_D:       "ABSD",
	FNEG_S:       "NEGF",
	FNEG_D:       "NEGD",
	FSQRT_S:      "SQRTF",
	FSQRT_D:      "SQRTD",
	FCOPYSIGN_S:  "FCOPYSGF",
	FCOPYSIGN_D:  "FCOPYSGD",
	FMAX_S:       "FMAXF",
	FMAX_D:       "FMAXD",
	FMIN_S:       "FMINF",
	FMIN_D:       "FMIND",
	FCLASS_S:     "FCLASSF",
	FCLASS_D:     "FCLASSD",
	FCMP_CEQ_S:   "CMPEQF",
	FCMP_CEQ_D:   "CMPEQD",
	FCMP_SLE_S:   "CMPGEF",
	FCMP_SLE_D:   "CMPGED",
	FCMP_SLT_S:   "CMPGTF",
	FCMP_SLT_D:   "CMPGTD",
	FCVT_D_S:     "MOVFD",
	FCVT_S_D:     "MOVDF",
	FFINT_S_W:    "FFINTFW",
	FFINT_S_L:    "FFINTFV",
	FFINT_D_W:    "FFINTDW",
	FFINT_D_L:    "FFINTDV",
	FTINTRM_L_D:  "FTINTRMVD",
	FTINTRM_L_S:  "FTINTRMVF",
	FTINTRM_W_D:  "FTINTRMWD",
	FTINTRM_W_S:  "FTINTRMWF",
	FTINTRNE_L_D: "FTINTRNEVD",
	FTINTRNE_L_S: "FTINTRNEVF",
	FTINTRNE_W_D: "FTINTRNEWD",
	FTINTRNE_W_S: "FTINTRNEWF",
	FTINTRP_L_D:  "FTINTRPVD",
	FTINTRP_L_S:  "FTINTRPVF",
	FTINTRP_W_D:  "FTINTRPWD",
	FTINTRP_W_S:  "FTINTRPWF",
	FTINTRZ_L_D:  "FTINTRZVD",
	FTINTRZ_L_S:  "FTINTRZVF",
	FTINTRZ_W_D:  "FTINTRZWD",
	FTINTRZ_W_S:  "FTINTRZWF",
	FTINT_L_D:    "FTINTVD",
	FTINT_L_S:    "FTINTVF",
	FTINT_W_D:    "FTINTWD",
	FTINT_W_S:    "FTINTWF",
	FRINT_S:      "FRINTS",
	FRINT_D:      "FRINTD",
	FMOV_S:       "MOVF",
	FMOV_D:       "MOVD",
	MOVGR2FR_W:   "MOVW",
	MOVGR2FR_D:   "MOVV",
	MOVFR2GR_S:   "MOVW",
	MOVFR2GR_D:   "MOVV",
	MOVGR2CF:     "MOVV",
	MOVCF2GR:     "MOVV",
	MOVFCSR2GR:   "MOVV",
	MOVGR2FCSR:   "MOVV",
	MOVFR2CF:     "MOVV",
	MOVCF2FR:     "MOVV",
	FLD_S:        "MOVF",
	FLD_D:        "MOVD",
	FST_S:        "MOVF",
	FST_D:        "MOVD",
	FLDX_S:       "MOVF",
	FLDX_D:       "MOVD",
	FSTX_S:       "MOVF",
	FSTX_D:       "MOVD",
}
