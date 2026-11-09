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

	// VMOVQ 0(R4), V2 / VMOVQ V2, 0(R4)
	case VLD, VST, XVLD, XVST:
		var off int32
		Iop := strings.ToUpper(inst.Op.String())
		mov := "VMOVQ"
		if strings.HasPrefix(Iop, "X") {
			mov = "XVMOVQ"
		}
		switch a := inst.Args[2].(type) {
		default:
			fmt.Printf("unexpected argument type: %s\n", Iop)
			return ""
		case Simm16:
			off = signumConvInt32(int32(a.Imm), a.Width)
		}
		if strings.Contains(Iop, "LD") {
			return fmt.Sprintf("%s %d(%s), %s", mov, off, args[1], args[0])
		}
		return fmt.Sprintf("%s %s, %d(%s)", mov, args[0], off, args[1])

	// VMOVQ (R4)(R5), V2 / VMOVQ V2, (R5)(R5)
	case VLDX, VSTX, XVLDX, XVSTX:
		Iop := strings.ToUpper(inst.Op.String())
		mov := "VMOVQ"
		if strings.HasPrefix(Iop, "X") {
			mov = "XVMOVQ"
		}
		if strings.Contains(Iop, "LD") {
			return fmt.Sprintf("%s (%s)(%s), %s", mov, args[1], args[2], args[0])
		}
		return fmt.Sprintf("%s %s, (%s)(%s)", mov, args[0], args[1], args[2])

	// VMOVQ V1.B[0], R4 / VMOVQ R4, V1.B[0]
	case VPICKVE2GR_B, VPICKVE2GR_H, VPICKVE2GR_W, VPICKVE2GR_D,
		VPICKVE2GR_BU, VPICKVE2GR_HU, VPICKVE2GR_WU, VPICKVE2GR_DU,
		XVPICKVE2GR_W, XVPICKVE2GR_D, XVPICKVE2GR_WU, XVPICKVE2GR_DU,
		VINSGR2VR_B, VINSGR2VR_H, VINSGR2VR_W, VINSGR2VR_D,
		XVINSGR2VR_W, XVINSGR2VR_D,
		XVINSVE0_W, XVINSVE0_D, XVPICKVE_W, XVPICKVE_D:
		Iop := strings.ToUpper(inst.Op.String())
		suf := strings.Split(Iop, ".")[1]
		suf = strings.ReplaceAll(suf, "D", "V")
		mov := "VMOVQ"
		if strings.HasPrefix(Iop, "X") {
			mov = "XVMOVQ"
		}
		var index uint32
		switch a := inst.Args[2].(type) {
		default:
			fmt.Printf("unexpected argument type: %s\n", Iop)
			return ""
		case Uimm:
			index = a.Imm
		}
		if strings.Contains(Iop, "INS") {
			return fmt.Sprintf("%s %s, %s.%s[%d]", mov, args[1], args[0], suf, index)
		}
		return fmt.Sprintf("%s %s.%s[%d], %s", mov, args[1], suf, index, args[0])

	// VMOVQ R4, V2.B16
	case VREPLGR2VR_B, VREPLGR2VR_H, VREPLGR2VR_W, VREPLGR2VR_D,
		XVREPLGR2VR_B, XVREPLGR2VR_H, XVREPLGR2VR_W, XVREPLGR2VR_D,
		XVREPLVE0_B, XVREPLVE0_H, XVREPLVE0_W, XVREPLVE0_D, XVREPLVE0_Q:
		Iop := strings.ToUpper(inst.Op.String())
		suf := strings.Split(Iop, ".")[1]
		suf = strings.ReplaceAll(suf, "D", "V")
		count := movCount[suf]
		mov := "VMOVQ"
		if strings.HasPrefix(Iop, "X") {
			mov = "XVMOVQ"
			count = 2 * count
		}
		return fmt.Sprintf("%s %s, %s.%s%d", mov, args[1], args[0], suf, count)

	// VMOVQ V1.B[0], V2.B16
	case VREPLVEI_B, VREPLVEI_H, VREPLVEI_W, VREPLVEI_D:
		Iop := strings.ToUpper(inst.Op.String())
		suf := strings.Split(Iop, ".")[1]
		suf = strings.ReplaceAll(suf, "D", "V")
		count := movCount[suf]
		var index uint32
		switch a := inst.Args[2].(type) {
		default:
			fmt.Printf("unexpected argument type: %s\n", Iop)
			return ""
		case Uimm:
			index = a.Imm
		}
		mov := "VMOVQ"
		return fmt.Sprintf("%s %s.%s[%d], %s.%s%d", mov, args[1], suf, index, args[0], suf, count)

	// VMOVQ 4(R4), V0.B16
	case VLDREPL_B, VLDREPL_H, VLDREPL_W, VLDREPL_D,
		XVLDREPL_B, XVLDREPL_H, XVLDREPL_W, XVLDREPL_D:
		Iop := strings.ToUpper(inst.Op.String())
		suf := strings.Split(Iop, ".")[1]
		suf = strings.ReplaceAll(suf, "D", "V")
		var off int32
		switch a := inst.Args[2].(type) {
		default:
			fmt.Printf("unexpected argument type: %s\n", Iop)
			return ""
		case Simm16:
			off = signumConvInt32(int32(a.Imm), a.Width)
		}
		count := movCount[suf]
		off *= SufWidth[suf]
		mov := "VMOVQ"
		if strings.HasPrefix(Iop, "X") {
			mov = "XVMOVQ"
			count *= 2
		}
		return fmt.Sprintf("%s %d(%s), %s.%s%d", mov, off, args[1], args[0], suf, count)

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

			case VSLLI_D, XVSLLI_D:
				if inst.Args[2].(Uimm).Imm == 0 {
					switch inst.Op {
					case VSLLI_D:
						op = "VMOVQ"
					case XVSLLI_D:
						op = "XVMOVQ"
					}
					args = args[1:]
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
		} else if regenum >= uint16(F0) && regenum <= uint16(F31) { // Float point register
			return fmt.Sprintf("F%d", regno)
		} else if regenum >= uint16(V0) && regenum <= uint16(V31) {
			return fmt.Sprintf("V%d", regno)
		} else {
			return fmt.Sprintf("X%d", regno)
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

	VSEQ_B:   "VSEQB",
	VSEQ_H:   "VSEQH",
	VSEQ_W:   "VSEQW",
	VSEQ_D:   "VSEQV",
	XVSEQ_B:  "XVSEQB",
	XVSEQ_H:  "XVSEQH",
	XVSEQ_W:  "XVSEQW",
	XVSEQ_D:  "XVSEQV",
	VSEQI_B:  "VSEQB",
	VSEQI_H:  "VSEQH",
	VSEQI_W:  "VSEQW",
	VSEQI_D:  "VSEQV",
	XVSEQI_B: "XVSEQB",
	XVSEQI_H: "XVSEQH",
	XVSEQI_W: "XVSEQW",
	XVSEQI_D: "XVSEQV",

	VSLT_B:   "VSLTB",
	VSLT_H:   "VSLTH",
	VSLT_W:   "VSLTW",
	VSLT_D:   "VSLTV",
	XVSLT_B:  "XVSLTB",
	XVSLT_H:  "XVSLTH",
	XVSLT_W:  "XVSLTW",
	XVSLT_D:  "XVSLTV",
	VSLTI_B:  "VSLTB",
	VSLTI_H:  "VSLTH",
	VSLTI_W:  "VSLTW",
	VSLTI_D:  "VSLTV",
	XVSLTI_B: "XVSLTB",
	XVSLTI_H: "XVSLTH",
	XVSLTI_W: "XVSLTW",
	XVSLTI_D: "XVSLTV",

	VSLT_BU:   "VSLTBU",
	VSLT_HU:   "VSLTHU",
	VSLT_WU:   "VSLTWU",
	VSLT_DU:   "VSLTVU",
	XVSLT_BU:  "XVSLTBU",
	XVSLT_HU:  "XVSLTHU",
	XVSLT_WU:  "XVSLTWU",
	XVSLT_DU:  "XVSLTVU",
	VSLTI_BU:  "VSLTBU",
	VSLTI_HU:  "VSLTHU",
	VSLTI_WU:  "VSLTWU",
	VSLTI_DU:  "VSLTVU",
	XVSLTI_BU: "XVSLTBU",
	XVSLTI_HU: "XVSLTHU",
	XVSLTI_WU: "XVSLTWU",
	XVSLTI_DU: "XVSLTVU",

	VPCNT_B:  "VPCNTB",
	VPCNT_H:  "VPCNTH",
	VPCNT_W:  "VPCNTW",
	VPCNT_D:  "VPCNTV",
	XVPCNT_B: "XVPCNTB",
	XVPCNT_H: "XVPCNTH",
	XVPCNT_W: "XVPCNTW",
	XVPCNT_D: "XVPCNTV",

	VAND_V:  "VANDV",
	VOR_V:   "VORV",
	VXOR_V:  "VXORV",
	VNOR_V:  "VNORV",
	VANDN_V: "VANDNV",
	VORN_V:  "VORNV",

	VANDI_B: "VANDB",
	VORI_B:  "VORB",
	VXORI_B: "VXORB",
	VNORI_B: "VNORB",

	XVAND_V:  "XVANDV",
	XVOR_V:   "XVORV",
	XVXOR_V:  "XVXORV",
	XVNOR_V:  "XVNORV",
	XVANDN_V: "XVANDNV",
	XVORN_V:  "XVORNV",

	XVANDI_B: "XVANDB",
	XVORI_B:  "XVORB",
	XVXORI_B: "XVXORB",
	XVNORI_B: "XVNORB",

	VSLL_B:   "VSLLB",
	VSLL_H:   "VSLLH",
	VSLL_W:   "VSLLW",
	VSLL_D:   "VSLLV",
	VSRL_B:   "VSRLB",
	VSRL_H:   "VSRLH",
	VSRL_W:   "VSRLW",
	VSRL_D:   "VSRLV",
	VSRA_B:   "VSRAB",
	VSRA_H:   "VSRAH",
	VSRA_W:   "VSRAW",
	VSRA_D:   "VSRAV",
	VROTR_B:  "VROTRB",
	VROTR_H:  "VROTRH",
	VROTR_W:  "VROTRW",
	VROTR_D:  "VROTRV",
	XVSLL_B:  "XVSLLB",
	XVSLL_H:  "XVSLLH",
	XVSLL_W:  "XVSLLW",
	XVSLL_D:  "XVSLLV",
	XVSRL_B:  "XVSRLB",
	XVSRL_H:  "XVSRLH",
	XVSRL_W:  "XVSRLW",
	XVSRL_D:  "XVSRLV",
	XVSRA_B:  "XVSRAB",
	XVSRA_H:  "XVSRAH",
	XVSRA_W:  "XVSRAW",
	XVSRA_D:  "XVSRAV",
	XVROTR_B: "XVROTRB",
	XVROTR_H: "XVROTRH",
	XVROTR_W: "XVROTRW",
	XVROTR_D: "XVROTRV",

	VSLLI_B:   "VSLLB",
	VSLLI_H:   "VSLLH",
	VSLLI_W:   "VSLLW",
	VSLLI_D:   "VSLLV",
	VSRLI_B:   "VSRLB",
	VSRLI_H:   "VSRLH",
	VSRLI_W:   "VSRLW",
	VSRLI_D:   "VSRLV",
	VSRAI_B:   "VSRAB",
	VSRAI_H:   "VSRAH",
	VSRAI_W:   "VSRAW",
	VSRAI_D:   "VSRAV",
	VROTRI_B:  "VROTRB",
	VROTRI_H:  "VROTRH",
	VROTRI_W:  "VROTRW",
	VROTRI_D:  "VROTRV",
	XVSLLI_B:  "XVSLLB",
	XVSLLI_H:  "XVSLLH",
	XVSLLI_W:  "XVSLLW",
	XVSLLI_D:  "XVSLLV",
	XVSRLI_B:  "XVSRLB",
	XVSRLI_H:  "XVSRLH",
	XVSRLI_W:  "XVSRLW",
	XVSRLI_D:  "XVSRLV",
	XVSRAI_B:  "XVSRAB",
	XVSRAI_H:  "XVSRAH",
	XVSRAI_W:  "XVSRAW",
	XVSRAI_D:  "XVSRAV",
	XVROTRI_B: "XVROTRB",
	XVROTRI_H: "XVROTRH",
	XVROTRI_W: "XVROTRW",
	XVROTRI_D: "XVROTRV",

	VADD_B:  "VADDB",
	VADD_H:  "VADDH",
	VADD_W:  "VADDW",
	VADD_D:  "VADDV",
	VADD_Q:  "VADDQ",
	VSUB_B:  "VSUBB",
	VSUB_H:  "VSUBH",
	VSUB_W:  "VSUBW",
	VSUB_D:  "VSUBV",
	VSUB_Q:  "VSUBQ",
	XVADD_B: "XVADDB",
	XVADD_H: "XVADDH",
	XVADD_W: "XVADDW",
	XVADD_D: "XVADDV",
	XVADD_Q: "XVADDQ",
	XVSUB_B: "XVSUBB",
	XVSUB_H: "XVSUBH",
	XVSUB_W: "XVSUBW",
	XVSUB_D: "XVSUBV",
	XVSUB_Q: "XVSUBQ",

	VADDI_BU:  "VADDBU",
	VADDI_HU:  "VADDHU",
	VADDI_WU:  "VADDWU",
	VADDI_DU:  "VADDVU",
	VSUBI_BU:  "VSUBBU",
	VSUBI_HU:  "VSUBHU",
	VSUBI_WU:  "VSUBWU",
	VSUBI_DU:  "VSUBVU",
	XVADDI_BU: "XVADDBU",
	XVADDI_HU: "XVADDHU",
	XVADDI_WU: "XVADDWU",
	XVADDI_DU: "XVADDVU",
	XVSUBI_BU: "XVSUBBU",
	XVSUBI_HU: "XVSUBHU",
	XVSUBI_WU: "XVSUBWU",
	XVSUBI_DU: "XVSUBVU",

	VSADD_B:  "VSADDB",
	VSADD_H:  "VSADDH",
	VSADD_W:  "VSADDW",
	VSADD_D:  "VSADDV",
	VSSUB_B:  "VSSUBB",
	VSSUB_H:  "VSSUBH",
	VSSUB_W:  "VSSUBW",
	VSSUB_D:  "VSSUBV",
	XVSADD_B: "XVSADDB",
	XVSADD_H: "XVSADDH",
	XVSADD_W: "XVSADDW",
	XVSADD_D: "XVSADDV",
	XVSSUB_B: "XVSSUBB",
	XVSSUB_H: "XVSSUBH",
	XVSSUB_W: "XVSSUBW",
	XVSSUB_D: "XVSSUBV",

	VSADD_BU:  "VSADDBU",
	VSADD_HU:  "VSADDHU",
	VSADD_WU:  "VSADDWU",
	VSADD_DU:  "VSADDVU",
	VSSUB_BU:  "VSSUBBU",
	VSSUB_HU:  "VSSUBHU",
	VSSUB_WU:  "VSSUBWU",
	VSSUB_DU:  "VSSUBVU",
	XVSADD_BU: "XVSADDBU",
	XVSADD_HU: "XVSADDHU",
	XVSADD_WU: "XVSADDWU",
	XVSADD_DU: "XVSADDVU",
	XVSSUB_BU: "XVSSUBBU",
	XVSSUB_HU: "XVSSUBHU",
	XVSSUB_WU: "XVSSUBWU",
	XVSSUB_DU: "XVSSUBVU",

	VILVL_B:  "VILVLB",
	VILVL_H:  "VILVLH",
	VILVL_W:  "VILVLW",
	VILVL_D:  "VILVLV",
	VILVH_B:  "VILVHB",
	VILVH_H:  "VILVHH",
	VILVH_W:  "VILVHW",
	VILVH_D:  "VILVHV",
	XVILVL_B: "XVILVLB",
	XVILVL_H: "XVILVLH",
	XVILVL_W: "XVILVLW",
	XVILVL_D: "XVILVLV",
	XVILVH_B: "XVILVHB",
	XVILVH_H: "XVILVHH",
	XVILVH_W: "XVILVHW",
	XVILVH_D: "XVILVHV",

	VMUL_B:   "VMULB",
	VMUL_H:   "VMULH",
	VMUL_W:   "VMULW",
	VMUL_D:   "VMULV",
	VMUH_B:   "VMUHB",
	VMUH_H:   "VMUHH",
	VMUH_W:   "VMUHW",
	VMUH_D:   "VMUHV",
	VMUH_BU:  "VMUHBU",
	VMUH_HU:  "VMUHHU",
	VMUH_WU:  "VMUHWU",
	VMUH_DU:  "VMUHVU",
	XVMUL_B:  "XVMULB",
	XVMUL_H:  "XVMULH",
	XVMUL_W:  "XVMULW",
	XVMUL_D:  "XVMULV",
	XVMUH_B:  "XVMUHB",
	XVMUH_H:  "XVMUHH",
	XVMUH_W:  "XVMUHW",
	XVMUH_D:  "XVMUHV",
	XVMUH_BU: "XVMUHBU",
	XVMUH_HU: "XVMUHHU",
	XVMUH_WU: "XVMUHWU",
	XVMUH_DU: "XVMUHVU",

	VDIV_B:   "VDIVB",
	VDIV_H:   "VDIVH",
	VDIV_W:   "VDIVW",
	VDIV_D:   "VDIVV",
	VDIV_BU:  "VDIVBU",
	VDIV_HU:  "VDIVHU",
	VDIV_WU:  "VDIVWU",
	VDIV_DU:  "VDIVVU",
	VMOD_B:   "VMODB",
	VMOD_H:   "VMODH",
	VMOD_W:   "VMODW",
	VMOD_D:   "VMODV",
	VMOD_BU:  "VMODBU",
	VMOD_HU:  "VMODHU",
	VMOD_WU:  "VMODWU",
	VMOD_DU:  "VMODVU",
	XVDIV_B:  "XVDIVB",
	XVDIV_H:  "XVDIVH",
	XVDIV_W:  "XVDIVW",
	XVDIV_D:  "XVDIVV",
	XVDIV_BU: "XVDIVBU",
	XVDIV_HU: "XVDIVHU",
	XVDIV_WU: "XVDIVWU",
	XVDIV_DU: "XVDIVVU",
	XVMOD_B:  "XVMODB",
	XVMOD_H:  "XVMODH",
	XVMOD_W:  "XVMODW",
	XVMOD_D:  "XVMODV",
	XVMOD_BU: "XVMODBU",
	XVMOD_HU: "XVMODHU",
	XVMOD_WU: "XVMODWU",
	XVMOD_DU: "XVMODVU",

	VFSQRT_S:   "VFSQRTF",
	VFSQRT_D:   "VFSQRTD",
	VFRECIP_S:  "VFRECIPF",
	VFRECIP_D:  "VFRECIPD",
	VFRSQRT_S:  "VFRSQRTF",
	VFRSQRT_D:  "VFRSQRTD",
	XVFSQRT_S:  "XVFSQRTF",
	XVFSQRT_D:  "XVFSQRTD",
	XVFRECIP_S: "XVFRECIPF",
	XVFRECIP_D: "XVFRECIPD",
	XVFRSQRT_S: "XVFRSQRTF",
	XVFRSQRT_D: "XVFRSQRTD",
	VNEG_B:     "VNEGB",
	VNEG_H:     "VNEGH",
	VNEG_W:     "VNEGW",
	VNEG_D:     "VNEGV",
	XVNEG_B:    "XVNEGB",
	XVNEG_H:    "XVNEGH",
	XVNEG_W:    "XVNEGW",
	XVNEG_D:    "XVNEGV",

	VMULWEV_H_B:   "VMULWEVHB",
	VMULWEV_W_H:   "VMULWEVWH",
	VMULWEV_D_W:   "VMULWEVVW",
	VMULWEV_Q_D:   "VMULWEVQV",
	VMULWOD_H_B:   "VMULWODHB",
	VMULWOD_W_H:   "VMULWODWH",
	VMULWOD_D_W:   "VMULWODVW",
	VMULWOD_Q_D:   "VMULWODQV",
	VMULWEV_H_BU:  "VMULWEVHBU",
	VMULWEV_W_HU:  "VMULWEVWHU",
	VMULWEV_D_WU:  "VMULWEVVWU",
	VMULWEV_Q_DU:  "VMULWEVQVU",
	VMULWOD_H_BU:  "VMULWODHBU",
	VMULWOD_W_HU:  "VMULWODWHU",
	VMULWOD_D_WU:  "VMULWODVWU",
	VMULWOD_Q_DU:  "VMULWODQVU",
	XVMULWEV_H_B:  "XVMULWEVHB",
	XVMULWEV_W_H:  "XVMULWEVWH",
	XVMULWEV_D_W:  "XVMULWEVVW",
	XVMULWEV_Q_D:  "XVMULWEVQV",
	XVMULWOD_H_B:  "XVMULWODHB",
	XVMULWOD_W_H:  "XVMULWODWH",
	XVMULWOD_D_W:  "XVMULWODVW",
	XVMULWOD_Q_D:  "XVMULWODQV",
	XVMULWEV_H_BU: "XVMULWEVHBU",
	XVMULWEV_W_HU: "XVMULWEVWHU",
	XVMULWEV_D_WU: "XVMULWEVVWU",
	XVMULWEV_Q_DU: "XVMULWEVQVU",
	XVMULWOD_H_BU: "XVMULWODHBU",
	XVMULWOD_W_HU: "XVMULWODWHU",
	XVMULWOD_D_WU: "XVMULWODVWU",
	XVMULWOD_Q_DU: "XVMULWODQVU",

	VMULWEV_H_BU_B:  "VMULWEVHBUB",
	VMULWEV_W_HU_H:  "VMULWEVWHUH",
	VMULWEV_D_WU_W:  "VMULWEVVWUW",
	VMULWEV_Q_DU_D:  "VMULWEVQVUV",
	VMULWOD_H_BU_B:  "VMULWODHBUB",
	VMULWOD_W_HU_H:  "VMULWODWHUH",
	VMULWOD_D_WU_W:  "VMULWODVWUW",
	VMULWOD_Q_DU_D:  "VMULWODQVUV",
	XVMULWEV_H_BU_B: "XVMULWEVHBUB",
	XVMULWEV_W_HU_H: "XVMULWEVWHUH",
	XVMULWEV_D_WU_W: "XVMULWEVVWUW",
	XVMULWEV_Q_DU_D: "XVMULWEVQVUV",
	XVMULWOD_H_BU_B: "XVMULWODHBUB",
	XVMULWOD_W_HU_H: "XVMULWODWHUH",
	XVMULWOD_D_WU_W: "XVMULWODVWUW",
	XVMULWOD_Q_DU_D: "XVMULWODQVUV",

	VADDWEV_H_B:  "VADDWEVHB",
	VADDWEV_W_H:  "VADDWEVWH",
	VADDWEV_D_W:  "VADDWEVVW",
	VADDWEV_Q_D:  "VADDWEVQV",
	VADDWOD_H_B:  "VADDWODHB",
	VADDWOD_W_H:  "VADDWODWH",
	VADDWOD_D_W:  "VADDWODVW",
	VADDWOD_Q_D:  "VADDWODQV",
	XVADDWEV_H_B: "XVADDWEVHB",
	XVADDWEV_W_H: "XVADDWEVWH",
	XVADDWEV_D_W: "XVADDWEVVW",
	XVADDWEV_Q_D: "XVADDWEVQV",
	XVADDWOD_H_B: "XVADDWODHB",
	XVADDWOD_W_H: "XVADDWODWH",
	XVADDWOD_D_W: "XVADDWODVW",
	XVADDWOD_Q_D: "XVADDWODQV",

	VSUBWEV_H_B:  "VSUBWEVHB",
	VSUBWEV_W_H:  "VSUBWEVWH",
	VSUBWEV_D_W:  "VSUBWEVVW",
	VSUBWEV_Q_D:  "VSUBWEVQV",
	VSUBWOD_H_B:  "VSUBWODHB",
	VSUBWOD_W_H:  "VSUBWODWH",
	VSUBWOD_D_W:  "VSUBWODVW",
	VSUBWOD_Q_D:  "VSUBWODQV",
	XVSUBWEV_H_B: "XVSUBWEVHB",
	XVSUBWEV_W_H: "XVSUBWEVWH",
	XVSUBWEV_D_W: "XVSUBWEVVW",
	XVSUBWEV_Q_D: "XVSUBWEVQV",
	XVSUBWOD_H_B: "XVSUBWODHB",
	XVSUBWOD_W_H: "XVSUBWODWH",
	XVSUBWOD_D_W: "XVSUBWODVW",
	XVSUBWOD_Q_D: "XVSUBWODQV",

	VADDWEV_H_BU:  "VADDWEVHBU",
	VADDWEV_W_HU:  "VADDWEVWHU",
	VADDWEV_D_WU:  "VADDWEVVWU",
	VADDWEV_Q_DU:  "VADDWEVQVU",
	VADDWOD_H_BU:  "VADDWODHBU",
	VADDWOD_W_HU:  "VADDWODWHU",
	VADDWOD_D_WU:  "VADDWODVWU",
	VADDWOD_Q_DU:  "VADDWODQVU",
	XVADDWEV_H_BU: "XVADDWEVHBU",
	XVADDWEV_W_HU: "XVADDWEVWHU",
	XVADDWEV_D_WU: "XVADDWEVVWU",
	XVADDWEV_Q_DU: "XVADDWEVQVU",
	XVADDWOD_H_BU: "XVADDWODHBU",
	XVADDWOD_W_HU: "XVADDWODWHU",
	XVADDWOD_D_WU: "XVADDWODVWU",
	XVADDWOD_Q_DU: "XVADDWODQVU",

	VSUBWEV_H_BU:  "VSUBWEVHBU",
	VSUBWEV_W_HU:  "VSUBWEVWHU",
	VSUBWEV_D_WU:  "VSUBWEVVWU",
	VSUBWEV_Q_DU:  "VSUBWEVQVU",
	VSUBWOD_H_BU:  "VSUBWODHBU",
	VSUBWOD_W_HU:  "VSUBWODWHU",
	VSUBWOD_D_WU:  "VSUBWODVWU",
	VSUBWOD_Q_DU:  "VSUBWODQVU",
	XVSUBWEV_H_BU: "XVSUBWEVHBU",
	XVSUBWEV_W_HU: "XVSUBWEVWHU",
	XVSUBWEV_D_WU: "XVSUBWEVVWU",
	XVSUBWEV_Q_DU: "XVSUBWEVQVU",
	XVSUBWOD_H_BU: "XVSUBWODHBU",
	XVSUBWOD_W_HU: "XVSUBWODWHU",
	XVSUBWOD_D_WU: "XVSUBWODVWU",
	XVSUBWOD_Q_DU: "XVSUBWODQVU",

	VMADD_B:  "VMADDB",
	VMADD_H:  "VMADDH",
	VMADD_W:  "VMADDW",
	VMADD_D:  "VMADDV",
	VMSUB_B:  "VMSUBB",
	VMSUB_H:  "VMSUBH",
	VMSUB_W:  "VMSUBW",
	VMSUB_D:  "VMSUBV",
	XVMADD_B: "XVMADDB",
	XVMADD_H: "XVMADDH",
	XVMADD_W: "XVMADDW",
	XVMADD_D: "XVMADDV",
	XVMSUB_B: "XVMSUBB",
	XVMSUB_H: "XVMSUBH",
	XVMSUB_W: "XVMSUBW",
	XVMSUB_D: "XVMSUBV",

	VMADDWEV_H_B:  "VMADDWEVHB",
	VMADDWEV_W_H:  "VMADDWEVWH",
	VMADDWEV_D_W:  "VMADDWEVVW",
	VMADDWEV_Q_D:  "VMADDWEVQV",
	VMADDWOD_H_B:  "VMADDWODHB",
	VMADDWOD_W_H:  "VMADDWODWH",
	VMADDWOD_D_W:  "VMADDWODVW",
	VMADDWOD_Q_D:  "VMADDWODQV",
	XVMADDWEV_H_B: "XVMADDWEVHB",
	XVMADDWEV_W_H: "XVMADDWEVWH",
	XVMADDWEV_D_W: "XVMADDWEVVW",
	XVMADDWEV_Q_D: "XVMADDWEVQV",
	XVMADDWOD_H_B: "XVMADDWODHB",
	XVMADDWOD_W_H: "XVMADDWODWH",
	XVMADDWOD_D_W: "XVMADDWODVW",
	XVMADDWOD_Q_D: "XVMADDWODQV",

	VMADDWEV_H_BU:  "VMADDWEVHBU",
	VMADDWEV_W_HU:  "VMADDWEVWHU",
	VMADDWEV_D_WU:  "VMADDWEVVWU",
	VMADDWEV_Q_DU:  "VMADDWEVQVU",
	VMADDWOD_H_BU:  "VMADDWODHBU",
	VMADDWOD_W_HU:  "VMADDWODWHU",
	VMADDWOD_D_WU:  "VMADDWODVWU",
	VMADDWOD_Q_DU:  "VMADDWODQVU",
	XVMADDWEV_H_BU: "XVMADDWEVHBU",
	XVMADDWEV_W_HU: "XVMADDWEVWHU",
	XVMADDWEV_D_WU: "XVMADDWEVVWU",
	XVMADDWEV_Q_DU: "XVMADDWEVQVU",
	XVMADDWOD_H_BU: "XVMADDWODHBU",
	XVMADDWOD_W_HU: "XVMADDWODWHU",
	XVMADDWOD_D_WU: "XVMADDWODVWU",
	XVMADDWOD_Q_DU: "XVMADDWODQVU",

	VMADDWEV_H_BU_B:  "VMADDWEVHBUB",
	VMADDWEV_W_HU_H:  "VMADDWEVWHUH",
	VMADDWEV_D_WU_W:  "VMADDWEVVWUW",
	VMADDWEV_Q_DU_D:  "VMADDWEVQVUV",
	VMADDWOD_H_BU_B:  "VMADDWODHBUB",
	VMADDWOD_W_HU_H:  "VMADDWODWHUH",
	VMADDWOD_D_WU_W:  "VMADDWODVWUW",
	VMADDWOD_Q_DU_D:  "VMADDWODQVUV",
	XVMADDWEV_H_BU_B: "XVMADDWEVHBUB",
	XVMADDWEV_W_HU_H: "XVMADDWEVWHUH",
	XVMADDWEV_D_WU_W: "XVMADDWEVVWUW",
	XVMADDWEV_Q_DU_D: "XVMADDWEVQVUV",
	XVMADDWOD_H_BU_B: "XVMADDWODHBUB",
	XVMADDWOD_W_HU_H: "XVMADDWODWHUH",
	XVMADDWOD_D_WU_W: "XVMADDWODVWUW",
	XVMADDWOD_Q_DU_D: "XVMADDWODQVUV",

	VSHUF4I_B:  "VSHUF4IB",
	VSHUF4I_H:  "VSHUF4IH",
	VSHUF4I_W:  "VSHUF4IW",
	VSHUF4I_D:  "VSHUF4IV",
	XVSHUF4I_B: "XVSHUF4IB",
	XVSHUF4I_H: "XVSHUF4IH",
	XVSHUF4I_W: "XVSHUF4IW",
	XVSHUF4I_D: "XVSHUF4IV",

	VSHUF_H:  "VSHUFH",
	VSHUF_W:  "VSHUFW",
	VSHUF_D:  "VSHUFV",
	XVSHUF_H: "XVSHUFH",
	XVSHUF_W: "XVSHUFW",
	XVSHUF_D: "XVSHUFV",
	VSHUF_B:  "VSHUFB",
	XVSHUF_B: "XVSHUFB",

	VPERMI_W:    "VPERMIW",
	XVPERMI_W:   "XVPERMIW",
	XVPERMI_D:   "XVPERMIV",
	XVPERMI_Q:   "XVPERMIQ",
	VEXTRINS_B:  "VEXTRINSB",
	VEXTRINS_H:  "VEXTRINSH",
	VEXTRINS_W:  "VEXTRINSW",
	VEXTRINS_D:  "VEXTRINSV",
	XVEXTRINS_B: "XVEXTRINSB",
	XVEXTRINS_H: "XVEXTRINSH",
	XVEXTRINS_W: "XVEXTRINSW",
	XVEXTRINS_D: "XVEXTRINSV",

	VSETEQZ_V:     "VSETEQV",
	VSETNEZ_V:     "VSETNEV",
	XVSETEQZ_V:    "XVSETEQV",
	XVSETNEZ_V:    "XVSETNEV",
	VSETANYEQZ_B:  "VSETANYEQB",
	VSETANYEQZ_H:  "VSETANYEQH",
	VSETANYEQZ_W:  "VSETANYEQW",
	VSETANYEQZ_D:  "VSETANYEQV",
	VSETALLNEZ_B:  "VSETALLNEB",
	VSETALLNEZ_H:  "VSETALLNEH",
	VSETALLNEZ_W:  "VSETALLNEW",
	VSETALLNEZ_D:  "VSETALLNEV",
	XVSETANYEQZ_B: "XVSETANYEQB",
	XVSETANYEQZ_H: "XVSETANYEQH",
	XVSETANYEQZ_W: "XVSETANYEQW",
	XVSETANYEQZ_D: "XVSETANYEQV",
	XVSETALLNEZ_B: "XVSETALLNEB",
	XVSETALLNEZ_H: "XVSETALLNEH",
	XVSETALLNEZ_W: "XVSETALLNEW",
	XVSETALLNEZ_D: "XVSETALLNEV",

	VFRINTRNE_S:  "VFRINTRNEF",
	VFRINTRNE_D:  "VFRINTRNED",
	VFRINTRZ_S:   "VFRINTRZF",
	VFRINTRZ_D:   "VFRINTRZD",
	VFRINTRP_S:   "VFRINTRPF",
	VFRINTRP_D:   "VFRINTRPD",
	VFRINTRM_S:   "VFRINTRMF",
	VFRINTRM_D:   "VFRINTRMD",
	VFRINT_S:     "VFRINTF",
	VFRINT_D:     "VFRINTD",
	XVFRINTRNE_S: "XVFRINTRNEF",
	XVFRINTRNE_D: "XVFRINTRNED",
	XVFRINTRZ_S:  "XVFRINTRZF",
	XVFRINTRZ_D:  "XVFRINTRZD",
	XVFRINTRP_S:  "XVFRINTRPF",
	XVFRINTRP_D:  "XVFRINTRPD",
	XVFRINTRM_S:  "XVFRINTRMF",
	XVFRINTRM_D:  "XVFRINTRMD",
	XVFRINT_S:    "XVFRINTF",
	XVFRINT_D:    "XVFRINTD",

	VFADD_S:  "VADDF",
	VFADD_D:  "VADDD",
	VFSUB_S:  "VSUBF",
	VFSUB_D:  "VSUBD",
	VFMUL_S:  "VMULF",
	VFMUL_D:  "VMULD",
	VFDIV_S:  "VDIVF",
	VFDIV_D:  "VDIVD",
	XVFADD_S: "XVADDF",
	XVFADD_D: "XVADDD",
	XVFSUB_S: "XVSUBF",
	XVFSUB_D: "XVSUBD",
	XVFMUL_S: "XVMULF",
	XVFMUL_D: "XVMULD",
	XVFDIV_S: "XVDIVF",
	XVFDIV_D: "XVDIVD",

	VFCLASS_S:  "VFCLASSF",
	VFCLASS_D:  "VFCLASSD",
	XVFCLASS_S: "XVFCLASSF",
	XVFCLASS_D: "XVFCLASSD",

	VBITCLR_B:  "VBITCLRB",
	VBITCLR_H:  "VBITCLRH",
	VBITCLR_W:  "VBITCLRW",
	VBITCLR_D:  "VBITCLRV",
	VBITSET_B:  "VBITSETB",
	VBITSET_H:  "VBITSETH",
	VBITSET_W:  "VBITSETW",
	VBITSET_D:  "VBITSETV",
	VBITREV_B:  "VBITREVB",
	VBITREV_H:  "VBITREVH",
	VBITREV_W:  "VBITREVW",
	VBITREV_D:  "VBITREVV",
	XVBITCLR_B: "XVBITCLRB",
	XVBITCLR_H: "XVBITCLRH",
	XVBITCLR_W: "XVBITCLRW",
	XVBITCLR_D: "XVBITCLRV",
	XVBITSET_B: "XVBITSETB",
	XVBITSET_H: "XVBITSETH",
	XVBITSET_W: "XVBITSETW",
	XVBITSET_D: "XVBITSETV",
	XVBITREV_B: "XVBITREVB",
	XVBITREV_H: "XVBITREVH",
	XVBITREV_W: "XVBITREVW",
	XVBITREV_D: "XVBITREVV",

	VBITCLRI_B:  "VBITCLRB",
	VBITCLRI_H:  "VBITCLRH",
	VBITCLRI_W:  "VBITCLRW",
	VBITCLRI_D:  "VBITCLRV",
	VBITSETI_B:  "VBITSETB",
	VBITSETI_H:  "VBITSETH",
	VBITSETI_W:  "VBITSETW",
	VBITSETI_D:  "VBITSETV",
	VBITREVI_B:  "VBITREVB",
	VBITREVI_H:  "VBITREVH",
	VBITREVI_W:  "VBITREVW",
	VBITREVI_D:  "VBITREVV",
	XVBITCLRI_B: "XVBITCLRB",
	XVBITCLRI_H: "XVBITCLRH",
	XVBITCLRI_W: "XVBITCLRW",
	XVBITCLRI_D: "XVBITCLRV",
	XVBITSETI_B: "XVBITSETB",
	XVBITSETI_H: "XVBITSETH",
	XVBITSETI_W: "XVBITSETW",
	XVBITSETI_D: "XVBITSETV",
	XVBITREVI_B: "XVBITREVB",
	XVBITREVI_H: "XVBITREVH",
	XVBITREVI_W: "XVBITREVW",
	XVBITREVI_D: "XVBITREVV",
}

var movCount = map[string]uint32{
	"B": 16,
	"H": 8,
	"W": 4,
	"V": 2,
	"Q": 1,
}
