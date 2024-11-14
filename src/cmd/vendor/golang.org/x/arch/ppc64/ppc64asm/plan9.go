// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ppc64asm

import (
	"fmt"
	"strings"
)

// GoSyntax returns the Go assembler syntax for the instruction.
// The pc is the program counter of the first instruction, used for expanding
// PC-relative addresses into absolute ones.
// The symname function queries the symbol table for the program
// being disassembled. It returns the name and base address of the symbol
// containing the target, if any; otherwise it returns "", 0.
func GoSyntax(inst Inst, pc uint64, symname func(uint64) (string, uint64)) string {
	if symname == nil {
		symname = func { "", 0 }
	}
	if inst.Op == 0 && inst.Enc == 0 {
		return "WORD $0"
	} else if inst.Op == 0 {
		return "?"
	}
	var args []string
	for i, a := range inst.Args[:] {
		if a == nil {
			break
		}
		if s := plan9Arg(&inst, i, pc, a, symname); s != "" {
			args = append(args, s)
		}
	}
	var op string
	op = plan9OpMap[inst.Op]
	if op == "" {
		op = strings.ToUpper(inst.Op.String())
		if op[len(op)-1] == '.' {
			op = op[:len(op)-1] + "CC"
		}
	}
	// laid out the instruction
	switch inst.Op {
	default: // dst, sA, sB, ...
		switch len(args) {
		case 0:
			return op
		case 1:
			return fmt.Sprintf("%s %s", op, args[0])
		case 2:
			if inst.Op == COPY || inst.Op == PASTECC {
				return op + " " + args[0] + "," + args[1]
			}
			return op + " " + args[1] + "," + args[0]
		case 3:
			if reverseOperandOrder(inst.Op) {
				return op + " " + args[2] + "," + args[1] + "," + args[0]
			}
		case 4:
			if reverseMiddleOps(inst.Op) {
				return op + " " + args[1] + "," + args[3] + "," + args[2] + "," + args[0]
			}
		}
		args = append(args, args[0])
		return op + " " + strings.Join(args[1:], ",")
	case PASTECC:
		// paste. has two input registers, and an L field, unlike other 3 operand instructions.
		return op + " " + args[0] + "," + args[1] + "," + args[2]
	case SYNC:
		if args[0] == "$1" {
			return "LWSYNC"
		}
		return "HWSYNC"

	case ISEL:
		return "ISEL " + args[3] + "," + args[1] + "," + args[2] + "," + args[0]

	// store instructions always have the memory operand at the end, no need to reorder
	// indexed stores handled separately
	case STB, STBU,
		STH, STHU,
		STW, STWU,
		STD, STDU,
		STFD, STFDU,
		STFS, STFSU,
		STQ, HASHST, HASHSTP:
		return op + " " + strings.Join(args, ",")

	case FCMPU, FCMPO, CMPD, CMPDI, CMPLD, CMPLDI, CMPW, CMPWI, CMPLW, CMPLWI:
		crf := int(inst.Args[0].(CondReg) - CR0)
		cmpstr := op + " " + args[1] + "," + args[2]
		if crf != 0 { // print CRx as the final operand if not implied (i.e BF != 0)
			cmpstr += "," + args[0]
		}
		return cmpstr

	case LIS:
		return "ADDIS $0," + args[1] + "," + args[0]
	// store instructions with index registers
	case STBX, STBUX, STHX, STHUX, STWX, STWUX, STDX, STDUX,
		STHBRX, STWBRX, STDBRX, STSWX, STFIWX:
		return "MOV" + op[2:len(op)-1] + " " + args[0] + ",(" + args[2] + ")(" + args[1] + ")"

	case STDCXCC, STWCXCC, STHCXCC, STBCXCC:
		return op + " " + args[0] + ",(" + args[2] + ")(" + args[1] + ")"

	case STXVX, STXVD2X, STXVW4X, STXVH8X, STXVB16X, STXSDX, STVX, STVXL, STVEBX, STVEHX, STVEWX, STXSIWX, STFDX, STFDUX, STFDPX, STFSX, STFSUX:
		return op + " " + args[0] + ",(" + args[2] + ")(" + args[1] + ")"

	case STXV:
		return op + " " + args[0] + "," + args[1]

	case STXVL, STXVLL:
		return op + " " + args[0] + "," + args[1] + "," + args[2]

	case LWAX, LWAUX, LWZX, LHZX, LBZX, LDX, LHAX, LHAUX, LDARX, LWARX, LHARX, LBARX, LFDX, LFDUX, LFSX, LFSUX, LDBRX, LWBRX, LHBRX, LDUX, LWZUX, LHZUX, LBZUX:
		if args[1] == "0" {
			return op + " (" + args[2] + ")," + args[0]
		}
		return op + " (" + args[2] + ")(" + args[1] + ")," + args[0]

	case LXVX, LXVD2X, LXVW4X, LXVH8X, LXVB16X, LVX, LVXL, LVSR, LVSL, LVEBX, LVEHX, LVEWX, LXSDX, LXSIWAX:
		return op + " (" + args[2] + ")(" + args[1] + ")," + args[0]

	case LXV:
		return op + " " + args[1] + "," + args[0]

	case LXVL, LXVLL:
		return op + " " + args[1] + "," + args[2] + "," + args[0]

	case DCBT, DCBTST, DCBZ, DCBST, ICBI:
		if args[0] == "0" || args[0] == "R0" {
			return op + " (" + args[1] + ")"
		}
		return op + " (" + args[1] + ")(" + args[0] + ")"

	// branch instructions needs additional handling
	case BCLR:
		if int(inst.Args[0].(Imm))&20 == 20 { // unconditional
			return "RET"
		}
		return op + " " + strings.Join(args, ", ")
	case BC:
		bo := int(inst.Args[0].(Imm))
		bi := int(inst.Args[1].(CondReg) - Cond0LT)
		bcname := condName[((bo&0x8)>>1)|(bi&0x3)]
		if bo&0x17 == 4 { // jump only a CR bit set/unset, no hints (at bits) set.
			if bi >= 4 {
				return fmt.Sprintf("B%s CR%d,%s", bcname, bi>>2, args[2])
			} else {
				return fmt.Sprintf("B%s %s", bcname, args[2])
			}
		}
		return op + " " + strings.Join(args, ",")
	case BCCTR:
		if int(inst.Args[0].(Imm))&20 == 20 { // unconditional
			return "BR (CTR)"
		}
		return op + " " + strings.Join(args, ", ")
	case BCCTRL:
		if int(inst.Args[0].(Imm))&20 == 20 { // unconditional
			return "BL (CTR)"
		}
		return op + " " + strings.Join(args, ",")
	case BCA, BCL, BCLA, BCLRL, BCTAR, BCTARL:
		return op + " " + strings.Join(args, ",")
	}
}

// plan9Arg formats arg (which is the argIndex's arg in inst) according to Plan 9 rules.
//
// NOTE: because Plan9Syntax is the only caller of this func, and it receives a copy
// of inst, it's ok to modify inst.Args here.
func plan9Arg(inst *Inst, argIndex int, pc uint64, arg Arg, symname func(uint64) (string, uint64)) string {
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
		if arg == R30 {
			return "g"
		}
		return strings.ToUpper(arg.String())
	case CondReg:
		// This op is left as its numerical value, not mapped onto CR + condition
		if inst.Op == ISEL {
			return fmt.Sprintf("$%d", (arg - Cond0LT))
		}
		bit := [4]string{"LT", "GT", "EQ", "SO"}[(arg-Cond0LT)%4]
		if arg <= Cond0SO {
			return bit
		} else if arg > Cond0SO && arg <= Cond7SO {
			return fmt.Sprintf("CR%d%s", int(arg-Cond0LT)/4, bit)
		} else {
			return fmt.Sprintf("CR%d", int(arg-CR0))
		}
	case Imm:
		return fmt.Sprintf("$%d", arg)
	case SpReg:
		switch arg {
		case 8:
			return "LR"
		case 9:
			return "CTR"
		}
		return fmt.Sprintf("SPR(%d)", int(arg))
	case PCRel:
		addr := pc + uint64(int64(arg))
		s, base := symname(addr)
		if s != "" && addr == base {
			return fmt.Sprintf("%s(SB)", s)
		}
		if inst.Op == BL && s != "" && (addr-base) == 8 {
			// When decoding an object built for PIE, a CALL targeting
			// a global entry point will be adjusted to the local entry
			// if any. For now, assume any symname+8 PC is a local call.
			return fmt.Sprintf("%s+%d(SB)", s, addr-base)
		}
		return fmt.Sprintf("%#x", addr)
	case Label:
		return fmt.Sprintf("%#x", int(arg))
	case Offset:
		reg := inst.Args[argIndex+1].(Reg)
		removeArg(inst, argIndex+1)
		if reg == R0 {
			return fmt.Sprintf("%d(0)", int(arg))
		}
		return fmt.Sprintf("%d(R%d)", int(arg), reg-R0)
	}
	return fmt.Sprintf("???(%v)", arg)
}

func reverseMiddleOps(op Op) bool {
	switch op {
	case FMADD, FMADDCC, FMADDS, FMADDSCC, FMSUB, FMSUBCC, FMSUBS, FMSUBSCC, FNMADD, FNMADDCC, FNMADDS, FNMADDSCC, FNMSUB, FNMSUBCC, FNMSUBS, FNMSUBSCC, FSEL, FSELCC:
		return true
	}
	return false
}

func reverseOperandOrder(op Op) bool {
	switch op {
	// Special case for SUBF, SUBFC: not reversed
	case ADD, ADDC, ADDE, ADDCC, ADDCCC:
		return true
	case MULLW, MULLWCC, MULHW, MULHWCC, MULLD, MULLDCC, MULHD, MULHDCC, MULLWO, MULLWOCC, MULHWU, MULHWUCC, MULLDO, MULLDOCC:
		return true
	case DIVD, DIVDCC, DIVDU, DIVDUCC, DIVDE, DIVDECC, DIVDEU, DIVDEUCC, DIVDO, DIVDOCC, DIVDUO, DIVDUOCC:
		return true
	case MODUD, MODSD, MODUW, MODSW:
		return true
	case FADD, FADDS, FSUB, FSUBS, FMUL, FMULS, FDIV, FDIVS, FMADD, FMADDS, FMSUB, FMSUBS, FNMADD, FNMADDS, FNMSUB, FNMSUBS, FMULSCC:
		return true
	case FADDCC, FADDSCC, FSUBCC, FMULCC, FDIVCC, FDIVSCC:
		return true
	case OR, ORCC, ORC, ORCCC, AND, ANDCC, ANDC, ANDCCC, XOR, XORCC, NAND, NANDCC, EQV, EQVCC, NOR, NORCC:
		return true
	case SLW, SLWCC, SLD, SLDCC, SRW, SRAW, SRWCC, SRAWCC, SRD, SRDCC, SRAD, SRADCC:
		return true
	}
	return false
}

// revCondMap maps a conditional register bit to its inverse, if possible.
var revCondMap = map[string]string{
	"LT": "GE", "GT": "LE", "EQ": "NE",
}

// Lookup table to map BI[0:1] and BO[3] to an extended mnemonic for CR ops.
// Bits 0-1 map to a bit with a CR field, and bit 2 selects the inverted (0)
// or regular (1) extended mnemonic.
var condName = []string{
	"GE",
	"LE",
	"NE",
	"NSO",
	"LT",
	"GT",
	"EQ",
	"SO",
}

// plan9OpMap maps an Op to its Plan 9 mnemonics, if different than its GNU mnemonics.
var plan9OpMap = map[Op]string{
	LWARX:     "LWAR",
	LDARX:     "LDAR",
	LHARX:     "LHAR",
	LBARX:     "LBAR",
	LWAX:      "MOVW",
	LHAX:      "MOVH",
	LWAUX:     "MOVWU",
	LHAU:      "MOVHU",
	LHAUX:     "MOVHU",
	LDX:       "MOVD",
	LDUX:      "MOVDU",
	LWZX:      "MOVWZ",
	LWZUX:     "MOVWZU",
	LHZX:      "MOVHZ",
	LHZUX:     "MOVHZU",
	LBZX:      "MOVBZ",
	LBZUX:     "MOVBZU",
	LDBRX:     "MOVDBR",
	LWBRX:     "MOVWBR",
	LHBRX:     "MOVHBR",
	MCRF:      "MOVFL",
	XORI:      "XOR",
	ORI:       "OR",
	ANDICC:    "ANDCC",
	ANDC:      "ANDN",
	ANDCCC:    "ANDNCC",
	ADDEO:     "ADDEV",
	ADDEOCC:   "ADDEVCC",
	ADDO:      "ADDV",
	ADDOCC:    "ADDVCC",
	ADDMEO:    "ADDMEV",
	ADDMEOCC:  "ADDMEVCC",
	ADDCO:     "ADDCV",
	ADDCOCC:   "ADDCVCC",
	ADDZEO:    "ADDZEV",
	ADDZEOCC:  "ADDZEVCC",
	SUBFME:    "SUBME",
	SUBFMECC:  "SUBMECC",
	SUBFZE:    "SUBZE",
	SUBFZECC:  "SUBZECC",
	SUBFZEO:   "SUBZEV",
	SUBFZEOCC: "SUBZEVCC",
	SUBF:      "SUB",
	SUBFC:     "SUBC",
	SUBFCC:    "SUBCC",
	SUBFCCC:   "SUBCCC",
	ORC:       "ORN",
	ORCCC:     "ORNCC",
	MULLWO:    "MULLWV",
	MULLWOCC:  "MULLWVCC",
	MULLDO:    "MULLDV",
	MULLDOCC:  "MULLDVCC",
	DIVDO:     "DIVDV",
	DIVDOCC:   "DIVDVCC",
	DIVDUO:    "DIVDUV",
	DIVDUOCC:  "DIVDUVCC",
	ADDI:      "ADD",
	MULLI:     "MULLD",
	SRADI:     "SRAD",
	STBCXCC:   "STBCCC",
	STWCXCC:   "STWCCC",
	STDCXCC:   "STDCCC",
	LI:        "MOVD",
	LBZ:       "MOVBZ", STB: "MOVB",
	LBZU: "MOVBZU", STBU: "MOVBU",
	LHZ: "MOVHZ", LHA: "MOVH", STH: "MOVH",
	LHZU: "MOVHZU", STHU: "MOVHU",
	LWZ: "MOVWZ", LWA: "MOVW", STW: "MOVW",
	LWZU: "MOVWZU", STWU: "MOVWU",
	LD: "MOVD", STD: "MOVD",
	LDU: "MOVDU", STDU: "MOVDU",
	LFD: "FMOVD", STFD: "FMOVD",
	LFS: "FMOVS", STFS: "FMOVS",
	LFDX: "FMOVD", STFDX: "FMOVD",
	LFDU: "FMOVDU", STFDU: "FMOVDU",
	LFDUX: "FMOVDU", STFDUX: "FMOVDU",
	LFSX: "FMOVS", STFSX: "FMOVS",
	LFSU: "FMOVSU", STFSU: "FMOVSU",
	LFSUX: "FMOVSU", STFSUX: "FMOVSU",
	CMPD: "CMP", CMPDI: "CMP",
	CMPW: "CMPW", CMPWI: "CMPW",
	CMPLD: "CMPU", CMPLDI: "CMPU",
	CMPLW: "CMPWU", CMPLWI: "CMPWU",
	MTSPR: "MOVD", MFSPR: "MOVD", // the width is ambiguous for SPRs
	B:  "BR",
	BL: "CALL",
}
