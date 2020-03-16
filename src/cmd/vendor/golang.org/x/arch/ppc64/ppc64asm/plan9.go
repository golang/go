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
		symname = func(uint64) (string, uint64) { return "", 0 }
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
			// In the case for some BC instructions, a CondReg arg has
			// both the CR and the branch condition encoded in its value.
			// plan9Arg will return a string with the string representation
			// of these values separated by a blank that will be treated
			// as 2 args from this point on.
			if strings.IndexByte(s, ' ') > 0 {
				t := strings.Split(s, " ")
				args = append(args, t[0])
				args = append(args, t[1])
			} else {
				args = append(args, s)
			}
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
		if len(args) == 0 {
			return op
		} else if len(args) == 1 {
			return fmt.Sprintf("%s %s", op, args[0])
		}
		args = append(args, args[0])
		return op + " " + strings.Join(args[1:], ",")
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
		STQ:
		return op + " " + strings.Join(args, ",")

	case CMPD, CMPDI, CMPLD, CMPLDI, CMPW, CMPWI, CMPLW, CMPLWI:
		if len(args) == 2 {
			return op + " " + args[0] + "," + args[1]
		} else if len(args) == 3 {
			return op + " " + args[0] + "," + args[1] + "," + args[2]
		}
		return op + " " + args[0] + " ??"

	case LIS:
		return "ADDIS $0," + args[1] + "," + args[0]
	// store instructions with index registers
	case STBX, STBUX, STHX, STHUX, STWX, STWUX, STDX, STDUX,
		STHBRX, STWBRX, STDBRX, STSWX, STFSX, STFSUX, STFDX, STFDUX, STFIWX, STFDPX:
		return "MOV" + op[2:len(op)-1] + " " + args[0] + ",(" + args[2] + ")(" + args[1] + ")"

	case STDCXCC, STWCXCC, STHCXCC, STBCXCC:
		return op + " " + args[0] + ",(" + args[2] + ")(" + args[1] + ")"

	case STXVD2X, STXVW4X:
		return op + " " + args[0] + ",(" + args[2] + ")(" + args[1] + ")"

	// load instructions with index registers
	case LBZX, LBZUX, LHZX, LHZUX, LWZX, LWZUX, LDX, LDUX,
		LHBRX, LWBRX, LDBRX, LSWX, LFSX, LFSUX, LFDX, LFDUX, LFIWAX, LFIWZX:
		return "MOV" + op[1:len(op)-1] + " (" + args[2] + ")(" + args[1] + ")," + args[0]

	case LDARX, LWARX, LHARX, LBARX:
		return op + " (" + args[2] + ")(" + args[1] + ")," + args[0]

	case LXVD2X, LXVW4X:
		return op + " (" + args[2] + ")(" + args[1] + ")," + args[0]

	case DCBT, DCBTST, DCBZ, DCBST:
		return op + " (" + args[1] + ")"

	// branch instructions needs additional handling
	case BCLR:
		if int(inst.Args[0].(Imm))&20 == 20 { // unconditional
			return "RET"
		}
		return op + " " + strings.Join(args, ", ")
	case BC:
		if int(inst.Args[0].(Imm))&0x1c == 12 { // jump on cond bit set
			if len(args) == 4 {
				return fmt.Sprintf("B%s %s,%s", args[1], args[2], args[3])
			}
			return fmt.Sprintf("B%s %s", args[1], args[2])
		} else if int(inst.Args[0].(Imm))&0x1c == 4 && revCondMap[args[1]] != "" { // jump on cond bit not set
			if len(args) == 4 {
				return fmt.Sprintf("B%s %s,%s", revCondMap[args[1]], args[2], args[3])
			}
			return fmt.Sprintf("B%s %s", revCondMap[args[1]], args[2])
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
// NOTE: because Plan9Syntax is the only caller of this func, and it receives a copy
//       of inst, it's ok to modify inst.Args here.
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
		if arg == CR0 && strings.HasPrefix(inst.Op.String(), "cmp") {
			return "" // don't show cr0 for cmp instructions
		} else if arg >= CR0 {
			return fmt.Sprintf("CR%d", int(arg-CR0))
		}
		bit := [4]string{"LT", "GT", "EQ", "SO"}[(arg-Cond0LT)%4]
		if arg <= Cond0SO {
			return bit
		}
		return fmt.Sprintf("%s CR%d", bit, int(arg-Cond0LT)/4)
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
		if s, base := symname(addr); s != "" && base == addr {
			return fmt.Sprintf("%s(SB)", s)
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

// revCondMap maps a conditional register bit to its inverse, if possible.
var revCondMap = map[string]string{
	"LT": "GE", "GT": "LE", "EQ": "NE",
}

// plan9OpMap maps an Op to its Plan 9 mnemonics, if different than its GNU mnemonics.
var plan9OpMap = map[Op]string{
	LWARX: "LWAR",
	LDARX: "LDAR",
	LHARX: "LHAR",
	LBARX: "LBAR",
	ADDI:  "ADD",
	SRADI: "SRAD",
	SUBF:  "SUB",
	LI:    "MOVD",
	LBZ:   "MOVBZ", STB: "MOVB",
	LBZU: "MOVBZU", STBU: "MOVBU",
	LHZ: "MOVHZ", LHA: "MOVH", STH: "MOVH",
	LHZU: "MOVHZU", STHU: "MOVHU",
	LWZ: "MOVWZ", LWA: "MOVW", STW: "MOVW",
	LWZU: "MOVWZU", STWU: "MOVWU",
	LD: "MOVD", STD: "MOVD",
	LDU: "MOVDU", STDU: "MOVDU",
	CMPD: "CMP", CMPDI: "CMP",
	CMPW: "CMPW", CMPWI: "CMPW",
	CMPLD: "CMPU", CMPLDI: "CMPU",
	CMPLW: "CMPWU", CMPLWI: "CMPWU",
	MTSPR: "MOVD", MFSPR: "MOVD", // the width is ambiguous for SPRs
	B:  "BR",
	BL: "CALL",
}
