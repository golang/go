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
	if inst.Op == 0 {
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
		return op + " " + strings.Join(args[1:], ", ")
	// store instructions always have the memory operand at the end, no need to reorder
	case STB, STBU, STBX, STBUX,
		STH, STHU, STHX, STHUX,
		STW, STWU, STWX, STWUX,
		STD, STDU, STDX, STDUX,
		STQ,
		STHBRX, STWBRX:
		return op + " " + strings.Join(args, ", ")
	// branch instructions needs additional handling
	case BCLR:
		if int(inst.Args[0].(Imm))&20 == 20 { // unconditional
			return "RET"
		}
		return op + " " + strings.Join(args, ", ")
	case BC:
		if int(inst.Args[0].(Imm))&0x1c == 12 { // jump on cond bit set
			return fmt.Sprintf("B%s %s", args[1], args[2])
		} else if int(inst.Args[0].(Imm))&0x1c == 4 && revCondMap[args[1]] != "" { // jump on cond bit not set
			return fmt.Sprintf("B%s %s", revCondMap[args[1]], args[2])
		}
		return op + " " + strings.Join(args, ", ")
	case BCCTR:
		if int(inst.Args[0].(Imm))&20 == 20 { // unconditional
			return "BR (CTR)"
		}
		return op + " " + strings.Join(args, ", ")
	case BCCTRL:
		if int(inst.Args[0].(Imm))&20 == 20 { // unconditional
			return "BL (CTR)"
		}
		return op + " " + strings.Join(args, ", ")
	case BCA, BCL, BCLA, BCLRL, BCTAR, BCTARL:
		return op + " " + strings.Join(args, ", ")
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
		if arg == CR0 && strings.HasPrefix(inst.Op.String(), "cmp") {
			return "" // don't show cr0 for cmp instructions
		} else if arg >= CR0 {
			return fmt.Sprintf("CR%d", int(arg-CR0))
		}
		bit := [4]string{"LT", "GT", "EQ", "SO"}[(arg-Cond0LT)%4]
		if arg <= Cond0SO {
			return bit
		}
		return fmt.Sprintf("4*CR%d+%s", int(arg-Cond0LT)/4, bit)
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
	LWARX: "LWAR", STWCX_: "STWCCC",
	LDARX: "LDAR", STDCX_: "STDCCC",
	LHARX: "LHAR", STHCX_: "STHCCC",
	LBARX: "LBAR", STBCX_: "STBCCC",
	ADDI: "ADD",
	ADD_: "ADDCC",
	LBZ:  "MOVBZ", STB: "MOVB",
	LBZU: "MOVBZU", STBU: "MOVBU", // TODO(minux): indexed forms are not handled
	LHZ: "MOVHZ", LHA: "MOVH", STH: "MOVH",
	LHZU: "MOVHZU", STHU: "MOVHU",
	LI:  "MOVD",
	LIS: "ADDIS",
	LWZ: "MOVWZ", LWA: "MOVW", STW: "MOVW",
	LWZU: "MOVWZU", STWU: "MOVWU",
	LD: "MOVD", STD: "MOVD",
	LDU: "MOVDU", STDU: "MOVDU",
	MTSPR: "MOVD", MFSPR: "MOVD", // the width is ambiguous for SPRs
	B:     "BR",
	BL:    "CALL",
	CMPLD: "CMPU", CMPLW: "CMPWU",
	CMPD: "CMP", CMPW: "CMPW",
}
