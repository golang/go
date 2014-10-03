// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package x86asm

import (
	"fmt"
	"strings"
)

// Plan9Syntax returns the Go assembler syntax for the instruction.
// The syntax was originally defined by Plan 9.
// The pc is the program counter of the instruction, used for expanding
// PC-relative addresses into absolute ones.
// The symname function queries the symbol table for the program
// being disassembled. Given a target address it returns the name and base
// address of the symbol containing the target, if any; otherwise it returns "", 0.
func Plan9Syntax(inst Inst, pc uint64, symname func(uint64) (string, uint64)) string {
	if symname == nil {
		symname = func(uint64) (string, uint64) { return "", 0 }
	}
	var args []string
	for i := len(inst.Args) - 1; i >= 0; i-- {
		a := inst.Args[i]
		if a == nil {
			continue
		}
		args = append(args, plan9Arg(&inst, pc, symname, a))
	}

	var last Prefix
	for _, p := range inst.Prefix {
		if p == 0 || p.IsREX() {
			break
		}
		last = p
	}

	prefix := ""
	switch last & 0xFF {
	case 0, 0x66, 0x67:
		// ignore
	case PrefixREPN:
		prefix += "REPNE "
	default:
		prefix += last.String() + " "
	}

	op := inst.Op.String()
	if plan9Suffix[inst.Op] {
		switch inst.DataSize {
		case 8:
			op += "B"
		case 16:
			op += "W"
		case 32:
			op += "L"
		case 64:
			op += "Q"
		}
	}

	if args != nil {
		op += " " + strings.Join(args, ", ")
	}

	return prefix + op
}

func plan9Arg(inst *Inst, pc uint64, symname func(uint64) (string, uint64), arg Arg) string {
	switch a := arg.(type) {
	case Reg:
		return plan9Reg[a]
	case Rel:
		if pc == 0 {
			break
		}
		// If the absolute address is the start of a symbol, use the name.
		// Otherwise use the raw address, so that things like relative
		// jumps show up as JMP 0x123 instead of JMP f+10(SB).
		// It is usually easier to search for 0x123 than to do the mental
		// arithmetic to find f+10.
		addr := pc + uint64(inst.Len) + uint64(a)
		if s, base := symname(addr); s != "" && addr == base {
			return fmt.Sprintf("%s(SB)", s)
		}
		return fmt.Sprintf("%#x", addr)

	case Imm:
		if s, base := symname(uint64(a)); s != "" {
			suffix := ""
			if uint64(a) != base {
				suffix = fmt.Sprintf("%+d", uint64(a)-base)
			}
			return fmt.Sprintf("$%s%s(SB)", s, suffix)
		}
		if inst.Mode == 32 {
			return fmt.Sprintf("$%#x", uint32(a))
		}
		if Imm(int32(a)) == a {
			return fmt.Sprintf("$%#x", int64(a))
		}
		return fmt.Sprintf("$%#x", uint64(a))
	case Mem:
		if a.Segment == 0 && a.Disp != 0 && a.Base == 0 && (a.Index == 0 || a.Scale == 0) {
			if s, base := symname(uint64(a.Disp)); s != "" {
				suffix := ""
				if uint64(a.Disp) != base {
					suffix = fmt.Sprintf("%+d", uint64(a.Disp)-base)
				}
				return fmt.Sprintf("%s%s(SB)", s, suffix)
			}
		}
		s := ""
		if a.Segment != 0 {
			s += fmt.Sprintf("%s:", plan9Reg[a.Segment])
		}
		if a.Disp != 0 {
			s += fmt.Sprintf("%#x", a.Disp)
		} else {
			s += "0"
		}
		if a.Base != 0 {
			s += fmt.Sprintf("(%s)", plan9Reg[a.Base])
		}
		if a.Index != 0 && a.Scale != 0 {
			s += fmt.Sprintf("(%s*%d)", plan9Reg[a.Index], a.Scale)
		}
		return s
	}
	return arg.String()
}

var plan9Suffix = [maxOp + 1]bool{
	ADC:       true,
	ADD:       true,
	AND:       true,
	BSF:       true,
	BSR:       true,
	BT:        true,
	BTC:       true,
	BTR:       true,
	BTS:       true,
	CMP:       true,
	CMPXCHG:   true,
	CVTSI2SD:  true,
	CVTSI2SS:  true,
	CVTSD2SI:  true,
	CVTSS2SI:  true,
	CVTTSD2SI: true,
	CVTTSS2SI: true,
	DEC:       true,
	DIV:       true,
	FLDENV:    true,
	FRSTOR:    true,
	IDIV:      true,
	IMUL:      true,
	IN:        true,
	INC:       true,
	LEA:       true,
	MOV:       true,
	MOVNTI:    true,
	MUL:       true,
	NEG:       true,
	NOP:       true,
	NOT:       true,
	OR:        true,
	OUT:       true,
	POP:       true,
	POPA:      true,
	PUSH:      true,
	PUSHA:     true,
	RCL:       true,
	RCR:       true,
	ROL:       true,
	ROR:       true,
	SAR:       true,
	SBB:       true,
	SHL:       true,
	SHLD:      true,
	SHR:       true,
	SHRD:      true,
	SUB:       true,
	TEST:      true,
	XADD:      true,
	XCHG:      true,
	XOR:       true,
}

var plan9Reg = [...]string{
	AL:   "AL",
	CL:   "CL",
	BL:   "BL",
	DL:   "DL",
	AH:   "AH",
	CH:   "CH",
	BH:   "BH",
	DH:   "DH",
	SPB:  "SP",
	BPB:  "BP",
	SIB:  "SI",
	DIB:  "DI",
	R8B:  "R8",
	R9B:  "R9",
	R10B: "R10",
	R11B: "R11",
	R12B: "R12",
	R13B: "R13",
	R14B: "R14",
	R15B: "R15",
	AX:   "AX",
	CX:   "CX",
	BX:   "BX",
	DX:   "DX",
	SP:   "SP",
	BP:   "BP",
	SI:   "SI",
	DI:   "DI",
	R8W:  "R8",
	R9W:  "R9",
	R10W: "R10",
	R11W: "R11",
	R12W: "R12",
	R13W: "R13",
	R14W: "R14",
	R15W: "R15",
	EAX:  "AX",
	ECX:  "CX",
	EDX:  "DX",
	EBX:  "BX",
	ESP:  "SP",
	EBP:  "BP",
	ESI:  "SI",
	EDI:  "DI",
	R8L:  "R8",
	R9L:  "R9",
	R10L: "R10",
	R11L: "R11",
	R12L: "R12",
	R13L: "R13",
	R14L: "R14",
	R15L: "R15",
	RAX:  "AX",
	RCX:  "CX",
	RDX:  "DX",
	RBX:  "BX",
	RSP:  "SP",
	RBP:  "BP",
	RSI:  "SI",
	RDI:  "DI",
	R8:   "R8",
	R9:   "R9",
	R10:  "R10",
	R11:  "R11",
	R12:  "R12",
	R13:  "R13",
	R14:  "R14",
	R15:  "R15",
	IP:   "IP",
	EIP:  "IP",
	RIP:  "IP",
	F0:   "F0",
	F1:   "F1",
	F2:   "F2",
	F3:   "F3",
	F4:   "F4",
	F5:   "F5",
	F6:   "F6",
	F7:   "F7",
	M0:   "M0",
	M1:   "M1",
	M2:   "M2",
	M3:   "M3",
	M4:   "M4",
	M5:   "M5",
	M6:   "M6",
	M7:   "M7",
	X0:   "X0",
	X1:   "X1",
	X2:   "X2",
	X3:   "X3",
	X4:   "X4",
	X5:   "X5",
	X6:   "X6",
	X7:   "X7",
	X8:   "X8",
	X9:   "X9",
	X10:  "X10",
	X11:  "X11",
	X12:  "X12",
	X13:  "X13",
	X14:  "X14",
	X15:  "X15",
	CS:   "CS",
	SS:   "SS",
	DS:   "DS",
	ES:   "ES",
	FS:   "FS",
	GS:   "GS",
	GDTR: "GDTR",
	IDTR: "IDTR",
	LDTR: "LDTR",
	MSW:  "MSW",
	TASK: "TASK",
	CR0:  "CR0",
	CR1:  "CR1",
	CR2:  "CR2",
	CR3:  "CR3",
	CR4:  "CR4",
	CR5:  "CR5",
	CR6:  "CR6",
	CR7:  "CR7",
	CR8:  "CR8",
	CR9:  "CR9",
	CR10: "CR10",
	CR11: "CR11",
	CR12: "CR12",
	CR13: "CR13",
	CR14: "CR14",
	CR15: "CR15",
	DR0:  "DR0",
	DR1:  "DR1",
	DR2:  "DR2",
	DR3:  "DR3",
	DR4:  "DR4",
	DR5:  "DR5",
	DR6:  "DR6",
	DR7:  "DR7",
	DR8:  "DR8",
	DR9:  "DR9",
	DR10: "DR10",
	DR11: "DR11",
	DR12: "DR12",
	DR13: "DR13",
	DR14: "DR14",
	DR15: "DR15",
	TR0:  "TR0",
	TR1:  "TR1",
	TR2:  "TR2",
	TR3:  "TR3",
	TR4:  "TR4",
	TR5:  "TR5",
	TR6:  "TR6",
	TR7:  "TR7",
}
