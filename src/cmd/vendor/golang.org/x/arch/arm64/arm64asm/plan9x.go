// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64asm

import (
	"fmt"
	"io"
	"sort"
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
	case LDR, LDRB, LDRH, LDRSB, LDRSH, LDRSW:
		// Check for PC-relative load.
		if offset, ok := inst.Args[1].(PCRel); ok {
			addr := pc + uint64(offset)
			if _, ok := inst.Args[0].(Reg); !ok {
				break
			}
			if s, base := symname(addr); s != "" && addr == base {
				args[1] = fmt.Sprintf("$%s(SB)", s)
			}
		}
	}

	// Move addressing mode into opcode suffix.
	suffix := ""
	switch inst.Op {
	case LDR, LDRB, LDRH, LDRSB, LDRSH, LDRSW, STR, STRB, STRH, STUR, STURB, STURH, LD1, ST1:
		switch mem := inst.Args[1].(type) {
		case MemImmediate:
			switch mem.Mode {
			case AddrOffset:
				// no suffix
			case AddrPreIndex:
				suffix = ".W"
			case AddrPostIndex, AddrPostReg:
				suffix = ".P"
			}
		}

	case STP, LDP:
		switch mem := inst.Args[2].(type) {
		case MemImmediate:
			switch mem.Mode {
			case AddrOffset:
				// no suffix
			case AddrPreIndex:
				suffix = ".W"
			case AddrPostIndex:
				suffix = ".P"
			}
		}
	}

	switch inst.Op {
	case BL:
		return "CALL " + args[0]

	case BLR:
		r := inst.Args[0].(Reg)
		regno := uint16(r) & 31
		return fmt.Sprintf("CALL (R%d)", regno)

	case RET:
		if r, ok := inst.Args[0].(Reg); ok && r == X30 {
			return "RET"
		}

	case B:
		if cond, ok := inst.Args[0].(Cond); ok {
			return "B" + cond.String() + " " + args[1]
		}
		return "JMP" + " " + args[0]

	case BR:
		r := inst.Args[0].(Reg)
		regno := uint16(r) & 31
		return fmt.Sprintf("JMP (R%d)", regno)

	case MOV:
		rno := -1
		switch a := inst.Args[0].(type) {
		case Reg:
			rno = int(a)
		case RegSP:
			rno = int(a)
		case RegisterWithArrangementAndIndex:
			op = "VMOV"
		case RegisterWithArrangement:
			op = "VMOV"
		}
		if rno >= 0 && rno <= int(WZR) {
			op = "MOVW"
		} else if rno >= int(X0) && rno <= int(XZR) {
			op = "MOVD"
		}
		if _, ok := inst.Args[1].(RegisterWithArrangementAndIndex); ok {
			op = "VMOV"
		}

	case LDR, LDUR:
		var rno uint16
		if r, ok := inst.Args[0].(Reg); ok {
			rno = uint16(r)
		} else {
			rno = uint16(inst.Args[0].(RegSP))
		}
		if rno <= uint16(WZR) {
			op = "MOVWU" + suffix
		} else if rno >= uint16(B0) && rno <= uint16(B31) {
			op = "FMOVB" + suffix
			args[0] = fmt.Sprintf("F%d", rno&31)
		} else if rno >= uint16(H0) && rno <= uint16(H31) {
			op = "FMOVH" + suffix
			args[0] = fmt.Sprintf("F%d", rno&31)
		} else if rno >= uint16(S0) && rno <= uint16(S31) {
			op = "FMOVS" + suffix
			args[0] = fmt.Sprintf("F%d", rno&31)
		} else if rno >= uint16(D0) && rno <= uint16(D31) {
			op = "FMOVD" + suffix
			args[0] = fmt.Sprintf("F%d", rno&31)
		} else if rno >= uint16(Q0) && rno <= uint16(Q31) {
			op = "FMOVQ" + suffix
			args[0] = fmt.Sprintf("F%d", rno&31)
		} else {
			op = "MOVD" + suffix
		}

	case LDRB:
		op = "MOVBU" + suffix

	case LDRH:
		op = "MOVHU" + suffix

	case LDRSW:
		op = "MOVW" + suffix

	case LDRSB:
		if r, ok := inst.Args[0].(Reg); ok {
			rno := uint16(r)
			if rno <= uint16(WZR) {
				op = "MOVBW" + suffix
			} else {
				op = "MOVB" + suffix
			}
		}
	case LDRSH:
		if r, ok := inst.Args[0].(Reg); ok {
			rno := uint16(r)
			if rno <= uint16(WZR) {
				op = "MOVHW" + suffix
			} else {
				op = "MOVH" + suffix
			}
		}
	case STR, STUR:
		var rno uint16
		if r, ok := inst.Args[0].(Reg); ok {
			rno = uint16(r)
		} else {
			rno = uint16(inst.Args[0].(RegSP))
		}
		if rno <= uint16(WZR) {
			op = "MOVW" + suffix
		} else if rno >= uint16(B0) && rno <= uint16(B31) {
			op = "FMOVB" + suffix
			args[0] = fmt.Sprintf("F%d", rno&31)
		} else if rno >= uint16(H0) && rno <= uint16(H31) {
			op = "FMOVH" + suffix
			args[0] = fmt.Sprintf("F%d", rno&31)
		} else if rno >= uint16(S0) && rno <= uint16(S31) {
			op = "FMOVS" + suffix
			args[0] = fmt.Sprintf("F%d", rno&31)
		} else if rno >= uint16(D0) && rno <= uint16(D31) {
			op = "FMOVD" + suffix
			args[0] = fmt.Sprintf("F%d", rno&31)
		} else if rno >= uint16(Q0) && rno <= uint16(Q31) {
			op = "FMOVQ" + suffix
			args[0] = fmt.Sprintf("F%d", rno&31)
		} else {
			op = "MOVD" + suffix
		}
		args[0], args[1] = args[1], args[0]

	case STRB, STURB:
		op = "MOVB" + suffix
		args[0], args[1] = args[1], args[0]

	case STRH, STURH:
		op = "MOVH" + suffix
		args[0], args[1] = args[1], args[0]

	case TBNZ, TBZ:
		args[0], args[1], args[2] = args[2], args[0], args[1]

	case MADD, MSUB, SMADDL, SMSUBL, UMADDL, UMSUBL:
		if r, ok := inst.Args[0].(Reg); ok {
			rno := uint16(r)
			if rno <= uint16(WZR) {
				op += "W"
			}
		}
		args[2], args[3] = args[3], args[2]
	case STLR:
		if r, ok := inst.Args[0].(Reg); ok {
			rno := uint16(r)
			if rno <= uint16(WZR) {
				op += "W"
			}
		}
		args[0], args[1] = args[1], args[0]

	case STLRB, STLRH:
		args[0], args[1] = args[1], args[0]

	case STLXR, STXR:
		if r, ok := inst.Args[1].(Reg); ok {
			rno := uint16(r)
			if rno <= uint16(WZR) {
				op += "W"
			}
		}
		args[1], args[2] = args[2], args[1]

	case STLXRB, STLXRH, STXRB, STXRH:
		args[1], args[2] = args[2], args[1]

	case BFI, BFXIL, SBFIZ, SBFX, UBFIZ, UBFX:
		if r, ok := inst.Args[0].(Reg); ok {
			rno := uint16(r)
			if rno <= uint16(WZR) {
				op += "W"
			}
		}
		args[1], args[2], args[3] = args[3], args[1], args[2]

	case LDAXP, LDXP:
		if r, ok := inst.Args[0].(Reg); ok {
			rno := uint16(r)
			if rno <= uint16(WZR) {
				op += "W"
			}
		}
		args[0] = fmt.Sprintf("(%s, %s)", args[0], args[1])
		args[1] = args[2]
		return op + " " + args[1] + ", " + args[0]

	case STP, LDP:
		args[0] = fmt.Sprintf("(%s, %s)", args[0], args[1])
		args[1] = args[2]

		rno, ok := inst.Args[0].(Reg)
		if !ok {
			rno = Reg(inst.Args[0].(RegSP))
		}
		if rno <= WZR {
			op = op + "W"
		} else if rno >= S0 && rno <= S31 {
			op = "F" + op + "S"
		} else if rno >= D0 && rno <= D31 {
			op = "F" + op + "D"
		} else if rno >= Q0 && rno <= Q31 {
			op = "F" + op + "Q"
		}
		op = op + suffix
		if inst.Op.String() == "STP" {
			return op + " " + args[0] + ", " + args[1]
		} else {
			return op + " " + args[1] + ", " + args[0]
		}

	case STLXP, STXP:
		if r, ok := inst.Args[1].(Reg); ok {
			rno := uint16(r)
			if rno <= uint16(WZR) {
				op += "W"
			}
		}
		args[1] = fmt.Sprintf("(%s, %s)", args[1], args[2])
		args[2] = args[3]
		return op + " " + args[1] + ", " + args[2] + ", " + args[0]

	case FCCMP, FCCMPE:
		args[0], args[1] = args[1], args[0]
		fallthrough

	case FCMP, FCMPE:
		if _, ok := inst.Args[1].(Imm); ok {
			args[1] = "$(0.0)"
		}
		fallthrough

	case FADD, FSUB, FMUL, FNMUL, FDIV, FMAX, FMIN, FMAXNM, FMINNM, FCSEL, FMADD, FMSUB, FNMADD, FNMSUB:
		if strings.HasSuffix(op, "MADD") || strings.HasSuffix(op, "MSUB") {
			args[2], args[3] = args[3], args[2]
		}
		if r, ok := inst.Args[0].(Reg); ok {
			rno := uint16(r)
			if rno >= uint16(S0) && rno <= uint16(S31) {
				op = fmt.Sprintf("%sS", op)
			} else if rno >= uint16(D0) && rno <= uint16(D31) {
				op = fmt.Sprintf("%sD", op)
			}
		}

	case FCVT:
		for i := 1; i >= 0; i-- {
			if r, ok := inst.Args[i].(Reg); ok {
				rno := uint16(r)
				if rno >= uint16(H0) && rno <= uint16(H31) {
					op = fmt.Sprintf("%sH", op)
				} else if rno >= uint16(S0) && rno <= uint16(S31) {
					op = fmt.Sprintf("%sS", op)
				} else if rno >= uint16(D0) && rno <= uint16(D31) {
					op = fmt.Sprintf("%sD", op)
				}
			}
		}

	case FABS, FNEG, FSQRT, FRINTN, FRINTP, FRINTM, FRINTZ, FRINTA, FRINTX, FRINTI:
		if r, ok := inst.Args[1].(Reg); ok {
			rno := uint16(r)
			if rno >= uint16(S0) && rno <= uint16(S31) {
				op = fmt.Sprintf("%sS", op)
			} else if rno >= uint16(D0) && rno <= uint16(D31) {
				op = fmt.Sprintf("%sD", op)
			}
		}

	case FCVTZS, FCVTZU, SCVTF, UCVTF:
		if _, ok := inst.Args[2].(Imm); !ok {
			for i := 1; i >= 0; i-- {
				if r, ok := inst.Args[i].(Reg); ok {
					rno := uint16(r)
					if rno >= uint16(S0) && rno <= uint16(S31) {
						op = fmt.Sprintf("%sS", op)
					} else if rno >= uint16(D0) && rno <= uint16(D31) {
						op = fmt.Sprintf("%sD", op)
					} else if rno <= uint16(WZR) {
						op += "W"
					}
				}
			}
		}

	case FMOV:
		for i := 0; i <= 1; i++ {
			if r, ok := inst.Args[i].(Reg); ok {
				rno := uint16(r)
				if rno >= uint16(S0) && rno <= uint16(S31) {
					op = fmt.Sprintf("%sS", op)
					break
				} else if rno >= uint16(D0) && rno <= uint16(D31) {
					op = fmt.Sprintf("%sD", op)
					break
				}
			}
		}

	case SYSL:
		op1 := int(inst.Args[1].(Imm).Imm)
		cn := int(inst.Args[2].(Imm_c))
		cm := int(inst.Args[3].(Imm_c))
		op2 := int(inst.Args[4].(Imm).Imm)
		sysregno := int32(op1<<16 | cn<<12 | cm<<8 | op2<<5)
		args[1] = fmt.Sprintf("$%d", sysregno)
		return op + " " + args[1] + ", " + args[0]

	case CBNZ, CBZ:
		if r, ok := inst.Args[0].(Reg); ok {
			rno := uint16(r)
			if rno <= uint16(WZR) {
				op += "W"
			}
		}
		args[0], args[1] = args[1], args[0]

	case ADR, ADRP:
		addr := int64(inst.Args[1].(PCRel))
		args[1] = fmt.Sprintf("%d(PC)", addr)

	case MSR:
		args[0] = inst.Args[0].String()

	case ST1:
		op = fmt.Sprintf("V%s", op) + suffix
		args[0], args[1] = args[1], args[0]

	case LD1:
		op = fmt.Sprintf("V%s", op) + suffix

	case UMOV:
		op = "VMOV"
	case NOP:
		op = "NOOP"

	default:
		index := sort.SearchStrings(noSuffixOpSet, op)
		if !(index < len(noSuffixOpSet) && noSuffixOpSet[index] == op) {
			rno := -1
			switch a := inst.Args[0].(type) {
			case Reg:
				rno = int(a)
			case RegSP:
				rno = int(a)
			case RegisterWithArrangement:
				op = fmt.Sprintf("V%s", op)
			}

			if rno >= int(B0) && rno <= int(Q31) && !strings.HasPrefix(op, "F") {
				op = fmt.Sprintf("V%s", op)
			}
			if rno >= 0 && rno <= int(WZR) {
				// Add "w" to opcode suffix.
				op += "W"
			}
		}
		op = op + suffix
	}

	// conditional instructions, replace args.
	if _, ok := inst.Args[3].(Cond); ok {
		if _, ok := inst.Args[2].(Reg); ok {
			args[1], args[2] = args[2], args[1]
		} else {
			args[0], args[2] = args[2], args[0]
		}
	}
	// Reverse args, placing dest last.
	for i, j := 0, len(args)-1; i < j; i, j = i+1, j-1 {
		args[i], args[j] = args[j], args[i]
	}

	if args != nil {
		op += " " + strings.Join(args, ", ")
	}

	return op
}

// No need add "W" to opcode suffix.
// Opcode must be inserted in ascending order.
var noSuffixOpSet = strings.Fields(`
AESD
AESE
AESIMC
AESMC
CRC32B
CRC32CB
CRC32CH
CRC32CW
CRC32CX
CRC32H
CRC32W
CRC32X
LDARB
LDARH
LDAXRB
LDAXRH
LDTRH
LDXRB
LDXRH
SHA1C
SHA1H
SHA1M
SHA1P
SHA1SU0
SHA1SU1
SHA256H
SHA256H2
SHA256SU0
SHA256SU1
`)

// floating point instructions without "F" prefix.
var fOpsWithoutFPrefix = map[Op]bool{
	LDP: true,
	STP: true,
}

func plan9Arg(inst *Inst, pc uint64, symname func(uint64) (string, uint64), arg Arg) string {
	switch a := arg.(type) {
	case Imm:
		return fmt.Sprintf("$%d", uint32(a.Imm))

	case Imm64:
		return fmt.Sprintf("$%d", int64(a.Imm))

	case ImmShift:
		if a.shift == 0 {
			return fmt.Sprintf("$%d", a.imm)
		}
		return fmt.Sprintf("$(%d<<%d)", a.imm, a.shift)

	case PCRel:
		addr := int64(pc) + int64(a)
		if s, base := symname(uint64(addr)); s != "" && uint64(addr) == base {
			return fmt.Sprintf("%s(SB)", s)
		}
		return fmt.Sprintf("%d(PC)", a/4)

	case Reg:
		regenum := uint16(a)
		regno := uint16(a) & 31

		if regenum >= uint16(B0) && regenum <= uint16(Q31) {
			if strings.HasPrefix(inst.Op.String(), "F") || strings.HasSuffix(inst.Op.String(), "CVTF") || fOpsWithoutFPrefix[inst.Op] {
				// FP registers are the same ones as SIMD registers
				// Print Fn for scalar variant to align with assembler (e.g., FCVT, SCVTF, UCVTF, etc.)
				return fmt.Sprintf("F%d", regno)
			} else {
				// Print Vn to align with assembler (e.g., SHA256H)
				return fmt.Sprintf("V%d", regno)
			}

		}
		return plan9gpr(a)

	case RegSP:
		regno := uint16(a) & 31
		if regno == 31 {
			return "RSP"
		}
		return fmt.Sprintf("R%d", regno)

	case RegExtshiftAmount:
		reg := plan9gpr(a.reg)
		extshift := ""
		amount := ""
		if a.extShift != ExtShift(0) {
			switch a.extShift {
			default:
				extshift = "." + a.extShift.String()

			case lsl:
				extshift = "<<"
				amount = fmt.Sprintf("%d", a.amount)
				return reg + extshift + amount

			case lsr:
				extshift = ">>"
				amount = fmt.Sprintf("%d", a.amount)
				return reg + extshift + amount

			case asr:
				extshift = "->"
				amount = fmt.Sprintf("%d", a.amount)
				return reg + extshift + amount
			case ror:
				extshift = "@>"
				amount = fmt.Sprintf("%d", a.amount)
				return reg + extshift + amount
			}
			if a.amount != 0 {
				amount = fmt.Sprintf("<<%d", a.amount)
			}
		}
		return reg + extshift + amount

	case MemImmediate:
		off := ""
		base := ""
		regno := uint16(a.Base) & 31
		if regno == 31 {
			base = "(RSP)"
		} else {
			base = fmt.Sprintf("(R%d)", regno)
		}
		if a.imm != 0 && a.Mode != AddrPostReg {
			off = fmt.Sprintf("%d", a.imm)
		} else if a.Mode == AddrPostReg {
			postR := fmt.Sprintf("(R%d)", a.imm)
			return base + postR
		}
		return off + base

	case MemExtend:
		base := ""
		index := ""
		regno := uint16(a.Base) & 31
		if regno == 31 {
			base = "(RSP)"
		} else {
			base = fmt.Sprintf("(R%d)", regno)
		}
		indexreg := plan9gpr(a.Index)

		if a.Extend == lsl {
			// Refer to ARM reference manual, for byte load/store(register), the index
			// shift amount must be 0, encoded in "S" as 0 if omitted, or as 1 if present.
			// a.Amount indicates the index shift amount, encoded in "S" field.
			// a.ShiftMustBeZero is set true indicates the index shift amount must be 0.
			// When a.ShiftMustBeZero is true, GNU syntax prints "[Xn, Xm lsl #0]" if "S"
			// equals to 1, or prints "[Xn, Xm]" if "S" equals to 0.
			if a.Amount != 0 && !a.ShiftMustBeZero {
				index = fmt.Sprintf("(%s<<%d)", indexreg, a.Amount)
			} else if a.ShiftMustBeZero && a.Amount == 1 {
				// When a.ShiftMustBeZero is ture, Go syntax prints "(Rm<<0)" if "a.Amount"
				// equals to 1.
				index = fmt.Sprintf("(%s<<0)", indexreg)
			} else {
				index = fmt.Sprintf("(%s)", indexreg)
			}
		} else {
			if a.Amount != 0 && !a.ShiftMustBeZero {
				index = fmt.Sprintf("(%s.%s<<%d)", indexreg, a.Extend.String(), a.Amount)
			} else {
				index = fmt.Sprintf("(%s.%s)", indexreg, a.Extend.String())
			}
		}

		return base + index

	case Cond:
		switch arg.String() {
		case "CS":
			return "HS"
		case "CC":
			return "LO"
		}

	case Imm_clrex:
		return fmt.Sprintf("$%d", uint32(a))

	case Imm_dcps:
		return fmt.Sprintf("$%d", uint32(a))

	case Imm_option:
		return fmt.Sprintf("$%d", uint8(a))

	case Imm_hint:
		return fmt.Sprintf("$%d", uint8(a))

	case Imm_fp:
		var s, pre, numerator, denominator int16
		var result float64
		if a.s == 0 {
			s = 1
		} else {
			s = -1
		}
		pre = s * int16(16+a.pre)
		if a.exp > 0 {
			numerator = (pre << uint8(a.exp))
			denominator = 16
		} else {
			numerator = pre
			denominator = (16 << uint8(-1*a.exp))
		}
		result = float64(numerator) / float64(denominator)
		return strings.TrimRight(fmt.Sprintf("$%f", result), "0")

	case RegisterWithArrangement:
		result := a.r.String()
		arrange := a.a.String()
		c := []rune(arrange)
		switch len(c) {
		case 3:
			c[1], c[2] = c[2], c[1] // .8B -> .B8
		case 4:
			c[1], c[2], c[3] = c[3], c[1], c[2] // 16B -> B16
		}
		arrange = string(c)
		result += arrange
		if a.cnt > 0 {
			result = "[" + result
			for i := 1; i < int(a.cnt); i++ {
				cur := V0 + Reg((uint16(a.r)-uint16(V0)+uint16(i))&31)
				result += ", " + cur.String() + arrange
			}
			result += "]"
		}
		return result

	case RegisterWithArrangementAndIndex:
		result := a.r.String()
		arrange := a.a.String()
		result += arrange
		if a.cnt > 1 {
			result = "[" + result
			for i := 1; i < int(a.cnt); i++ {
				cur := V0 + Reg((uint16(a.r)-uint16(V0)+uint16(i))&31)
				result += ", " + cur.String() + arrange
			}
			result += "]"
		}
		return fmt.Sprintf("%s[%d]", result, a.index)

	case Systemreg:
		return fmt.Sprintf("$%d", uint32(a.op0&1)<<14|uint32(a.op1&7)<<11|uint32(a.cn&15)<<7|uint32(a.cm&15)<<3|uint32(a.op2)&7)

	case Imm_prfop:
		if strings.Contains(a.String(), "#") {
			return fmt.Sprintf("$%d", a)
		}
	case sysOp:
		result := a.op.String()
		if a.r != 0 {
			result += ", " + plan9gpr(a.r)
		}
		return result
	}

	return strings.ToUpper(arg.String())
}

// Convert a general-purpose register to plan9 assembly format.
func plan9gpr(r Reg) string {
	regno := uint16(r) & 31
	if regno == 31 {
		return "ZR"
	}
	return fmt.Sprintf("R%d", regno)
}
