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
	// bit 3 of index is a negated check.
	condBit = [8]string{
		"lt", "gt", "eq", "so",
		"ge", "le", "ne", "ns"}
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
	opName := inst.Op.String()
	argList := inst.Args[:]

	switch opName {
	case "bc", "bcl", "bca", "bcla", "bclr", "bclrl", "bcctr", "bcctrl", "bctar", "bctarl":
		sfx := inst.Op.String()[2:]
		bo := int(inst.Args[0].(Imm))
		bi := inst.Args[1].(CondReg)
		atsfx := [4]string{"", "?", "-", "+"}
		decsfx := [2]string{"dnz", "dz"}

		//BO field is... complicated (z == ignored bit, at == prediction hint)
		//Paraphrased from ISA 3.1 Book I Section 2.4:
		//
		//0000z -> decrement ctr, b if ctr != 0 and CRbi == 0
		//0001z -> decrement ctr, b if ctr == 0 and CRbi == 0
		//001at -> b if CRbi == 0
		//0100z -> decrement ctr, b if ctr != 0 and CRbi == 1
		//0101z -> decrement ctr, b if ctr == 0 and CRbi == 1
		//011at -> b if CRbi == 1
		//1a00t -> decrement ctr, b if ctr != 0
		//1a01t -> decrement ctr, b if ctr == 0
		//1z1zz -> b always

		// Decoding (in this order) we get
		// BO & 0b00100 == 0b00000 -> dz if BO[1], else dnz (not simplified for bcctrl forms)
		// BO & 0b10000 == 0b10000 -> (bc and bca forms not simplified), at = B[4]B[0] if B[2] != 0, done
		// BO & 0b10000 == 0b00000 -> t if BO[3], else f
		// BO & 0b10100 == 0b00100 -> at = B[0:1]

		// BI fields rename as follows:
		// less than            : lt BI%4==0 && test == t
		// less than or equal   : le BI%4==1 && test == f
		// equal 		: eq BI%4==2 && test == t
		// greater than or equal: ge BI%4==0 && test == f
		// greater than		: gt BI%4==1 && test == t
		// not less than	: nl BI%4==0 && test == f
		// not equal		: ne BI%4==2 && test == f
		// not greater than	: ng BI%4==1 && test == f
		// summary overflow	: so BI%4==3 && test == t
		// not summary overflow : ns BI%4==3 && test == f
		// unordered		: un BI%4==3 && test == t
		// not unordered	: nu BI%4==3 && test == f
		//
		// Note, there are only 8 possible tests, but quite a few more
		// ways to name fields.  For simplicity, we choose those in condBit.

		at := 0   // 0 == no hint, 1 == reserved, 2 == not likely, 3 == likely
		form := 1 // 1 == n/a,  0 == cr bit not set, 4 == cr bit set
		cr := (bi - Cond0LT) / 4
		bh := -1 // Only for lr/tar/ctr variants.
		switch opName {
		case "bclr", "bclrl", "bcctr", "bcctrl", "bctar", "bctarl":
			bh = int(inst.Args[2].(Imm))
		}

		if bo&0x14 == 0x14 {
			if bo == 0x14 && bi == Cond0LT { // preferred form of unconditional branch
				// Likewise, avoid printing fake b/ba/bl/bla
				if opName != "bc" && opName != "bca" && opName != "bcl" && opName != "bcla" {
					startArg = 2
				}
			}
		} else if bo&0x04 == 0 { // ctr is decremented
			if opName != "bcctr" && opName != "bcctrl" {
				startArg = 1
				tf := ""
				if bo&0x10 == 0x00 {
					tf = "f"
					if bo&0x08 == 0x08 {
						tf = "t"
					}
				}
				sfx = decsfx[(bo>>1)&1] + tf + sfx
			}
			if bo&0x10 == 0x10 {
				if opName != "bcctr" && opName != "bcctrl" {
					startArg = 2
				}
				if bi != Cond0LT {
					// A non-zero BI bit was encoded, but ignored by BO
					startArg = 0
				}
				at = ((bo & 0x8) >> 2) | (bo & 0x1)
			} else if bo&0x4 == 0x4 {
				at = bo & 0x3
			}
		} else if bo&0x10 == 0x10 { // BI field is not used
			if opName != "bca" && opName != "bc" {
				at = ((bo & 0x8) >> 2) | (bo & 0x1)
				startArg = 2
			}
			// If BI is encoded as a bit other than 0, no mnemonic.
			if bo&0x14 == 0x14 {
				startArg = 0
			}
		} else {
			form = (bo & 0x8) >> 1
			startArg = 2
			if bo&0x14 == 0x04 {
				at = bo & 0x3
			}
		}
		sfx += atsfx[at]

		if form != 1 {
			bit := int((bi-Cond0LT)%4) | (^form)&0x4
			sfx = condBit[bit] + sfx
		}

		if at != 1 && startArg > 0 && bh <= 0 {
			str := fmt.Sprintf("b%s", sfx)
			if startArg > 1 && (cr != 0 || bh > 0) {
				str += fmt.Sprintf(" cr%d", cr)
				sep = ","
			}
			buf.WriteString(str)
			if startArg < 2 && bh == 0 {
				str := fmt.Sprintf(" %s",
					gnuArg(&inst, 1, inst.Args[1], PC))
				buf.WriteString(str)
				startArg = 3
			} else if bh == 0 {
				startArg = 3
			}
		} else {
			if startArg == 0 || bh > 0 || at == 1 {
				buf.WriteString(inst.Op.String())
				buf.WriteString(atsfx[at])
				startArg = 0
			} else {
				buf.WriteString("b" + sfx)
			}
			if bh == 0 {
				str := fmt.Sprintf(" %d,%s", bo, gnuArg(&inst, 1, inst.Args[1], PC))
				buf.WriteString(str)
				startArg = 3
			}
		}

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

	case "mtfsfi", "mtfsfi.":
		buf.WriteString(opName)
		l := inst.Args[2].(Imm)
		if l == 0 {
			// L == 0 is an extended mnemonic for the same.
			asm := fmt.Sprintf(" %s,%s",
				gnuArg(&inst, 0, inst.Args[0], PC),
				gnuArg(&inst, 1, inst.Args[1], PC))
			buf.WriteString(asm)
			startArg = 3
		}

	case "paste.":
		buf.WriteString(opName)
		l := inst.Args[2].(Imm)
		if l == 1 {
			// L == 1 is an extended mnemonic for the same.
			asm := fmt.Sprintf(" %s,%s",
				gnuArg(&inst, 0, inst.Args[0], PC),
				gnuArg(&inst, 1, inst.Args[1], PC))
			buf.WriteString(asm)
			startArg = 3
		}

	case "mtfsf", "mtfsf.":
		buf.WriteString(opName)
		l := inst.Args[3].(Imm)
		if l == 0 {
			// L == 0 is an extended mnemonic for the same.
			asm := fmt.Sprintf(" %s,%s,%s",
				gnuArg(&inst, 0, inst.Args[0], PC),
				gnuArg(&inst, 1, inst.Args[1], PC),
				gnuArg(&inst, 2, inst.Args[2], PC))
			buf.WriteString(asm)
			startArg = 4
		}

	case "sync":
		lsc := inst.Args[0].(Imm)<<4 | inst.Args[1].(Imm)
		switch lsc {
		case 0x00:
			buf.WriteString("hwsync")
			startArg = 2
		case 0x10:
			buf.WriteString("lwsync")
			startArg = 2
		default:
			buf.WriteString(opName)
		}

	case "lbarx", "lharx", "lwarx", "ldarx":
		// If EH == 0, omit printing EH.
		eh := inst.Args[3].(Imm)
		if eh == 0 {
			argList = inst.Args[:3]
		}
		buf.WriteString(inst.Op.String())

	case "paddi":
		// There are several extended mnemonics.  Notably, "pla" is
		// the only valid mnemonic for paddi (R=1), In this case, RA must
		// always be 0.  Otherwise it is invalid.
		r := inst.Args[3].(Imm)
		ra := inst.Args[1].(Reg)
		str := opName
		if ra == R0 {
			name := []string{"pli", "pla"}
			str = fmt.Sprintf("%s %s,%s",
				name[r&1],
				gnuArg(&inst, 0, inst.Args[0], PC),
				gnuArg(&inst, 2, inst.Args[2], PC))
			startArg = 4
		} else {
			str = fmt.Sprintf("%s %s,%s,%s", opName,
				gnuArg(&inst, 0, inst.Args[0], PC),
				gnuArg(&inst, 1, inst.Args[1], PC),
				gnuArg(&inst, 2, inst.Args[2], PC))
			startArg = 4
			if r == 1 {
				// This is an illegal encoding (ra != 0 && r == 1) on ISA 3.1.
				v := uint64(inst.Enc)<<32 | uint64(inst.SuffixEnc)
				return fmt.Sprintf(".quad 0x%x", v)
			}
		}
		buf.WriteString(str)

	default:
		// Prefixed load/stores do not print the displacement register when R==1 (they are PCrel).
		// This also implies RA should be 0.  Likewise, when R==0, printing of R can be omitted.
		if strings.HasPrefix(opName, "pl") || strings.HasPrefix(opName, "pst") {
			r := inst.Args[3].(Imm)
			ra := inst.Args[2].(Reg)
			d := inst.Args[1].(Offset)
			if r == 1 && ra == R0 {
				str := fmt.Sprintf("%s %s,%d", opName, gnuArg(&inst, 0, inst.Args[0], PC), d)
				buf.WriteString(str)
				startArg = 4
			} else {
				str := fmt.Sprintf("%s %s,%d(%s)", opName,
					gnuArg(&inst, 0, inst.Args[0], PC),
					d,
					gnuArg(&inst, 2, inst.Args[2], PC))
				if r == 1 {
					// This is an invalid encoding (ra != 0 && r == 1) on ISA 3.1.
					v := uint64(inst.Enc)<<32 | uint64(inst.SuffixEnc)
					return fmt.Sprintf(".quad 0x%x", v)
				}
				buf.WriteString(str)
				startArg = 4
			}
		} else {
			buf.WriteString(opName)
		}
	}
	for i, arg := range argList {
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
		return fmt.Sprintf("4*cr%d+%s", int(arg-Cond0LT)/4, bit)
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
	case LBARX, LWARX, LHARX, LDARX:
		return true
	}
	return false
}
