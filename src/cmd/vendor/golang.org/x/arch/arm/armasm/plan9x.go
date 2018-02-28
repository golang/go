// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package armasm

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"strings"
)

// GoSyntax returns the Go assembler syntax for the instruction.
// The syntax was originally defined by Plan 9.
// The pc is the program counter of the instruction, used for expanding
// PC-relative addresses into absolute ones.
// The symname function queries the symbol table for the program
// being disassembled. Given a target address it returns the name and base
// address of the symbol containing the target, if any; otherwise it returns "", 0.
// The reader r should read from the text segment using text addresses
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

	switch inst.Op &^ 15 {
	case LDR_EQ, LDRB_EQ, LDRH_EQ, LDRSB_EQ, LDRSH_EQ, VLDR_EQ:
		// Check for RET
		reg, _ := inst.Args[0].(Reg)
		mem, _ := inst.Args[1].(Mem)
		if inst.Op&^15 == LDR_EQ && reg == R15 && mem.Base == SP && mem.Sign == 0 && mem.Mode == AddrPostIndex {
			return fmt.Sprintf("RET%s #%d", op[3:], mem.Offset)
		}

		// Check for PC-relative load.
		if mem.Base == PC && mem.Sign == 0 && mem.Mode == AddrOffset && text != nil {
			addr := uint32(pc) + 8 + uint32(mem.Offset)
			buf := make([]byte, 8)
			switch inst.Op &^ 15 {
			case LDRB_EQ, LDRSB_EQ:
				if _, err := text.ReadAt(buf[:1], int64(addr)); err != nil {
					break
				}
				args[1] = fmt.Sprintf("$%#x", buf[0])

			case LDRH_EQ, LDRSH_EQ:
				if _, err := text.ReadAt(buf[:2], int64(addr)); err != nil {
					break
				}
				args[1] = fmt.Sprintf("$%#x", binary.LittleEndian.Uint16(buf))

			case LDR_EQ:
				if _, err := text.ReadAt(buf[:4], int64(addr)); err != nil {
					break
				}
				x := binary.LittleEndian.Uint32(buf)
				if s, base := symname(uint64(x)); s != "" && uint64(x) == base {
					args[1] = fmt.Sprintf("$%s(SB)", s)
				} else {
					args[1] = fmt.Sprintf("$%#x", x)
				}

			case VLDR_EQ:
				switch {
				case strings.HasPrefix(args[0], "D"): // VLDR.F64
					if _, err := text.ReadAt(buf, int64(addr)); err != nil {
						break
					}
					args[1] = fmt.Sprintf("$%f", math.Float64frombits(binary.LittleEndian.Uint64(buf)))
				case strings.HasPrefix(args[0], "S"): // VLDR.F32
					if _, err := text.ReadAt(buf[:4], int64(addr)); err != nil {
						break
					}
					args[1] = fmt.Sprintf("$%f", math.Float32frombits(binary.LittleEndian.Uint32(buf)))
				default:
					panic(fmt.Sprintf("wrong FP register: %v", inst))
				}
			}
		}
	}

	// Move addressing mode into opcode suffix.
	suffix := ""
	switch inst.Op &^ 15 {
	case LDR_EQ, LDRB_EQ, LDRSB_EQ, LDRH_EQ, LDRSH_EQ, STR_EQ, STRB_EQ, STRH_EQ, VLDR_EQ, VSTR_EQ:
		mem, _ := inst.Args[1].(Mem)
		switch mem.Mode {
		case AddrOffset, AddrLDM:
			// no suffix
		case AddrPreIndex, AddrLDM_WB:
			suffix = ".W"
		case AddrPostIndex:
			suffix = ".P"
		}
		off := ""
		if mem.Offset != 0 {
			off = fmt.Sprintf("%#x", mem.Offset)
		}
		base := fmt.Sprintf("(R%d)", int(mem.Base))
		index := ""
		if mem.Sign != 0 {
			sign := ""
			if mem.Sign < 0 {
				suffix += ".U"
			}
			shift := ""
			if mem.Count != 0 {
				shift = fmt.Sprintf("%s%d", plan9Shift[mem.Shift], mem.Count)
			}
			index = fmt.Sprintf("(%sR%d%s)", sign, int(mem.Index), shift)
		}
		args[1] = off + base + index
	}

	// Reverse args, placing dest last.
	for i, j := 0, len(args)-1; i < j; i, j = i+1, j-1 {
		args[i], args[j] = args[j], args[i]
	}
	// For MLA-like instructions, the addend is the third operand.
	switch inst.Op &^ 15 {
	case SMLAWT_EQ, SMLAWB_EQ, MLA_EQ, MLA_S_EQ, MLS_EQ, SMMLA_EQ, SMMLS_EQ, SMLABB_EQ, SMLATB_EQ, SMLABT_EQ, SMLATT_EQ, SMLAD_EQ, SMLAD_X_EQ, SMLSD_EQ, SMLSD_X_EQ:
		args = []string{args[1], args[2], args[0], args[3]}
	}

	switch inst.Op &^ 15 {
	case MOV_EQ:
		op = "MOVW" + op[3:]

	case LDR_EQ:
		op = "MOVW" + op[3:] + suffix
	case LDRB_EQ:
		op = "MOVBU" + op[4:] + suffix
	case LDRSB_EQ:
		op = "MOVBS" + op[5:] + suffix
	case LDRH_EQ:
		op = "MOVHU" + op[4:] + suffix
	case LDRSH_EQ:
		op = "MOVHS" + op[5:] + suffix
	case VLDR_EQ:
		switch {
		case strings.HasPrefix(args[1], "D"): // VLDR.F64
			op = "MOVD" + op[4:] + suffix
			args[1] = "F" + args[1][1:] // Dx -> Fx
		case strings.HasPrefix(args[1], "S"): // VLDR.F32
			op = "MOVF" + op[4:] + suffix
			if inst.Args[0].(Reg)&1 == 0 { // Sx -> Fy, y = x/2, if x is even
				args[1] = fmt.Sprintf("F%d", (inst.Args[0].(Reg)-S0)/2)
			}
		default:
			panic(fmt.Sprintf("wrong FP register: %v", inst))
		}

	case STR_EQ:
		op = "MOVW" + op[3:] + suffix
		args[0], args[1] = args[1], args[0]
	case STRB_EQ:
		op = "MOVB" + op[4:] + suffix
		args[0], args[1] = args[1], args[0]
	case STRH_EQ:
		op = "MOVH" + op[4:] + suffix
		args[0], args[1] = args[1], args[0]
	case VSTR_EQ:
		switch {
		case strings.HasPrefix(args[1], "D"): // VSTR.F64
			op = "MOVD" + op[4:] + suffix
			args[1] = "F" + args[1][1:] // Dx -> Fx
		case strings.HasPrefix(args[1], "S"): // VSTR.F32
			op = "MOVF" + op[4:] + suffix
			if inst.Args[0].(Reg)&1 == 0 { // Sx -> Fy, y = x/2, if x is even
				args[1] = fmt.Sprintf("F%d", (inst.Args[0].(Reg)-S0)/2)
			}
		default:
			panic(fmt.Sprintf("wrong FP register: %v", inst))
		}
		args[0], args[1] = args[1], args[0]
	}

	if args != nil {
		op += " " + strings.Join(args, ", ")
	}

	return op
}

// assembler syntax for the various shifts.
// @x> is a lie; the assembler uses @> 0
// instead of @x> 1, but i wanted to be clear that it
// was a different operation (rotate right extended, not rotate right).
var plan9Shift = []string{"<<", ">>", "->", "@>", "@x>"}

func plan9Arg(inst *Inst, pc uint64, symname func(uint64) (string, uint64), arg Arg) string {
	switch a := arg.(type) {
	case Endian:

	case Imm:
		return fmt.Sprintf("$%d", uint32(a))

	case Mem:

	case PCRel:
		addr := uint32(pc) + 8 + uint32(a)
		if s, base := symname(uint64(addr)); s != "" && uint64(addr) == base {
			return fmt.Sprintf("%s(SB)", s)
		}
		return fmt.Sprintf("%#x", addr)

	case Reg:
		if a < 16 {
			return fmt.Sprintf("R%d", int(a))
		}

	case RegList:
		var buf bytes.Buffer
		start := -2
		end := -2
		fmt.Fprintf(&buf, "[")
		flush := func() {
			if start >= 0 {
				if buf.Len() > 1 {
					fmt.Fprintf(&buf, ",")
				}
				if start == end {
					fmt.Fprintf(&buf, "R%d", start)
				} else {
					fmt.Fprintf(&buf, "R%d-R%d", start, end)
				}
				start = -2
				end = -2
			}
		}
		for i := 0; i < 16; i++ {
			if a&(1<<uint(i)) != 0 {
				if i == end+1 {
					end++
					continue
				}
				start = i
				end = i
			} else {
				flush()
			}
		}
		flush()
		fmt.Fprintf(&buf, "]")
		return buf.String()

	case RegShift:
		return fmt.Sprintf("R%d%s$%d", int(a.Reg), plan9Shift[a.Shift], int(a.Count))

	case RegShiftReg:
		return fmt.Sprintf("R%d%sR%d", int(a.Reg), plan9Shift[a.Shift], int(a.RegCount))
	}
	return strings.ToUpper(arg.String())
}
