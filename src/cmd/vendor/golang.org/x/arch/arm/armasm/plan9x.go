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
	case PLD, PLI, PLD_W:
		if mem, ok := inst.Args[0].(Mem); ok {
			args[0], suffix = memOpTrans(mem)
		} else {
			panic(fmt.Sprintf("illegal instruction: %v", inst))
		}
	case LDR_EQ, LDRB_EQ, LDRSB_EQ, LDRH_EQ, LDRSH_EQ, STR_EQ, STRB_EQ, STRH_EQ, VLDR_EQ, VSTR_EQ, LDREX_EQ, LDREXH_EQ, LDREXB_EQ:
		if mem, ok := inst.Args[1].(Mem); ok {
			args[1], suffix = memOpTrans(mem)
		} else {
			panic(fmt.Sprintf("illegal instruction: %v", inst))
		}
	case SWP_EQ, SWP_B_EQ, STREX_EQ, STREXB_EQ, STREXH_EQ:
		if mem, ok := inst.Args[2].(Mem); ok {
			args[2], suffix = memOpTrans(mem)
		} else {
			panic(fmt.Sprintf("illegal instruction: %v", inst))
		}
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
	// For STREX like instructions, the memory operands comes first.
	switch inst.Op &^ 15 {
	case STREX_EQ, STREXB_EQ, STREXH_EQ, SWP_EQ, SWP_B_EQ:
		args = []string{args[1], args[0], args[2]}
	}

	// special process for FP instructions
	op, args = fpTrans(&inst, op, args)

	// LDR/STR like instructions -> MOV like
	switch inst.Op &^ 15 {
	case MOV_EQ:
		op = "MOVW" + op[3:]
	case LDR_EQ, MSR_EQ, MRS_EQ:
		op = "MOVW" + op[3:] + suffix
	case VMRS_EQ, VMSR_EQ:
		op = "MOVW" + op[4:] + suffix
	case LDRB_EQ, UXTB_EQ:
		op = "MOVBU" + op[4:] + suffix
	case LDRSB_EQ:
		op = "MOVBS" + op[5:] + suffix
	case SXTB_EQ:
		op = "MOVBS" + op[4:] + suffix
	case LDRH_EQ, UXTH_EQ:
		op = "MOVHU" + op[4:] + suffix
	case LDRSH_EQ:
		op = "MOVHS" + op[5:] + suffix
	case SXTH_EQ:
		op = "MOVHS" + op[4:] + suffix
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
		args[0], args[1] = args[1], args[0]
	default:
		op = op + suffix
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

// convert memory operand from GNU syntax to Plan 9 syntax, for example,
// [r5] -> (R5)
// [r6, #4080] -> 0xff0(R6)
// [r2, r0, ror #1] -> (R2)(R0@>1)
// inst [r2, -r0, ror #1] -> INST.U (R2)(R0@>1)
// input:
//
//	a memory operand
//
// return values:
//
//	corresponding memory operand in Plan 9 syntax
//	.W/.P/.U suffix
func memOpTrans(mem Mem) (string, string) {
	suffix := ""
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
	return off + base + index, suffix
}

type goFPInfo struct {
	op        Op
	transArgs []int  // indexes of arguments which need transformation
	gnuName   string // instruction name in GNU syntax
	goName    string // instruction name in Plan 9 syntax
}

var fpInst []goFPInfo = []goFPInfo{
	{VADD_EQ_F32, []int{2, 1, 0}, "VADD", "ADDF"},
	{VADD_EQ_F64, []int{2, 1, 0}, "VADD", "ADDD"},
	{VSUB_EQ_F32, []int{2, 1, 0}, "VSUB", "SUBF"},
	{VSUB_EQ_F64, []int{2, 1, 0}, "VSUB", "SUBD"},
	{VMUL_EQ_F32, []int{2, 1, 0}, "VMUL", "MULF"},
	{VMUL_EQ_F64, []int{2, 1, 0}, "VMUL", "MULD"},
	{VNMUL_EQ_F32, []int{2, 1, 0}, "VNMUL", "NMULF"},
	{VNMUL_EQ_F64, []int{2, 1, 0}, "VNMUL", "NMULD"},
	{VMLA_EQ_F32, []int{2, 1, 0}, "VMLA", "MULAF"},
	{VMLA_EQ_F64, []int{2, 1, 0}, "VMLA", "MULAD"},
	{VMLS_EQ_F32, []int{2, 1, 0}, "VMLS", "MULSF"},
	{VMLS_EQ_F64, []int{2, 1, 0}, "VMLS", "MULSD"},
	{VNMLA_EQ_F32, []int{2, 1, 0}, "VNMLA", "NMULAF"},
	{VNMLA_EQ_F64, []int{2, 1, 0}, "VNMLA", "NMULAD"},
	{VNMLS_EQ_F32, []int{2, 1, 0}, "VNMLS", "NMULSF"},
	{VNMLS_EQ_F64, []int{2, 1, 0}, "VNMLS", "NMULSD"},
	{VDIV_EQ_F32, []int{2, 1, 0}, "VDIV", "DIVF"},
	{VDIV_EQ_F64, []int{2, 1, 0}, "VDIV", "DIVD"},
	{VNEG_EQ_F32, []int{1, 0}, "VNEG", "NEGF"},
	{VNEG_EQ_F64, []int{1, 0}, "VNEG", "NEGD"},
	{VABS_EQ_F32, []int{1, 0}, "VABS", "ABSF"},
	{VABS_EQ_F64, []int{1, 0}, "VABS", "ABSD"},
	{VSQRT_EQ_F32, []int{1, 0}, "VSQRT", "SQRTF"},
	{VSQRT_EQ_F64, []int{1, 0}, "VSQRT", "SQRTD"},
	{VCMP_EQ_F32, []int{1, 0}, "VCMP", "CMPF"},
	{VCMP_EQ_F64, []int{1, 0}, "VCMP", "CMPD"},
	{VCMP_E_EQ_F32, []int{1, 0}, "VCMP.E", "CMPF"},
	{VCMP_E_EQ_F64, []int{1, 0}, "VCMP.E", "CMPD"},
	{VLDR_EQ, []int{1}, "VLDR", "MOV"},
	{VSTR_EQ, []int{1}, "VSTR", "MOV"},
	{VMOV_EQ_F32, []int{1, 0}, "VMOV", "MOVF"},
	{VMOV_EQ_F64, []int{1, 0}, "VMOV", "MOVD"},
	{VMOV_EQ_32, []int{1, 0}, "VMOV", "MOVW"},
	{VMOV_EQ, []int{1, 0}, "VMOV", "MOVW"},
	{VCVT_EQ_F64_F32, []int{1, 0}, "VCVT", "MOVFD"},
	{VCVT_EQ_F32_F64, []int{1, 0}, "VCVT", "MOVDF"},
	{VCVT_EQ_F32_U32, []int{1, 0}, "VCVT", "MOVWF.U"},
	{VCVT_EQ_F32_S32, []int{1, 0}, "VCVT", "MOVWF"},
	{VCVT_EQ_S32_F32, []int{1, 0}, "VCVT", "MOVFW"},
	{VCVT_EQ_U32_F32, []int{1, 0}, "VCVT", "MOVFW.U"},
	{VCVT_EQ_F64_U32, []int{1, 0}, "VCVT", "MOVWD.U"},
	{VCVT_EQ_F64_S32, []int{1, 0}, "VCVT", "MOVWD"},
	{VCVT_EQ_S32_F64, []int{1, 0}, "VCVT", "MOVDW"},
	{VCVT_EQ_U32_F64, []int{1, 0}, "VCVT", "MOVDW.U"},
}

// convert FP instructions from GNU syntax to Plan 9 syntax, for example,
// vadd.f32 s0, s3, s4 -> ADDF F0, S3, F2
// vsub.f64 d0, d2, d4 -> SUBD F0, F2, F4
// vldr s2, [r11] -> MOVF (R11), F1
// inputs: instruction name and arguments in GNU syntax
// return values: corresponding instruction name and arguments in Plan 9 syntax
func fpTrans(inst *Inst, op string, args []string) (string, []string) {
	for _, fp := range fpInst {
		if inst.Op&^15 == fp.op {
			// remove gnu syntax suffixes
			op = strings.Replace(op, ".F32", "", -1)
			op = strings.Replace(op, ".F64", "", -1)
			op = strings.Replace(op, ".S32", "", -1)
			op = strings.Replace(op, ".U32", "", -1)
			op = strings.Replace(op, ".32", "", -1)
			// compose op name
			if fp.op == VLDR_EQ || fp.op == VSTR_EQ {
				switch {
				case strings.HasPrefix(args[fp.transArgs[0]], "D"):
					op = "MOVD" + op[len(fp.gnuName):]
				case strings.HasPrefix(args[fp.transArgs[0]], "S"):
					op = "MOVF" + op[len(fp.gnuName):]
				default:
					panic(fmt.Sprintf("wrong FP register: %v", inst))
				}
			} else {
				op = fp.goName + op[len(fp.gnuName):]
			}
			// transform registers
			for ix, ri := range fp.transArgs {
				switch {
				case strings.HasSuffix(args[ri], "[1]"): // MOVW Rx, Dy[1]
					break
				case strings.HasSuffix(args[ri], "[0]"): // Dx[0] -> Fx
					args[ri] = strings.Replace(args[ri], "[0]", "", -1)
					fallthrough
				case strings.HasPrefix(args[ri], "D"): // Dx -> Fx
					args[ri] = "F" + args[ri][1:]
				case strings.HasPrefix(args[ri], "S"):
					if inst.Args[ix].(Reg)&1 == 0 { // Sx -> Fy, y = x/2, if x is even
						args[ri] = fmt.Sprintf("F%d", (inst.Args[ix].(Reg)-S0)/2)
					}
				case strings.HasPrefix(args[ri], "$"): // CMPF/CMPD $0, Fx
					break
				case strings.HasPrefix(args[ri], "R"): // MOVW Rx, Dy[1]
					break
				default:
					panic(fmt.Sprintf("wrong FP register: %v", inst))
				}
			}
			break
		}
	}
	return op, args
}
