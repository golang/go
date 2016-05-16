// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package armasm

import (
	"bytes"
	"fmt"
	"strings"
)

var saveDot = strings.NewReplacer(
	".F16", "_dot_F16",
	".F32", "_dot_F32",
	".F64", "_dot_F64",
	".S32", "_dot_S32",
	".U32", "_dot_U32",
	".FXS", "_dot_S",
	".FXU", "_dot_U",
	".32", "_dot_32",
)

// GNUSyntax returns the GNU assembler syntax for the instruction, as defined by GNU binutils.
// This form typically matches the syntax defined in the ARM Reference Manual.
func GNUSyntax(inst Inst) string {
	var buf bytes.Buffer
	op := inst.Op.String()
	op = saveDot.Replace(op)
	op = strings.Replace(op, ".", "", -1)
	op = strings.Replace(op, "_dot_", ".", -1)
	op = strings.ToLower(op)
	buf.WriteString(op)
	sep := " "
	for i, arg := range inst.Args {
		if arg == nil {
			break
		}
		text := gnuArg(&inst, i, arg)
		if text == "" {
			continue
		}
		buf.WriteString(sep)
		sep = ", "
		buf.WriteString(text)
	}
	return buf.String()
}

func gnuArg(inst *Inst, argIndex int, arg Arg) string {
	switch inst.Op &^ 15 {
	case LDRD_EQ, LDREXD_EQ, STRD_EQ:
		if argIndex == 1 {
			// second argument in consecutive pair not printed
			return ""
		}
	case STREXD_EQ:
		if argIndex == 2 {
			// second argument in consecutive pair not printed
			return ""
		}
	}

	switch arg := arg.(type) {
	case Imm:
		switch inst.Op &^ 15 {
		case BKPT_EQ:
			return fmt.Sprintf("%#04x", uint32(arg))
		case SVC_EQ:
			return fmt.Sprintf("%#08x", uint32(arg))
		}
		return fmt.Sprintf("#%d", int32(arg))

	case ImmAlt:
		return fmt.Sprintf("#%d, %d", arg.Val, arg.Rot)

	case Mem:
		R := gnuArg(inst, -1, arg.Base)
		X := ""
		if arg.Sign != 0 {
			X = ""
			if arg.Sign < 0 {
				X = "-"
			}
			X += gnuArg(inst, -1, arg.Index)
			if arg.Shift == ShiftLeft && arg.Count == 0 {
				// nothing
			} else if arg.Shift == RotateRightExt {
				X += ", rrx"
			} else {
				X += fmt.Sprintf(", %s #%d", strings.ToLower(arg.Shift.String()), arg.Count)
			}
		} else {
			X = fmt.Sprintf("#%d", arg.Offset)
		}

		switch arg.Mode {
		case AddrOffset:
			if X == "#0" {
				return fmt.Sprintf("[%s]", R)
			}
			return fmt.Sprintf("[%s, %s]", R, X)
		case AddrPreIndex:
			return fmt.Sprintf("[%s, %s]!", R, X)
		case AddrPostIndex:
			return fmt.Sprintf("[%s], %s", R, X)
		case AddrLDM:
			if X == "#0" {
				return R
			}
		case AddrLDM_WB:
			if X == "#0" {
				return R + "!"
			}
		}
		return fmt.Sprintf("[%s Mode(%d) %s]", R, int(arg.Mode), X)

	case PCRel:
		return fmt.Sprintf(".%+#x", int32(arg)+4)

	case Reg:
		switch inst.Op &^ 15 {
		case LDREX_EQ:
			if argIndex == 0 {
				return fmt.Sprintf("r%d", int32(arg))
			}
		}
		switch arg {
		case R10:
			return "sl"
		case R11:
			return "fp"
		case R12:
			return "ip"
		}

	case RegList:
		var buf bytes.Buffer
		fmt.Fprintf(&buf, "{")
		sep := ""
		for i := 0; i < 16; i++ {
			if arg&(1<<uint(i)) != 0 {
				fmt.Fprintf(&buf, "%s%s", sep, gnuArg(inst, -1, Reg(i)))
				sep = ", "
			}
		}
		fmt.Fprintf(&buf, "}")
		return buf.String()

	case RegShift:
		if arg.Shift == ShiftLeft && arg.Count == 0 {
			return gnuArg(inst, -1, arg.Reg)
		}
		if arg.Shift == RotateRightExt {
			return gnuArg(inst, -1, arg.Reg) + ", rrx"
		}
		return fmt.Sprintf("%s, %s #%d", gnuArg(inst, -1, arg.Reg), strings.ToLower(arg.Shift.String()), arg.Count)

	case RegShiftReg:
		return fmt.Sprintf("%s, %s %s", gnuArg(inst, -1, arg.Reg), strings.ToLower(arg.Shift.String()), gnuArg(inst, -1, arg.RegCount))

	}
	return strings.ToLower(arg.String())
}
