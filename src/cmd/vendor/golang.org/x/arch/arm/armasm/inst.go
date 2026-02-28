// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package armasm

import (
	"bytes"
	"fmt"
)

// A Mode is an instruction execution mode.
type Mode int

const (
	_ Mode = iota
	ModeARM
	ModeThumb
)

func (m Mode) String() string {
	switch m {
	case ModeARM:
		return "ARM"
	case ModeThumb:
		return "Thumb"
	}
	return fmt.Sprintf("Mode(%d)", int(m))
}

// An Op is an ARM opcode.
type Op uint16

// NOTE: The actual Op values are defined in tables.go.
// They are chosen to simplify instruction decoding and
// are not a dense packing from 0 to N, although the
// density is high, probably at least 90%.

func (op Op) String() string {
	if op >= Op(len(opstr)) || opstr[op] == "" {
		return fmt.Sprintf("Op(%d)", int(op))
	}
	return opstr[op]
}

// An Inst is a single instruction.
type Inst struct {
	Op   Op     // Opcode mnemonic
	Enc  uint32 // Raw encoding bits.
	Len  int    // Length of encoding in bytes.
	Args Args   // Instruction arguments, in ARM manual order.
}

func (i Inst) String() string {
	var buf bytes.Buffer
	buf.WriteString(i.Op.String())
	for j, arg := range i.Args {
		if arg == nil {
			break
		}
		if j == 0 {
			buf.WriteString(" ")
		} else {
			buf.WriteString(", ")
		}
		buf.WriteString(arg.String())
	}
	return buf.String()
}

// An Args holds the instruction arguments.
// If an instruction has fewer than 4 arguments,
// the final elements in the array are nil.
type Args [4]Arg

// An Arg is a single instruction argument, one of these types:
// Endian, Imm, Mem, PCRel, Reg, RegList, RegShift, RegShiftReg.
type Arg interface {
	IsArg()
	String() string
}

type Float32Imm float32

func (Float32Imm) IsArg() {}

func (f Float32Imm) String() string {
	return fmt.Sprintf("#%v", float32(f))
}

type Float64Imm float32

func (Float64Imm) IsArg() {}

func (f Float64Imm) String() string {
	return fmt.Sprintf("#%v", float64(f))
}

// An Imm is an integer constant.
type Imm uint32

func (Imm) IsArg() {}

func (i Imm) String() string {
	return fmt.Sprintf("#%#x", uint32(i))
}

// An ImmAlt is an alternate encoding of an integer constant.
type ImmAlt struct {
	Val uint8
	Rot uint8
}

func (ImmAlt) IsArg() {}

func (i ImmAlt) Imm() Imm {
	v := uint32(i.Val)
	r := uint(i.Rot)
	return Imm(v>>r | v<<(32-r))
}

func (i ImmAlt) String() string {
	return fmt.Sprintf("#%#x, %d", i.Val, i.Rot)
}

// A Label is a text (code) address.
type Label uint32

func (Label) IsArg() {}

func (i Label) String() string {
	return fmt.Sprintf("%#x", uint32(i))
}

// A Reg is a single register.
// The zero value denotes R0, not the absence of a register.
type Reg uint8

const (
	R0 Reg = iota
	R1
	R2
	R3
	R4
	R5
	R6
	R7
	R8
	R9
	R10
	R11
	R12
	R13
	R14
	R15

	S0
	S1
	S2
	S3
	S4
	S5
	S6
	S7
	S8
	S9
	S10
	S11
	S12
	S13
	S14
	S15
	S16
	S17
	S18
	S19
	S20
	S21
	S22
	S23
	S24
	S25
	S26
	S27
	S28
	S29
	S30
	S31

	D0
	D1
	D2
	D3
	D4
	D5
	D6
	D7
	D8
	D9
	D10
	D11
	D12
	D13
	D14
	D15
	D16
	D17
	D18
	D19
	D20
	D21
	D22
	D23
	D24
	D25
	D26
	D27
	D28
	D29
	D30
	D31

	APSR
	APSR_nzcv
	FPSCR

	SP = R13
	LR = R14
	PC = R15
)

func (Reg) IsArg() {}

func (r Reg) String() string {
	switch r {
	case APSR:
		return "APSR"
	case APSR_nzcv:
		return "APSR_nzcv"
	case FPSCR:
		return "FPSCR"
	case SP:
		return "SP"
	case PC:
		return "PC"
	case LR:
		return "LR"
	}
	if R0 <= r && r <= R15 {
		return fmt.Sprintf("R%d", int(r-R0))
	}
	if S0 <= r && r <= S31 {
		return fmt.Sprintf("S%d", int(r-S0))
	}
	if D0 <= r && r <= D31 {
		return fmt.Sprintf("D%d", int(r-D0))
	}
	return fmt.Sprintf("Reg(%d)", int(r))
}

// A RegX represents a fraction of a multi-value register.
// The Index field specifies the index number,
// but the size of the fraction is not specified.
// It must be inferred from the instruction and the register type.
// For example, in a VMOV instruction, RegX{D5, 1} represents
// the top 32 bits of the 64-bit D5 register.
type RegX struct {
	Reg   Reg
	Index int
}

func (RegX) IsArg() {}

func (r RegX) String() string {
	return fmt.Sprintf("%s[%d]", r.Reg, r.Index)
}

// A RegList is a register list.
// Bits at indexes x = 0 through 15 indicate whether the corresponding Rx register is in the list.
type RegList uint16

func (RegList) IsArg() {}

func (r RegList) String() string {
	var buf bytes.Buffer
	fmt.Fprintf(&buf, "{")
	sep := ""
	for i := 0; i < 16; i++ {
		if r&(1<<uint(i)) != 0 {
			fmt.Fprintf(&buf, "%s%s", sep, Reg(i).String())
			sep = ","
		}
	}
	fmt.Fprintf(&buf, "}")
	return buf.String()
}

// An Endian is the argument to the SETEND instruction.
type Endian uint8

const (
	LittleEndian Endian = 0
	BigEndian    Endian = 1
)

func (Endian) IsArg() {}

func (e Endian) String() string {
	if e != 0 {
		return "BE"
	}
	return "LE"
}

// A Shift describes an ARM shift operation.
type Shift uint8

const (
	ShiftLeft        Shift = 0 // left shift
	ShiftRight       Shift = 1 // logical (unsigned) right shift
	ShiftRightSigned Shift = 2 // arithmetic (signed) right shift
	RotateRight      Shift = 3 // right rotate
	RotateRightExt   Shift = 4 // right rotate through carry (Count will always be 1)
)

var shiftName = [...]string{
	"LSL", "LSR", "ASR", "ROR", "RRX",
}

func (s Shift) String() string {
	if s < 5 {
		return shiftName[s]
	}
	return fmt.Sprintf("Shift(%d)", int(s))
}

// A RegShift is a register shifted by a constant.
type RegShift struct {
	Reg   Reg
	Shift Shift
	Count uint8
}

func (RegShift) IsArg() {}

func (r RegShift) String() string {
	return fmt.Sprintf("%s %s #%d", r.Reg, r.Shift, r.Count)
}

// A RegShiftReg is a register shifted by a register.
type RegShiftReg struct {
	Reg      Reg
	Shift    Shift
	RegCount Reg
}

func (RegShiftReg) IsArg() {}

func (r RegShiftReg) String() string {
	return fmt.Sprintf("%s %s %s", r.Reg, r.Shift, r.RegCount)
}

// A PCRel describes a memory address (usually a code label)
// as a distance relative to the program counter.
// TODO(rsc): Define which program counter (PC+4? PC+8? PC?).
type PCRel int32

func (PCRel) IsArg() {}

func (r PCRel) String() string {
	return fmt.Sprintf("PC%+#x", int32(r))
}

// An AddrMode is an ARM addressing mode.
type AddrMode uint8

const (
	_             AddrMode = iota
	AddrPostIndex          // [R], X – use address R, set R = R + X
	AddrPreIndex           // [R, X]! – use address R + X, set R = R + X
	AddrOffset             // [R, X] – use address R + X
	AddrLDM                // R – [R] but formats as R, for LDM/STM only
	AddrLDM_WB             // R! - [R], X where X is instruction-specific amount, for LDM/STM only
)

// A Mem is a memory reference made up of a base R and index expression X.
// The effective memory address is R or R+X depending on AddrMode.
// The index expression is X = Sign*(Index Shift Count) + Offset,
// but in any instruction either Sign = 0 or Offset = 0.
type Mem struct {
	Base   Reg
	Mode   AddrMode
	Sign   int8
	Index  Reg
	Shift  Shift
	Count  uint8
	Offset int16
}

func (Mem) IsArg() {}

func (m Mem) String() string {
	R := m.Base.String()
	X := ""
	if m.Sign != 0 {
		X = "+"
		if m.Sign < 0 {
			X = "-"
		}
		X += m.Index.String()
		if m.Shift != ShiftLeft || m.Count != 0 {
			X += fmt.Sprintf(", %s #%d", m.Shift, m.Count)
		}
	} else {
		X = fmt.Sprintf("#%d", m.Offset)
	}

	switch m.Mode {
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
	return fmt.Sprintf("[%s Mode(%d) %s]", R, int(m.Mode), X)
}
