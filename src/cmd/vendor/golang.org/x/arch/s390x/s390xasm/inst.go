// Copyright 2024 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package s390xasm

import (
	"bytes"
	"fmt"
	"strings"
)

type Inst struct {
	Op   Op     // Opcode mnemonic
	Enc  uint64 // Raw encoding bits (if Len == 8, this is the prefix word)
	Len  int    // Length of encoding in bytes.
	Args Args   // Instruction arguments, in Power ISA manual order.
}

func (i Inst) String(pc uint64) string {
	var buf bytes.Buffer
	var rxb_check bool
	m := i.Op.String()
	if strings.HasPrefix(m, "v") || strings.Contains(m, "wfc") || strings.Contains(m, "wfk") {
		rxb_check = true
	}
	mnemonic := HandleExtndMnemonic(&i)
	buf.WriteString(fmt.Sprintf("%s", mnemonic))
	for j, arg := range i.Args {
		if arg == nil {
			break
		}
		if j == 0 {
			buf.WriteString(" ")
		} else {
			switch arg.(type) {
			case VReg, Reg:
				if _, ok := i.Args[j-1].(Disp12); ok {
					buf.WriteString("")
				} else if _, ok := i.Args[j-1].(Disp20); ok {
					buf.WriteString("")
				} else {
					buf.WriteString(",")
				}
			case Base:
				if _, ok := i.Args[j-1].(VReg); ok {
					buf.WriteString(",")
				} else if _, ok := i.Args[j-1].(Reg); ok {
					buf.WriteString(",")
				}
			case Index, Len:
			default:
				buf.WriteString(",")
			}
		}
		buf.WriteString(arg.String(pc))
		if rxb_check && i.Args[j+2] == nil {
			break
		}
	}
	return buf.String()
}

// An Op is an instruction operation.
type Op uint16

func (o Op) String() string {
	if int(o) >= len(opstr) || opstr[o] == "" {
		return fmt.Sprintf("Op(%d)", int(o))
	}
	return opstr[o]
}

// An Arg is a single instruction argument.
// One of these types: Reg, Base, Index, Disp20, Disp12, Len, Mask, Sign8, Sign16, Sign32, RegIm12, RegIm16, RegIm24, RegIm32.
type Arg interface {
	IsArg()
	String(pc uint64) string
}

// An Args holds the instruction arguments.
// If an instruction has fewer than 6 arguments,
// the final elements in the array are nil.
type Args [8]Arg

// Base represents an 4-bit Base Register field
type Base uint8

const (
	B0 Base = iota
	B1
	B2
	B3
	B4
	B5
	B6
	B7
	B8
	B9
	B10
	B11
	B12
	B13
	B14
	B15
)

func (Base) IsArg() {}
func (r Base) String(pc uint64) string {
	switch {
	case B1 <= r && r <= B15:
		s := "%"
		return fmt.Sprintf("%sr%d)", s, int(r-B0))
	case B0 == r:
		return fmt.Sprintf("")
	default:
		return fmt.Sprintf("Base(%d)", int(r))
	}
}

// Index represents an 4-bit Index Register field
type Index uint8

const (
	X0 Index = iota
	X1
	X2
	X3
	X4
	X5
	X6
	X7
	X8
	X9
	X10
	X11
	X12
	X13
	X14
	X15
)

func (Index) IsArg() {}
func (r Index) String(pc uint64) string {
	switch {
	case X1 <= r && r <= X15:
		s := "%"
		return fmt.Sprintf("%sr%d,", s, int(r-X0))
	case X0 == r:
		return fmt.Sprintf("")
	default:
		return fmt.Sprintf("Base(%d)", int(r))
	}
}

// Disp20 represents an 20-bit Unsigned Displacement
type Disp20 uint32

func (Disp20) IsArg() {}
func (r Disp20) String(pc uint64) string {
	if (r>>19)&0x01 == 1 {
		return fmt.Sprintf("%d(", int32(r|0xfff<<20))
	} else {
		return fmt.Sprintf("%d(", int32(r))
	}
}

// Disp12 represents an 12-bit Unsigned Displacement
type Disp12 uint16

func (Disp12) IsArg() {}
func (r Disp12) String(pc uint64) string {
	return fmt.Sprintf("%d(", r)
}

// RegIm12 represents an 12-bit Register immediate number.
type RegIm12 uint16

func (RegIm12) IsArg() {}
func (r RegIm12) String(pc uint64) string {
	if (r>>11)&0x01 == 1 {
		return fmt.Sprintf("%#x", pc+(2*uint64(int16(r|0xf<<12))))
	} else {
		return fmt.Sprintf("%#x", pc+(2*uint64(int16(r))))
	}
}

// RegIm16 represents an 16-bit Register immediate number.
type RegIm16 uint16

func (RegIm16) IsArg() {}
func (r RegIm16) String(pc uint64) string {
	return fmt.Sprintf("%#x", pc+(2*uint64(int16(r))))
}

// RegIm24 represents an 24-bit Register immediate number.
type RegIm24 uint32

func (RegIm24) IsArg() {}
func (r RegIm24) String(pc uint64) string {
	if (r>>23)&0x01 == 1 {
		return fmt.Sprintf("%#x", pc+(2*uint64(int32(r|0xff<<24))))
	} else {
		return fmt.Sprintf("%#x", pc+(2*uint64(int32(r))))
	}
}

// RegIm32 represents an 32-bit Register immediate number.
type RegIm32 uint32

func (RegIm32) IsArg() {}
func (r RegIm32) String(pc uint64) string {
	return fmt.Sprintf("%#x", pc+(2*uint64(int32(r))))
}

// A Reg is a single register. The zero value means R0, not the absence of a register.
// It also includes special registers.
type Reg uint16

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
	F0
	F1
	F2
	F3
	F4
	F5
	F6
	F7
	F8
	F9
	F10
	F11
	F12
	F13
	F14
	F15
	A0
	A1
	A2
	A3
	A4
	A5
	A6
	A7
	A8
	A9
	A10
	A11
	A12
	A13
	A14
	A15
	C0
	C1
	C2
	C3
	C4
	C5
	C6
	C7
	C8
	C9
	C10
	C11
	C12
	C13
	C14
	C15
)

func (Reg) IsArg() {}
func (r Reg) String(pc uint64) string {
	s := "%"
	switch {
	case R0 <= r && r <= R15:
		return fmt.Sprintf("%sr%d", s, int(r-R0))
	case F0 <= r && r <= F15:
		return fmt.Sprintf("%sf%d", s, int(r-F0))
	case A0 <= r && r <= A15:
		return fmt.Sprintf("%sa%d", s, int(r-A0))
	case C0 <= r && r <= C15:
		return fmt.Sprintf("%sc%d", s, int(r-C0))
	default:
		return fmt.Sprintf("Reg(%d)", int(r))
	}
}

// VReg is a vector register. The zero value means V0, not the absence of a register.

type VReg uint8

const (
	V0 VReg = iota
	V1
	V2
	V3
	V4
	V5
	V6
	V7
	V8
	V9
	V10
	V11
	V12
	V13
	V14
	V15
	V16
	V17
	V18
	V19
	V20
	V21
	V22
	V23
	V24
	V25
	V26
	V27
	V28
	V29
	V30
	V31
)

func (VReg) IsArg() {}
func (r VReg) String(pc uint64) string {
	s := "%"
	if V0 <= r && r <= V31 {
		return fmt.Sprintf("%sv%d", s, int(r-V0))
	} else {
		return fmt.Sprintf("VReg(%d)", int(r))
	}
}

// Imm represents an immediate number.
type Imm uint32

func (Imm) IsArg() {}
func (i Imm) String(pc uint64) string {
	return fmt.Sprintf("%d", uint32(i))
}

// Sign8 represents an 8-bit signed immediate number.
type Sign8 int8

func (Sign8) IsArg() {}
func (i Sign8) String(pc uint64) string {
	return fmt.Sprintf("%d", i)
}

// Sign16 represents an 16-bit signed immediate number.
type Sign16 int16

func (Sign16) IsArg() {}
func (i Sign16) String(pc uint64) string {
	return fmt.Sprintf("%d", i)
}

// Sign32 represents an 32-bit signed immediate number.
type Sign32 int32

func (Sign32) IsArg() {}
func (i Sign32) String(pc uint64) string {
	return fmt.Sprintf("%d", i)
}

// Mask represents an 4-bit mask value
type Mask uint8

func (Mask) IsArg() {}
func (i Mask) String(pc uint64) string {
	return fmt.Sprintf("%d", i)
}

// Len represents an 8-bit type holds 4/8-bit Len argument
type Len uint8

func (Len) IsArg() {}
func (i Len) String(pc uint64) string {
	return fmt.Sprintf("%d,", uint16(i)+1)
}
