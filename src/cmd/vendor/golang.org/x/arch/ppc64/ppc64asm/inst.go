// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ppc64asm

import (
	"bytes"
	"fmt"
)

type Inst struct {
	Op   Op     // Opcode mnemonic
	Enc  uint32 // Raw encoding bits
	Len  int    // Length of encoding in bytes.
	Args Args   // Instruction arguments, in Power ISA manual order.
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

// An Op is an instruction operation.
type Op uint16

func (o Op) String() string {
	if int(o) >= len(opstr) || opstr[o] == "" {
		return fmt.Sprintf("Op(%d)", int(o))
	}
	return opstr[o]
}

// An Arg is a single instruction argument, one of these types: Reg, CondReg, SpReg, Imm, PCRel, Label, or Offset.
type Arg interface {
	IsArg()
	String() string
}

// An Args holds the instruction arguments.
// If an instruction has fewer than 4 arguments,
// the final elements in the array are nil.
type Args [5]Arg

// A Reg is a single register. The zero value means R0, not the absence of a register.
// It also includes special registers.
type Reg uint16

const (
	_ Reg = iota
	R0
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
	R16
	R17
	R18
	R19
	R20
	R21
	R22
	R23
	R24
	R25
	R26
	R27
	R28
	R29
	R30
	R31
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
	F16
	F17
	F18
	F19
	F20
	F21
	F22
	F23
	F24
	F25
	F26
	F27
	F28
	F29
	F30
	F31
	V0 // VSX extension, F0 is V0[0:63].
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
	VS0
	VS1
	VS2
	VS3
	VS4
	VS5
	VS6
	VS7
	VS8
	VS9
	VS10
	VS11
	VS12
	VS13
	VS14
	VS15
	VS16
	VS17
	VS18
	VS19
	VS20
	VS21
	VS22
	VS23
	VS24
	VS25
	VS26
	VS27
	VS28
	VS29
	VS30
	VS31
	VS32
	VS33
	VS34
	VS35
	VS36
	VS37
	VS38
	VS39
	VS40
	VS41
	VS42
	VS43
	VS44
	VS45
	VS46
	VS47
	VS48
	VS49
	VS50
	VS51
	VS52
	VS53
	VS54
	VS55
	VS56
	VS57
	VS58
	VS59
	VS60
	VS61
	VS62
	VS63
)

func (Reg) IsArg() {}
func (r Reg) String() string {
	switch {
	case R0 <= r && r <= R31:
		return fmt.Sprintf("r%d", int(r-R0))
	case F0 <= r && r <= F31:
		return fmt.Sprintf("f%d", int(r-F0))
	case V0 <= r && r <= V31:
		return fmt.Sprintf("v%d", int(r-V0))
	case VS0 <= r && r <= VS63:
		return fmt.Sprintf("vs%d", int(r-VS0))
	default:
		return fmt.Sprintf("Reg(%d)", int(r))
	}
}

// CondReg is a bit or field in the conditon register.
type CondReg int8

const (
	_ CondReg = iota
	// Condition Regster bits
	Cond0LT
	Cond0GT
	Cond0EQ
	Cond0SO
	Cond1LT
	Cond1GT
	Cond1EQ
	Cond1SO
	Cond2LT
	Cond2GT
	Cond2EQ
	Cond2SO
	Cond3LT
	Cond3GT
	Cond3EQ
	Cond3SO
	Cond4LT
	Cond4GT
	Cond4EQ
	Cond4SO
	Cond5LT
	Cond5GT
	Cond5EQ
	Cond5SO
	Cond6LT
	Cond6GT
	Cond6EQ
	Cond6SO
	Cond7LT
	Cond7GT
	Cond7EQ
	Cond7SO
	// Condition Register Fields
	CR0
	CR1
	CR2
	CR3
	CR4
	CR5
	CR6
	CR7
)

func (CondReg) IsArg() {}
func (c CondReg) String() string {
	switch {
	default:
		return fmt.Sprintf("CondReg(%d)", int(c))
	case c >= CR0:
		return fmt.Sprintf("CR%d", int(c-CR0))
	case c >= Cond0LT && c < CR0:
		return fmt.Sprintf("Cond%d%s", int((c-Cond0LT)/4), [4]string{"LT", "GT", "EQ", "SO"}[(c-Cond0LT)%4])
	}
}

// SpReg is a special register, its meaning depends on Op.
type SpReg uint16

const (
	SpRegZero SpReg = 0
)

func (SpReg) IsArg() {}
func (s SpReg) String() string {
	return fmt.Sprintf("SpReg(%d)", int(s))
}

// PCRel is a PC-relative offset, used only in branch instructions.
type PCRel int32

func (PCRel) IsArg() {}
func (r PCRel) String() string {
	return fmt.Sprintf("PC%+#x", int32(r))
}

// A Label is a code (text) address, used only in absolute branch instructions.
type Label uint32

func (Label) IsArg() {}
func (l Label) String() string {
	return fmt.Sprintf("%#x", uint32(l))
}

// Imm represents an immediate number.
type Imm int32

func (Imm) IsArg() {}
func (i Imm) String() string {
	return fmt.Sprintf("%d", int32(i))
}

// Offset represents a memory offset immediate.
type Offset int32

func (Offset) IsArg() {}
func (o Offset) String() string {
	return fmt.Sprintf("%+d", int32(o))
}
