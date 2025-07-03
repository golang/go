// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loong64asm

import (
	"fmt"
	"strings"
)

// An Inst is a single instruction.
type Inst struct {
	Op   Op     // Opcode mnemonic
	Enc  uint32 // Raw encoding bits.
	Args Args   // Instruction arguments, in Loong64 manual order.
}

func (i Inst) String() string {
	var op string = i.Op.String()
	var args []string

	for _, arg := range i.Args {
		if arg == nil {
			break
		}
		args = append(args, arg.String())
	}

	switch i.Op {
	case OR:
		if i.Args[2].(Reg) == R0 {
			op = "move"
			args = args[0:2]
		}

	case ANDI:
		if i.Args[0].(Reg) == R0 && i.Args[1].(Reg) == R0 {
			return "nop"
		}

	case JIRL:
		if i.Args[0].(Reg) == R0 && i.Args[1].(Reg) == R1 && i.Args[2].(OffsetSimm).Imm == 0 {
			return "ret"
		} else if i.Args[0].(Reg) == R0 && i.Args[2].(OffsetSimm).Imm == 0 {
			return "jr " + args[1]
		}

	case BLT:
		if i.Args[0].(Reg) == R0 {
			op = "bgtz"
			args = args[1:]
		} else if i.Args[1].(Reg) == R0 {
			op = "bltz"
			args = append(args[:1], args[2:]...)
		}

	case BGE:
		if i.Args[0].(Reg) == R0 {
			op = "blez"
			args = args[1:]
		} else if i.Args[1].(Reg) == R0 {
			op = "bgez"
			args = append(args[:1], args[2:]...)
		}
	}

	if len(args) == 0 {
		return op
	} else {
		return op + " " + strings.Join(args, ", ")
	}
}

// An Op is an Loong64 opcode.
type Op uint16

// NOTE: The actual Op values are defined in tables.go.
// They are chosen to simplify instruction decoding and
// are not a dense packing from 0 to N, although the
// density is high, probably at least 90%.
func (op Op) String() string {
	if (op >= Op(len(opstr))) || (opstr[op] == "") {
		return fmt.Sprintf("Op(%d)", int(op))
	}

	return opstr[op]
}

// An Args holds the instruction arguments.
// If an instruction has fewer than 5 arguments,
// the final elements in the array are nil.
type Args [5]Arg

// An Arg is a single instruction argument
type Arg interface {
	String() string
}

// A Reg is a single register.
// The zero value denotes R0, not the absence of a register.
type Reg uint16

const (
	// General-purpose register
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

	// Float point register
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
)

func (r Reg) String() string {
	switch {
	case r == R0:
		return "$zero"

	case r == R1:
		return "$ra"

	case r == R2:
		return "$tp"

	case r == R3:
		return "$sp"

	case (r >= R4) && (r <= R11):
		return fmt.Sprintf("$a%d", int(r-R4))

	case (r >= R12) && (r <= R20):
		return fmt.Sprintf("$t%d", int(r-R12))

	case r == R21:
		return "$r21"

	case r == R22:
		return "$fp"

	case (r >= R23) && (r <= R31):
		return fmt.Sprintf("$s%d", int(r-R23))

	case (r >= F0) && (r <= F7):
		return fmt.Sprintf("$fa%d", int(r-F0))

	case (r >= F8) && (r <= F23):
		return fmt.Sprintf("$ft%d", int(r-F8))

	case (r >= F24) && (r <= F31):
		return fmt.Sprintf("$fs%d", int(r-F24))

	default:
		return fmt.Sprintf("Unknown(%d)", int(r))
	}
}

// float control status register
type Fcsr uint8

const (
	FCSR0 Fcsr = iota
	FCSR1
	FCSR2
	FCSR3
)

func (f Fcsr) String() string {
	return fmt.Sprintf("$fcsr%d", uint8(f))
}

// float condition flags register
type Fcc uint8

const (
	FCC0 Fcc = iota
	FCC1
	FCC2
	FCC3
	FCC4
	FCC5
	FCC6
	FCC7
)

func (f Fcc) String() string {
	return fmt.Sprintf("$fcc%d", uint8(f))
}

// An Imm is an integer constant.
type Uimm struct {
	Imm     uint32
	Decimal bool
}

func (i Uimm) String() string {
	if i.Decimal == true {
		return fmt.Sprintf("%d", i.Imm)
	} else {
		return fmt.Sprintf("%#x", i.Imm)
	}
}

type Simm16 struct {
	Imm   int16
	Width uint8
}

func (si Simm16) String() string {
	return fmt.Sprintf("%d", int32(si.Imm))
}

type Simm32 struct {
	Imm   int32
	Width uint8
}

func (si Simm32) String() string {
	return fmt.Sprintf("%d", int32(si.Imm))
}

type OffsetSimm struct {
	Imm   int32
	Width uint8
}

func (o OffsetSimm) String() string {
	return fmt.Sprintf("%d", int32(o.Imm))
}

type SaSimm int16

func (s SaSimm) String() string {
	return fmt.Sprintf("%#x", int(s))
}

type CodeSimm int16

func (c CodeSimm) String() string {
	return fmt.Sprintf("%#x", int(c))
}
