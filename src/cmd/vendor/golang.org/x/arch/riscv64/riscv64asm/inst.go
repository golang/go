// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package riscv64asm

import (
	"fmt"
	"strings"
)

// An Op is a RISC-V opcode.
type Op uint16

// NOTE: The actual Op values are defined in tables.go.
func (op Op) String() string {
	if op >= Op(len(opstr)) || opstr[op] == "" {
		return fmt.Sprintf("Op(%d)", op)
	}

	return opstr[op]
}

// An Arg is a single instruction argument.
type Arg interface {
	String() string
}

// An Args holds the instruction arguments.
// If an instruction has fewer than 6 arguments,
// the final elements in the array are nil.
type Args [6]Arg

// An Inst is a single instruction.
type Inst struct {
	Op   Op     // Opcode mnemonic.
	Enc  uint32 // Raw encoding bits.
	Args Args   // Instruction arguments, in RISC-V mamual order.
	Len  int    // Length of encoded instruction in bytes
}

func (i Inst) String() string {
	var args []string
	for _, arg := range i.Args {
		if arg == nil {
			break
		}
		args = append(args, arg.String())
	}

	if len(args) == 0 {
		return i.Op.String()
	}
	return i.Op.String() + " " + strings.Join(args, ",")
}

// A Reg is a single register.
// The zero value denotes X0, not the absence of a register.
type Reg uint16

const (
	// General-purpose registers
	X0 Reg = iota
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
	X16
	X17
	X18
	X19
	X20
	X21
	X22
	X23
	X24
	X25
	X26
	X27
	X28
	X29
	X30
	X31

	// Floating point registers
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

	// Vector registers
	V0
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

func (r Reg) String() string {
	switch {
	case r >= X0 && r <= X31:
		return fmt.Sprintf("x%d", r)

	case r >= F0 && r <= F31:
		return fmt.Sprintf("f%d", r-F0)

	case r >= V0 && r <= V31:
		return fmt.Sprintf("v%d", r-V0)

	default:
		return fmt.Sprintf("Unknown(%d)", r)
	}
}

// A CSR is a single control and status register.
// Use stringer to generate CSR match table.
//
//go:generate stringer -type=CSR
type CSR uint16

const (
	// Control status register
	USTATUS        CSR = 0x0000
	FFLAGS         CSR = 0x0001
	FRM            CSR = 0x0002
	FCSR           CSR = 0x0003
	UIE            CSR = 0x0004
	UTVEC          CSR = 0x0005
	UTVT           CSR = 0x0007
	VSTART         CSR = 0x0008
	VXSAT          CSR = 0x0009
	VXRM           CSR = 0x000a
	VCSR           CSR = 0x000f
	USCRATCH       CSR = 0x0040
	UEPC           CSR = 0x0041
	UCAUSE         CSR = 0x0042
	UTVAL          CSR = 0x0043
	UIP            CSR = 0x0044
	UNXTI          CSR = 0x0045
	UINTSTATUS     CSR = 0x0046
	USCRATCHCSW    CSR = 0x0048
	USCRATCHCSWL   CSR = 0x0049
	SSTATUS        CSR = 0x0100
	SEDELEG        CSR = 0x0102
	SIDELEG        CSR = 0x0103
	SIE            CSR = 0x0104
	STVEC          CSR = 0x0105
	SCOUNTEREN     CSR = 0x0106
	STVT           CSR = 0x0107
	SSCRATCH       CSR = 0x0140
	SEPC           CSR = 0x0141
	SCAUSE         CSR = 0x0142
	STVAL          CSR = 0x0143
	SIP            CSR = 0x0144
	SNXTI          CSR = 0x0145
	SINTSTATUS     CSR = 0x0146
	SSCRATCHCSW    CSR = 0x0148
	SSCRATCHCSWL   CSR = 0x0149
	SATP           CSR = 0x0180
	VSSTATUS       CSR = 0x0200
	VSIE           CSR = 0x0204
	VSTVEC         CSR = 0x0205
	VSSCRATCH      CSR = 0x0240
	VSEPC          CSR = 0x0241
	VSCAUSE        CSR = 0x0242
	VSTVAL         CSR = 0x0243
	VSIP           CSR = 0x0244
	VSATP          CSR = 0x0280
	MSTATUS        CSR = 0x0300
	MISA           CSR = 0x0301
	MEDELEG        CSR = 0x0302
	MIDELEG        CSR = 0x0303
	MIE            CSR = 0x0304
	MTVEC          CSR = 0x0305
	MCOUNTEREN     CSR = 0x0306
	MTVT           CSR = 0x0307
	MSTATUSH       CSR = 0x0310
	MCOUNTINHIBIT  CSR = 0x0320
	MHPMEVENT3     CSR = 0x0323
	MHPMEVENT4     CSR = 0x0324
	MHPMEVENT5     CSR = 0x0325
	MHPMEVENT6     CSR = 0x0326
	MHPMEVENT7     CSR = 0x0327
	MHPMEVENT8     CSR = 0x0328
	MHPMEVENT9     CSR = 0x0329
	MHPMEVENT10    CSR = 0x032a
	MHPMEVENT11    CSR = 0x032b
	MHPMEVENT12    CSR = 0x032c
	MHPMEVENT13    CSR = 0x032d
	MHPMEVENT14    CSR = 0x032e
	MHPMEVENT15    CSR = 0x032f
	MHPMEVENT16    CSR = 0x0330
	MHPMEVENT17    CSR = 0x0331
	MHPMEVENT18    CSR = 0x0332
	MHPMEVENT19    CSR = 0x0333
	MHPMEVENT20    CSR = 0x0334
	MHPMEVENT21    CSR = 0x0335
	MHPMEVENT22    CSR = 0x0336
	MHPMEVENT23    CSR = 0x0337
	MHPMEVENT24    CSR = 0x0338
	MHPMEVENT25    CSR = 0x0339
	MHPMEVENT26    CSR = 0x033a
	MHPMEVENT27    CSR = 0x033b
	MHPMEVENT28    CSR = 0x033c
	MHPMEVENT29    CSR = 0x033d
	MHPMEVENT30    CSR = 0x033e
	MHPMEVENT31    CSR = 0x033f
	MSCRATCH       CSR = 0x0340
	MEPC           CSR = 0x0341
	MCAUSE         CSR = 0x0342
	MTVAL          CSR = 0x0343
	MIP            CSR = 0x0344
	MNXTI          CSR = 0x0345
	MINTSTATUS     CSR = 0x0346
	MSCRATCHCSW    CSR = 0x0348
	MSCRATCHCSWL   CSR = 0x0349
	MTINST         CSR = 0x034a
	MTVAL2         CSR = 0x034b
	PMPCFG0        CSR = 0x03a0
	PMPCFG1        CSR = 0x03a1
	PMPCFG2        CSR = 0x03a2
	PMPCFG3        CSR = 0x03a3
	PMPADDR0       CSR = 0x03b0
	PMPADDR1       CSR = 0x03b1
	PMPADDR2       CSR = 0x03b2
	PMPADDR3       CSR = 0x03b3
	PMPADDR4       CSR = 0x03b4
	PMPADDR5       CSR = 0x03b5
	PMPADDR6       CSR = 0x03b6
	PMPADDR7       CSR = 0x03b7
	PMPADDR8       CSR = 0x03b8
	PMPADDR9       CSR = 0x03b9
	PMPADDR10      CSR = 0x03ba
	PMPADDR11      CSR = 0x03bb
	PMPADDR12      CSR = 0x03bc
	PMPADDR13      CSR = 0x03bd
	PMPADDR14      CSR = 0x03be
	PMPADDR15      CSR = 0x03bf
	HSTATUS        CSR = 0x0600
	HEDELEG        CSR = 0x0602
	HIDELEG        CSR = 0x0603
	HIE            CSR = 0x0604
	HTIMEDELTA     CSR = 0x0605
	HCOUNTEREN     CSR = 0x0606
	HGEIE          CSR = 0x0607
	HTIMEDELTAH    CSR = 0x0615
	HTVAL          CSR = 0x0643
	HIP            CSR = 0x0644
	HVIP           CSR = 0x0645
	HTINST         CSR = 0x064a
	HGATP          CSR = 0x0680
	TSELECT        CSR = 0x07a0
	TDATA1         CSR = 0x07a1
	TDATA2         CSR = 0x07a2
	TDATA3         CSR = 0x07a3
	TINFO          CSR = 0x07a4
	TCONTROL       CSR = 0x07a5
	MCONTEXT       CSR = 0x07a8
	MNOISE         CSR = 0x07a9
	SCONTEXT       CSR = 0x07aa
	DCSR           CSR = 0x07b0
	DPC            CSR = 0x07b1
	DSCRATCH0      CSR = 0x07b2
	DSCRATCH1      CSR = 0x07b3
	MCYCLE         CSR = 0x0b00
	MINSTRET       CSR = 0x0b02
	MHPMCOUNTER3   CSR = 0x0b03
	MHPMCOUNTER4   CSR = 0x0b04
	MHPMCOUNTER5   CSR = 0x0b05
	MHPMCOUNTER6   CSR = 0x0b06
	MHPMCOUNTER7   CSR = 0x0b07
	MHPMCOUNTER8   CSR = 0x0b08
	MHPMCOUNTER9   CSR = 0x0b09
	MHPMCOUNTER10  CSR = 0x0b0a
	MHPMCOUNTER11  CSR = 0x0b0b
	MHPMCOUNTER12  CSR = 0x0b0c
	MHPMCOUNTER13  CSR = 0x0b0d
	MHPMCOUNTER14  CSR = 0x0b0e
	MHPMCOUNTER15  CSR = 0x0b0f
	MHPMCOUNTER16  CSR = 0x0b10
	MHPMCOUNTER17  CSR = 0x0b11
	MHPMCOUNTER18  CSR = 0x0b12
	MHPMCOUNTER19  CSR = 0x0b13
	MHPMCOUNTER20  CSR = 0x0b14
	MHPMCOUNTER21  CSR = 0x0b15
	MHPMCOUNTER22  CSR = 0x0b16
	MHPMCOUNTER23  CSR = 0x0b17
	MHPMCOUNTER24  CSR = 0x0b18
	MHPMCOUNTER25  CSR = 0x0b19
	MHPMCOUNTER26  CSR = 0x0b1a
	MHPMCOUNTER27  CSR = 0x0b1b
	MHPMCOUNTER28  CSR = 0x0b1c
	MHPMCOUNTER29  CSR = 0x0b1d
	MHPMCOUNTER30  CSR = 0x0b1e
	MHPMCOUNTER31  CSR = 0x0b1f
	MCYCLEH        CSR = 0x0b80
	MINSTRETH      CSR = 0x0b82
	MHPMCOUNTER3H  CSR = 0x0b83
	MHPMCOUNTER4H  CSR = 0x0b84
	MHPMCOUNTER5H  CSR = 0x0b85
	MHPMCOUNTER6H  CSR = 0x0b86
	MHPMCOUNTER7H  CSR = 0x0b87
	MHPMCOUNTER8H  CSR = 0x0b88
	MHPMCOUNTER9H  CSR = 0x0b89
	MHPMCOUNTER10H CSR = 0x0b8a
	MHPMCOUNTER11H CSR = 0x0b8b
	MHPMCOUNTER12H CSR = 0x0b8c
	MHPMCOUNTER13H CSR = 0x0b8d
	MHPMCOUNTER14H CSR = 0x0b8e
	MHPMCOUNTER15H CSR = 0x0b8f
	MHPMCOUNTER16H CSR = 0x0b90
	MHPMCOUNTER17H CSR = 0x0b91
	MHPMCOUNTER18H CSR = 0x0b92
	MHPMCOUNTER19H CSR = 0x0b93
	MHPMCOUNTER20H CSR = 0x0b94
	MHPMCOUNTER21H CSR = 0x0b95
	MHPMCOUNTER22H CSR = 0x0b96
	MHPMCOUNTER23H CSR = 0x0b97
	MHPMCOUNTER24H CSR = 0x0b98
	MHPMCOUNTER25H CSR = 0x0b99
	MHPMCOUNTER26H CSR = 0x0b9a
	MHPMCOUNTER27H CSR = 0x0b9b
	MHPMCOUNTER28H CSR = 0x0b9c
	MHPMCOUNTER29H CSR = 0x0b9d
	MHPMCOUNTER30H CSR = 0x0b9e
	MHPMCOUNTER31H CSR = 0x0b9f
	CYCLE          CSR = 0x0c00
	TIME           CSR = 0x0c01
	INSTRET        CSR = 0x0c02
	HPMCOUNTER3    CSR = 0x0c03
	HPMCOUNTER4    CSR = 0x0c04
	HPMCOUNTER5    CSR = 0x0c05
	HPMCOUNTER6    CSR = 0x0c06
	HPMCOUNTER7    CSR = 0x0c07
	HPMCOUNTER8    CSR = 0x0c08
	HPMCOUNTER9    CSR = 0x0c09
	HPMCOUNTER10   CSR = 0x0c0a
	HPMCOUNTER11   CSR = 0x0c0b
	HPMCOUNTER12   CSR = 0x0c0c
	HPMCOUNTER13   CSR = 0x0c0d
	HPMCOUNTER14   CSR = 0x0c0e
	HPMCOUNTER15   CSR = 0x0c0f
	HPMCOUNTER16   CSR = 0x0c10
	HPMCOUNTER17   CSR = 0x0c11
	HPMCOUNTER18   CSR = 0x0c12
	HPMCOUNTER19   CSR = 0x0c13
	HPMCOUNTER20   CSR = 0x0c14
	HPMCOUNTER21   CSR = 0x0c15
	HPMCOUNTER22   CSR = 0x0c16
	HPMCOUNTER23   CSR = 0x0c17
	HPMCOUNTER24   CSR = 0x0c18
	HPMCOUNTER25   CSR = 0x0c19
	HPMCOUNTER26   CSR = 0x0c1a
	HPMCOUNTER27   CSR = 0x0c1b
	HPMCOUNTER28   CSR = 0x0c1c
	HPMCOUNTER29   CSR = 0x0c1d
	HPMCOUNTER30   CSR = 0x0c1e
	HPMCOUNTER31   CSR = 0x0c1f
	VL             CSR = 0x0c20
	VTYPE          CSR = 0x0c21
	VLENB          CSR = 0x0c22
	CYCLEH         CSR = 0x0c80
	TIMEH          CSR = 0x0c81
	INSTRETH       CSR = 0x0c82
	HPMCOUNTER3H   CSR = 0x0c83
	HPMCOUNTER4H   CSR = 0x0c84
	HPMCOUNTER5H   CSR = 0x0c85
	HPMCOUNTER6H   CSR = 0x0c86
	HPMCOUNTER7H   CSR = 0x0c87
	HPMCOUNTER8H   CSR = 0x0c88
	HPMCOUNTER9H   CSR = 0x0c89
	HPMCOUNTER10H  CSR = 0x0c8a
	HPMCOUNTER11H  CSR = 0x0c8b
	HPMCOUNTER12H  CSR = 0x0c8c
	HPMCOUNTER13H  CSR = 0x0c8d
	HPMCOUNTER14H  CSR = 0x0c8e
	HPMCOUNTER15H  CSR = 0x0c8f
	HPMCOUNTER16H  CSR = 0x0c90
	HPMCOUNTER17H  CSR = 0x0c91
	HPMCOUNTER18H  CSR = 0x0c92
	HPMCOUNTER19H  CSR = 0x0c93
	HPMCOUNTER20H  CSR = 0x0c94
	HPMCOUNTER21H  CSR = 0x0c95
	HPMCOUNTER22H  CSR = 0x0c96
	HPMCOUNTER23H  CSR = 0x0c97
	HPMCOUNTER24H  CSR = 0x0c98
	HPMCOUNTER25H  CSR = 0x0c99
	HPMCOUNTER26H  CSR = 0x0c9a
	HPMCOUNTER27H  CSR = 0x0c9b
	HPMCOUNTER28H  CSR = 0x0c9c
	HPMCOUNTER29H  CSR = 0x0c9d
	HPMCOUNTER30H  CSR = 0x0c9e
	HPMCOUNTER31H  CSR = 0x0c9f
	HGEIP          CSR = 0x0e12
	MVENDORID      CSR = 0x0f11
	MARCHID        CSR = 0x0f12
	MIMPID         CSR = 0x0f13
	MHARTID        CSR = 0x0f14
	MENTROPY       CSR = 0x0f15
)

// An Uimm is an unsigned immediate number
type Uimm struct {
	Imm     uint32 // 32-bit unsigned integer
	Decimal bool   // Print format of the immediate, either decimal or hexadecimal
}

func (ui Uimm) String() string {
	if ui.Decimal {
		return fmt.Sprintf("%d", ui.Imm)
	}
	return fmt.Sprintf("%#x", ui.Imm)
}

// A Simm is a signed immediate number
type Simm struct {
	Imm     int32 // 32-bit signed integer
	Decimal bool  // Print format of the immediate, either decimal or hexadecimal
	Width   uint8 // Actual width of the Simm
}

func (si Simm) String() string {
	if si.Decimal {
		return fmt.Sprintf("%d", si.Imm)
	}
	return fmt.Sprintf("%#x", si.Imm)
}

// A RegPtr is an address register with no offset
type RegPtr struct {
	reg Reg // Avoid promoted String method
}

func (regPtr RegPtr) String() string {
	return fmt.Sprintf("(%s)", regPtr.reg)
}

// A RegOffset is a register with offset value
type RegOffset struct {
	OfsReg Reg
	Ofs    Simm
}

func (regofs RegOffset) String() string {
	return fmt.Sprintf("%s(%s)", regofs.Ofs, regofs.OfsReg)
}

// A MemOrder is a memory order hint in fence instruction
type MemOrder uint8

func (memOrder MemOrder) String() string {
	var str string
	if memOrder<<7>>7 == 1 {
		str += "i"
	}
	if memOrder>>1<<7>>7 == 1 {
		str += "o"
	}
	if memOrder>>2<<7>>7 == 1 {
		str += "r"
	}
	if memOrder>>3<<7>>7 == 1 {
		str += "w"
	}
	return str
}

// A VType represents the vtype field of VSETIVLI and VSETVLI instructions
type VType uint32

var vlmulName = []string{"M1", "M2", "M4", "M8", "", "MF8", "MF4", "MF2"}
var vsewName = []string{"E8", "E16", "E32", "E64", "", "", "", ""}
var vtaName = []string{"TU", "TA"}
var vmaName = []string{"MU", "MA"}

func (vtype VType) String() string {

	vlmul := vtype & 0x7
	vsew := (vtype >> 3) & 0x7
	vta := (vtype >> 6) & 0x1
	vma := (vtype >> 7) & 0x1

	return fmt.Sprintf("%s, %s, %s, %s", vsewName[vsew], vlmulName[vlmul], vtaName[vta], vmaName[vma])
}
