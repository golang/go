// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package x86asm implements decoding of x86 machine code.
package x86asm

import (
	"bytes"
	"fmt"
)

// An Inst is a single instruction.
type Inst struct {
	Prefix   Prefixes // Prefixes applied to the instruction.
	Op       Op       // Opcode mnemonic
	Opcode   uint32   // Encoded opcode bits, left aligned (first byte is Opcode>>24, etc)
	Args     Args     // Instruction arguments, in Intel order
	Mode     int      // processor mode in bits: 16, 32, or 64
	AddrSize int      // address size in bits: 16, 32, or 64
	DataSize int      // operand size in bits: 16, 32, or 64
	MemBytes int      // size of memory argument in bytes: 1, 2, 4, 8, 16, and so on.
	Len      int      // length of encoded instruction in bytes
	PCRel    int      // length of PC-relative address in instruction encoding
	PCRelOff int      // index of start of PC-relative address in instruction encoding
}

// Prefixes is an array of prefixes associated with a single instruction.
// The prefixes are listed in the same order as found in the instruction:
// each prefix byte corresponds to one slot in the array. The first zero
// in the array marks the end of the prefixes.
type Prefixes [14]Prefix

// A Prefix represents an Intel instruction prefix.
// The low 8 bits are the actual prefix byte encoding,
// and the top 8 bits contain distinguishing bits and metadata.
type Prefix uint16

const (
	// Metadata about the role of a prefix in an instruction.
	PrefixImplicit Prefix = 0x8000 // prefix is implied by instruction text
	PrefixIgnored  Prefix = 0x4000 // prefix is ignored: either irrelevant or overridden by a later prefix
	PrefixInvalid  Prefix = 0x2000 // prefix makes entire instruction invalid (bad LOCK)

	// Memory segment overrides.
	PrefixES Prefix = 0x26 // ES segment override
	PrefixCS Prefix = 0x2E // CS segment override
	PrefixSS Prefix = 0x36 // SS segment override
	PrefixDS Prefix = 0x3E // DS segment override
	PrefixFS Prefix = 0x64 // FS segment override
	PrefixGS Prefix = 0x65 // GS segment override

	// Branch prediction.
	PrefixPN Prefix = 0x12E // predict not taken (conditional branch only)
	PrefixPT Prefix = 0x13E // predict taken (conditional branch only)

	// Size attributes.
	PrefixDataSize Prefix = 0x66 // operand size override
	PrefixData16   Prefix = 0x166
	PrefixData32   Prefix = 0x266
	PrefixAddrSize Prefix = 0x67 // address size override
	PrefixAddr16   Prefix = 0x167
	PrefixAddr32   Prefix = 0x267

	// One of a kind.
	PrefixLOCK     Prefix = 0xF0 // lock
	PrefixREPN     Prefix = 0xF2 // repeat not zero
	PrefixXACQUIRE Prefix = 0x1F2
	PrefixBND      Prefix = 0x2F2
	PrefixREP      Prefix = 0xF3 // repeat
	PrefixXRELEASE Prefix = 0x1F3

	// The REX prefixes must be in the range [PrefixREX, PrefixREX+0x10).
	// the other bits are set or not according to the intended use.
	PrefixREX       Prefix = 0x40 // REX 64-bit extension prefix
	PrefixREXW      Prefix = 0x08 // extension bit W (64-bit instruction width)
	PrefixREXR      Prefix = 0x04 // extension bit R (r field in modrm)
	PrefixREXX      Prefix = 0x02 // extension bit X (index field in sib)
	PrefixREXB      Prefix = 0x01 // extension bit B (r/m field in modrm or base field in sib)
	PrefixVEX2Bytes Prefix = 0xC5 // Short form of vex prefix
	PrefixVEX3Bytes Prefix = 0xC4 // Long form of vex prefix
)

// IsREX reports whether p is a REX prefix byte.
func (p Prefix) IsREX() bool {
	return p&0xF0 == PrefixREX
}

func (p Prefix) IsVEX() bool {
	return p&0xFF == PrefixVEX2Bytes || p&0xFF == PrefixVEX3Bytes
}

func (p Prefix) String() string {
	p &^= PrefixImplicit | PrefixIgnored | PrefixInvalid
	if s := prefixNames[p]; s != "" {
		return s
	}

	if p.IsREX() {
		s := "REX."
		if p&PrefixREXW != 0 {
			s += "W"
		}
		if p&PrefixREXR != 0 {
			s += "R"
		}
		if p&PrefixREXX != 0 {
			s += "X"
		}
		if p&PrefixREXB != 0 {
			s += "B"
		}
		return s
	}

	return fmt.Sprintf("Prefix(%#x)", int(p))
}

// An Op is an x86 opcode.
type Op uint32

func (op Op) String() string {
	i := int(op)
	if i < 0 || i >= len(opNames) || opNames[i] == "" {
		return fmt.Sprintf("Op(%d)", i)
	}
	return opNames[i]
}

// An Args holds the instruction arguments.
// If an instruction has fewer than 4 arguments,
// the final elements in the array are nil.
type Args [4]Arg

// An Arg is a single instruction argument,
// one of these types: Reg, Mem, Imm, Rel.
type Arg interface {
	String() string
	isArg()
}

// Note that the implements of Arg that follow are all sized
// so that on a 64-bit machine the data can be inlined in
// the interface value instead of requiring an allocation.

// A Reg is a single register.
// The zero Reg value has no name but indicates ``no register.''
type Reg uint8

const (
	_ Reg = iota

	// 8-bit
	AL
	CL
	DL
	BL
	AH
	CH
	DH
	BH
	SPB
	BPB
	SIB
	DIB
	R8B
	R9B
	R10B
	R11B
	R12B
	R13B
	R14B
	R15B

	// 16-bit
	AX
	CX
	DX
	BX
	SP
	BP
	SI
	DI
	R8W
	R9W
	R10W
	R11W
	R12W
	R13W
	R14W
	R15W

	// 32-bit
	EAX
	ECX
	EDX
	EBX
	ESP
	EBP
	ESI
	EDI
	R8L
	R9L
	R10L
	R11L
	R12L
	R13L
	R14L
	R15L

	// 64-bit
	RAX
	RCX
	RDX
	RBX
	RSP
	RBP
	RSI
	RDI
	R8
	R9
	R10
	R11
	R12
	R13
	R14
	R15

	// Instruction pointer.
	IP  // 16-bit
	EIP // 32-bit
	RIP // 64-bit

	// 387 floating point registers.
	F0
	F1
	F2
	F3
	F4
	F5
	F6
	F7

	// MMX registers.
	M0
	M1
	M2
	M3
	M4
	M5
	M6
	M7

	// XMM registers.
	X0
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

	// Segment registers.
	ES
	CS
	SS
	DS
	FS
	GS

	// System registers.
	GDTR
	IDTR
	LDTR
	MSW
	TASK

	// Control registers.
	CR0
	CR1
	CR2
	CR3
	CR4
	CR5
	CR6
	CR7
	CR8
	CR9
	CR10
	CR11
	CR12
	CR13
	CR14
	CR15

	// Debug registers.
	DR0
	DR1
	DR2
	DR3
	DR4
	DR5
	DR6
	DR7
	DR8
	DR9
	DR10
	DR11
	DR12
	DR13
	DR14
	DR15

	// Task registers.
	TR0
	TR1
	TR2
	TR3
	TR4
	TR5
	TR6
	TR7
)

const regMax = TR7

func (Reg) isArg() {}

func (r Reg) String() string {
	i := int(r)
	if i < 0 || i >= len(regNames) || regNames[i] == "" {
		return fmt.Sprintf("Reg(%d)", i)
	}
	return regNames[i]
}

// A Mem is a memory reference.
// The general form is Segment:[Base+Scale*Index+Disp].
type Mem struct {
	Segment Reg
	Base    Reg
	Scale   uint8
	Index   Reg
	Disp    int64
}

func (Mem) isArg() {}

func (m Mem) String() string {
	var base, plus, scale, index, disp string

	if m.Base != 0 {
		base = m.Base.String()
	}
	if m.Scale != 0 {
		if m.Base != 0 {
			plus = "+"
		}
		if m.Scale > 1 {
			scale = fmt.Sprintf("%d*", m.Scale)
		}
		index = m.Index.String()
	}
	if m.Disp != 0 || m.Base == 0 && m.Scale == 0 {
		disp = fmt.Sprintf("%+#x", m.Disp)
	}
	return "[" + base + plus + scale + index + disp + "]"
}

// A Rel is an offset relative to the current instruction pointer.
type Rel int32

func (Rel) isArg() {}

func (r Rel) String() string {
	return fmt.Sprintf(".%+d", r)
}

// An Imm is an integer constant.
type Imm int64

func (Imm) isArg() {}

func (i Imm) String() string {
	return fmt.Sprintf("%#x", int64(i))
}

func (i Inst) String() string {
	var buf bytes.Buffer
	for _, p := range i.Prefix {
		if p == 0 {
			break
		}
		if p&PrefixImplicit != 0 {
			continue
		}
		fmt.Fprintf(&buf, "%v ", p)
	}
	fmt.Fprintf(&buf, "%v", i.Op)
	sep := " "
	for _, v := range i.Args {
		if v == nil {
			break
		}
		fmt.Fprintf(&buf, "%s%v", sep, v)
		sep = ", "
	}
	return buf.String()
}

func isReg(a Arg) bool {
	_, ok := a.(Reg)
	return ok
}

func isSegReg(a Arg) bool {
	r, ok := a.(Reg)
	return ok && ES <= r && r <= GS
}

func isMem(a Arg) bool {
	_, ok := a.(Mem)
	return ok
}

func isImm(a Arg) bool {
	_, ok := a.(Imm)
	return ok
}

func regBytes(a Arg) int {
	r, ok := a.(Reg)
	if !ok {
		return 0
	}
	if AL <= r && r <= R15B {
		return 1
	}
	if AX <= r && r <= R15W {
		return 2
	}
	if EAX <= r && r <= R15L {
		return 4
	}
	if RAX <= r && r <= R15 {
		return 8
	}
	return 0
}

func isSegment(p Prefix) bool {
	switch p {
	case PrefixCS, PrefixDS, PrefixES, PrefixFS, PrefixGS, PrefixSS:
		return true
	}
	return false
}

// The Op definitions and string list are in tables.go.

var prefixNames = map[Prefix]string{
	PrefixCS:       "CS",
	PrefixDS:       "DS",
	PrefixES:       "ES",
	PrefixFS:       "FS",
	PrefixGS:       "GS",
	PrefixSS:       "SS",
	PrefixLOCK:     "LOCK",
	PrefixREP:      "REP",
	PrefixREPN:     "REPN",
	PrefixAddrSize: "ADDRSIZE",
	PrefixDataSize: "DATASIZE",
	PrefixAddr16:   "ADDR16",
	PrefixData16:   "DATA16",
	PrefixAddr32:   "ADDR32",
	PrefixData32:   "DATA32",
	PrefixBND:      "BND",
	PrefixXACQUIRE: "XACQUIRE",
	PrefixXRELEASE: "XRELEASE",
	PrefixREX:      "REX",
	PrefixPT:       "PT",
	PrefixPN:       "PN",
}

var regNames = [...]string{
	AL:   "AL",
	CL:   "CL",
	BL:   "BL",
	DL:   "DL",
	AH:   "AH",
	CH:   "CH",
	BH:   "BH",
	DH:   "DH",
	SPB:  "SPB",
	BPB:  "BPB",
	SIB:  "SIB",
	DIB:  "DIB",
	R8B:  "R8B",
	R9B:  "R9B",
	R10B: "R10B",
	R11B: "R11B",
	R12B: "R12B",
	R13B: "R13B",
	R14B: "R14B",
	R15B: "R15B",
	AX:   "AX",
	CX:   "CX",
	BX:   "BX",
	DX:   "DX",
	SP:   "SP",
	BP:   "BP",
	SI:   "SI",
	DI:   "DI",
	R8W:  "R8W",
	R9W:  "R9W",
	R10W: "R10W",
	R11W: "R11W",
	R12W: "R12W",
	R13W: "R13W",
	R14W: "R14W",
	R15W: "R15W",
	EAX:  "EAX",
	ECX:  "ECX",
	EDX:  "EDX",
	EBX:  "EBX",
	ESP:  "ESP",
	EBP:  "EBP",
	ESI:  "ESI",
	EDI:  "EDI",
	R8L:  "R8L",
	R9L:  "R9L",
	R10L: "R10L",
	R11L: "R11L",
	R12L: "R12L",
	R13L: "R13L",
	R14L: "R14L",
	R15L: "R15L",
	RAX:  "RAX",
	RCX:  "RCX",
	RDX:  "RDX",
	RBX:  "RBX",
	RSP:  "RSP",
	RBP:  "RBP",
	RSI:  "RSI",
	RDI:  "RDI",
	R8:   "R8",
	R9:   "R9",
	R10:  "R10",
	R11:  "R11",
	R12:  "R12",
	R13:  "R13",
	R14:  "R14",
	R15:  "R15",
	IP:   "IP",
	EIP:  "EIP",
	RIP:  "RIP",
	F0:   "F0",
	F1:   "F1",
	F2:   "F2",
	F3:   "F3",
	F4:   "F4",
	F5:   "F5",
	F6:   "F6",
	F7:   "F7",
	M0:   "M0",
	M1:   "M1",
	M2:   "M2",
	M3:   "M3",
	M4:   "M4",
	M5:   "M5",
	M6:   "M6",
	M7:   "M7",
	X0:   "X0",
	X1:   "X1",
	X2:   "X2",
	X3:   "X3",
	X4:   "X4",
	X5:   "X5",
	X6:   "X6",
	X7:   "X7",
	X8:   "X8",
	X9:   "X9",
	X10:  "X10",
	X11:  "X11",
	X12:  "X12",
	X13:  "X13",
	X14:  "X14",
	X15:  "X15",
	CS:   "CS",
	SS:   "SS",
	DS:   "DS",
	ES:   "ES",
	FS:   "FS",
	GS:   "GS",
	GDTR: "GDTR",
	IDTR: "IDTR",
	LDTR: "LDTR",
	MSW:  "MSW",
	TASK: "TASK",
	CR0:  "CR0",
	CR1:  "CR1",
	CR2:  "CR2",
	CR3:  "CR3",
	CR4:  "CR4",
	CR5:  "CR5",
	CR6:  "CR6",
	CR7:  "CR7",
	CR8:  "CR8",
	CR9:  "CR9",
	CR10: "CR10",
	CR11: "CR11",
	CR12: "CR12",
	CR13: "CR13",
	CR14: "CR14",
	CR15: "CR15",
	DR0:  "DR0",
	DR1:  "DR1",
	DR2:  "DR2",
	DR3:  "DR3",
	DR4:  "DR4",
	DR5:  "DR5",
	DR6:  "DR6",
	DR7:  "DR7",
	DR8:  "DR8",
	DR9:  "DR9",
	DR10: "DR10",
	DR11: "DR11",
	DR12: "DR12",
	DR13: "DR13",
	DR14: "DR14",
	DR15: "DR15",
	TR0:  "TR0",
	TR1:  "TR1",
	TR2:  "TR2",
	TR3:  "TR3",
	TR4:  "TR4",
	TR5:  "TR5",
	TR6:  "TR6",
	TR7:  "TR7",
}
