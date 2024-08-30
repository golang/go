// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64asm

import (
	"fmt"
	"strings"
)

// An Op is an ARM64 opcode.
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
	Args Args   // Instruction arguments, in ARM manual order.
}

func (i Inst) String() string {
	var args []string
	for _, arg := range i.Args {
		if arg == nil {
			break
		}
		args = append(args, arg.String())
	}
	return i.Op.String() + " " + strings.Join(args, ", ")
}

// An Args holds the instruction arguments.
// If an instruction has fewer than 5 arguments,
// the final elements in the array are nil.
type Args [5]Arg

// An Arg is a single instruction argument, one of these types:
// Reg, RegSP, ImmShift, RegExtshiftAmount, PCRel, MemImmediate,
// MemExtend, Imm, Imm64, Imm_hint, Imm_clrex, Imm_dcps, Cond,
// Imm_c, Imm_option, Imm_prfop, Pstatefield, Systemreg, Imm_fp
// RegisterWithArrangement, RegisterWithArrangementAndIndex.
type Arg interface {
	isArg()
	String() string
}

// A Reg is a single register.
// The zero value denotes W0, not the absence of a register.
type Reg uint16

const (
	W0 Reg = iota
	W1
	W2
	W3
	W4
	W5
	W6
	W7
	W8
	W9
	W10
	W11
	W12
	W13
	W14
	W15
	W16
	W17
	W18
	W19
	W20
	W21
	W22
	W23
	W24
	W25
	W26
	W27
	W28
	W29
	W30
	WZR

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
	XZR

	B0
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
	B16
	B17
	B18
	B19
	B20
	B21
	B22
	B23
	B24
	B25
	B26
	B27
	B28
	B29
	B30
	B31

	H0
	H1
	H2
	H3
	H4
	H5
	H6
	H7
	H8
	H9
	H10
	H11
	H12
	H13
	H14
	H15
	H16
	H17
	H18
	H19
	H20
	H21
	H22
	H23
	H24
	H25
	H26
	H27
	H28
	H29
	H30
	H31

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

	Q0
	Q1
	Q2
	Q3
	Q4
	Q5
	Q6
	Q7
	Q8
	Q9
	Q10
	Q11
	Q12
	Q13
	Q14
	Q15
	Q16
	Q17
	Q18
	Q19
	Q20
	Q21
	Q22
	Q23
	Q24
	Q25
	Q26
	Q27
	Q28
	Q29
	Q30
	Q31

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

	WSP = WZR // These are different registers with the same encoding.
	SP  = XZR // These are different registers with the same encoding.
)

func (Reg) isArg() {}

func (r Reg) String() string {
	switch {
	case r == WZR:
		return "WZR"
	case r == XZR:
		return "XZR"
	case W0 <= r && r <= W30:
		return fmt.Sprintf("W%d", int(r-W0))
	case X0 <= r && r <= X30:
		return fmt.Sprintf("X%d", int(r-X0))

	case B0 <= r && r <= B31:
		return fmt.Sprintf("B%d", int(r-B0))
	case H0 <= r && r <= H31:
		return fmt.Sprintf("H%d", int(r-H0))
	case S0 <= r && r <= S31:
		return fmt.Sprintf("S%d", int(r-S0))
	case D0 <= r && r <= D31:
		return fmt.Sprintf("D%d", int(r-D0))
	case Q0 <= r && r <= Q31:
		return fmt.Sprintf("Q%d", int(r-Q0))

	case V0 <= r && r <= V31:
		return fmt.Sprintf("V%d", int(r-V0))
	default:
		return fmt.Sprintf("Reg(%d)", int(r))
	}
}

// A RegSP represent a register and X31/W31 is regarded as SP/WSP.
type RegSP Reg

func (RegSP) isArg() {}

func (r RegSP) String() string {
	switch Reg(r) {
	case WSP:
		return "WSP"
	case SP:
		return "SP"
	default:
		return Reg(r).String()
	}
}

type ImmShift struct {
	imm   uint16
	shift uint8
}

func (ImmShift) isArg() {}

func (is ImmShift) String() string {
	if is.shift == 0 {
		return fmt.Sprintf("#%#x", is.imm)
	}
	if is.shift < 128 {
		return fmt.Sprintf("#%#x, LSL #%d", is.imm, is.shift)
	}
	return fmt.Sprintf("#%#x, MSL #%d", is.imm, is.shift-128)
}

type ExtShift uint8

const (
	_ ExtShift = iota
	uxtb
	uxth
	uxtw
	uxtx
	sxtb
	sxth
	sxtw
	sxtx
	lsl
	lsr
	asr
	ror
)

func (extShift ExtShift) String() string {
	switch extShift {
	case uxtb:
		return "UXTB"

	case uxth:
		return "UXTH"

	case uxtw:
		return "UXTW"

	case uxtx:
		return "UXTX"

	case sxtb:
		return "SXTB"

	case sxth:
		return "SXTH"

	case sxtw:
		return "SXTW"

	case sxtx:
		return "SXTX"

	case lsl:
		return "LSL"

	case lsr:
		return "LSR"

	case asr:
		return "ASR"

	case ror:
		return "ROR"
	}
	return ""
}

type RegExtshiftAmount struct {
	reg       Reg
	extShift  ExtShift
	amount    uint8
	show_zero bool
}

func (RegExtshiftAmount) isArg() {}

func (rea RegExtshiftAmount) String() string {
	buf := rea.reg.String()
	if rea.extShift != ExtShift(0) {
		buf += ", " + rea.extShift.String()
		if rea.amount != 0 {
			buf += fmt.Sprintf(" #%d", rea.amount)
		} else {
			if rea.show_zero == true {
				buf += fmt.Sprintf(" #%d", rea.amount)
			}
		}
	}
	return buf
}

// A PCRel describes a memory address (usually a code label)
// as a distance relative to the program counter.
type PCRel int64

func (PCRel) isArg() {}

func (r PCRel) String() string {
	return fmt.Sprintf(".%+#x", uint64(r))
}

// An AddrMode is an ARM addressing mode.
type AddrMode uint8

const (
	_             AddrMode = iota
	AddrPostIndex          // [R], X - use address R, set R = R + X
	AddrPreIndex           // [R, X]! - use address R + X, set R = R + X
	AddrOffset             // [R, X] - use address R + X
	AddrPostReg            // [Rn], Rm - - use address Rn, set Rn = Rn + Rm
)

// A MemImmediate is a memory reference made up of a base R and immediate X.
// The effective memory address is R or R+X depending on AddrMode.
type MemImmediate struct {
	Base RegSP
	Mode AddrMode
	imm  int32
}

func (MemImmediate) isArg() {}

func (m MemImmediate) String() string {
	R := m.Base.String()
	X := fmt.Sprintf("#%d", m.imm)

	switch m.Mode {
	case AddrOffset:
		if X == "#0" {
			return fmt.Sprintf("[%s]", R)
		}
		return fmt.Sprintf("[%s,%s]", R, X)
	case AddrPreIndex:
		return fmt.Sprintf("[%s,%s]!", R, X)
	case AddrPostIndex:
		return fmt.Sprintf("[%s],%s", R, X)
	case AddrPostReg:
		post := Reg(X0) + Reg(m.imm)
		postR := post.String()
		return fmt.Sprintf("[%s], %s", R, postR)
	}
	return fmt.Sprintf("unimplemented!")
}

// A MemExtend is a memory reference made up of a base R and index expression X.
// The effective memory address is R or R+X depending on Index, Extend and Amount.
type MemExtend struct {
	Base   RegSP
	Index  Reg
	Extend ExtShift
	// Amount indicates the index shift amount (but also see ShiftMustBeZero field below).
	Amount uint8
	// Refer to ARM reference manual, for byte load/store(register), the index
	// shift amount must be 0, encoded in "S" as 0 if omitted, or as 1 if present.
	// a.ShiftMustBeZero is set true indicates the index shift amount must be 0.
	// In GNU syntax, a #0 shift amount is printed if Amount is 1 but ShiftMustBeZero
	// is true; #0 is not printed if Amount is 0 and ShiftMustBeZero is true.
	// Both cases represent shift by 0 bit.
	ShiftMustBeZero bool
}

func (MemExtend) isArg() {}

func (m MemExtend) String() string {
	Rbase := m.Base.String()
	RIndex := m.Index.String()
	if m.ShiftMustBeZero {
		if m.Amount != 0 {
			return fmt.Sprintf("[%s,%s,%s #0]", Rbase, RIndex, m.Extend.String())
		} else {
			if m.Extend != lsl {
				return fmt.Sprintf("[%s,%s,%s]", Rbase, RIndex, m.Extend.String())
			} else {
				return fmt.Sprintf("[%s,%s]", Rbase, RIndex)
			}
		}
	} else {
		if m.Amount != 0 {
			return fmt.Sprintf("[%s,%s,%s #%d]", Rbase, RIndex, m.Extend.String(), m.Amount)
		} else {
			if m.Extend != lsl {
				return fmt.Sprintf("[%s,%s,%s]", Rbase, RIndex, m.Extend.String())
			} else {
				return fmt.Sprintf("[%s,%s]", Rbase, RIndex)
			}
		}
	}
}

// An Imm is an integer constant.
type Imm struct {
	Imm     uint32
	Decimal bool
}

func (Imm) isArg() {}

func (i Imm) String() string {
	if !i.Decimal {
		return fmt.Sprintf("#%#x", i.Imm)
	} else {
		return fmt.Sprintf("#%d", i.Imm)
	}
}

type Imm64 struct {
	Imm     uint64
	Decimal bool
}

func (Imm64) isArg() {}

func (i Imm64) String() string {
	if !i.Decimal {
		return fmt.Sprintf("#%#x", i.Imm)
	} else {
		return fmt.Sprintf("#%d", i.Imm)
	}
}

// An Imm_hint is an integer constant for HINT instruction.
type Imm_hint uint8

func (Imm_hint) isArg() {}

func (i Imm_hint) String() string {
	return fmt.Sprintf("#%#x", uint32(i))
}

// An Imm_clrex is an integer constant for CLREX instruction.
type Imm_clrex uint8

func (Imm_clrex) isArg() {}

func (i Imm_clrex) String() string {
	if i == 15 {
		return ""
	}
	return fmt.Sprintf("#%#x", uint32(i))
}

// An Imm_dcps is an integer constant for DCPS[123] instruction.
type Imm_dcps uint16

func (Imm_dcps) isArg() {}

func (i Imm_dcps) String() string {
	if i == 0 {
		return ""
	}
	return fmt.Sprintf("#%#x", uint32(i))
}

// Standard conditions.
type Cond struct {
	Value  uint8
	Invert bool
}

func (Cond) isArg() {}

func (c Cond) String() string {
	cond31 := c.Value >> 1
	invert := bool((c.Value & 1) == 1)
	invert = (invert != c.Invert)
	switch cond31 {
	case 0:
		if invert {
			return "NE"
		} else {
			return "EQ"
		}
	case 1:
		if invert {
			return "CC"
		} else {
			return "CS"
		}
	case 2:
		if invert {
			return "PL"
		} else {
			return "MI"
		}
	case 3:
		if invert {
			return "VC"
		} else {
			return "VS"
		}
	case 4:
		if invert {
			return "LS"
		} else {
			return "HI"
		}
	case 5:
		if invert {
			return "LT"
		} else {
			return "GE"
		}
	case 6:
		if invert {
			return "LE"
		} else {
			return "GT"
		}
	case 7:
		return "AL"
	}
	return ""
}

// An Imm_c is an integer constant for SYS/SYSL/TLBI instruction.
type Imm_c uint8

func (Imm_c) isArg() {}

func (i Imm_c) String() string {
	return fmt.Sprintf("C%d", uint8(i))
}

// An Imm_option is an integer constant for DMB/DSB/ISB instruction.
type Imm_option uint8

func (Imm_option) isArg() {}

func (i Imm_option) String() string {
	switch uint8(i) {
	case 15:
		return "SY"
	case 14:
		return "ST"
	case 13:
		return "LD"
	case 11:
		return "ISH"
	case 10:
		return "ISHST"
	case 9:
		return "ISHLD"
	case 7:
		return "NSH"
	case 6:
		return "NSHST"
	case 5:
		return "NSHLD"
	case 3:
		return "OSH"
	case 2:
		return "OSHST"
	case 1:
		return "OSHLD"
	}
	return fmt.Sprintf("#%#02x", uint8(i))
}

// An Imm_prfop is an integer constant for PRFM instruction.
type Imm_prfop uint8

func (Imm_prfop) isArg() {}

func (i Imm_prfop) String() string {
	prf_type := (i >> 3) & (1<<2 - 1)
	prf_target := (i >> 1) & (1<<2 - 1)
	prf_policy := i & 1
	var result string

	switch prf_type {
	case 0:
		result = "PLD"
	case 1:
		result = "PLI"
	case 2:
		result = "PST"
	case 3:
		return fmt.Sprintf("#%#02x", uint8(i))
	}
	switch prf_target {
	case 0:
		result += "L1"
	case 1:
		result += "L2"
	case 2:
		result += "L3"
	case 3:
		return fmt.Sprintf("#%#02x", uint8(i))
	}
	if prf_policy == 0 {
		result += "KEEP"
	} else {
		result += "STRM"
	}
	return result
}

type Pstatefield uint8

const (
	SPSel Pstatefield = iota
	DAIFSet
	DAIFClr
)

func (Pstatefield) isArg() {}

func (p Pstatefield) String() string {
	switch p {
	case SPSel:
		return "SPSel"
	case DAIFSet:
		return "DAIFSet"
	case DAIFClr:
		return "DAIFClr"
	default:
		return "unimplemented"
	}
}

type Systemreg struct {
	op0 uint8
	op1 uint8
	cn  uint8
	cm  uint8
	op2 uint8
}

func (Systemreg) isArg() {}

func (s Systemreg) String() string {
	return fmt.Sprintf("S%d_%d_C%d_C%d_%d",
		s.op0, s.op1, s.cn, s.cm, s.op2)
}

// An Imm_fp is a signed floating-point constant.
type Imm_fp struct {
	s   uint8
	exp int8
	pre uint8
}

func (Imm_fp) isArg() {}

func (i Imm_fp) String() string {
	var s, pre, numerator, denominator int16
	var result float64
	if i.s == 0 {
		s = 1
	} else {
		s = -1
	}
	pre = s * int16(16+i.pre)
	if i.exp > 0 {
		numerator = (pre << uint8(i.exp))
		denominator = 16
	} else {
		numerator = pre
		denominator = (16 << uint8(-1*i.exp))
	}
	result = float64(numerator) / float64(denominator)
	return fmt.Sprintf("#%.18e", result)
}

type Arrangement uint8

const (
	_ Arrangement = iota
	ArrangementB
	Arrangement8B
	Arrangement16B
	ArrangementH
	Arrangement4H
	Arrangement8H
	ArrangementS
	Arrangement2S
	Arrangement4S
	ArrangementD
	Arrangement1D
	Arrangement2D
	Arrangement1Q
)

func (a Arrangement) String() (result string) {
	switch a {
	case ArrangementB:
		result = ".B"
	case Arrangement8B:
		result = ".8B"
	case Arrangement16B:
		result = ".16B"
	case ArrangementH:
		result = ".H"
	case Arrangement4H:
		result = ".4H"
	case Arrangement8H:
		result = ".8H"
	case ArrangementS:
		result = ".S"
	case Arrangement2S:
		result = ".2S"
	case Arrangement4S:
		result = ".4S"
	case ArrangementD:
		result = ".D"
	case Arrangement1D:
		result = ".1D"
	case Arrangement2D:
		result = ".2D"
	case Arrangement1Q:
		result = ".1Q"
	}
	return
}

// Register with arrangement: <Vd>.<T>, { <Vt>.8B, <Vt2>.8B},
type RegisterWithArrangement struct {
	r   Reg
	a   Arrangement
	cnt uint8
}

func (RegisterWithArrangement) isArg() {}

func (r RegisterWithArrangement) String() string {
	result := r.r.String()
	result += r.a.String()
	if r.cnt > 0 {
		result = "{" + result
		if r.cnt == 2 {
			r1 := V0 + Reg((uint16(r.r)-uint16(V0)+1)&31)
			result += ", " + r1.String() + r.a.String()
		} else if r.cnt > 2 {
			if (uint16(r.cnt) + ((uint16(r.r) - uint16(V0)) & 31)) > 32 {
				for i := 1; i < int(r.cnt); i++ {
					cur := V0 + Reg((uint16(r.r)-uint16(V0)+uint16(i))&31)
					result += ", " + cur.String() + r.a.String()
				}
			} else {
				r1 := V0 + Reg((uint16(r.r)-uint16(V0)+uint16(r.cnt)-1)&31)
				result += "-" + r1.String() + r.a.String()
			}
		}
		result += "}"
	}
	return result
}

// Register with arrangement and index:
//
//	<Vm>.<Ts>[<index>],
//	{ <Vt>.B, <Vt2>.B }[<index>].
type RegisterWithArrangementAndIndex struct {
	r     Reg
	a     Arrangement
	index uint8
	cnt   uint8
}

func (RegisterWithArrangementAndIndex) isArg() {}

func (r RegisterWithArrangementAndIndex) String() string {
	result := r.r.String()
	result += r.a.String()
	if r.cnt > 0 {
		result = "{" + result
		if r.cnt == 2 {
			r1 := V0 + Reg((uint16(r.r)-uint16(V0)+1)&31)
			result += ", " + r1.String() + r.a.String()
		} else if r.cnt > 2 {
			if (uint16(r.cnt) + ((uint16(r.r) - uint16(V0)) & 31)) > 32 {
				for i := 1; i < int(r.cnt); i++ {
					cur := V0 + Reg((uint16(r.r)-uint16(V0)+uint16(i))&31)
					result += ", " + cur.String() + r.a.String()
				}
			} else {
				r1 := V0 + Reg((uint16(r.r)-uint16(V0)+uint16(r.cnt)-1)&31)
				result += "-" + r1.String() + r.a.String()
			}
		}
		result += "}"
	}
	return fmt.Sprintf("%s[%d]", result, r.index)
}

type sysOp struct {
	op          sysInstFields
	r           Reg
	hasOperand2 bool
}

func (s sysOp) isArg() {}

func (s sysOp) String() string {
	result := s.op.String()
	// If s.hasOperand2 is false, the value in the register
	// specified by s.r is ignored.
	if s.hasOperand2 {
		result += ", " + s.r.String()
	}
	return result
}

type sysInstFields struct {
	op1 uint8
	cn  uint8
	cm  uint8
	op2 uint8
}

type sysInstAttrs struct {
	typ         sys
	name        string
	hasOperand2 bool
}

func (s sysInstFields) isArg() {}

func (s sysInstFields) getAttrs() sysInstAttrs {
	attrs, ok := sysInstsAttrs[sysInstFields{s.op1, s.cn, s.cm, s.op2}]
	if !ok {
		return sysInstAttrs{typ: sys_SYS}
	}
	return attrs
}

func (s sysInstFields) String() string {
	return s.getAttrs().name
}

func (s sysInstFields) getType() sys {
	return s.getAttrs().typ
}

var sysInstsAttrs = map[sysInstFields]sysInstAttrs{
	sysInstFields{0, 8, 3, 0}:  {sys_TLBI, "VMALLE1IS", false},
	sysInstFields{0, 8, 3, 1}:  {sys_TLBI, "VAE1IS", true},
	sysInstFields{0, 8, 3, 2}:  {sys_TLBI, "ASIDE1IS", true},
	sysInstFields{0, 8, 3, 3}:  {sys_TLBI, "VAAE1IS", true},
	sysInstFields{0, 8, 3, 5}:  {sys_TLBI, "VALE1IS", true},
	sysInstFields{0, 8, 3, 7}:  {sys_TLBI, "VAALE1IS", true},
	sysInstFields{0, 8, 7, 0}:  {sys_TLBI, "VMALLE1", false},
	sysInstFields{0, 8, 7, 1}:  {sys_TLBI, "VAE1", true},
	sysInstFields{0, 8, 7, 2}:  {sys_TLBI, "ASIDE1", true},
	sysInstFields{0, 8, 7, 3}:  {sys_TLBI, "VAAE1", true},
	sysInstFields{0, 8, 7, 5}:  {sys_TLBI, "VALE1", true},
	sysInstFields{0, 8, 7, 7}:  {sys_TLBI, "VAALE1", true},
	sysInstFields{4, 8, 0, 1}:  {sys_TLBI, "IPAS2E1IS", true},
	sysInstFields{4, 8, 0, 5}:  {sys_TLBI, "IPAS2LE1IS", true},
	sysInstFields{4, 8, 3, 0}:  {sys_TLBI, "ALLE2IS", false},
	sysInstFields{4, 8, 3, 1}:  {sys_TLBI, "VAE2IS", true},
	sysInstFields{4, 8, 3, 4}:  {sys_TLBI, "ALLE1IS", false},
	sysInstFields{4, 8, 3, 5}:  {sys_TLBI, "VALE2IS", true},
	sysInstFields{4, 8, 3, 6}:  {sys_TLBI, "VMALLS12E1IS", false},
	sysInstFields{4, 8, 4, 1}:  {sys_TLBI, "IPAS2E1", true},
	sysInstFields{4, 8, 4, 5}:  {sys_TLBI, "IPAS2LE1", true},
	sysInstFields{4, 8, 7, 0}:  {sys_TLBI, "ALLE2", false},
	sysInstFields{4, 8, 7, 1}:  {sys_TLBI, "VAE2", true},
	sysInstFields{4, 8, 7, 4}:  {sys_TLBI, "ALLE1", false},
	sysInstFields{4, 8, 7, 5}:  {sys_TLBI, "VALE2", true},
	sysInstFields{4, 8, 7, 6}:  {sys_TLBI, "VMALLS12E1", false},
	sysInstFields{6, 8, 3, 0}:  {sys_TLBI, "ALLE3IS", false},
	sysInstFields{6, 8, 3, 1}:  {sys_TLBI, "VAE3IS", true},
	sysInstFields{6, 8, 3, 5}:  {sys_TLBI, "VALE3IS", true},
	sysInstFields{6, 8, 7, 0}:  {sys_TLBI, "ALLE3", false},
	sysInstFields{6, 8, 7, 1}:  {sys_TLBI, "VAE3", true},
	sysInstFields{6, 8, 7, 5}:  {sys_TLBI, "VALE3", true},
	sysInstFields{0, 8, 1, 0}:  {sys_TLBI, "VMALLE1OS", false},
	sysInstFields{0, 8, 1, 1}:  {sys_TLBI, "VAE1OS", true},
	sysInstFields{0, 8, 1, 2}:  {sys_TLBI, "ASIDE1OS", true},
	sysInstFields{0, 8, 1, 3}:  {sys_TLBI, "VAAE1OS", true},
	sysInstFields{0, 8, 1, 5}:  {sys_TLBI, "VALE1OS", true},
	sysInstFields{0, 8, 1, 7}:  {sys_TLBI, "VAALE1OS", true},
	sysInstFields{0, 8, 2, 1}:  {sys_TLBI, "RVAE1IS", true},
	sysInstFields{0, 8, 2, 3}:  {sys_TLBI, "RVAAE1IS", true},
	sysInstFields{0, 8, 2, 5}:  {sys_TLBI, "RVALE1IS", true},
	sysInstFields{0, 8, 2, 7}:  {sys_TLBI, "RVAALE1IS", true},
	sysInstFields{0, 8, 5, 1}:  {sys_TLBI, "RVAE1OS", true},
	sysInstFields{0, 8, 5, 3}:  {sys_TLBI, "RVAAE1OS", true},
	sysInstFields{0, 8, 5, 5}:  {sys_TLBI, "RVALE1OS", true},
	sysInstFields{0, 8, 5, 7}:  {sys_TLBI, "RVAALE1OS", true},
	sysInstFields{0, 8, 6, 1}:  {sys_TLBI, "RVAE1", true},
	sysInstFields{0, 8, 6, 3}:  {sys_TLBI, "RVAAE1", true},
	sysInstFields{0, 8, 6, 5}:  {sys_TLBI, "RVALE1", true},
	sysInstFields{0, 8, 6, 7}:  {sys_TLBI, "RVAALE1", true},
	sysInstFields{4, 8, 0, 2}:  {sys_TLBI, "RIPAS2E1IS", true},
	sysInstFields{4, 8, 0, 6}:  {sys_TLBI, "RIPAS2LE1IS", true},
	sysInstFields{4, 8, 1, 0}:  {sys_TLBI, "ALLE2OS", false},
	sysInstFields{4, 8, 1, 1}:  {sys_TLBI, "VAE2OS", true},
	sysInstFields{4, 8, 1, 4}:  {sys_TLBI, "ALLE1OS", false},
	sysInstFields{4, 8, 1, 5}:  {sys_TLBI, "VALE2OS", true},
	sysInstFields{4, 8, 1, 6}:  {sys_TLBI, "VMALLS12E1OS", false},
	sysInstFields{4, 8, 2, 1}:  {sys_TLBI, "RVAE2IS", true},
	sysInstFields{4, 8, 2, 5}:  {sys_TLBI, "RVALE2IS", true},
	sysInstFields{4, 8, 4, 0}:  {sys_TLBI, "IPAS2E1OS", true},
	sysInstFields{4, 8, 4, 2}:  {sys_TLBI, "RIPAS2E1", true},
	sysInstFields{4, 8, 4, 3}:  {sys_TLBI, "RIPAS2E1OS", true},
	sysInstFields{4, 8, 4, 4}:  {sys_TLBI, "IPAS2LE1OS", true},
	sysInstFields{4, 8, 4, 6}:  {sys_TLBI, "RIPAS2LE1", true},
	sysInstFields{4, 8, 4, 7}:  {sys_TLBI, "RIPAS2LE1OS", true},
	sysInstFields{4, 8, 5, 1}:  {sys_TLBI, "RVAE2OS", true},
	sysInstFields{4, 8, 5, 5}:  {sys_TLBI, "RVALE2OS", true},
	sysInstFields{4, 8, 6, 1}:  {sys_TLBI, "RVAE2", true},
	sysInstFields{4, 8, 6, 5}:  {sys_TLBI, "RVALE2", true},
	sysInstFields{6, 8, 1, 0}:  {sys_TLBI, "ALLE3OS", false},
	sysInstFields{6, 8, 1, 1}:  {sys_TLBI, "VAE3OS", true},
	sysInstFields{6, 8, 1, 5}:  {sys_TLBI, "VALE3OS", true},
	sysInstFields{6, 8, 2, 1}:  {sys_TLBI, "RVAE3IS", true},
	sysInstFields{6, 8, 2, 5}:  {sys_TLBI, "RVALE3IS", true},
	sysInstFields{6, 8, 5, 1}:  {sys_TLBI, "RVAE3OS", true},
	sysInstFields{6, 8, 5, 5}:  {sys_TLBI, "RVALE3OS", true},
	sysInstFields{6, 8, 6, 1}:  {sys_TLBI, "RVAE3", true},
	sysInstFields{6, 8, 6, 5}:  {sys_TLBI, "RVALE3", true},
	sysInstFields{0, 7, 6, 1}:  {sys_DC, "IVAC", true},
	sysInstFields{0, 7, 6, 2}:  {sys_DC, "ISW", true},
	sysInstFields{0, 7, 10, 2}: {sys_DC, "CSW", true},
	sysInstFields{0, 7, 14, 2}: {sys_DC, "CISW", true},
	sysInstFields{3, 7, 4, 1}:  {sys_DC, "ZVA", true},
	sysInstFields{3, 7, 10, 1}: {sys_DC, "CVAC", true},
	sysInstFields{3, 7, 11, 1}: {sys_DC, "CVAU", true},
	sysInstFields{3, 7, 14, 1}: {sys_DC, "CIVAC", true},
	sysInstFields{0, 7, 6, 3}:  {sys_DC, "IGVAC", true},
	sysInstFields{0, 7, 6, 4}:  {sys_DC, "IGSW", true},
	sysInstFields{0, 7, 6, 5}:  {sys_DC, "IGDVAC", true},
	sysInstFields{0, 7, 6, 6}:  {sys_DC, "IGDSW", true},
	sysInstFields{0, 7, 10, 4}: {sys_DC, "CGSW", true},
	sysInstFields{0, 7, 10, 6}: {sys_DC, "CGDSW", true},
	sysInstFields{0, 7, 14, 4}: {sys_DC, "CIGSW", true},
	sysInstFields{0, 7, 14, 6}: {sys_DC, "CIGDSW", true},
	sysInstFields{3, 7, 4, 3}:  {sys_DC, "GVA", true},
	sysInstFields{3, 7, 4, 4}:  {sys_DC, "GZVA", true},
	sysInstFields{3, 7, 10, 3}: {sys_DC, "CGVAC", true},
	sysInstFields{3, 7, 10, 5}: {sys_DC, "CGDVAC", true},
	sysInstFields{3, 7, 12, 3}: {sys_DC, "CGVAP", true},
	sysInstFields{3, 7, 12, 5}: {sys_DC, "CGDVAP", true},
	sysInstFields{3, 7, 13, 3}: {sys_DC, "CGVADP", true},
	sysInstFields{3, 7, 13, 5}: {sys_DC, "CGDVADP", true},
	sysInstFields{3, 7, 14, 3}: {sys_DC, "CIGVAC", true},
	sysInstFields{3, 7, 14, 5}: {sys_DC, "CIGDVAC", true},
	sysInstFields{3, 7, 12, 1}: {sys_DC, "CVAP", true},
	sysInstFields{3, 7, 13, 1}: {sys_DC, "CVADP", true},
}
