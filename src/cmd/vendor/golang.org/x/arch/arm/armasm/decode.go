// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package armasm

import (
	"encoding/binary"
	"fmt"
)

// An instFormat describes the format of an instruction encoding.
// An instruction with 32-bit value x matches the format if x&mask == value
// and the condition matches.
// The condition matches if x>>28 == 0xF && value>>28==0xF
// or if x>>28 != 0xF and value>>28 == 0.
// If x matches the format, then the rest of the fields describe how to interpret x.
// The opBits describe bits that should be extracted from x and added to the opcode.
// For example opBits = 0x1234 means that the value
//	(2 bits at offset 1) followed by (4 bits at offset 3)
// should be added to op.
// Finally the args describe how to decode the instruction arguments.
// args is stored as a fixed-size array; if there are fewer than len(args) arguments,
// args[i] == 0 marks the end of the argument list.
type instFormat struct {
	mask     uint32
	value    uint32
	priority int8
	op       Op
	opBits   uint64
	args     instArgs
}

type instArgs [4]instArg

var (
	errMode    = fmt.Errorf("unsupported execution mode")
	errShort   = fmt.Errorf("truncated instruction")
	errUnknown = fmt.Errorf("unknown instruction")
)

var decoderCover []bool

// Decode decodes the leading bytes in src as a single instruction.
func Decode(src []byte, mode Mode) (inst Inst, err error) {
	if mode != ModeARM {
		return Inst{}, errMode
	}
	if len(src) < 4 {
		return Inst{}, errShort
	}

	if decoderCover == nil {
		decoderCover = make([]bool, len(instFormats))
	}

	x := binary.LittleEndian.Uint32(src)

	// The instFormat table contains both conditional and unconditional instructions.
	// Considering only the top 4 bits, the conditional instructions use mask=0, value=0,
	// while the unconditional instructions use mask=f, value=f.
	// Prepare a version of x with the condition cleared to 0 in conditional instructions
	// and then assume mask=f during matching.
	const condMask = 0xf0000000
	xNoCond := x
	if x&condMask != condMask {
		xNoCond &^= condMask
	}
	var priority int8
Search:
	for i := range instFormats {
		f := &instFormats[i]
		if xNoCond&(f.mask|condMask) != f.value || f.priority <= priority {
			continue
		}
		delta := uint32(0)
		deltaShift := uint(0)
		for opBits := f.opBits; opBits != 0; opBits >>= 16 {
			n := uint(opBits & 0xFF)
			off := uint((opBits >> 8) & 0xFF)
			delta |= (x >> off) & (1<<n - 1) << deltaShift
			deltaShift += n
		}
		op := f.op + Op(delta)

		// Special case: BKPT encodes with condition but cannot have one.
		if op&^15 == BKPT_EQ && op != BKPT {
			continue Search
		}

		var args Args
		for j, aop := range f.args {
			if aop == 0 {
				break
			}
			arg := decodeArg(aop, x)
			if arg == nil { // cannot decode argument
				continue Search
			}
			args[j] = arg
		}

		decoderCover[i] = true

		inst = Inst{
			Op:   op,
			Args: args,
			Enc:  x,
			Len:  4,
		}
		priority = f.priority
		continue Search
	}
	if inst.Op != 0 {
		return inst, nil
	}
	return Inst{}, errUnknown
}

// An instArg describes the encoding of a single argument.
// In the names used for arguments, _p_ means +, _m_ means -,
// _pm_ means ± (usually keyed by the U bit).
// The _W suffix indicates a general addressing mode based on the P and W bits.
// The _offset and _postindex suffixes force the given addressing mode.
// The rest should be somewhat self-explanatory, at least given
// the decodeArg function.
type instArg uint8

const (
	_ instArg = iota
	arg_APSR
	arg_FPSCR
	arg_Dn_half
	arg_R1_0
	arg_R1_12
	arg_R2_0
	arg_R2_12
	arg_R_0
	arg_R_12
	arg_R_12_nzcv
	arg_R_16
	arg_R_16_WB
	arg_R_8
	arg_R_rotate
	arg_R_shift_R
	arg_R_shift_imm
	arg_SP
	arg_Sd
	arg_Sd_Dd
	arg_Dd_Sd
	arg_Sm
	arg_Sm_Dm
	arg_Sn
	arg_Sn_Dn
	arg_const
	arg_endian
	arg_fbits
	arg_fp_0
	arg_imm24
	arg_imm5
	arg_imm5_32
	arg_imm5_nz
	arg_imm_12at8_4at0
	arg_imm_4at16_12at0
	arg_imm_vfp
	arg_label24
	arg_label24H
	arg_label_m_12
	arg_label_p_12
	arg_label_pm_12
	arg_label_pm_4_4
	arg_lsb_width
	arg_mem_R
	arg_mem_R_pm_R_W
	arg_mem_R_pm_R_postindex
	arg_mem_R_pm_R_shift_imm_W
	arg_mem_R_pm_R_shift_imm_offset
	arg_mem_R_pm_R_shift_imm_postindex
	arg_mem_R_pm_imm12_W
	arg_mem_R_pm_imm12_offset
	arg_mem_R_pm_imm12_postindex
	arg_mem_R_pm_imm8_W
	arg_mem_R_pm_imm8_postindex
	arg_mem_R_pm_imm8at0_offset
	arg_option
	arg_registers
	arg_registers1
	arg_registers2
	arg_satimm4
	arg_satimm5
	arg_satimm4m1
	arg_satimm5m1
	arg_widthm1
)

// decodeArg decodes the arg described by aop from the instruction bits x.
// It returns nil if x cannot be decoded according to aop.
func decodeArg(aop instArg, x uint32) Arg {
	switch aop {
	default:
		return nil

	case arg_APSR:
		return APSR
	case arg_FPSCR:
		return FPSCR

	case arg_R_0:
		return Reg(x & (1<<4 - 1))
	case arg_R_8:
		return Reg((x >> 8) & (1<<4 - 1))
	case arg_R_12:
		return Reg((x >> 12) & (1<<4 - 1))
	case arg_R_16:
		return Reg((x >> 16) & (1<<4 - 1))

	case arg_R_12_nzcv:
		r := Reg((x >> 12) & (1<<4 - 1))
		if r == R15 {
			return APSR_nzcv
		}
		return r

	case arg_R_16_WB:
		mode := AddrLDM
		if (x>>21)&1 != 0 {
			mode = AddrLDM_WB
		}
		return Mem{Base: Reg((x >> 16) & (1<<4 - 1)), Mode: mode}

	case arg_R_rotate:
		Rm := Reg(x & (1<<4 - 1))
		typ, count := decodeShift(x)
		// ROR #0 here means ROR #0, but decodeShift rewrites to RRX #1.
		if typ == RotateRightExt {
			return Rm
		}
		return RegShift{Rm, typ, count}

	case arg_R_shift_R:
		Rm := Reg(x & (1<<4 - 1))
		Rs := Reg((x >> 8) & (1<<4 - 1))
		typ := Shift((x >> 5) & (1<<2 - 1))
		return RegShiftReg{Rm, typ, Rs}

	case arg_R_shift_imm:
		Rm := Reg(x & (1<<4 - 1))
		typ, count := decodeShift(x)
		if typ == ShiftLeft && count == 0 {
			return Rm
		}
		return RegShift{Rm, typ, count}

	case arg_R1_0:
		return Reg((x & (1<<4 - 1)))
	case arg_R1_12:
		return Reg(((x >> 12) & (1<<4 - 1)))
	case arg_R2_0:
		return Reg((x & (1<<4 - 1)) | 1)
	case arg_R2_12:
		return Reg(((x >> 12) & (1<<4 - 1)) | 1)

	case arg_SP:
		return SP

	case arg_Sd_Dd:
		v := (x >> 12) & (1<<4 - 1)
		vx := (x >> 22) & 1
		sz := (x >> 8) & 1
		if sz != 0 {
			return D0 + Reg(vx<<4+v)
		} else {
			return S0 + Reg(v<<1+vx)
		}

	case arg_Dd_Sd:
		return decodeArg(arg_Sd_Dd, x^(1<<8))

	case arg_Sd:
		v := (x >> 12) & (1<<4 - 1)
		vx := (x >> 22) & 1
		return S0 + Reg(v<<1+vx)

	case arg_Sm_Dm:
		v := (x >> 0) & (1<<4 - 1)
		vx := (x >> 5) & 1
		sz := (x >> 8) & 1
		if sz != 0 {
			return D0 + Reg(vx<<4+v)
		} else {
			return S0 + Reg(v<<1+vx)
		}

	case arg_Sm:
		v := (x >> 0) & (1<<4 - 1)
		vx := (x >> 5) & 1
		return S0 + Reg(v<<1+vx)

	case arg_Dn_half:
		v := (x >> 16) & (1<<4 - 1)
		vx := (x >> 7) & 1
		return RegX{D0 + Reg(vx<<4+v), int((x >> 21) & 1)}

	case arg_Sn_Dn:
		v := (x >> 16) & (1<<4 - 1)
		vx := (x >> 7) & 1
		sz := (x >> 8) & 1
		if sz != 0 {
			return D0 + Reg(vx<<4+v)
		} else {
			return S0 + Reg(v<<1+vx)
		}

	case arg_Sn:
		v := (x >> 16) & (1<<4 - 1)
		vx := (x >> 7) & 1
		return S0 + Reg(v<<1+vx)

	case arg_const:
		v := x & (1<<8 - 1)
		rot := (x >> 8) & (1<<4 - 1) * 2
		if rot > 0 && v&3 == 0 {
			// could rotate less
			return ImmAlt{uint8(v), uint8(rot)}
		}
		if rot >= 24 && ((v<<(32-rot))&0xFF)>>(32-rot) == v {
			// could wrap around to rot==0.
			return ImmAlt{uint8(v), uint8(rot)}
		}
		return Imm(v>>rot | v<<(32-rot))

	case arg_endian:
		return Endian((x >> 9) & 1)

	case arg_fbits:
		return Imm((16 << ((x >> 7) & 1)) - ((x&(1<<4-1))<<1 | (x>>5)&1))

	case arg_fp_0:
		return Imm(0)

	case arg_imm24:
		return Imm(x & (1<<24 - 1))

	case arg_imm5:
		return Imm((x >> 7) & (1<<5 - 1))

	case arg_imm5_32:
		x = (x >> 7) & (1<<5 - 1)
		if x == 0 {
			x = 32
		}
		return Imm(x)

	case arg_imm5_nz:
		x = (x >> 7) & (1<<5 - 1)
		if x == 0 {
			return nil
		}
		return Imm(x)

	case arg_imm_4at16_12at0:
		return Imm((x>>16)&(1<<4-1)<<12 | x&(1<<12-1))

	case arg_imm_12at8_4at0:
		return Imm((x>>8)&(1<<12-1)<<4 | x&(1<<4-1))

	case arg_imm_vfp:
		x = (x>>16)&(1<<4-1)<<4 | x&(1<<4-1)
		return Imm(x)

	case arg_label24:
		imm := (x & (1<<24 - 1)) << 2
		return PCRel(int32(imm<<6) >> 6)

	case arg_label24H:
		h := (x >> 24) & 1
		imm := (x&(1<<24-1))<<2 | h<<1
		return PCRel(int32(imm<<6) >> 6)

	case arg_label_m_12:
		d := int32(x & (1<<12 - 1))
		return Mem{Base: PC, Mode: AddrOffset, Offset: int16(-d)}

	case arg_label_p_12:
		d := int32(x & (1<<12 - 1))
		return Mem{Base: PC, Mode: AddrOffset, Offset: int16(d)}

	case arg_label_pm_12:
		d := int32(x & (1<<12 - 1))
		u := (x >> 23) & 1
		if u == 0 {
			d = -d
		}
		return Mem{Base: PC, Mode: AddrOffset, Offset: int16(d)}

	case arg_label_pm_4_4:
		d := int32((x>>8)&(1<<4-1)<<4 | x&(1<<4-1))
		u := (x >> 23) & 1
		if u == 0 {
			d = -d
		}
		return PCRel(d)

	case arg_lsb_width:
		lsb := (x >> 7) & (1<<5 - 1)
		msb := (x >> 16) & (1<<5 - 1)
		if msb < lsb || msb >= 32 {
			return nil
		}
		return Imm(msb + 1 - lsb)

	case arg_mem_R:
		Rn := Reg((x >> 16) & (1<<4 - 1))
		return Mem{Base: Rn, Mode: AddrOffset}

	case arg_mem_R_pm_R_postindex:
		// Treat [<Rn>],+/-<Rm> like [<Rn>,+/-<Rm>{,<shift>}]{!}
		// by forcing shift bits to <<0 and P=0, W=0 (postindex=true).
		return decodeArg(arg_mem_R_pm_R_shift_imm_W, x&^((1<<7-1)<<5|1<<24|1<<21))

	case arg_mem_R_pm_R_W:
		// Treat [<Rn>,+/-<Rm>]{!} like [<Rn>,+/-<Rm>{,<shift>}]{!}
		// by forcing shift bits to <<0.
		return decodeArg(arg_mem_R_pm_R_shift_imm_W, x&^((1<<7-1)<<5))

	case arg_mem_R_pm_R_shift_imm_offset:
		// Treat [<Rn>],+/-<Rm>{,<shift>} like [<Rn>,+/-<Rm>{,<shift>}]{!}
		// by forcing P=1, W=0 (index=false, wback=false).
		return decodeArg(arg_mem_R_pm_R_shift_imm_W, x&^(1<<21)|1<<24)

	case arg_mem_R_pm_R_shift_imm_postindex:
		// Treat [<Rn>],+/-<Rm>{,<shift>} like [<Rn>,+/-<Rm>{,<shift>}]{!}
		// by forcing P=0, W=0 (postindex=true).
		return decodeArg(arg_mem_R_pm_R_shift_imm_W, x&^(1<<24|1<<21))

	case arg_mem_R_pm_R_shift_imm_W:
		Rn := Reg((x >> 16) & (1<<4 - 1))
		Rm := Reg(x & (1<<4 - 1))
		typ, count := decodeShift(x)
		u := (x >> 23) & 1
		w := (x >> 21) & 1
		p := (x >> 24) & 1
		if p == 0 && w == 1 {
			return nil
		}
		sign := int8(+1)
		if u == 0 {
			sign = -1
		}
		mode := AddrMode(uint8(p<<1) | uint8(w^1))
		return Mem{Base: Rn, Mode: mode, Sign: sign, Index: Rm, Shift: typ, Count: count}

	case arg_mem_R_pm_imm12_offset:
		// Treat [<Rn>,#+/-<imm12>] like [<Rn>{,#+/-<imm12>}]{!}
		// by forcing P=1, W=0 (index=false, wback=false).
		return decodeArg(arg_mem_R_pm_imm12_W, x&^(1<<21)|1<<24)

	case arg_mem_R_pm_imm12_postindex:
		// Treat [<Rn>],#+/-<imm12> like [<Rn>{,#+/-<imm12>}]{!}
		// by forcing P=0, W=0 (postindex=true).
		return decodeArg(arg_mem_R_pm_imm12_W, x&^(1<<24|1<<21))

	case arg_mem_R_pm_imm12_W:
		Rn := Reg((x >> 16) & (1<<4 - 1))
		u := (x >> 23) & 1
		w := (x >> 21) & 1
		p := (x >> 24) & 1
		if p == 0 && w == 1 {
			return nil
		}
		sign := int8(+1)
		if u == 0 {
			sign = -1
		}
		imm := int16(x & (1<<12 - 1))
		mode := AddrMode(uint8(p<<1) | uint8(w^1))
		return Mem{Base: Rn, Mode: mode, Offset: int16(sign) * imm}

	case arg_mem_R_pm_imm8_postindex:
		// Treat [<Rn>],#+/-<imm8> like [<Rn>{,#+/-<imm8>}]{!}
		// by forcing P=0, W=0 (postindex=true).
		return decodeArg(arg_mem_R_pm_imm8_W, x&^(1<<24|1<<21))

	case arg_mem_R_pm_imm8_W:
		Rn := Reg((x >> 16) & (1<<4 - 1))
		u := (x >> 23) & 1
		w := (x >> 21) & 1
		p := (x >> 24) & 1
		if p == 0 && w == 1 {
			return nil
		}
		sign := int8(+1)
		if u == 0 {
			sign = -1
		}
		imm := int16((x>>8)&(1<<4-1)<<4 | x&(1<<4-1))
		mode := AddrMode(uint8(p<<1) | uint8(w^1))
		return Mem{Base: Rn, Mode: mode, Offset: int16(sign) * imm}

	case arg_mem_R_pm_imm8at0_offset:
		Rn := Reg((x >> 16) & (1<<4 - 1))
		u := (x >> 23) & 1
		sign := int8(+1)
		if u == 0 {
			sign = -1
		}
		imm := int16(x&(1<<8-1)) << 2
		return Mem{Base: Rn, Mode: AddrOffset, Offset: int16(sign) * imm}

	case arg_option:
		return Imm(x & (1<<4 - 1))

	case arg_registers:
		return RegList(x & (1<<16 - 1))

	case arg_registers2:
		x &= 1<<16 - 1
		n := 0
		for i := 0; i < 16; i++ {
			if x>>uint(i)&1 != 0 {
				n++
			}
		}
		if n < 2 {
			return nil
		}
		return RegList(x)

	case arg_registers1:
		Rt := (x >> 12) & (1<<4 - 1)
		return RegList(1 << Rt)

	case arg_satimm4:
		return Imm((x >> 16) & (1<<4 - 1))

	case arg_satimm5:
		return Imm((x >> 16) & (1<<5 - 1))

	case arg_satimm4m1:
		return Imm((x>>16)&(1<<4-1) + 1)

	case arg_satimm5m1:
		return Imm((x>>16)&(1<<5-1) + 1)

	case arg_widthm1:
		return Imm((x>>16)&(1<<5-1) + 1)

	}
}

// decodeShift decodes the shift-by-immediate encoded in x.
func decodeShift(x uint32) (Shift, uint8) {
	count := (x >> 7) & (1<<5 - 1)
	typ := Shift((x >> 5) & (1<<2 - 1))
	switch typ {
	case ShiftRight, ShiftRightSigned:
		if count == 0 {
			count = 32
		}
	case RotateRight:
		if count == 0 {
			typ = RotateRightExt
			count = 1
		}
	}
	return typ, uint8(count)
}
