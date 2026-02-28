// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loong64asm

import (
	"encoding/binary"
	"fmt"
)

type instArgs [5]instArg

// An instFormat describes the format of an instruction encoding.
type instFormat struct {
	mask  uint32
	value uint32
	op    Op
	// args describe how to decode the instruction arguments.
	// args is stored as a fixed-size array.
	// if there are fewer than len(args) arguments, args[i] == 0 marks
	// the end of the argument list.
	args instArgs
}

var (
	errShort   = fmt.Errorf("truncated instruction")
	errUnknown = fmt.Errorf("unknown instruction")
)

var decoderCover []bool

func init() {
	decoderCover = make([]bool, len(instFormats))
}

// Decode decodes the 4 bytes in src as a single instruction.
func Decode(src []byte) (inst Inst, err error) {
	if len(src) < 4 {
		return Inst{}, errShort
	}

	x := binary.LittleEndian.Uint32(src)

Search:
	for i := range instFormats {
		f := &instFormats[i]

		if (x & f.mask) != f.value {
			continue
		}

		// Decode args.
		var args Args
		for j, aop := range f.args {
			if aop == 0 {
				break
			}

			arg := decodeArg(aop, x, i)
			if arg == nil {
				// Cannot decode argument
				continue Search
			}

			args[j] = arg
		}

		decoderCover[i] = true
		inst = Inst{
			Op:   f.op,
			Args: args,
			Enc:  x,
		}
		return inst, nil
	}

	return Inst{}, errUnknown
}

// decodeArg decodes the arg described by aop from the instruction bits x.
// It returns nil if x cannot be decoded according to aop.
func decodeArg(aop instArg, x uint32, index int) Arg {
	switch aop {
	case arg_fd:
		return F0 + Reg(x&((1<<5)-1))

	case arg_fj:
		return F0 + Reg((x>>5)&((1<<5)-1))

	case arg_fk:
		return F0 + Reg((x>>10)&((1<<5)-1))

	case arg_fa:
		return F0 + Reg((x>>15)&((1<<5)-1))

	case arg_rd:
		return R0 + Reg(x&((1<<5)-1))

	case arg_rj:
		return R0 + Reg((x>>5)&((1<<5)-1))

	case arg_rk:
		return R0 + Reg((x>>10)&((1<<5)-1))

	case arg_fcsr_4_0:
		return FCSR0 + Fcsr(x&((1<<5)-1))

	case arg_fcsr_9_5:
		return FCSR0 + Fcsr((x>>5)&((1<<5)-1))

	case arg_cd:
		return FCC0 + Fcc(x&((1<<3)-1))

	case arg_cj:
		return FCC0 + Fcc((x>>5)&((1<<3)-1))

	case arg_ca:
		return FCC0 + Fcc((x>>15)&((1<<3)-1))

	case arg_op_4_0:
		tmp := x & ((1 << 5) - 1)
		return Uimm{tmp, false}

	case arg_csr_23_10:
		tmp := (x >> 10) & ((1 << 14) - 1)
		return Uimm{tmp, false}

	case arg_sa2_16_15:
		f := &instFormats[index]
		tmp := SaSimm((x >> 15) & ((1 << 2) - 1))
		if (f.op == ALSL_D) || (f.op == ALSL_W) || (f.op == ALSL_WU) {
			return tmp + 1
		} else {
			return tmp + 0
		}

	case arg_sa3_17_15:
		return SaSimm((x >> 15) & ((1 << 3) - 1))

	case arg_code_4_0:
		return CodeSimm(x & ((1 << 5) - 1))

	case arg_code_14_0:
		return CodeSimm(x & ((1 << 15) - 1))

	case arg_ui5_14_10:
		tmp := (x >> 10) & ((1 << 5) - 1)
		return Uimm{tmp, false}

	case arg_ui6_15_10:
		tmp := (x >> 10) & ((1 << 6) - 1)
		return Uimm{tmp, false}

	case arg_ui12_21_10:
		tmp := ((x >> 10) & ((1 << 12) - 1) & 0xfff)
		return Uimm{tmp, false}

	case arg_lsbw:
		tmp := (x >> 10) & ((1 << 5) - 1)
		return Uimm{tmp, false}

	case arg_msbw:
		tmp := (x >> 16) & ((1 << 5) - 1)
		return Uimm{tmp, false}

	case arg_lsbd:
		tmp := (x >> 10) & ((1 << 6) - 1)
		return Uimm{tmp, false}

	case arg_msbd:
		tmp := (x >> 16) & ((1 << 6) - 1)
		return Uimm{tmp, false}

	case arg_hint_4_0:
		tmp := x & ((1 << 5) - 1)
		return Uimm{tmp, false}

	case arg_hint_14_0:
		tmp := x & ((1 << 15) - 1)
		return Uimm{tmp, false}

	case arg_level_14_0:
		tmp := x & ((1 << 15) - 1)
		return Uimm{tmp, false}

	case arg_level_17_10:
		tmp := (x >> 10) & ((1 << 8) - 1)
		return Uimm{tmp, false}

	case arg_seq_17_10:
		tmp := (x >> 10) & ((1 << 8) - 1)
		return Uimm{tmp, false}

	case arg_si12_21_10:
		var tmp int16

		// no int12, so sign-extend a 12-bit signed to 16-bit signed
		if (x & 0x200000) == 0x200000 {
			tmp = int16(((x >> 10) & ((1 << 12) - 1)) | 0xf000)
		} else {
			tmp = int16(((x >> 10) & ((1 << 12) - 1)) | 0x0000)
		}
		return Simm16{tmp, 12}

	case arg_si14_23_10:
		var tmp int32
		if (x & 0x800000) == 0x800000 {
			tmp = int32((((x >> 10) & ((1 << 14) - 1)) << 2) | 0xffff0000)
		} else {
			tmp = int32((((x >> 10) & ((1 << 14) - 1)) << 2) | 0x00000000)
		}
		return Simm32{tmp, 14}

	case arg_si16_25_10:
		var tmp int32

		if (x & 0x2000000) == 0x2000000 {
			tmp = int32(((x >> 10) & ((1 << 16) - 1)) | 0xffff0000)
		} else {
			tmp = int32(((x >> 10) & ((1 << 16) - 1)) | 0x00000000)
		}

		return Simm32{tmp, 16}

	case arg_si20_24_5:
		var tmp int32
		if (x & 0x1000000) == 0x1000000 {
			tmp = int32(((x >> 5) & ((1 << 20) - 1)) | 0xfff00000)
		} else {
			tmp = int32(((x >> 5) & ((1 << 20) - 1)) | 0x00000000)
		}
		return Simm32{tmp, 20}

	case arg_offset_20_0:
		var tmp int32

		if (x & 0x10) == 0x10 {
			tmp = int32(((((x << 16) | ((x >> 10) & ((1 << 16) - 1))) & ((1 << 21) - 1)) << 2) | 0xff800000)
		} else {
			tmp = int32((((x << 16) | ((x >> 10) & ((1 << 16) - 1))) & ((1 << 21) - 1)) << 2)
		}

		return OffsetSimm{tmp, 21}

	case arg_offset_15_0:
		var tmp int32
		if (x & 0x2000000) == 0x2000000 {
			tmp = int32((((x >> 10) & ((1 << 16) - 1)) << 2) | 0xfffc0000)
		} else {
			tmp = int32((((x >> 10) & ((1 << 16) - 1)) << 2) | 0x00000000)
		}

		return OffsetSimm{tmp, 16}

	case arg_offset_25_0:
		var tmp int32

		if (x & 0x200) == 0x200 {
			tmp = int32(((((x << 16) | ((x >> 10) & ((1 << 16) - 1))) & ((1 << 26) - 1)) << 2) | 0xf0000000)
		} else {
			tmp = int32(((((x << 16) | ((x >> 10) & ((1 << 16) - 1))) & ((1 << 26) - 1)) << 2) | 0x00000000)
		}

		return OffsetSimm{tmp, 26}
	default:
		return nil
	}
}
