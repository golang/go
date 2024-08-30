// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64asm

import (
	"encoding/binary"
	"fmt"
)

type instArgs [5]instArg

// An instFormat describes the format of an instruction encoding.
// An instruction with 32-bit value x matches the format if x&mask == value
// and the predicator: canDecode(x) return true.
type instFormat struct {
	mask  uint32
	value uint32
	op    Op
	// args describe how to decode the instruction arguments.
	// args is stored as a fixed-size array.
	// if there are fewer than len(args) arguments, args[i] == 0 marks
	// the end of the argument list.
	args      instArgs
	canDecode func(instr uint32) bool
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
		if x&f.mask != f.value {
			continue
		}
		if f.canDecode != nil && !f.canDecode(x) {
			continue
		}
		// Decode args.
		var args Args
		for j, aop := range f.args {
			if aop == 0 {
				break
			}
			arg := decodeArg(aop, x)
			if arg == nil { // Cannot decode argument
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
func decodeArg(aop instArg, x uint32) Arg {
	switch aop {
	default:
		return nil

	case arg_Da:
		return D0 + Reg((x>>10)&(1<<5-1))

	case arg_Dd:
		return D0 + Reg(x&(1<<5-1))

	case arg_Dm:
		return D0 + Reg((x>>16)&(1<<5-1))

	case arg_Dn:
		return D0 + Reg((x>>5)&(1<<5-1))

	case arg_Hd:
		return H0 + Reg(x&(1<<5-1))

	case arg_Hn:
		return H0 + Reg((x>>5)&(1<<5-1))

	case arg_IAddSub:
		imm12 := (x >> 10) & (1<<12 - 1)
		shift := (x >> 22) & (1<<2 - 1)
		if shift > 1 {
			return nil
		}
		shift = shift * 12
		return ImmShift{uint16(imm12), uint8(shift)}

	case arg_Sa:
		return S0 + Reg((x>>10)&(1<<5-1))

	case arg_Sd:
		return S0 + Reg(x&(1<<5-1))

	case arg_Sm:
		return S0 + Reg((x>>16)&(1<<5-1))

	case arg_Sn:
		return S0 + Reg((x>>5)&(1<<5-1))

	case arg_Wa:
		return W0 + Reg((x>>10)&(1<<5-1))

	case arg_Wd:
		return W0 + Reg(x&(1<<5-1))

	case arg_Wds:
		return RegSP(W0) + RegSP(x&(1<<5-1))

	case arg_Wm:
		return W0 + Reg((x>>16)&(1<<5-1))

	case arg_Rm_extend__UXTB_0__UXTH_1__UXTW_2__LSL_UXTX_3__SXTB_4__SXTH_5__SXTW_6__SXTX_7__0_4:
		return handle_ExtendedRegister(x, true)

	case arg_Wm_extend__UXTB_0__UXTH_1__LSL_UXTW_2__UXTX_3__SXTB_4__SXTH_5__SXTW_6__SXTX_7__0_4:
		return handle_ExtendedRegister(x, false)

	case arg_Wn:
		return W0 + Reg((x>>5)&(1<<5-1))

	case arg_Wns:
		return RegSP(W0) + RegSP((x>>5)&(1<<5-1))

	case arg_Xa:
		return X0 + Reg((x>>10)&(1<<5-1))

	case arg_Xd:
		return X0 + Reg(x&(1<<5-1))

	case arg_Xds:
		return RegSP(X0) + RegSP(x&(1<<5-1))

	case arg_Xm:
		return X0 + Reg((x>>16)&(1<<5-1))

	case arg_Wm_shift__LSL_0__LSR_1__ASR_2__0_31:
		return handle_ImmediateShiftedRegister(x, 31, true, false)

	case arg_Wm_shift__LSL_0__LSR_1__ASR_2__ROR_3__0_31:
		return handle_ImmediateShiftedRegister(x, 31, true, true)

	case arg_Xm_shift__LSL_0__LSR_1__ASR_2__0_63:
		return handle_ImmediateShiftedRegister(x, 63, false, false)

	case arg_Xm_shift__LSL_0__LSR_1__ASR_2__ROR_3__0_63:
		return handle_ImmediateShiftedRegister(x, 63, false, true)

	case arg_Xn:
		return X0 + Reg((x>>5)&(1<<5-1))

	case arg_Xns:
		return RegSP(X0) + RegSP((x>>5)&(1<<5-1))

	case arg_slabel_imm14_2:
		imm14 := ((x >> 5) & (1<<14 - 1))
		return PCRel(((int64(imm14) << 2) << 48) >> 48)

	case arg_slabel_imm19_2:
		imm19 := ((x >> 5) & (1<<19 - 1))
		return PCRel(((int64(imm19) << 2) << 43) >> 43)

	case arg_slabel_imm26_2:
		imm26 := (x & (1<<26 - 1))
		return PCRel(((int64(imm26) << 2) << 36) >> 36)

	case arg_slabel_immhi_immlo_0:
		immhi := ((x >> 5) & (1<<19 - 1))
		immlo := ((x >> 29) & (1<<2 - 1))
		immhilo := (immhi)<<2 | immlo
		return PCRel((int64(immhilo) << 43) >> 43)

	case arg_slabel_immhi_immlo_12:
		immhi := ((x >> 5) & (1<<19 - 1))
		immlo := ((x >> 29) & (1<<2 - 1))
		immhilo := (immhi)<<2 | immlo
		return PCRel(((int64(immhilo) << 12) << 31) >> 31)

	case arg_Xns_mem:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		return MemImmediate{Rn, AddrOffset, 0}

	case arg_Xns_mem_extend_m__UXTW_2__LSL_3__SXTW_6__SXTX_7__0_0__1_1:
		return handle_MemExtend(x, 1, false)

	case arg_Xns_mem_extend_m__UXTW_2__LSL_3__SXTW_6__SXTX_7__0_0__2_1:
		return handle_MemExtend(x, 2, false)

	case arg_Xns_mem_extend_m__UXTW_2__LSL_3__SXTW_6__SXTX_7__0_0__3_1:
		return handle_MemExtend(x, 3, false)

	case arg_Xns_mem_extend_m__UXTW_2__LSL_3__SXTW_6__SXTX_7__absent_0__0_1:
		return handle_MemExtend(x, 1, true)

	case arg_Xns_mem_optional_imm12_1_unsigned:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		imm12 := (x >> 10) & (1<<12 - 1)
		return MemImmediate{Rn, AddrOffset, int32(imm12)}

	case arg_Xns_mem_optional_imm12_2_unsigned:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		imm12 := (x >> 10) & (1<<12 - 1)
		return MemImmediate{Rn, AddrOffset, int32(imm12 << 1)}

	case arg_Xns_mem_optional_imm12_4_unsigned:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		imm12 := (x >> 10) & (1<<12 - 1)
		return MemImmediate{Rn, AddrOffset, int32(imm12 << 2)}

	case arg_Xns_mem_optional_imm12_8_unsigned:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		imm12 := (x >> 10) & (1<<12 - 1)
		return MemImmediate{Rn, AddrOffset, int32(imm12 << 3)}

	case arg_Xns_mem_optional_imm7_4_signed:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		imm7 := (x >> 15) & (1<<7 - 1)
		return MemImmediate{Rn, AddrOffset, ((int32(imm7 << 2)) << 23) >> 23}

	case arg_Xns_mem_optional_imm7_8_signed:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		imm7 := (x >> 15) & (1<<7 - 1)
		return MemImmediate{Rn, AddrOffset, ((int32(imm7 << 3)) << 22) >> 22}

	case arg_Xns_mem_optional_imm9_1_signed:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		imm9 := (x >> 12) & (1<<9 - 1)
		return MemImmediate{Rn, AddrOffset, (int32(imm9) << 23) >> 23}

	case arg_Xns_mem_post_imm7_4_signed:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		imm7 := (x >> 15) & (1<<7 - 1)
		return MemImmediate{Rn, AddrPostIndex, ((int32(imm7 << 2)) << 23) >> 23}

	case arg_Xns_mem_post_imm7_8_signed:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		imm7 := (x >> 15) & (1<<7 - 1)
		return MemImmediate{Rn, AddrPostIndex, ((int32(imm7 << 3)) << 22) >> 22}

	case arg_Xns_mem_post_imm9_1_signed:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		imm9 := (x >> 12) & (1<<9 - 1)
		return MemImmediate{Rn, AddrPostIndex, ((int32(imm9)) << 23) >> 23}

	case arg_Xns_mem_wb_imm7_4_signed:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		imm7 := (x >> 15) & (1<<7 - 1)
		return MemImmediate{Rn, AddrPreIndex, ((int32(imm7 << 2)) << 23) >> 23}

	case arg_Xns_mem_wb_imm7_8_signed:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		imm7 := (x >> 15) & (1<<7 - 1)
		return MemImmediate{Rn, AddrPreIndex, ((int32(imm7 << 3)) << 22) >> 22}

	case arg_Xns_mem_wb_imm9_1_signed:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		imm9 := (x >> 12) & (1<<9 - 1)
		return MemImmediate{Rn, AddrPreIndex, ((int32(imm9)) << 23) >> 23}

	case arg_Ws:
		return W0 + Reg((x>>16)&(1<<5-1))

	case arg_Wt:
		return W0 + Reg(x&(1<<5-1))

	case arg_Wt2:
		return W0 + Reg((x>>10)&(1<<5-1))

	case arg_Xs:
		return X0 + Reg((x>>16)&(1<<5-1))

	case arg_Xt:
		return X0 + Reg(x&(1<<5-1))

	case arg_Xt2:
		return X0 + Reg((x>>10)&(1<<5-1))

	case arg_immediate_0_127_CRm_op2:
		crm_op2 := (x >> 5) & (1<<7 - 1)
		return Imm_hint(crm_op2)

	case arg_immediate_0_15_CRm:
		crm := (x >> 8) & (1<<4 - 1)
		return Imm{crm, false}

	case arg_immediate_0_15_nzcv:
		nzcv := x & (1<<4 - 1)
		return Imm{nzcv, false}

	case arg_immediate_0_31_imm5:
		imm5 := (x >> 16) & (1<<5 - 1)
		return Imm{imm5, false}

	case arg_immediate_0_31_immr:
		immr := (x >> 16) & (1<<6 - 1)
		if immr > 31 {
			return nil
		}
		return Imm{immr, false}

	case arg_immediate_0_31_imms:
		imms := (x >> 10) & (1<<6 - 1)
		if imms > 31 {
			return nil
		}
		return Imm{imms, true}

	case arg_immediate_0_63_b5_b40:
		b5 := (x >> 31) & 1
		b40 := (x >> 19) & (1<<5 - 1)
		return Imm{(b5 << 5) | b40, true}

	case arg_immediate_0_63_immr:
		immr := (x >> 16) & (1<<6 - 1)
		return Imm{immr, false}

	case arg_immediate_0_63_imms:
		imms := (x >> 10) & (1<<6 - 1)
		return Imm{imms, true}

	case arg_immediate_0_65535_imm16:
		imm16 := (x >> 5) & (1<<16 - 1)
		return Imm{imm16, false}

	case arg_immediate_0_7_op1:
		op1 := (x >> 16) & (1<<3 - 1)
		return Imm{op1, true}

	case arg_immediate_0_7_op2:
		op2 := (x >> 5) & (1<<3 - 1)
		return Imm{op2, true}

	case arg_immediate_ASR_SBFM_32M_bitfield_0_31_immr:
		immr := (x >> 16) & (1<<6 - 1)
		if immr > 31 {
			return nil
		}
		return Imm{immr, true}

	case arg_immediate_ASR_SBFM_64M_bitfield_0_63_immr:
		immr := (x >> 16) & (1<<6 - 1)
		return Imm{immr, true}

	case arg_immediate_BFI_BFM_32M_bitfield_lsb_32_immr:
		immr := (x >> 16) & (1<<6 - 1)
		if immr > 31 {
			return nil
		}
		return Imm{32 - immr, true}

	case arg_immediate_BFI_BFM_32M_bitfield_width_32_imms:
		imms := (x >> 10) & (1<<6 - 1)
		if imms > 31 {
			return nil
		}
		return Imm{imms + 1, true}

	case arg_immediate_BFI_BFM_64M_bitfield_lsb_64_immr:
		immr := (x >> 16) & (1<<6 - 1)
		return Imm{64 - immr, true}

	case arg_immediate_BFI_BFM_64M_bitfield_width_64_imms:
		imms := (x >> 10) & (1<<6 - 1)
		return Imm{imms + 1, true}

	case arg_immediate_BFXIL_BFM_32M_bitfield_lsb_32_immr:
		immr := (x >> 16) & (1<<6 - 1)
		if immr > 31 {
			return nil
		}
		return Imm{immr, true}

	case arg_immediate_BFXIL_BFM_32M_bitfield_width_32_imms:
		immr := (x >> 16) & (1<<6 - 1)
		imms := (x >> 10) & (1<<6 - 1)
		width := imms - immr + 1
		if width < 1 || width > 32-immr {
			return nil
		}
		return Imm{width, true}

	case arg_immediate_BFXIL_BFM_64M_bitfield_lsb_64_immr:
		immr := (x >> 16) & (1<<6 - 1)
		return Imm{immr, true}

	case arg_immediate_BFXIL_BFM_64M_bitfield_width_64_imms:
		immr := (x >> 16) & (1<<6 - 1)
		imms := (x >> 10) & (1<<6 - 1)
		width := imms - immr + 1
		if width < 1 || width > 64-immr {
			return nil
		}
		return Imm{width, true}

	case arg_immediate_bitmask_32_imms_immr:
		return handle_bitmasks(x, 32)

	case arg_immediate_bitmask_64_N_imms_immr:
		return handle_bitmasks(x, 64)

	case arg_immediate_LSL_UBFM_32M_bitfield_0_31_immr:
		imms := (x >> 10) & (1<<6 - 1)
		shift := 31 - imms
		if shift > 31 {
			return nil
		}
		return Imm{shift, true}

	case arg_immediate_LSL_UBFM_64M_bitfield_0_63_immr:
		imms := (x >> 10) & (1<<6 - 1)
		shift := 63 - imms
		if shift > 63 {
			return nil
		}
		return Imm{shift, true}

	case arg_immediate_LSR_UBFM_32M_bitfield_0_31_immr:
		immr := (x >> 16) & (1<<6 - 1)
		if immr > 31 {
			return nil
		}
		return Imm{immr, true}

	case arg_immediate_LSR_UBFM_64M_bitfield_0_63_immr:
		immr := (x >> 16) & (1<<6 - 1)
		return Imm{immr, true}

	case arg_immediate_optional_0_15_CRm:
		crm := (x >> 8) & (1<<4 - 1)
		return Imm_clrex(crm)

	case arg_immediate_optional_0_65535_imm16:
		imm16 := (x >> 5) & (1<<16 - 1)
		return Imm_dcps(imm16)

	case arg_immediate_OptLSL_amount_16_0_16:
		imm16 := (x >> 5) & (1<<16 - 1)
		hw := (x >> 21) & (1<<2 - 1)
		shift := hw * 16
		if shift > 16 {
			return nil
		}
		return ImmShift{uint16(imm16), uint8(shift)}

	case arg_immediate_OptLSL_amount_16_0_48:
		imm16 := (x >> 5) & (1<<16 - 1)
		hw := (x >> 21) & (1<<2 - 1)
		shift := hw * 16
		return ImmShift{uint16(imm16), uint8(shift)}

	case arg_immediate_SBFIZ_SBFM_32M_bitfield_lsb_32_immr:
		immr := (x >> 16) & (1<<6 - 1)
		if immr > 31 {
			return nil
		}
		return Imm{32 - immr, true}

	case arg_immediate_SBFIZ_SBFM_32M_bitfield_width_32_imms:
		imms := (x >> 10) & (1<<6 - 1)
		if imms > 31 {
			return nil
		}
		return Imm{imms + 1, true}

	case arg_immediate_SBFIZ_SBFM_64M_bitfield_lsb_64_immr:
		immr := (x >> 16) & (1<<6 - 1)
		return Imm{64 - immr, true}

	case arg_immediate_SBFIZ_SBFM_64M_bitfield_width_64_imms:
		imms := (x >> 10) & (1<<6 - 1)
		return Imm{imms + 1, true}

	case arg_immediate_SBFX_SBFM_32M_bitfield_lsb_32_immr:
		immr := (x >> 16) & (1<<6 - 1)
		if immr > 31 {
			return nil
		}
		return Imm{immr, true}

	case arg_immediate_SBFX_SBFM_32M_bitfield_width_32_imms:
		immr := (x >> 16) & (1<<6 - 1)
		imms := (x >> 10) & (1<<6 - 1)
		width := imms - immr + 1
		if width < 1 || width > 32-immr {
			return nil
		}
		return Imm{width, true}

	case arg_immediate_SBFX_SBFM_64M_bitfield_lsb_64_immr:
		immr := (x >> 16) & (1<<6 - 1)
		return Imm{immr, true}

	case arg_immediate_SBFX_SBFM_64M_bitfield_width_64_imms:
		immr := (x >> 16) & (1<<6 - 1)
		imms := (x >> 10) & (1<<6 - 1)
		width := imms - immr + 1
		if width < 1 || width > 64-immr {
			return nil
		}
		return Imm{width, true}

	case arg_immediate_shift_32_implicit_imm16_hw:
		imm16 := (x >> 5) & (1<<16 - 1)
		hw := (x >> 21) & (1<<2 - 1)
		shift := hw * 16
		if shift > 16 {
			return nil
		}
		result := uint32(imm16) << shift
		return Imm{result, false}

	case arg_immediate_shift_32_implicit_inverse_imm16_hw:
		imm16 := (x >> 5) & (1<<16 - 1)
		hw := (x >> 21) & (1<<2 - 1)
		shift := hw * 16
		if shift > 16 {
			return nil
		}
		result := uint32(imm16) << shift
		return Imm{^result, false}

	case arg_immediate_shift_64_implicit_imm16_hw:
		imm16 := (x >> 5) & (1<<16 - 1)
		hw := (x >> 21) & (1<<2 - 1)
		shift := hw * 16
		result := uint64(imm16) << shift
		return Imm64{result, false}

	case arg_immediate_shift_64_implicit_inverse_imm16_hw:
		imm16 := (x >> 5) & (1<<16 - 1)
		hw := (x >> 21) & (1<<2 - 1)
		shift := hw * 16
		result := uint64(imm16) << shift
		return Imm64{^result, false}

	case arg_immediate_UBFIZ_UBFM_32M_bitfield_lsb_32_immr:
		immr := (x >> 16) & (1<<6 - 1)
		if immr > 31 {
			return nil
		}
		return Imm{32 - immr, true}

	case arg_immediate_UBFIZ_UBFM_32M_bitfield_width_32_imms:
		imms := (x >> 10) & (1<<6 - 1)
		if imms > 31 {
			return nil
		}
		return Imm{imms + 1, true}

	case arg_immediate_UBFIZ_UBFM_64M_bitfield_lsb_64_immr:
		immr := (x >> 16) & (1<<6 - 1)
		return Imm{64 - immr, true}

	case arg_immediate_UBFIZ_UBFM_64M_bitfield_width_64_imms:
		imms := (x >> 10) & (1<<6 - 1)
		return Imm{imms + 1, true}

	case arg_immediate_UBFX_UBFM_32M_bitfield_lsb_32_immr:
		immr := (x >> 16) & (1<<6 - 1)
		if immr > 31 {
			return nil
		}
		return Imm{immr, true}

	case arg_immediate_UBFX_UBFM_32M_bitfield_width_32_imms:
		immr := (x >> 16) & (1<<6 - 1)
		imms := (x >> 10) & (1<<6 - 1)
		width := imms - immr + 1
		if width < 1 || width > 32-immr {
			return nil
		}
		return Imm{width, true}

	case arg_immediate_UBFX_UBFM_64M_bitfield_lsb_64_immr:
		immr := (x >> 16) & (1<<6 - 1)
		return Imm{immr, true}

	case arg_immediate_UBFX_UBFM_64M_bitfield_width_64_imms:
		immr := (x >> 16) & (1<<6 - 1)
		imms := (x >> 10) & (1<<6 - 1)
		width := imms - immr + 1
		if width < 1 || width > 64-immr {
			return nil
		}
		return Imm{width, true}

	case arg_Rt_31_1__W_0__X_1:
		b5 := (x >> 31) & 1
		Rt := x & (1<<5 - 1)
		if b5 == 0 {
			return W0 + Reg(Rt)
		} else {
			return X0 + Reg(Rt)
		}

	case arg_cond_AllowALNV_Normal:
		cond := (x >> 12) & (1<<4 - 1)
		return Cond{uint8(cond), false}

	case arg_conditional:
		cond := x & (1<<4 - 1)
		return Cond{uint8(cond), false}

	case arg_cond_NotAllowALNV_Invert:
		cond := (x >> 12) & (1<<4 - 1)
		if (cond >> 1) == 7 {
			return nil
		}
		return Cond{uint8(cond), true}

	case arg_Cm:
		CRm := (x >> 8) & (1<<4 - 1)
		return Imm_c(CRm)

	case arg_Cn:
		CRn := (x >> 12) & (1<<4 - 1)
		return Imm_c(CRn)

	case arg_option_DMB_BO_system_CRm:
		CRm := (x >> 8) & (1<<4 - 1)
		return Imm_option(CRm)

	case arg_option_DSB_BO_system_CRm:
		CRm := (x >> 8) & (1<<4 - 1)
		return Imm_option(CRm)

	case arg_option_ISB_BI_system_CRm:
		CRm := (x >> 8) & (1<<4 - 1)
		if CRm == 15 {
			return Imm_option(CRm)
		}
		return Imm{CRm, false}

	case arg_prfop_Rt:
		Rt := x & (1<<5 - 1)
		return Imm_prfop(Rt)

	case arg_pstatefield_op1_op2__SPSel_05__DAIFSet_36__DAIFClr_37:
		op1 := (x >> 16) & (1<<3 - 1)
		op2 := (x >> 5) & (1<<3 - 1)
		if (op1 == 0) && (op2 == 5) {
			return SPSel
		} else if (op1 == 3) && (op2 == 6) {
			return DAIFSet
		} else if (op1 == 3) && (op2 == 7) {
			return DAIFClr
		}
		return nil

	case arg_sysreg_o0_op1_CRn_CRm_op2:
		op0 := (x >> 19) & (1<<2 - 1)
		op1 := (x >> 16) & (1<<3 - 1)
		CRn := (x >> 12) & (1<<4 - 1)
		CRm := (x >> 8) & (1<<4 - 1)
		op2 := (x >> 5) & (1<<3 - 1)
		return Systemreg{uint8(op0), uint8(op1), uint8(CRn), uint8(CRm), uint8(op2)}

	case arg_sysop_AT_SYS_CR_system:
		//TODO: system instruction
		return nil

	case arg_sysop_SYS_CR_system:
		//TODO: system instruction
		return nil

	case arg_sysop_DC_SYS_CR_system, arg_sysop_TLBI_SYS_CR_system:
		op1 := (x >> 16) & 7
		cn := (x >> 12) & 15
		cm := (x >> 8) & 15
		op2 := (x >> 5) & 7
		sysInst := sysInstFields{uint8(op1), uint8(cn), uint8(cm), uint8(op2)}
		attrs := sysInst.getAttrs()
		reg := int(x & 31)
		if !attrs.hasOperand2 {
			if reg == 31 {
				return sysOp{sysInst, 0, false}
			}
			// This instruction is undefined if the Rt field is not set to 31.
			return nil
		}
		return sysOp{sysInst, X0 + Reg(reg), true}

	case arg_Bt:
		return B0 + Reg(x&(1<<5-1))

	case arg_Dt:
		return D0 + Reg(x&(1<<5-1))

	case arg_Dt2:
		return D0 + Reg((x>>10)&(1<<5-1))

	case arg_Ht:
		return H0 + Reg(x&(1<<5-1))

	case arg_immediate_0_63_immh_immb__UIntimmhimmb64_8:
		immh := (x >> 19) & (1<<4 - 1)
		if (immh & 8) == 0 {
			return nil
		}
		immb := (x >> 16) & (1<<3 - 1)
		return Imm{(immh << 3) + immb - 64, true}

	case arg_immediate_0_width_immh_immb__SEEAdvancedSIMDmodifiedimmediate_0__UIntimmhimmb8_1__UIntimmhimmb16_2__UIntimmhimmb32_4:
		immh := (x >> 19) & (1<<4 - 1)
		immb := (x >> 16) & (1<<3 - 1)
		if immh == 1 {
			return Imm{(immh << 3) + immb - 8, true}
		} else if (immh >> 1) == 1 {
			return Imm{(immh << 3) + immb - 16, true}
		} else if (immh >> 2) == 1 {
			return Imm{(immh << 3) + immb - 32, true}
		} else {
			return nil
		}

	case arg_immediate_0_width_immh_immb__SEEAdvancedSIMDmodifiedimmediate_0__UIntimmhimmb8_1__UIntimmhimmb16_2__UIntimmhimmb32_4__UIntimmhimmb64_8:
		fallthrough

	case arg_immediate_0_width_m1_immh_immb__UIntimmhimmb8_1__UIntimmhimmb16_2__UIntimmhimmb32_4__UIntimmhimmb64_8:
		immh := (x >> 19) & (1<<4 - 1)
		immb := (x >> 16) & (1<<3 - 1)
		if immh == 1 {
			return Imm{(immh << 3) + immb - 8, true}
		} else if (immh >> 1) == 1 {
			return Imm{(immh << 3) + immb - 16, true}
		} else if (immh >> 2) == 1 {
			return Imm{(immh << 3) + immb - 32, true}
		} else if (immh >> 3) == 1 {
			return Imm{(immh << 3) + immb - 64, true}
		} else {
			return nil
		}

	case arg_immediate_0_width_size__8_0__16_1__32_2:
		size := (x >> 22) & (1<<2 - 1)
		switch size {
		case 0:
			return Imm{8, true}
		case 1:
			return Imm{16, true}
		case 2:
			return Imm{32, true}
		default:
			return nil
		}

	case arg_immediate_1_64_immh_immb__128UIntimmhimmb_8:
		immh := (x >> 19) & (1<<4 - 1)
		if (immh & 8) == 0 {
			return nil
		}
		immb := (x >> 16) & (1<<3 - 1)
		return Imm{128 - ((immh << 3) + immb), true}

	case arg_immediate_1_width_immh_immb__16UIntimmhimmb_1__32UIntimmhimmb_2__64UIntimmhimmb_4:
		fallthrough

	case arg_immediate_1_width_immh_immb__SEEAdvancedSIMDmodifiedimmediate_0__16UIntimmhimmb_1__32UIntimmhimmb_2__64UIntimmhimmb_4:
		immh := (x >> 19) & (1<<4 - 1)
		immb := (x >> 16) & (1<<3 - 1)
		if immh == 1 {
			return Imm{16 - ((immh << 3) + immb), true}
		} else if (immh >> 1) == 1 {
			return Imm{32 - ((immh << 3) + immb), true}
		} else if (immh >> 2) == 1 {
			return Imm{64 - ((immh << 3) + immb), true}
		} else {
			return nil
		}

	case arg_immediate_1_width_immh_immb__SEEAdvancedSIMDmodifiedimmediate_0__16UIntimmhimmb_1__32UIntimmhimmb_2__64UIntimmhimmb_4__128UIntimmhimmb_8:
		immh := (x >> 19) & (1<<4 - 1)
		immb := (x >> 16) & (1<<3 - 1)
		if immh == 1 {
			return Imm{16 - ((immh << 3) + immb), true}
		} else if (immh >> 1) == 1 {
			return Imm{32 - ((immh << 3) + immb), true}
		} else if (immh >> 2) == 1 {
			return Imm{64 - ((immh << 3) + immb), true}
		} else if (immh >> 3) == 1 {
			return Imm{128 - ((immh << 3) + immb), true}
		} else {
			return nil
		}

	case arg_immediate_8x8_a_b_c_d_e_f_g_h:
		var imm uint64
		if x&(1<<5) != 0 {
			imm = (1 << 8) - 1
		} else {
			imm = 0
		}
		if x&(1<<6) != 0 {
			imm += ((1 << 8) - 1) << 8
		}
		if x&(1<<7) != 0 {
			imm += ((1 << 8) - 1) << 16
		}
		if x&(1<<8) != 0 {
			imm += ((1 << 8) - 1) << 24
		}
		if x&(1<<9) != 0 {
			imm += ((1 << 8) - 1) << 32
		}
		if x&(1<<16) != 0 {
			imm += ((1 << 8) - 1) << 40
		}
		if x&(1<<17) != 0 {
			imm += ((1 << 8) - 1) << 48
		}
		if x&(1<<18) != 0 {
			imm += ((1 << 8) - 1) << 56
		}
		return Imm64{imm, false}

	case arg_immediate_exp_3_pre_4_a_b_c_d_e_f_g_h:
		pre := (x >> 5) & (1<<4 - 1)
		exp := 1 - ((x >> 17) & 1)
		exp = (exp << 2) + (((x >> 16) & 1) << 1) + ((x >> 9) & 1)
		s := ((x >> 18) & 1)
		return Imm_fp{uint8(s), int8(exp) - 3, uint8(pre)}

	case arg_immediate_exp_3_pre_4_imm8:
		pre := (x >> 13) & (1<<4 - 1)
		exp := 1 - ((x >> 19) & 1)
		exp = (exp << 2) + ((x >> 17) & (1<<2 - 1))
		s := ((x >> 20) & 1)
		return Imm_fp{uint8(s), int8(exp) - 3, uint8(pre)}

	case arg_immediate_fbits_min_1_max_0_sub_0_immh_immb__64UIntimmhimmb_4__128UIntimmhimmb_8:
		fallthrough

	case arg_immediate_fbits_min_1_max_0_sub_0_immh_immb__SEEAdvancedSIMDmodifiedimmediate_0__64UIntimmhimmb_4__128UIntimmhimmb_8:
		immh := (x >> 19) & (1<<4 - 1)
		immb := (x >> 16) & (1<<3 - 1)
		if (immh >> 2) == 1 {
			return Imm{64 - ((immh << 3) + immb), true}
		} else if (immh >> 3) == 1 {
			return Imm{128 - ((immh << 3) + immb), true}
		} else {
			return nil
		}

	case arg_immediate_fbits_min_1_max_32_sub_64_scale:
		scale := (x >> 10) & (1<<6 - 1)
		fbits := 64 - scale
		if fbits > 32 {
			return nil
		}
		return Imm{fbits, true}

	case arg_immediate_fbits_min_1_max_64_sub_64_scale:
		scale := (x >> 10) & (1<<6 - 1)
		fbits := 64 - scale
		return Imm{fbits, true}

	case arg_immediate_floatzero:
		return Imm{0, true}

	case arg_immediate_index_Q_imm4__imm4lt20gt_00__imm4_10:
		Q := (x >> 30) & 1
		imm4 := (x >> 11) & (1<<4 - 1)
		if Q == 1 || (imm4>>3) == 0 {
			return Imm{imm4, true}
		} else {
			return nil
		}

	case arg_immediate_MSL__a_b_c_d_e_f_g_h_cmode__8_0__16_1:
		var shift uint8
		imm8 := (x >> 16) & (1<<3 - 1)
		imm8 = (imm8 << 5) | ((x >> 5) & (1<<5 - 1))
		if (x>>12)&1 == 0 {
			shift = 8 + 128
		} else {
			shift = 16 + 128
		}
		return ImmShift{uint16(imm8), shift}

	case arg_immediate_OptLSL__a_b_c_d_e_f_g_h_cmode__0_0__8_1:
		imm8 := (x >> 16) & (1<<3 - 1)
		imm8 = (imm8 << 5) | ((x >> 5) & (1<<5 - 1))
		cmode1 := (x >> 13) & 1
		shift := 8 * cmode1
		return ImmShift{uint16(imm8), uint8(shift)}

	case arg_immediate_OptLSL__a_b_c_d_e_f_g_h_cmode__0_0__8_1__16_2__24_3:
		imm8 := (x >> 16) & (1<<3 - 1)
		imm8 = (imm8 << 5) | ((x >> 5) & (1<<5 - 1))
		cmode1 := (x >> 13) & (1<<2 - 1)
		shift := 8 * cmode1
		return ImmShift{uint16(imm8), uint8(shift)}

	case arg_immediate_OptLSLZero__a_b_c_d_e_f_g_h:
		imm8 := (x >> 16) & (1<<3 - 1)
		imm8 = (imm8 << 5) | ((x >> 5) & (1<<5 - 1))
		return ImmShift{uint16(imm8), 0}

	case arg_immediate_zero:
		return Imm{0, true}

	case arg_Qd:
		return Q0 + Reg(x&(1<<5-1))

	case arg_Qn:
		return Q0 + Reg((x>>5)&(1<<5-1))

	case arg_Qt:
		return Q0 + Reg(x&(1<<5-1))

	case arg_Qt2:
		return Q0 + Reg((x>>10)&(1<<5-1))

	case arg_Rn_16_5__W_1__W_2__W_4__X_8:
		imm5 := (x >> 16) & (1<<5 - 1)
		if ((imm5 & 1) == 1) || ((imm5 & 2) == 2) || ((imm5 & 4) == 4) {
			return W0 + Reg((x>>5)&(1<<5-1))
		} else if (imm5 & 8) == 8 {
			return X0 + Reg((x>>5)&(1<<5-1))
		} else {
			return nil
		}

	case arg_St:
		return S0 + Reg(x&(1<<5-1))

	case arg_St2:
		return S0 + Reg((x>>10)&(1<<5-1))

	case arg_Vd_16_5__B_1__H_2__S_4__D_8:
		imm5 := (x >> 16) & (1<<5 - 1)
		Rd := x & (1<<5 - 1)
		if imm5&1 == 1 {
			return B0 + Reg(Rd)
		} else if imm5&2 == 2 {
			return H0 + Reg(Rd)
		} else if imm5&4 == 4 {
			return S0 + Reg(Rd)
		} else if imm5&8 == 8 {
			return D0 + Reg(Rd)
		} else {
			return nil
		}

	case arg_Vd_19_4__B_1__H_2__S_4:
		immh := (x >> 19) & (1<<4 - 1)
		Rd := x & (1<<5 - 1)
		if immh == 1 {
			return B0 + Reg(Rd)
		} else if immh>>1 == 1 {
			return H0 + Reg(Rd)
		} else if immh>>2 == 1 {
			return S0 + Reg(Rd)
		} else {
			return nil
		}

	case arg_Vd_19_4__B_1__H_2__S_4__D_8:
		immh := (x >> 19) & (1<<4 - 1)
		Rd := x & (1<<5 - 1)
		if immh == 1 {
			return B0 + Reg(Rd)
		} else if immh>>1 == 1 {
			return H0 + Reg(Rd)
		} else if immh>>2 == 1 {
			return S0 + Reg(Rd)
		} else if immh>>3 == 1 {
			return D0 + Reg(Rd)
		} else {
			return nil
		}

	case arg_Vd_19_4__D_8:
		immh := (x >> 19) & (1<<4 - 1)
		Rd := x & (1<<5 - 1)
		if immh>>3 == 1 {
			return D0 + Reg(Rd)
		} else {
			return nil
		}

	case arg_Vd_19_4__S_4__D_8:
		immh := (x >> 19) & (1<<4 - 1)
		Rd := x & (1<<5 - 1)
		if immh>>2 == 1 {
			return S0 + Reg(Rd)
		} else if immh>>3 == 1 {
			return D0 + Reg(Rd)
		} else {
			return nil
		}

	case arg_Vd_22_1__S_0:
		sz := (x >> 22) & 1
		Rd := x & (1<<5 - 1)
		if sz == 0 {
			return S0 + Reg(Rd)
		} else {
			return nil
		}

	case arg_Vd_22_1__S_0__D_1:
		sz := (x >> 22) & 1
		Rd := x & (1<<5 - 1)
		if sz == 0 {
			return S0 + Reg(Rd)
		} else {
			return D0 + Reg(Rd)
		}

	case arg_Vd_22_1__S_1:
		sz := (x >> 22) & 1
		Rd := x & (1<<5 - 1)
		if sz == 1 {
			return S0 + Reg(Rd)
		} else {
			return nil
		}

	case arg_Vd_22_2__B_0__H_1__S_2:
		size := (x >> 22) & (1<<2 - 1)
		Rd := x & (1<<5 - 1)
		if size == 0 {
			return B0 + Reg(Rd)
		} else if size == 1 {
			return H0 + Reg(Rd)
		} else if size == 2 {
			return S0 + Reg(Rd)
		} else {
			return nil
		}

	case arg_Vd_22_2__B_0__H_1__S_2__D_3:
		size := (x >> 22) & (1<<2 - 1)
		Rd := x & (1<<5 - 1)
		if size == 0 {
			return B0 + Reg(Rd)
		} else if size == 1 {
			return H0 + Reg(Rd)
		} else if size == 2 {
			return S0 + Reg(Rd)
		} else {
			return D0 + Reg(Rd)
		}

	case arg_Vd_22_2__D_3:
		size := (x >> 22) & (1<<2 - 1)
		Rd := x & (1<<5 - 1)
		if size == 3 {
			return D0 + Reg(Rd)
		} else {
			return nil
		}

	case arg_Vd_22_2__H_0__S_1__D_2:
		size := (x >> 22) & (1<<2 - 1)
		Rd := x & (1<<5 - 1)
		if size == 0 {
			return H0 + Reg(Rd)
		} else if size == 1 {
			return S0 + Reg(Rd)
		} else if size == 2 {
			return D0 + Reg(Rd)
		} else {
			return nil
		}

	case arg_Vd_22_2__H_1__S_2:
		size := (x >> 22) & (1<<2 - 1)
		Rd := x & (1<<5 - 1)
		if size == 1 {
			return H0 + Reg(Rd)
		} else if size == 2 {
			return S0 + Reg(Rd)
		} else {
			return nil
		}

	case arg_Vd_22_2__S_1__D_2:
		size := (x >> 22) & (1<<2 - 1)
		Rd := x & (1<<5 - 1)
		if size == 1 {
			return S0 + Reg(Rd)
		} else if size == 2 {
			return D0 + Reg(Rd)
		} else {
			return nil
		}

	case arg_Vd_arrangement_16B:
		Rd := x & (1<<5 - 1)
		return RegisterWithArrangement{V0 + Reg(Rd), Arrangement16B, 0}

	case arg_Vd_arrangement_2D:
		Rd := x & (1<<5 - 1)
		return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2D, 0}

	case arg_Vd_arrangement_4S:
		Rd := x & (1<<5 - 1)
		return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4S, 0}

	case arg_Vd_arrangement_D_index__1:
		Rd := x & (1<<5 - 1)
		return RegisterWithArrangementAndIndex{V0 + Reg(Rd), ArrangementD, 1, 0}

	case arg_Vd_arrangement_imm5___B_1__H_2__S_4__D_8_index__imm5__imm5lt41gt_1__imm5lt42gt_2__imm5lt43gt_4__imm5lt4gt_8_1:
		var a Arrangement
		var index uint32
		Rd := x & (1<<5 - 1)
		imm5 := (x >> 16) & (1<<5 - 1)
		if imm5&1 == 1 {
			a = ArrangementB
			index = imm5 >> 1
		} else if imm5&2 == 2 {
			a = ArrangementH
			index = imm5 >> 2
		} else if imm5&4 == 4 {
			a = ArrangementS
			index = imm5 >> 3
		} else if imm5&8 == 8 {
			a = ArrangementD
			index = imm5 >> 4
		} else {
			return nil
		}
		return RegisterWithArrangementAndIndex{V0 + Reg(Rd), a, uint8(index), 0}

	case arg_Vd_arrangement_imm5_Q___8B_10__16B_11__4H_20__8H_21__2S_40__4S_41__2D_81:
		Rd := x & (1<<5 - 1)
		imm5 := (x >> 16) & (1<<5 - 1)
		Q := (x >> 30) & 1
		if imm5&1 == 1 {
			if Q == 0 {
				return RegisterWithArrangement{V0 + Reg(Rd), Arrangement8B, 0}
			} else {
				return RegisterWithArrangement{V0 + Reg(Rd), Arrangement16B, 0}
			}
		} else if imm5&2 == 2 {
			if Q == 0 {
				return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4H, 0}
			} else {
				return RegisterWithArrangement{V0 + Reg(Rd), Arrangement8H, 0}
			}
		} else if imm5&4 == 4 {
			if Q == 0 {
				return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2S, 0}
			} else {
				return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4S, 0}
			}
		} else if (imm5&8 == 8) && (Q == 1) {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2D, 0}
		} else {
			return nil
		}

	case arg_Vd_arrangement_immh_Q___SEEAdvancedSIMDmodifiedimmediate_00__2S_40__4S_41__2D_81:
		Rd := x & (1<<5 - 1)
		immh := (x >> 19) & (1<<4 - 1)
		Q := (x >> 30) & 1
		if immh>>2 == 1 {
			if Q == 0 {
				return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2S, 0}
			} else {
				return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4S, 0}
			}
		} else if immh>>3 == 1 {
			if Q == 1 {
				return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2D, 0}
			}
		}
		return nil

	case arg_Vd_arrangement_immh_Q___SEEAdvancedSIMDmodifiedimmediate_00__8B_10__16B_11__4H_20__8H_21__2S_40__4S_41:
		Rd := x & (1<<5 - 1)
		immh := (x >> 19) & (1<<4 - 1)
		Q := (x >> 30) & 1
		if immh == 1 {
			if Q == 0 {
				return RegisterWithArrangement{V0 + Reg(Rd), Arrangement8B, 0}
			} else {
				return RegisterWithArrangement{V0 + Reg(Rd), Arrangement16B, 0}
			}
		} else if immh>>1 == 1 {
			if Q == 0 {
				return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4H, 0}
			} else {
				return RegisterWithArrangement{V0 + Reg(Rd), Arrangement8H, 0}
			}
		} else if immh>>2 == 1 {
			if Q == 0 {
				return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2S, 0}
			} else {
				return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4S, 0}
			}
		}
		return nil

	case arg_Vd_arrangement_immh_Q___SEEAdvancedSIMDmodifiedimmediate_00__8B_10__16B_11__4H_20__8H_21__2S_40__4S_41__2D_81:
		Rd := x & (1<<5 - 1)
		immh := (x >> 19) & (1<<4 - 1)
		Q := (x >> 30) & 1
		if immh == 1 {
			if Q == 0 {
				return RegisterWithArrangement{V0 + Reg(Rd), Arrangement8B, 0}
			} else {
				return RegisterWithArrangement{V0 + Reg(Rd), Arrangement16B, 0}
			}
		} else if immh>>1 == 1 {
			if Q == 0 {
				return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4H, 0}
			} else {
				return RegisterWithArrangement{V0 + Reg(Rd), Arrangement8H, 0}
			}
		} else if immh>>2 == 1 {
			if Q == 0 {
				return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2S, 0}
			} else {
				return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4S, 0}
			}
		} else if immh>>3 == 1 {
			if Q == 1 {
				return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2D, 0}
			}
		}
		return nil

	case arg_Vd_arrangement_immh___SEEAdvancedSIMDmodifiedimmediate_0__8H_1__4S_2__2D_4:
		Rd := x & (1<<5 - 1)
		immh := (x >> 19) & (1<<4 - 1)
		if immh == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement8H, 0}
		} else if immh>>1 == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4S, 0}
		} else if immh>>2 == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2D, 0}
		}
		return nil

	case arg_Vd_arrangement_Q___2S_0__4S_1:
		Rd := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		if Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2S, 0}
		} else {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4S, 0}
		}

	case arg_Vd_arrangement_Q___4H_0__8H_1:
		Rd := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		if Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4H, 0}
		} else {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement8H, 0}
		}

	case arg_Vd_arrangement_Q___8B_0__16B_1:
		Rd := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		if Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement8B, 0}
		} else {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement16B, 0}
		}

	case arg_Vd_arrangement_Q_sz___2S_00__4S_10__2D_11:
		Rd := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		sz := (x >> 22) & 1
		if sz == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2S, 0}
		} else if sz == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4S, 0}
		} else if sz == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2D, 0}
		}
		return nil

	case arg_Vd_arrangement_size___4S_1__2D_2:
		Rd := x & (1<<5 - 1)
		size := (x >> 22) & 3
		if size == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4S, 0}
		} else if size == 2 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2D, 0}
		}
		return nil

	case arg_Vd_arrangement_size___8H_0__1Q_3:
		Rd := x & (1<<5 - 1)
		size := (x >> 22) & 3
		if size == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement8H, 0}
		} else if size == 3 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement1Q, 0}
		}
		return nil

	case arg_Vd_arrangement_size___8H_0__4S_1__2D_2:
		Rd := x & (1<<5 - 1)
		size := (x >> 22) & 3
		if size == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement8H, 0}
		} else if size == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4S, 0}
		} else if size == 2 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2D, 0}
		}
		return nil

	case arg_Vd_arrangement_size_Q___4H_00__8H_01__2S_10__4S_11__1D_20__2D_21:
		Rd := x & (1<<5 - 1)
		size := (x >> 22) & 3
		Q := (x >> 30) & 1
		if size == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4H, 0}
		} else if size == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement8H, 0}
		} else if size == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2S, 0}
		} else if size == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4S, 0}
		} else if size == 2 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement1D, 0}
		} else if size == 2 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2D, 0}
		}
		return nil

	case arg_Vd_arrangement_size_Q___4H_10__8H_11__2S_20__4S_21:
		Rd := x & (1<<5 - 1)
		size := (x >> 22) & 3
		Q := (x >> 30) & 1
		if size == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4H, 0}
		} else if size == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement8H, 0}
		} else if size == 2 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2S, 0}
		} else if size == 2 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4S, 0}
		}
		return nil

	case arg_Vd_arrangement_size_Q___8B_00__16B_01:
		Rd := x & (1<<5 - 1)
		size := (x >> 22) & 3
		Q := (x >> 30) & 1
		if size == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement8B, 0}
		} else if size == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement16B, 0}
		}
		return nil

	case arg_Vd_arrangement_size_Q___8B_00__16B_01__4H_10__8H_11:
		Rd := x & (1<<5 - 1)
		size := (x >> 22) & 3
		Q := (x >> 30) & 1
		if size == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement8B, 0}
		} else if size == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement16B, 0}
		} else if size == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4H, 0}
		} else if size == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement8H, 0}
		}
		return nil

	case arg_Vd_arrangement_size_Q___8B_00__16B_01__4H_10__8H_11__2S_20__4S_21:
		Rd := x & (1<<5 - 1)
		size := (x >> 22) & 3
		Q := (x >> 30) & 1
		if size == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement8B, 0}
		} else if size == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement16B, 0}
		} else if size == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4H, 0}
		} else if size == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement8H, 0}
		} else if size == 2 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2S, 0}
		} else if size == 2 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4S, 0}
		}
		return nil

	case arg_Vd_arrangement_size_Q___8B_00__16B_01__4H_10__8H_11__2S_20__4S_21__2D_31:
		Rd := x & (1<<5 - 1)
		size := (x >> 22) & 3
		Q := (x >> 30) & 1
		if size == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement8B, 0}
		} else if size == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement16B, 0}
		} else if size == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4H, 0}
		} else if size == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement8H, 0}
		} else if size == 2 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2S, 0}
		} else if size == 2 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4S, 0}
		} else if size == 3 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2D, 0}
		}
		return nil

	case arg_Vd_arrangement_sz___4S_0__2D_1:
		Rd := x & (1<<5 - 1)
		sz := (x >> 22) & 1
		if sz == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4S, 0}
		} else {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2D, 0}
		}

	case arg_Vd_arrangement_sz_Q___2S_00__4S_01:
		Rd := x & (1<<5 - 1)
		sz := (x >> 22) & 1
		Q := (x >> 30) & 1
		if sz == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2S, 0}
		} else if sz == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4S, 0}
		}
		return nil

	case arg_Vd_arrangement_sz_Q___2S_00__4S_01__2D_11:
		Rd := x & (1<<5 - 1)
		sz := (x >> 22) & 1
		Q := (x >> 30) & 1
		if sz == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2S, 0}
		} else if sz == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4S, 0}
		} else if sz == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2D, 0}
		}
		return nil

	case arg_Vd_arrangement_sz_Q___2S_10__4S_11:
		Rd := x & (1<<5 - 1)
		sz := (x >> 22) & 1
		Q := (x >> 30) & 1
		if sz == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2S, 0}
		} else if sz == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4S, 0}
		}
		return nil

	case arg_Vd_arrangement_sz_Q___4H_00__8H_01__2S_10__4S_11:
		Rd := x & (1<<5 - 1)
		sz := (x >> 22) & 1
		Q := (x >> 30) & 1
		if sz == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4H, 0}
		} else if sz == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement8H, 0}
		} else if sz == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement2S, 0}
		} else /* sz == 1 && Q == 1 */ {
			return RegisterWithArrangement{V0 + Reg(Rd), Arrangement4S, 0}
		}

	case arg_Vm_22_1__S_0__D_1:
		sz := (x >> 22) & 1
		Rm := (x >> 16) & (1<<5 - 1)
		if sz == 0 {
			return S0 + Reg(Rm)
		} else {
			return D0 + Reg(Rm)
		}

	case arg_Vm_22_2__B_0__H_1__S_2__D_3:
		size := (x >> 22) & (1<<2 - 1)
		Rm := (x >> 16) & (1<<5 - 1)
		if size == 0 {
			return B0 + Reg(Rm)
		} else if size == 1 {
			return H0 + Reg(Rm)
		} else if size == 2 {
			return S0 + Reg(Rm)
		} else {
			return D0 + Reg(Rm)
		}

	case arg_Vm_22_2__D_3:
		size := (x >> 22) & (1<<2 - 1)
		Rm := (x >> 16) & (1<<5 - 1)
		if size == 3 {
			return D0 + Reg(Rm)
		} else {
			return nil
		}

	case arg_Vm_22_2__H_1__S_2:
		size := (x >> 22) & (1<<2 - 1)
		Rm := (x >> 16) & (1<<5 - 1)
		if size == 1 {
			return H0 + Reg(Rm)
		} else if size == 2 {
			return S0 + Reg(Rm)
		} else {
			return nil
		}

	case arg_Vm_arrangement_4S:
		Rm := (x >> 16) & (1<<5 - 1)
		return RegisterWithArrangement{V0 + Reg(Rm), Arrangement4S, 0}

	case arg_Vm_arrangement_Q___8B_0__16B_1:
		Rm := (x >> 16) & (1<<5 - 1)
		Q := (x >> 30) & 1
		if Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement8B, 0}
		} else {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement16B, 0}
		}

	case arg_Vm_arrangement_size___8H_0__4S_1__2D_2:
		Rm := (x >> 16) & (1<<5 - 1)
		size := (x >> 22) & 3
		if size == 0 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement8H, 0}
		} else if size == 1 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement4S, 0}
		} else if size == 2 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement2D, 0}
		}
		return nil

	case arg_Vm_arrangement_size___H_1__S_2_index__size_L_H_M__HLM_1__HL_2_1:
		var a Arrangement
		var index uint32
		var vm uint32
		Rm := (x >> 16) & (1<<4 - 1)
		size := (x >> 22) & 3
		H := (x >> 11) & 1
		L := (x >> 21) & 1
		M := (x >> 20) & 1
		if size == 1 {
			a = ArrangementH
			index = (H << 2) | (L << 1) | M
			vm = Rm
		} else if size == 2 {
			a = ArrangementS
			index = (H << 1) | L
			vm = (M << 4) | Rm
		} else {
			return nil
		}
		return RegisterWithArrangementAndIndex{V0 + Reg(vm), a, uint8(index), 0}

	case arg_Vm_arrangement_size_Q___4H_10__8H_11__2S_20__4S_21:
		Rm := (x >> 16) & (1<<5 - 1)
		size := (x >> 22) & 3
		Q := (x >> 30) & 1
		if size == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement4H, 0}
		} else if size == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement8H, 0}
		} else if size == 2 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement2S, 0}
		} else if size == 2 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement4S, 0}
		}
		return nil

	case arg_Vm_arrangement_size_Q___8B_00__16B_01:
		Rm := (x >> 16) & (1<<5 - 1)
		size := (x >> 22) & 3
		Q := (x >> 30) & 1
		if size == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement8B, 0}
		} else if size == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement16B, 0}
		}
		return nil

	case arg_Vm_arrangement_size_Q___8B_00__16B_01__1D_30__2D_31:
		Rm := (x >> 16) & (1<<5 - 1)
		size := (x >> 22) & 3
		Q := (x >> 30) & 1
		if size == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement8B, 0}
		} else if size == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement16B, 0}
		} else if size == 3 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement1D, 0}
		} else if size == 3 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement2D, 0}
		}
		return nil

	case arg_Vm_arrangement_size_Q___8B_00__16B_01__4H_10__8H_11__2S_20__4S_21:
		Rm := (x >> 16) & (1<<5 - 1)
		size := (x >> 22) & 3
		Q := (x >> 30) & 1
		if size == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement8B, 0}
		} else if size == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement16B, 0}
		} else if size == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement4H, 0}
		} else if size == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement8H, 0}
		} else if size == 2 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement2S, 0}
		} else if size == 2 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement4S, 0}
		}
		return nil

	case arg_Vm_arrangement_size_Q___8B_00__16B_01__4H_10__8H_11__2S_20__4S_21__2D_31:
		Rm := (x >> 16) & (1<<5 - 1)
		size := (x >> 22) & 3
		Q := (x >> 30) & 1
		if size == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement8B, 0}
		} else if size == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement16B, 0}
		} else if size == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement4H, 0}
		} else if size == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement8H, 0}
		} else if size == 2 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement2S, 0}
		} else if size == 2 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement4S, 0}
		} else if size == 3 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement2D, 0}
		}
		return nil

	case arg_Vm_arrangement_sz_Q___2S_00__4S_01__2D_11:
		Rm := (x >> 16) & (1<<5 - 1)
		sz := (x >> 22) & 1
		Q := (x >> 30) & 1
		if sz == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement2S, 0}
		} else if sz == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement4S, 0}
		} else if sz == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rm), Arrangement2D, 0}
		}
		return nil

	case arg_Vm_arrangement_sz___S_0__D_1_index__sz_L_H__HL_00__H_10_1:
		var a Arrangement
		var index uint32
		Rm := (x >> 16) & (1<<5 - 1)
		sz := (x >> 22) & 1
		H := (x >> 11) & 1
		L := (x >> 21) & 1
		if sz == 0 {
			a = ArrangementS
			index = (H << 1) | L
		} else if sz == 1 && L == 0 {
			a = ArrangementD
			index = H
		} else {
			return nil
		}
		return RegisterWithArrangementAndIndex{V0 + Reg(Rm), a, uint8(index), 0}

	case arg_Vn_19_4__B_1__H_2__S_4__D_8:
		immh := (x >> 19) & (1<<4 - 1)
		Rn := (x >> 5) & (1<<5 - 1)
		if immh == 1 {
			return B0 + Reg(Rn)
		} else if immh>>1 == 1 {
			return H0 + Reg(Rn)
		} else if immh>>2 == 1 {
			return S0 + Reg(Rn)
		} else if immh>>3 == 1 {
			return D0 + Reg(Rn)
		} else {
			return nil
		}

	case arg_Vn_19_4__D_8:
		immh := (x >> 19) & (1<<4 - 1)
		Rn := (x >> 5) & (1<<5 - 1)
		if immh>>3 == 1 {
			return D0 + Reg(Rn)
		} else {
			return nil
		}

	case arg_Vn_19_4__H_1__S_2__D_4:
		immh := (x >> 19) & (1<<4 - 1)
		Rn := (x >> 5) & (1<<5 - 1)
		if immh == 1 {
			return H0 + Reg(Rn)
		} else if immh>>1 == 1 {
			return S0 + Reg(Rn)
		} else if immh>>2 == 1 {
			return D0 + Reg(Rn)
		} else {
			return nil
		}

	case arg_Vn_19_4__S_4__D_8:
		immh := (x >> 19) & (1<<4 - 1)
		Rn := (x >> 5) & (1<<5 - 1)
		if immh>>2 == 1 {
			return S0 + Reg(Rn)
		} else if immh>>3 == 1 {
			return D0 + Reg(Rn)
		} else {
			return nil
		}

	case arg_Vn_1_arrangement_16B:
		Rn := (x >> 5) & (1<<5 - 1)
		return RegisterWithArrangement{V0 + Reg(Rn), Arrangement16B, 1}

	case arg_Vn_22_1__D_1:
		sz := (x >> 22) & 1
		Rn := (x >> 5) & (1<<5 - 1)
		if sz == 1 {
			return D0 + Reg(Rn)
		}
		return nil

	case arg_Vn_22_1__S_0__D_1:
		sz := (x >> 22) & 1
		Rn := (x >> 5) & (1<<5 - 1)
		if sz == 0 {
			return S0 + Reg(Rn)
		} else {
			return D0 + Reg(Rn)
		}

	case arg_Vn_22_2__B_0__H_1__S_2__D_3:
		size := (x >> 22) & (1<<2 - 1)
		Rn := (x >> 5) & (1<<5 - 1)
		if size == 0 {
			return B0 + Reg(Rn)
		} else if size == 1 {
			return H0 + Reg(Rn)
		} else if size == 2 {
			return S0 + Reg(Rn)
		} else {
			return D0 + Reg(Rn)
		}

	case arg_Vn_22_2__D_3:
		size := (x >> 22) & (1<<2 - 1)
		Rn := (x >> 5) & (1<<5 - 1)
		if size == 3 {
			return D0 + Reg(Rn)
		} else {
			return nil
		}

	case arg_Vn_22_2__H_0__S_1__D_2:
		size := (x >> 22) & (1<<2 - 1)
		Rn := (x >> 5) & (1<<5 - 1)
		if size == 0 {
			return H0 + Reg(Rn)
		} else if size == 1 {
			return S0 + Reg(Rn)
		} else if size == 2 {
			return D0 + Reg(Rn)
		} else {
			return nil
		}

	case arg_Vn_22_2__H_1__S_2:
		size := (x >> 22) & (1<<2 - 1)
		Rn := (x >> 5) & (1<<5 - 1)
		if size == 1 {
			return H0 + Reg(Rn)
		} else if size == 2 {
			return S0 + Reg(Rn)
		} else {
			return nil
		}

	case arg_Vn_2_arrangement_16B:
		Rn := (x >> 5) & (1<<5 - 1)
		return RegisterWithArrangement{V0 + Reg(Rn), Arrangement16B, 2}

	case arg_Vn_3_arrangement_16B:
		Rn := (x >> 5) & (1<<5 - 1)
		return RegisterWithArrangement{V0 + Reg(Rn), Arrangement16B, 3}

	case arg_Vn_4_arrangement_16B:
		Rn := (x >> 5) & (1<<5 - 1)
		return RegisterWithArrangement{V0 + Reg(Rn), Arrangement16B, 4}

	case arg_Vn_arrangement_16B:
		Rn := (x >> 5) & (1<<5 - 1)
		return RegisterWithArrangement{V0 + Reg(Rn), Arrangement16B, 0}

	case arg_Vn_arrangement_4S:
		Rn := (x >> 5) & (1<<5 - 1)
		return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4S, 0}

	case arg_Vn_arrangement_D_index__1:
		Rn := (x >> 5) & (1<<5 - 1)
		return RegisterWithArrangementAndIndex{V0 + Reg(Rn), ArrangementD, 1, 0}

	case arg_Vn_arrangement_D_index__imm5_1:
		Rn := (x >> 5) & (1<<5 - 1)
		index := (x >> 20) & 1
		return RegisterWithArrangementAndIndex{V0 + Reg(Rn), ArrangementD, uint8(index), 0}

	case arg_Vn_arrangement_imm5___B_1__H_2_index__imm5__imm5lt41gt_1__imm5lt42gt_2_1:
		var a Arrangement
		var index uint32
		Rn := (x >> 5) & (1<<5 - 1)
		imm5 := (x >> 16) & (1<<5 - 1)
		if imm5&1 == 1 {
			a = ArrangementB
			index = imm5 >> 1
		} else if imm5&2 == 2 {
			a = ArrangementH
			index = imm5 >> 2
		} else {
			return nil
		}
		return RegisterWithArrangementAndIndex{V0 + Reg(Rn), a, uint8(index), 0}

	case arg_Vn_arrangement_imm5___B_1__H_2__S_4__D_8_index__imm5_imm4__imm4lt30gt_1__imm4lt31gt_2__imm4lt32gt_4__imm4lt3gt_8_1:
		var a Arrangement
		var index uint32
		Rn := (x >> 5) & (1<<5 - 1)
		imm5 := (x >> 16) & (1<<5 - 1)
		imm4 := (x >> 11) & (1<<4 - 1)
		if imm5&1 == 1 {
			a = ArrangementB
			index = imm4
		} else if imm5&2 == 2 {
			a = ArrangementH
			index = imm4 >> 1
		} else if imm5&4 == 4 {
			a = ArrangementS
			index = imm4 >> 2
		} else if imm5&8 == 8 {
			a = ArrangementD
			index = imm4 >> 3
		} else {
			return nil
		}
		return RegisterWithArrangementAndIndex{V0 + Reg(Rn), a, uint8(index), 0}

	case arg_Vn_arrangement_imm5___B_1__H_2__S_4__D_8_index__imm5__imm5lt41gt_1__imm5lt42gt_2__imm5lt43gt_4__imm5lt4gt_8_1:
		var a Arrangement
		var index uint32
		Rn := (x >> 5) & (1<<5 - 1)
		imm5 := (x >> 16) & (1<<5 - 1)
		if imm5&1 == 1 {
			a = ArrangementB
			index = imm5 >> 1
		} else if imm5&2 == 2 {
			a = ArrangementH
			index = imm5 >> 2
		} else if imm5&4 == 4 {
			a = ArrangementS
			index = imm5 >> 3
		} else if imm5&8 == 8 {
			a = ArrangementD
			index = imm5 >> 4
		} else {
			return nil
		}
		return RegisterWithArrangementAndIndex{V0 + Reg(Rn), a, uint8(index), 0}

	case arg_Vn_arrangement_imm5___B_1__H_2__S_4_index__imm5__imm5lt41gt_1__imm5lt42gt_2__imm5lt43gt_4_1:
		var a Arrangement
		var index uint32
		Rn := (x >> 5) & (1<<5 - 1)
		imm5 := (x >> 16) & (1<<5 - 1)
		if imm5&1 == 1 {
			a = ArrangementB
			index = imm5 >> 1
		} else if imm5&2 == 2 {
			a = ArrangementH
			index = imm5 >> 2
		} else if imm5&4 == 4 {
			a = ArrangementS
			index = imm5 >> 3
		} else {
			return nil
		}
		return RegisterWithArrangementAndIndex{V0 + Reg(Rn), a, uint8(index), 0}

	case arg_Vn_arrangement_imm5___D_8_index__imm5_1:
		var a Arrangement
		var index uint32
		Rn := (x >> 5) & (1<<5 - 1)
		imm5 := (x >> 16) & (1<<5 - 1)
		if imm5&15 == 8 {
			a = ArrangementD
			index = imm5 >> 4
		} else {
			return nil
		}
		return RegisterWithArrangementAndIndex{V0 + Reg(Rn), a, uint8(index), 0}

	case arg_Vn_arrangement_immh_Q___SEEAdvancedSIMDmodifiedimmediate_00__2S_40__4S_41__2D_81:
		Rn := (x >> 5) & (1<<5 - 1)
		immh := (x >> 19) & (1<<4 - 1)
		Q := (x >> 30) & 1
		if immh>>2 == 1 {
			if Q == 0 {
				return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2S, 0}
			} else {
				return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4S, 0}
			}
		} else if immh>>3 == 1 {
			if Q == 1 {
				return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2D, 0}
			}
		}
		return nil

	case arg_Vn_arrangement_immh_Q___SEEAdvancedSIMDmodifiedimmediate_00__8B_10__16B_11__4H_20__8H_21__2S_40__4S_41:
		Rn := (x >> 5) & (1<<5 - 1)
		immh := (x >> 19) & (1<<4 - 1)
		Q := (x >> 30) & 1
		if immh == 1 {
			if Q == 0 {
				return RegisterWithArrangement{V0 + Reg(Rn), Arrangement8B, 0}
			} else {
				return RegisterWithArrangement{V0 + Reg(Rn), Arrangement16B, 0}
			}
		} else if immh>>1 == 1 {
			if Q == 0 {
				return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4H, 0}
			} else {
				return RegisterWithArrangement{V0 + Reg(Rn), Arrangement8H, 0}
			}
		} else if immh>>2 == 1 {
			if Q == 0 {
				return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2S, 0}
			} else {
				return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4S, 0}
			}
		}
		return nil

	case arg_Vn_arrangement_immh_Q___SEEAdvancedSIMDmodifiedimmediate_00__8B_10__16B_11__4H_20__8H_21__2S_40__4S_41__2D_81:
		Rn := (x >> 5) & (1<<5 - 1)
		immh := (x >> 19) & (1<<4 - 1)
		Q := (x >> 30) & 1
		if immh == 1 {
			if Q == 0 {
				return RegisterWithArrangement{V0 + Reg(Rn), Arrangement8B, 0}
			} else {
				return RegisterWithArrangement{V0 + Reg(Rn), Arrangement16B, 0}
			}
		} else if immh>>1 == 1 {
			if Q == 0 {
				return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4H, 0}
			} else {
				return RegisterWithArrangement{V0 + Reg(Rn), Arrangement8H, 0}
			}
		} else if immh>>2 == 1 {
			if Q == 0 {
				return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2S, 0}
			} else {
				return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4S, 0}
			}
		} else if immh>>3 == 1 {
			if Q == 1 {
				return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2D, 0}
			}
		}
		return nil

	case arg_Vn_arrangement_immh___SEEAdvancedSIMDmodifiedimmediate_0__8H_1__4S_2__2D_4:
		Rn := (x >> 5) & (1<<5 - 1)
		immh := (x >> 19) & (1<<4 - 1)
		if immh == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement8H, 0}
		} else if immh>>1 == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4S, 0}
		} else if immh>>2 == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2D, 0}
		}
		return nil

	case arg_Vn_arrangement_Q___8B_0__16B_1:
		Rn := (x >> 5) & (1<<5 - 1)
		Q := (x >> 30) & 1
		if Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement8B, 0}
		} else {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement16B, 0}
		}

	case arg_Vn_arrangement_Q_sz___2S_00__4S_10__2D_11:
		Rn := (x >> 5) & (1<<5 - 1)
		Q := (x >> 30) & 1
		sz := (x >> 22) & 1
		if sz == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2S, 0}
		} else if sz == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4S, 0}
		} else if sz == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2D, 0}
		}
		return nil

	case arg_Vn_arrangement_Q_sz___4S_10:
		Rn := (x >> 5) & (1<<5 - 1)
		Q := (x >> 30) & 1
		sz := (x >> 22) & 1
		if sz == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4S, 0}
		}
		return nil

	case arg_Vn_arrangement_S_index__imm5__imm5lt41gt_1__imm5lt42gt_2__imm5lt43gt_4_1:
		var index uint32
		Rn := (x >> 5) & (1<<5 - 1)
		imm5 := (x >> 16) & (1<<5 - 1)
		index = imm5 >> 3
		return RegisterWithArrangementAndIndex{V0 + Reg(Rn), ArrangementS, uint8(index), 0}

	case arg_Vn_arrangement_size___2D_3:
		Rn := (x >> 5) & (1<<5 - 1)
		size := (x >> 22) & 3
		if size == 3 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2D, 0}
		}
		return nil

	case arg_Vn_arrangement_size___8H_0__4S_1__2D_2:
		Rn := (x >> 5) & (1<<5 - 1)
		size := (x >> 22) & 3
		if size == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement8H, 0}
		} else if size == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4S, 0}
		} else if size == 2 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2D, 0}
		}
		return nil

	case arg_Vn_arrangement_size_Q___4H_10__8H_11__2S_20__4S_21:
		Rn := (x >> 5) & (1<<5 - 1)
		size := (x >> 22) & 3
		Q := (x >> 30) & 1
		if size == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4H, 0}
		} else if size == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement8H, 0}
		} else if size == 2 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2S, 0}
		} else if size == 2 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4S, 0}
		}
		return nil

	case arg_Vn_arrangement_size_Q___8B_00__16B_01:
		Rn := (x >> 5) & (1<<5 - 1)
		size := (x >> 22) & 3
		Q := (x >> 30) & 1
		if size == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement8B, 0}
		} else if size == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement16B, 0}
		}
		return nil

	case arg_Vn_arrangement_size_Q___8B_00__16B_01__1D_30__2D_31:
		Rn := (x >> 5) & (1<<5 - 1)
		size := (x >> 22) & 3
		Q := (x >> 30) & 1
		if size == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement8B, 0}
		} else if size == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement16B, 0}
		} else if size == 3 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement1D, 0}
		} else if size == 3 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2D, 0}
		}
		return nil

	case arg_Vn_arrangement_size_Q___8B_00__16B_01__4H_10__8H_11:
		Rn := (x >> 5) & (1<<5 - 1)
		size := (x >> 22) & 3
		Q := (x >> 30) & 1
		if size == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement8B, 0}
		} else if size == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement16B, 0}
		} else if size == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4H, 0}
		} else if size == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement8H, 0}
		}
		return nil

	case arg_Vn_arrangement_size_Q___8B_00__16B_01__4H_10__8H_11__2S_20__4S_21:
		Rn := (x >> 5) & (1<<5 - 1)
		size := (x >> 22) & 3
		Q := (x >> 30) & 1
		if size == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement8B, 0}
		} else if size == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement16B, 0}
		} else if size == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4H, 0}
		} else if size == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement8H, 0}
		} else if size == 2 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2S, 0}
		} else if size == 2 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4S, 0}
		}
		return nil

	case arg_Vn_arrangement_size_Q___8B_00__16B_01__4H_10__8H_11__2S_20__4S_21__2D_31:
		Rn := (x >> 5) & (1<<5 - 1)
		size := (x >> 22) & 3
		Q := (x >> 30) & 1
		if size == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement8B, 0}
		} else if size == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement16B, 0}
		} else if size == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4H, 0}
		} else if size == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement8H, 0}
		} else if size == 2 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2S, 0}
		} else if size == 2 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4S, 0}
		} else if size == 3 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2D, 0}
		}
		return nil

	case arg_Vn_arrangement_size_Q___8B_00__16B_01__4H_10__8H_11__4S_21:
		Rn := (x >> 5) & (1<<5 - 1)
		size := (x >> 22) & 3
		Q := (x >> 30) & 1
		if size == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement8B, 0}
		} else if size == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement16B, 0}
		} else if size == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4H, 0}
		} else if size == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement8H, 0}
		} else if size == 2 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4S, 0}
		}
		return nil

	case arg_Vn_arrangement_sz___2D_1:
		Rn := (x >> 5) & (1<<5 - 1)
		sz := (x >> 22) & 1
		if sz == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2D, 0}
		}
		return nil

	case arg_Vn_arrangement_sz___2S_0__2D_1:
		Rn := (x >> 5) & (1<<5 - 1)
		sz := (x >> 22) & 1
		if sz == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2S, 0}
		} else {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2D, 0}
		}

	case arg_Vn_arrangement_sz___4S_0__2D_1:
		Rn := (x >> 5) & (1<<5 - 1)
		sz := (x >> 22) & 1
		if sz == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4S, 0}
		} else {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2D, 0}
		}

	case arg_Vn_arrangement_sz_Q___2S_00__4S_01:
		Rn := (x >> 5) & (1<<5 - 1)
		sz := (x >> 22) & 1
		Q := (x >> 30) & 1
		if sz == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2S, 0}
		} else if sz == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4S, 0}
		}
		return nil

	case arg_Vn_arrangement_sz_Q___2S_00__4S_01__2D_11:
		Rn := (x >> 5) & (1<<5 - 1)
		sz := (x >> 22) & 1
		Q := (x >> 30) & 1
		if sz == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2S, 0}
		} else if sz == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4S, 0}
		} else if sz == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2D, 0}
		}
		return nil

	case arg_Vn_arrangement_sz_Q___4H_00__8H_01__2S_10__4S_11:
		Rn := (x >> 5) & (1<<5 - 1)
		sz := (x >> 22) & 1
		Q := (x >> 30) & 1
		if sz == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4H, 0}
		} else if sz == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement8H, 0}
		} else if sz == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement2S, 0}
		} else /* sz == 1 && Q == 1 */ {
			return RegisterWithArrangement{V0 + Reg(Rn), Arrangement4S, 0}
		}

	case arg_Vt_1_arrangement_B_index__Q_S_size_1:
		Rt := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		S := (x >> 12) & 1
		size := (x >> 10) & 3
		index := (Q << 3) | (S << 2) | (size)
		return RegisterWithArrangementAndIndex{V0 + Reg(Rt), ArrangementB, uint8(index), 1}

	case arg_Vt_1_arrangement_D_index__Q_1:
		Rt := x & (1<<5 - 1)
		index := (x >> 30) & 1
		return RegisterWithArrangementAndIndex{V0 + Reg(Rt), ArrangementD, uint8(index), 1}

	case arg_Vt_1_arrangement_H_index__Q_S_size_1:
		Rt := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		S := (x >> 12) & 1
		size := (x >> 11) & 1
		index := (Q << 2) | (S << 1) | (size)
		return RegisterWithArrangementAndIndex{V0 + Reg(Rt), ArrangementH, uint8(index), 1}

	case arg_Vt_1_arrangement_S_index__Q_S_1:
		Rt := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		S := (x >> 12) & 1
		index := (Q << 1) | S
		return RegisterWithArrangementAndIndex{V0 + Reg(Rt), ArrangementS, uint8(index), 1}

	case arg_Vt_1_arrangement_size_Q___8B_00__16B_01__4H_10__8H_11__2S_20__4S_21__1D_30__2D_31:
		Rt := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		size := (x >> 10) & 3
		if size == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement8B, 1}
		} else if size == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement16B, 1}
		} else if size == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement4H, 1}
		} else if size == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement8H, 1}
		} else if size == 2 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement2S, 1}
		} else if size == 2 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement4S, 1}
		} else if size == 3 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement1D, 1}
		} else /* size == 3 && Q == 1 */ {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement2D, 1}
		}

	case arg_Vt_2_arrangement_B_index__Q_S_size_1:
		Rt := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		S := (x >> 12) & 1
		size := (x >> 10) & 3
		index := (Q << 3) | (S << 2) | (size)
		return RegisterWithArrangementAndIndex{V0 + Reg(Rt), ArrangementB, uint8(index), 2}

	case arg_Vt_2_arrangement_D_index__Q_1:
		Rt := x & (1<<5 - 1)
		index := (x >> 30) & 1
		return RegisterWithArrangementAndIndex{V0 + Reg(Rt), ArrangementD, uint8(index), 2}

	case arg_Vt_2_arrangement_H_index__Q_S_size_1:
		Rt := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		S := (x >> 12) & 1
		size := (x >> 11) & 1
		index := (Q << 2) | (S << 1) | (size)
		return RegisterWithArrangementAndIndex{V0 + Reg(Rt), ArrangementH, uint8(index), 2}

	case arg_Vt_2_arrangement_S_index__Q_S_1:
		Rt := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		S := (x >> 12) & 1
		index := (Q << 1) | S
		return RegisterWithArrangementAndIndex{V0 + Reg(Rt), ArrangementS, uint8(index), 2}

	case arg_Vt_2_arrangement_size_Q___8B_00__16B_01__4H_10__8H_11__2S_20__4S_21__1D_30__2D_31:
		Rt := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		size := (x >> 10) & 3
		if size == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement8B, 2}
		} else if size == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement16B, 2}
		} else if size == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement4H, 2}
		} else if size == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement8H, 2}
		} else if size == 2 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement2S, 2}
		} else if size == 2 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement4S, 2}
		} else if size == 3 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement1D, 2}
		} else /* size == 3 && Q == 1 */ {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement2D, 2}
		}

	case arg_Vt_2_arrangement_size_Q___8B_00__16B_01__4H_10__8H_11__2S_20__4S_21__2D_31:
		Rt := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		size := (x >> 10) & 3
		if size == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement8B, 2}
		} else if size == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement16B, 2}
		} else if size == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement4H, 2}
		} else if size == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement8H, 2}
		} else if size == 2 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement2S, 2}
		} else if size == 2 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement4S, 2}
		} else if size == 3 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement2D, 2}
		}
		return nil

	case arg_Vt_3_arrangement_B_index__Q_S_size_1:
		Rt := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		S := (x >> 12) & 1
		size := (x >> 10) & 3
		index := (Q << 3) | (S << 2) | (size)
		return RegisterWithArrangementAndIndex{V0 + Reg(Rt), ArrangementB, uint8(index), 3}

	case arg_Vt_3_arrangement_D_index__Q_1:
		Rt := x & (1<<5 - 1)
		index := (x >> 30) & 1
		return RegisterWithArrangementAndIndex{V0 + Reg(Rt), ArrangementD, uint8(index), 3}

	case arg_Vt_3_arrangement_H_index__Q_S_size_1:
		Rt := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		S := (x >> 12) & 1
		size := (x >> 11) & 1
		index := (Q << 2) | (S << 1) | (size)
		return RegisterWithArrangementAndIndex{V0 + Reg(Rt), ArrangementH, uint8(index), 3}

	case arg_Vt_3_arrangement_S_index__Q_S_1:
		Rt := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		S := (x >> 12) & 1
		index := (Q << 1) | S
		return RegisterWithArrangementAndIndex{V0 + Reg(Rt), ArrangementS, uint8(index), 3}

	case arg_Vt_3_arrangement_size_Q___8B_00__16B_01__4H_10__8H_11__2S_20__4S_21__1D_30__2D_31:
		Rt := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		size := (x >> 10) & 3
		if size == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement8B, 3}
		} else if size == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement16B, 3}
		} else if size == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement4H, 3}
		} else if size == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement8H, 3}
		} else if size == 2 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement2S, 3}
		} else if size == 2 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement4S, 3}
		} else if size == 3 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement1D, 3}
		} else /* size == 3 && Q == 1 */ {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement2D, 3}
		}

	case arg_Vt_3_arrangement_size_Q___8B_00__16B_01__4H_10__8H_11__2S_20__4S_21__2D_31:
		Rt := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		size := (x >> 10) & 3
		if size == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement8B, 3}
		} else if size == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement16B, 3}
		} else if size == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement4H, 3}
		} else if size == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement8H, 3}
		} else if size == 2 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement2S, 3}
		} else if size == 2 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement4S, 3}
		} else if size == 3 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement2D, 3}
		}
		return nil

	case arg_Vt_4_arrangement_B_index__Q_S_size_1:
		Rt := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		S := (x >> 12) & 1
		size := (x >> 10) & 3
		index := (Q << 3) | (S << 2) | (size)
		return RegisterWithArrangementAndIndex{V0 + Reg(Rt), ArrangementB, uint8(index), 4}

	case arg_Vt_4_arrangement_D_index__Q_1:
		Rt := x & (1<<5 - 1)
		index := (x >> 30) & 1
		return RegisterWithArrangementAndIndex{V0 + Reg(Rt), ArrangementD, uint8(index), 4}

	case arg_Vt_4_arrangement_H_index__Q_S_size_1:
		Rt := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		S := (x >> 12) & 1
		size := (x >> 11) & 1
		index := (Q << 2) | (S << 1) | (size)
		return RegisterWithArrangementAndIndex{V0 + Reg(Rt), ArrangementH, uint8(index), 4}

	case arg_Vt_4_arrangement_S_index__Q_S_1:
		Rt := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		S := (x >> 12) & 1
		index := (Q << 1) | S
		return RegisterWithArrangementAndIndex{V0 + Reg(Rt), ArrangementS, uint8(index), 4}

	case arg_Vt_4_arrangement_size_Q___8B_00__16B_01__4H_10__8H_11__2S_20__4S_21__1D_30__2D_31:
		Rt := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		size := (x >> 10) & 3
		if size == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement8B, 4}
		} else if size == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement16B, 4}
		} else if size == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement4H, 4}
		} else if size == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement8H, 4}
		} else if size == 2 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement2S, 4}
		} else if size == 2 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement4S, 4}
		} else if size == 3 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement1D, 4}
		} else /* size == 3 && Q == 1 */ {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement2D, 4}
		}

	case arg_Vt_4_arrangement_size_Q___8B_00__16B_01__4H_10__8H_11__2S_20__4S_21__2D_31:
		Rt := x & (1<<5 - 1)
		Q := (x >> 30) & 1
		size := (x >> 10) & 3
		if size == 0 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement8B, 4}
		} else if size == 0 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement16B, 4}
		} else if size == 1 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement4H, 4}
		} else if size == 1 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement8H, 4}
		} else if size == 2 && Q == 0 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement2S, 4}
		} else if size == 2 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement4S, 4}
		} else if size == 3 && Q == 1 {
			return RegisterWithArrangement{V0 + Reg(Rt), Arrangement2D, 4}
		}
		return nil

	case arg_Xns_mem_extend_m__UXTW_2__LSL_3__SXTW_6__SXTX_7__0_0__4_1:
		return handle_MemExtend(x, 4, false)

	case arg_Xns_mem_offset:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		return MemImmediate{Rn, AddrOffset, 0}

	case arg_Xns_mem_optional_imm12_16_unsigned:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		imm12 := (x >> 10) & (1<<12 - 1)
		return MemImmediate{Rn, AddrOffset, int32(imm12 << 4)}

	case arg_Xns_mem_optional_imm7_16_signed:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		imm7 := (x >> 15) & (1<<7 - 1)
		return MemImmediate{Rn, AddrOffset, ((int32(imm7 << 4)) << 21) >> 21}

	case arg_Xns_mem_post_fixedimm_1:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		return MemImmediate{Rn, AddrPostIndex, 1}

	case arg_Xns_mem_post_fixedimm_12:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		return MemImmediate{Rn, AddrPostIndex, 12}

	case arg_Xns_mem_post_fixedimm_16:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		return MemImmediate{Rn, AddrPostIndex, 16}

	case arg_Xns_mem_post_fixedimm_2:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		return MemImmediate{Rn, AddrPostIndex, 2}

	case arg_Xns_mem_post_fixedimm_24:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		return MemImmediate{Rn, AddrPostIndex, 24}

	case arg_Xns_mem_post_fixedimm_3:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		return MemImmediate{Rn, AddrPostIndex, 3}

	case arg_Xns_mem_post_fixedimm_32:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		return MemImmediate{Rn, AddrPostIndex, 32}

	case arg_Xns_mem_post_fixedimm_4:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		return MemImmediate{Rn, AddrPostIndex, 4}

	case arg_Xns_mem_post_fixedimm_6:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		return MemImmediate{Rn, AddrPostIndex, 6}

	case arg_Xns_mem_post_fixedimm_8:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		return MemImmediate{Rn, AddrPostIndex, 8}

	case arg_Xns_mem_post_imm7_16_signed:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		imm7 := (x >> 15) & (1<<7 - 1)
		return MemImmediate{Rn, AddrPostIndex, ((int32(imm7 << 4)) << 21) >> 21}

	case arg_Xns_mem_post_Q__16_0__32_1:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		Q := (x >> 30) & 1
		return MemImmediate{Rn, AddrPostIndex, int32((Q + 1) * 16)}

	case arg_Xns_mem_post_Q__24_0__48_1:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		Q := (x >> 30) & 1
		return MemImmediate{Rn, AddrPostIndex, int32((Q + 1) * 24)}

	case arg_Xns_mem_post_Q__32_0__64_1:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		Q := (x >> 30) & 1
		return MemImmediate{Rn, AddrPostIndex, int32((Q + 1) * 32)}

	case arg_Xns_mem_post_Q__8_0__16_1:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		Q := (x >> 30) & 1
		return MemImmediate{Rn, AddrPostIndex, int32((Q + 1) * 8)}

	case arg_Xns_mem_post_size__1_0__2_1__4_2__8_3:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		size := (x >> 10) & 3
		return MemImmediate{Rn, AddrPostIndex, int32(1 << size)}

	case arg_Xns_mem_post_size__2_0__4_1__8_2__16_3:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		size := (x >> 10) & 3
		return MemImmediate{Rn, AddrPostIndex, int32(2 << size)}

	case arg_Xns_mem_post_size__3_0__6_1__12_2__24_3:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		size := (x >> 10) & 3
		return MemImmediate{Rn, AddrPostIndex, int32(3 << size)}

	case arg_Xns_mem_post_size__4_0__8_1__16_2__32_3:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		size := (x >> 10) & 3
		return MemImmediate{Rn, AddrPostIndex, int32(4 << size)}

	case arg_Xns_mem_post_Xm:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		Rm := (x >> 16) & (1<<5 - 1)
		return MemImmediate{Rn, AddrPostReg, int32(Rm)}

	case arg_Xns_mem_wb_imm7_16_signed:
		Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
		imm7 := (x >> 15) & (1<<7 - 1)
		return MemImmediate{Rn, AddrPreIndex, ((int32(imm7 << 4)) << 21) >> 21}
	}
}

func handle_ExtendedRegister(x uint32, has_width bool) Arg {
	s := (x >> 29) & 1
	rm := (x >> 16) & (1<<5 - 1)
	option := (x >> 13) & (1<<3 - 1)
	imm3 := (x >> 10) & (1<<3 - 1)
	rn := (x >> 5) & (1<<5 - 1)
	rd := x & (1<<5 - 1)
	is_32bit := !has_width
	var rea RegExtshiftAmount
	if has_width {
		if option&0x3 != 0x3 {
			rea.reg = W0 + Reg(rm)
		} else {
			rea.reg = X0 + Reg(rm)
		}
	} else {
		rea.reg = W0 + Reg(rm)
	}
	switch option {
	case 0:
		rea.extShift = uxtb
	case 1:
		rea.extShift = uxth
	case 2:
		if is_32bit && (rn == 31 || (s == 0 && rd == 31)) {
			if imm3 != 0 {
				rea.extShift = lsl
			} else {
				rea.extShift = ExtShift(0)
			}
		} else {
			rea.extShift = uxtw
		}
	case 3:
		if !is_32bit && (rn == 31 || (s == 0 && rd == 31)) {
			if imm3 != 0 {
				rea.extShift = lsl
			} else {
				rea.extShift = ExtShift(0)
			}
		} else {
			rea.extShift = uxtx
		}
	case 4:
		rea.extShift = sxtb
	case 5:
		rea.extShift = sxth
	case 6:
		rea.extShift = sxtw
	case 7:
		rea.extShift = sxtx
	}
	rea.show_zero = false
	rea.amount = uint8(imm3)
	return rea
}

func handle_ImmediateShiftedRegister(x uint32, max uint8, is_w, has_ror bool) Arg {
	var rsa RegExtshiftAmount
	if is_w {
		rsa.reg = W0 + Reg((x>>16)&(1<<5-1))
	} else {
		rsa.reg = X0 + Reg((x>>16)&(1<<5-1))
	}
	switch (x >> 22) & 0x3 {
	case 0:
		rsa.extShift = lsl
	case 1:
		rsa.extShift = lsr
	case 2:
		rsa.extShift = asr
	case 3:
		if has_ror {
			rsa.extShift = ror
		} else {
			return nil
		}
	}
	rsa.show_zero = true
	rsa.amount = uint8((x >> 10) & (1<<6 - 1))
	if rsa.amount == 0 && rsa.extShift == lsl {
		rsa.extShift = ExtShift(0)
	} else if rsa.amount > max {
		return nil
	}
	return rsa
}

func handle_MemExtend(x uint32, mult uint8, absent bool) Arg {
	var extend ExtShift
	var Rm Reg
	option := (x >> 13) & (1<<3 - 1)
	Rn := RegSP(X0) + RegSP(x>>5&(1<<5-1))
	if (option & 1) != 0 {
		Rm = Reg(X0) + Reg(x>>16&(1<<5-1))
	} else {
		Rm = Reg(W0) + Reg(x>>16&(1<<5-1))
	}
	switch option {
	default:
		return nil
	case 2:
		extend = uxtw
	case 3:
		extend = lsl
	case 6:
		extend = sxtw
	case 7:
		extend = sxtx
	}
	amount := (uint8((x >> 12) & 1)) * mult
	return MemExtend{Rn, Rm, extend, amount, absent}
}

func handle_bitmasks(x uint32, datasize uint8) Arg {
	var length, levels, esize, i uint8
	var welem, wmask uint64
	n := (x >> 22) & 1
	imms := uint8((x >> 10) & (1<<6 - 1))
	immr := uint8((x >> 16) & (1<<6 - 1))
	if n != 0 {
		length = 6
	} else if (imms & 32) == 0 {
		length = 5
	} else if (imms & 16) == 0 {
		length = 4
	} else if (imms & 8) == 0 {
		length = 3
	} else if (imms & 4) == 0 {
		length = 2
	} else if (imms & 2) == 0 {
		length = 1
	} else {
		return nil
	}
	levels = 1<<length - 1
	s := imms & levels
	r := immr & levels
	esize = 1 << length
	if esize > datasize {
		return nil
	}
	welem = 1<<(s+1) - 1
	ror := (welem >> r) | (welem << (esize - r))
	ror &= ((1 << esize) - 1)
	wmask = 0
	for i = 0; i < datasize; i += esize {
		wmask = (wmask << esize) | ror
	}
	return Imm64{wmask, false}
}
