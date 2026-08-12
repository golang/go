// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

import (
	"math"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/block"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/arm64"
	"internal/abi"
)

// loadByType returns the load instruction of the given type.
func loadByType(t *types.Type) obj.As {
	if t.IsSIMD() {
		switch t.Size() {
		case 16:
			return arm64.AFMOVQ // Use FMOVQ (LDR Q) for 128-bit SIMD loads
		case 8:
			return arm64.APLDR
		case 32:
			return arm64.AZLDR
		}
	} else if t.IsFloat() {
		switch t.Size() {
		case 4:
			return arm64.AFMOVS
		case 8:
			return arm64.AFMOVD
		}
	} else {
		switch t.Size() {
		case 1:
			if t.IsSigned() {
				return arm64.AMOVB
			} else {
				return arm64.AMOVBU
			}
		case 2:
			if t.IsSigned() {
				return arm64.AMOVH
			} else {
				return arm64.AMOVHU
			}
		case 4:
			if t.IsSigned() {
				return arm64.AMOVW
			} else {
				return arm64.AMOVWU
			}
		case 8:
			return arm64.AMOVD
		}
	}
	panic("bad load type")
}

// storeByType returns the store instruction of the given type.
func storeByType(t *types.Type) obj.As {
	if t.IsSIMD() {
		switch t.Size() {
		case 16:
			return arm64.AFMOVQ // Use FMOVQ (STR Q) for 128-bit SIMD stores
		case 8:
			return arm64.APSTR
		case 32:
			return arm64.AZSTR
		}
	} else if t.IsFloat() {
		switch t.Size() {
		case 4:
			return arm64.AFMOVS
		case 8:
			return arm64.AFMOVD
		}
	} else {
		switch t.Size() {
		case 1:
			return arm64.AMOVB
		case 2:
			return arm64.AMOVH
		case 4:
			return arm64.AMOVW
		case 8:
			return arm64.AMOVD
		}
	}
	panic("bad store type")
}

// loadByType2 returns an opcode that can load consecutive memory locations into 2 registers with type t.
// returns obj.AXXX if no such opcode exists.
func loadByType2(t *types.Type) obj.As {
	if t.IsFloat() {
		switch t.Size() {
		case 4:
			return arm64.AFLDPS
		case 8:
			return arm64.AFLDPD
		}
	} else {
		switch t.Size() {
		case 4:
			return arm64.ALDPW
		case 8:
			return arm64.ALDP
		}
	}
	return obj.AXXX
}

// storeByType2 returns an opcode that can store registers with type t into 2 consecutive memory locations.
// returns obj.AXXX if no such opcode exists.
func storeByType2(t *types.Type) obj.As {
	if t.IsFloat() {
		switch t.Size() {
		case 4:
			return arm64.AFSTPS
		case 8:
			return arm64.AFSTPD
		}
	} else {
		switch t.Size() {
		case 4:
			return arm64.ASTPW
		case 8:
			return arm64.ASTP
		}
	}
	return obj.AXXX
}

// makeshift encodes a register shifted by a constant, used as an Offset in Prog.
func makeshift(v *ssa.Value, reg int16, typ int64, s int64) int64 {
	if s < 0 || s >= 64 {
		v.Fatalf("shift out of range: %d", s)
	}
	return int64(reg&31)<<16 | typ | (s&63)<<10
}

// genshift generates a Prog for r = r0 op (r1 shifted by n).
func genshift(s *ssagen.State, v *ssa.Value, as obj.As, r0, r1, r int16, typ int64, n int64) *obj.Prog {
	p := s.Prog(as)
	p.From.Type = obj.TYPE_SHIFT
	p.From.Offset = makeshift(v, r1, typ, n)
	p.Reg = r0
	if r != 0 {
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	}
	return p
}

// generate the memory operand for the indexed load/store instructions.
// base and idx are registers.
func genIndexedOperand(op ssaop.Op, base, idx int16) obj.Addr {
	// Reg: base register, Index: (shifted) index register
	mop := obj.Addr{Type: obj.TYPE_MEM, Reg: base}
	switch op {
	case ssaop.OpARM64MOVDloadidx8, ssaop.OpARM64MOVDstoreidx8,
		ssaop.OpARM64FMOVDloadidx8, ssaop.OpARM64FMOVDstoreidx8:
		mop.Index = arm64.REG_LSL | 3<<5 | idx&31
	case ssaop.OpARM64MOVWloadidx4, ssaop.OpARM64MOVWUloadidx4, ssaop.OpARM64MOVWstoreidx4,
		ssaop.OpARM64FMOVSloadidx4, ssaop.OpARM64FMOVSstoreidx4:
		mop.Index = arm64.REG_LSL | 2<<5 | idx&31
	case ssaop.OpARM64MOVHloadidx2, ssaop.OpARM64MOVHUloadidx2, ssaop.OpARM64MOVHstoreidx2:
		mop.Index = arm64.REG_LSL | 1<<5 | idx&31
	default: // not shifted
		mop.Index = idx
	}
	return mop
}

const simdSVEVectorLengthScaled int16 = -32768

// simdRegArng encodes ssa value's register with specified simd arrangement
func simdRegArng(reg int16, arng int16) int16 {
	if reg < arm64.REG_F0 || arm64.REG_F31 < reg {
		base.Fatalf("expected fp register: r%d", reg)
	}
	var err error
	if reg, err = arm64.RegisterArrangement(reg, arng, false); err != nil {
		base.Fatalf("bad simd register arrangement: %v", err)
	}
	return reg
}

// simdRegElem encodes ssa value's reference to a vector register element
func simdRegElem(reg int16, arng int16, idx int16) (res obj.Addr) {
	if reg < arm64.REG_F0 || arm64.REG_F31 < reg {
		base.Fatalf("expected fp register: r%d", reg)
	}
	elem, err := arm64.RegisterArrangement(reg, arng, true /*indexing*/)
	if err != nil {
		base.Fatalf("bad simd register indexing arrangement: %v", err)
	}
	res.Type = obj.TYPE_REG
	res.Class = arm64.C_ELEM
	res.Index = idx
	res.Reg = elem
	return
}

// allLanes converts an element arrangement to its 128-bit vector arrangement.
// e.g., ARNG_B -> ARNG_16B, ARNG_S -> ARNG_4S
func allLanes(arng int16) int16 {
	switch arng {
	case arm64.ARNG_B:
		return arm64.ARNG_16B
	case arm64.ARNG_H:
		return arm64.ARNG_8H
	case arm64.ARNG_S:
		return arm64.ARNG_4S
	case arm64.ARNG_D:
		return arm64.ARNG_2D
	default:
		base.Fatalf("unsupported element arrangement: %d", arng)
		return 0
	}
}

// arngNarrow converts arng to its narrow (halved element width and vector width) arrangement.
func arngNarrow(arng int16) int16 {
	switch arng {
	case arm64.ARNG_8H:
		return arm64.ARNG_8B
	case arm64.ARNG_4S:
		return arm64.ARNG_4H
	case arm64.ARNG_2D:
		return arm64.ARNG_2S
	default:
		base.Fatalf("unsupported narrow input arrangement: %d", arng)
		return 0
	}
}

// arngLong converts a half-lane arrangement to its long (doubled element width and vector width) arrangement.
func arngLong(arng int16) int16 {
	switch arng {
	case arm64.ARNG_8B:
		return arm64.ARNG_8H
	case arm64.ARNG_4H:
		return arm64.ARNG_4S
	case arm64.ARNG_2S:
		return arm64.ARNG_2D
	case arm64.ARNG_1D:
		return arm64.ARNG_1Q
	default:
		base.Fatalf("unsupported long input arrangement: %d", arng)
		return 0
	}
}

// arngHalfLanes converts a full-width arrangement to its half-lane (64-bit) arrangement.
// Same element width, half the lanes. Used for long base variant sources.
func arngHalfLanes(arng int16) int16 {
	switch arng {
	case arm64.ARNG_16B:
		return arm64.ARNG_8B
	case arm64.ARNG_8H:
		return arm64.ARNG_4H
	case arm64.ARNG_4S:
		return arm64.ARNG_2S
	case arm64.ARNG_2D:
		return arm64.ARNG_1D
	default:
		base.Fatalf("unsupported halfLanes input arrangement: %d", arng)
		return 0
	}
}

// arngTwiceLanes converts a half-lane (64-bit) arrangement to its full-width arrangement.
// Same element width, double the lanes. Inverse of arngHalfLanes.
func arngTwiceLanes(arng int16) int16 {
	switch arng {
	case arm64.ARNG_8B:
		return arm64.ARNG_16B
	case arm64.ARNG_4H:
		return arm64.ARNG_8H
	case arm64.ARNG_2S:
		return arm64.ARNG_4S
	default:
		base.Fatalf("unsupported twiceLanes input arrangement: %d", arng)
		return 0
	}
}

// simdV01Imm generates a VMOVI-like instruction, e.g. VMOVI $0, V0.B16
func simdV01Imm(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = int64(v.AuxUInt8())
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdRegArng(v.Reg(), arrangement)
	return p
}

// simdV11Asm generates element-wise unary vector operations with explicit asm, e.g. VMOV V1.B16, V0.B16
func simdV11Asm(s *ssagen.State, asm obj.As, src, dst int16, arrangement int16) *obj.Prog {
	p := s.Prog(asm)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdRegArng(src, arrangement)
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdRegArng(dst, arrangement)
	return p
}

// simdV11 generates element-wise unary vector operations, e.g. VCNT V1.B8, V0.B8
func simdV11(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	return simdV11Asm(s, v.Op.Asm(), v.Args[0].Reg(), v.Reg(), arrangement)
}

// simdV11Imm generates a unary vector operation with immediate constant,
// e.g. VUSHR $3, V1.B16, V0.B16
func simdV11Imm(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = int64(v.AuxUInt8())
	p.Reg = simdRegArng(v.Args[0].Reg(), arrangement)
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdRegArng(v.Reg(), arrangement)
	return p
}

// simdV11ImmIn1 generates a broadcast1ToN instruction,
// e.g. VDUP V1.S[0], V0.S4 (duplicate element 0 to all lanes)
// The arrangement parameter specifies the element arrangement (e.g., ARNG_S, ARNG_D)
func simdV11ImmIn1(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From = simdRegElem(v.Args[0].Reg(), arrangement, int16(v.AuxUInt8()))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdRegArng(v.Reg(), allLanes(arrangement))
	return p
}

// simdV11Scalar generates vector-to-scalar reduction operations, e.g. VUADDLV V1.B8, V0
func simdV11Scalar(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdRegArng(v.Args[0].Reg(), arrangement)
	p.To.Type = obj.TYPE_REG
	p.To.Reg = v.Reg() - arm64.REG_F0 + arm64.REG_V0
	return p
}

// simdV11ScalarImmIn1 generates a SIMD instruction with indexed input and
// scalar-in-vector-register output, e.g. VDUP V1.S[1], V0
// The arrangement parameter specifies the source arrangement (e.g., S, D)
func simdV11ScalarImmIn1(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From = simdRegElem(v.Args[0].Reg(), arrangement, int16(v.AuxUInt8()))
	p.To.Type = obj.TYPE_REG
	p.To.Reg = v.Reg() - arm64.REG_F0 + arm64.REG_V0
	p.To.Class = arm64.C_VREG
	return p
}

// simdV21 generates element-wise binary vector operations, e.g. VFADD V1.S4, V2.S4, V0.S4
func simdV21(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdRegArng(v.Args[1].Reg(), arrangement)
	p.Reg = simdRegArng(v.Args[0].Reg(), arrangement)
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdRegArng(v.Reg(), arrangement)
	return p
}

// simdV21Imm generates a binary instruction with immediate, e.g. EXT $imm, Vm.16B, Vn.16B, Vd.16B
func simdV21Imm(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = int64(v.AuxUInt8())
	p.Reg = simdRegArng(v.Args[0].Reg(), arrangement)
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdRegArng(v.Reg(), arrangement)
	p.AddRestSource(obj.Addr{Type: obj.TYPE_REG, Reg: simdRegArng(v.Args[1].Reg(), arrangement)})
	return p
}

// simdV31ResultInArg0 generates a destructive 3-register instruction,
// e.g. VBIT Vm.16B, Vn.16B, Vd.16B.
func simdV31ResultInArg0(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdRegArng(v.Args[2].Reg(), arrangement)
	p.Reg = simdRegArng(v.Args[1].Reg(), arrangement)
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdRegArng(v.Reg(), arrangement)
	return p
}

// simdV21List generates a binary instruction with register list, e.g. TBL Vm.Ta, {Vn.B16}, Vd.Ta.
func simdV21List(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	if v.Op.Asm() != arm64.AVTBL { // TODO: support other instructions as needed.
		panic("simdV21List: expected VTBL")
	}
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdRegArng(v.Args[1].Reg(), arrangement)
	// TBL requires B16 table arrangement.
	// Also, multi-element register lists are not supported by regalloc.
	const listB16 = int64(1 << 30)
	regList, _ := arm64.RegisterListOffset(int(v.Args[0].Reg()&31), 1, listB16, 0)
	p.AddRestSource(obj.Addr{Type: obj.TYPE_REGLIST, Offset: regList})
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdRegArng(v.Reg(), arrangement)
	return p
}

// simdV31ResultInArg0List generates a destructive 3-register instruction
// with register list, e.g. TBX Vm.Ta, {Vn.B16}, Vd.Ta.
func simdV31ResultInArg0List(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	if v.Op.Asm() != arm64.AVTBX { // TODO: support other instructions as needed.
		panic("simdV31ResultInArg0List: expected VTBX")
	}
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdRegArng(v.Args[2].Reg(), arrangement)
	// TBX requires B16 table arrangement.
	// Also, multi-element register lists are not supported by regalloc.
	const listB16 = int64(1 << 30)
	regList, _ := arm64.RegisterListOffset(int(v.Args[1].Reg()&31), 1, listB16, 0)
	p.AddRestSource(obj.Addr{Type: obj.TYPE_REGLIST, Offset: regList})
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdRegArng(v.Reg(), arrangement)
	return p
}

// simdVfpvResultInArg0ImmOutIn1 generates vector floating-point SetElem,
// e.g. VMOV V2.S[0], V1.S[3] (INS element instruction)
// The arrangement parameter specifies the vector element arrangement (e.g., S, D)
func simdVfpvResultInArg0ImmOutIn1(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.To = simdRegElem(v.Reg(), arrangement, int16(v.AuxUInt8()))
	p.From = simdRegElem(v.Args[1].Reg(), arrangement, 0)
	return p
}

// simdVgpImmIn1 generates vector GetElem instruction VMOV V1.S[2], R0
// The arrangement parameter specifies the vector element arrangement (e.g., S, D)
func simdVgpImmIn1(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From = simdRegElem(v.Args[0].Reg(), arrangement, int16(v.AuxUInt8()))
	p.To.Reg = v.Reg()
	p.To.Type = obj.TYPE_REG
	return p
}

// simdVgpvResultInArg0ImmOutIn0 generates vector SetElem, e.g. VMOV R0, V1.S[2] (INS general instruction)
// The arrangement parameter specifies the vector element arrangement (e.g., S, D)
func simdVgpvResultInArg0ImmOutIn0(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.To = simdRegElem(v.Reg(), arrangement, int16(v.AuxUInt8()))
	p.From.Reg = v.Args[1].Reg()
	p.From.Type = obj.TYPE_REG
	return p
}

// Narrow and long lowering helpers

// simdV11Narrow generates a pure narrowing instruction, e.g. XTN Vn.8H, Vd.8B
func simdV11Narrow(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdRegArng(v.Args[0].Reg(), arrangement)
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdRegArng(v.Reg(), arngNarrow(arrangement))
	return p
}

// simdV21Narrow2 generates a destructive (updating upper half only) narrow "2" instruction,
// e.g. XTN2 V1.4S, V0.8H. The arrangement parameter specifies the source arrangement.
func simdV21Narrow2(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdRegArng(v.Args[1].Reg(), arrangement)
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdRegArng(v.Reg(), arngTwiceLanes(arngNarrow(arrangement)))
	return p
}

// simdV11ImmNarrow generates a pure narrowing instruction with immediate, e.g. SHRN $imm, V1.4S, V0.8B
// The arrangement parameter specifies the source arrangement.
func simdV11ImmNarrow(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = int64(v.AuxUInt8())
	p.Reg = simdRegArng(v.Args[0].Reg(), arrangement)
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdRegArng(v.Reg(), arngNarrow(arrangement))
	return p
}

// simdV21ImmNarrow2 generates a destructive (updating upper half only) narrow "2" instruction
// with immediate, e.g. SHRN2 $imm, V1.4S, V0.16B. The arrangement parameter specifies the source arrangement.
func simdV21ImmNarrow2(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = int64(v.AuxUInt8())
	p.Reg = simdRegArng(v.Args[1].Reg(), arrangement)
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdRegArng(v.Reg(), arngTwiceLanes(arngNarrow(arrangement)))
	return p
}

// simdV11Long generates a unary long instruction, e.g. SXTL V1.4H, V0.8H
// The instruction reads the lower half of the source, the destination has 2x element size.
func simdV11Long(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	src := arngHalfLanes(arrangement)
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdRegArng(v.Args[0].Reg(), src)
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdRegArng(v.Reg(), arngLong(src))
	return p
}

// simdV11Long2 generates a unary long "2" instruction, e.g. SXTL2 V1.4S, V0.2D
// The instruction reads the upper half of the source, the destination has 2x element size.
func simdV11Long2(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdRegArng(v.Args[0].Reg(), arrangement)
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdRegArng(v.Reg(), arngLong(arngHalfLanes(arrangement)))
	return p
}

// simdV11ImmLong generates a long instruction with immediate, e.g. USHLL $imm, V1.4H, V0.8H
// The instruction reads the lower half of the source, the destination has 2x element size.
func simdV11ImmLong(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	src := arngHalfLanes(arrangement)
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = int64(v.AuxUInt8())
	p.Reg = simdRegArng(v.Args[0].Reg(), src)
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdRegArng(v.Reg(), arngLong(src))
	return p
}

// simdV11ImmLong2 generates a long "2" instruction with immediate, e.g. USHLL2 $imm, V1.4S, V0.2D
// The instruction reads the upper half of the source, the destination has 2x element size.
func simdV11ImmLong2(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = int64(v.AuxUInt8())
	p.Reg = simdRegArng(v.Args[0].Reg(), arrangement)
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdRegArng(v.Reg(), arngLong(arngHalfLanes(arrangement)))
	return p
}

// simdV21Long generates a binary long instruction, e.g. UMULL V1.4H, V2.4H, V0.8H
// The instruction reads lower halves of its sources, the destination has 2x element size.
func simdV21Long(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	src := arngHalfLanes(arrangement)
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdRegArng(v.Args[1].Reg(), src)
	p.Reg = simdRegArng(v.Args[0].Reg(), src)
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdRegArng(v.Reg(), arngLong(src))
	return p
}

// simdV21Long2 generates a binary long "2" instruction, e.g. UMULL2 V1.4S, V2.4S, V0.2D
// The instruction reads upper halves of its sources, the destination has 2x element size.
func simdV21Long2(s *ssagen.State, v *ssa.Value, arrangement int16) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = simdRegArng(v.Args[1].Reg(), arrangement)
	p.Reg = simdRegArng(v.Args[0].Reg(), arrangement)
	p.To.Type = obj.TYPE_REG
	p.To.Reg = simdRegArng(v.Reg(), arngLong(arngHalfLanes(arrangement)))
	return p
}

func ssaGenValue(s *ssagen.State, v *ssa.Value) {
	switch v.Op {
	case ssaop.OpCopy, ssaop.OpARM64MOVDreg:
		if v.Type.IsMemory() {
			return
		}
		x := v.Args[0].Reg()
		y := v.Reg()
		if x == y {
			return
		}
		as := arm64.AMOVD
		if v.Type.IsFloat() {
			switch v.Type.Size() {
			case 4:
				as = arm64.AFMOVS
			case 8:
				as = arm64.AFMOVD
			default:
				panic("bad float size")
			}
		} else if v.Type.IsSIMD() {
			if v.Type.Size() == 16 {
				simdV11Asm(s, arm64.AVMOV, x, y, arm64.ARNG_16B)
				return
			} else if v.Type.Size() == 32 {
				// Z->Z
				p := s.Prog(arm64.AZORR)
				p.From.Type = obj.TYPE_REG
				p.From.Reg = zregArng(x, arm64.ARNG_D)
				p.AddRestSourceReg(zregArng(x, arm64.ARNG_D))
				p.To.Type = obj.TYPE_REG
				p.To.Reg = zregArng(y, arm64.ARNG_D)
				return
			} else if v.Type.Size() == 8 {
				// P->P
				p := s.Prog(arm64.APORR)
				p.From.Type = obj.TYPE_REG
				if x < arm64.REG_P0 || x > arm64.REG_P15 {
					panic("bad P reg")
				}
				p.From.Reg = pregArng(x, arm64.ARNG_B)
				p.AddRestSourceReg(pregArng(x, arm64.ARNG_B))
				p.AddRestSourceReg(pregMask(x, arm64.PRED_Z))
				p.To.Type = obj.TYPE_REG
				p.To.Reg = pregArng(y, arm64.ARNG_B)
				return
			} else {
				panic("bad simd size")
			}
		}
		p := s.Prog(as)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = x
		p.To.Type = obj.TYPE_REG
		p.To.Reg = y
	case ssaop.OpARM64MOVDnop, ssaop.OpARM64ZERO:
		// nothing to do
	case ssaop.OpARM64VMOVI16B:
		simdV01Imm(s, v, arm64.ARNG_16B)
	case ssaop.OpLoadReg:
		if v.Type.IsFlags() {
			v.Fatalf("load flags not implemented: %v", v.LongString())
			return
		}
		if v.Type.IsSIMD() && (v.Type.Size() == 32 || v.Type.Size() == 8) {
			// SVE Z/P reload: reach the slot through a register.
			from := sveStackAddr(s, v.Args[0])
			p := s.Prog(loadByType(v.Type))
			p.From = from
			p.To.Type = obj.TYPE_REG
			p.To.Reg = pzreg(v.Reg())
		} else {
			p := s.Prog(loadByType(v.Type))
			ssagen.AddrAuto(&p.From, v.Args[0])
			p.To.Type = obj.TYPE_REG
			p.To.Reg = v.Reg()
		}
	case ssaop.OpStoreReg:
		if v.Type.IsFlags() {
			v.Fatalf("store flags not implemented: %v", v.LongString())
			return
		}
		if v.Type.IsSIMD() && (v.Type.Size() == 32 || v.Type.Size() == 8) {
			// SVE Z/P spill: reach the slot through a register.
			to := sveStackAddr(s, v)
			p := s.Prog(storeByType(v.Type))
			p.From.Type = obj.TYPE_REG
			p.From.Reg = pzreg(v.Args[0].Reg())
			p.To = to
		} else {
			p := s.Prog(storeByType(v.Type))
			p.From.Type = obj.TYPE_REG
			p.From.Reg = v.Args[0].Reg()
			ssagen.AddrAuto(&p.To, v)
		}
	case ssaop.OpArgIntReg, ssaop.OpArgFloatReg:
		ssagen.CheckArgReg(v)
		// The assembler needs to wrap the entry safepoint/stack growth code with spill/unspill
		// The loop only runs once.
		args := v.Block.Func.RegArgs
		if len(args) == 0 {
			break
		}
		v.Block.Func.RegArgs = nil // prevent from running again

		for i := 0; i < len(args); i++ {
			a := args[i]
			// Offset by size of the saved LR slot.
			addr := ssagen.SpillSlotAddr(a, arm64.REGSP, base.Ctxt.Arch.FixedFrameSize)
			// Look for double-register operations if we can.
			if i < len(args)-1 {
				b := args[i+1]
				if a.Type.Size() == b.Type.Size() &&
					a.Type.IsFloat() == b.Type.IsFloat() &&
					b.Offset == a.Offset+a.Type.Size() {
					ld := loadByType2(a.Type)
					st := storeByType2(a.Type)
					if ld != obj.AXXX && st != obj.AXXX {
						s.FuncInfo().AddSpill(obj.RegSpill{Reg: a.Reg, Reg2: b.Reg, Addr: addr, Unspill: ld, Spill: st})
						i++ // b is done also, skip it.
						continue
					}
				}
			}
			reg := a.Reg
			if a.Type.IsSIMD() && (a.Type.Size() == 32 || a.Type.Size() == 8) {
				reg = pzreg(reg)
				addr.Scale = simdSVEVectorLengthScaled
			}
			// Pass the spill/unspill information along to the assembler.
			s.FuncInfo().AddSpill(obj.RegSpill{Reg: reg, Addr: addr, Unspill: loadByType(a.Type), Spill: storeByType(a.Type)})
		}

	case ssaop.OpARM64ADD,
		ssaop.OpARM64SUB,
		ssaop.OpARM64AND,
		ssaop.OpARM64OR,
		ssaop.OpARM64XOR,
		ssaop.OpARM64BIC,
		ssaop.OpARM64EON,
		ssaop.OpARM64ORN,
		ssaop.OpARM64MUL,
		ssaop.OpARM64MULW,
		ssaop.OpARM64MNEG,
		ssaop.OpARM64MNEGW,
		ssaop.OpARM64MULH,
		ssaop.OpARM64UMULH,
		ssaop.OpARM64MULL,
		ssaop.OpARM64UMULL,
		ssaop.OpARM64DIV,
		ssaop.OpARM64UDIV,
		ssaop.OpARM64DIVW,
		ssaop.OpARM64UDIVW,
		ssaop.OpARM64MOD,
		ssaop.OpARM64UMOD,
		ssaop.OpARM64MODW,
		ssaop.OpARM64UMODW,
		ssaop.OpARM64SLL,
		ssaop.OpARM64SRL,
		ssaop.OpARM64SRA,
		ssaop.OpARM64FADDS,
		ssaop.OpARM64FADDD,
		ssaop.OpARM64FSUBS,
		ssaop.OpARM64FSUBD,
		ssaop.OpARM64FMULS,
		ssaop.OpARM64FMULD,
		ssaop.OpARM64FNMULS,
		ssaop.OpARM64FNMULD,
		ssaop.OpARM64FDIVS,
		ssaop.OpARM64FDIVD,
		ssaop.OpARM64FMINS,
		ssaop.OpARM64FMIND,
		ssaop.OpARM64FMAXS,
		ssaop.OpARM64FMAXD,
		ssaop.OpARM64ROR,
		ssaop.OpARM64RORW:
		r := v.Reg()
		r1 := v.Args[0].Reg()
		r2 := v.Args[1].Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r2
		p.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssaop.OpARM64ZLDRload:
		// Whole-register load of a scalable vector, e.g. ZLDR (VL*0)(R0), Z1.
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.From.Scale = simdSVEVectorLengthScaled
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = pzreg(v.Reg())
	case ssaop.OpARM64ZSTRstore:
		// Whole-register VL-scaled store of a scalable vector: ZSTR Zn, (mem).
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = pzreg(v.Args[1].Reg())
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.To.Scale = simdSVEVectorLengthScaled
		ssagen.AddAux(&p.To, v)
	case ssaop.OpARM64PLDRload:
		// Whole-register VL-scaled load of a predicate, e.g. PLDR (VL*0)(R0), P0.
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.From.Scale = simdSVEVectorLengthScaled
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARM64PSTRstore:
		// Whole-register VL-scaled store of a predicate, e.g. PSTR P0, (VL*0)(R0).
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.To.Scale = simdSVEVectorLengthScaled
		ssagen.AddAux(&p.To, v)
	case ssaop.OpARM64ZDUPBconst:
		// Broadcast an 8-bit immediate to every byte lane (ZeroSIMD uses [0]).
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = zregArng(v.Reg(), arm64.ARNG_B)
	case ssaop.OpARM64RDVL:
		// Read the vector length in bytes into a GP register, e.g. RDVL $1, R0.
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARM64PWHILELTB:
		simdPWHILELT(s, v, arm64.ARNG_B)
	case ssaop.OpARM64PWHILELTH:
		simdPWHILELT(s, v, arm64.ARNG_H)
	case ssaop.OpARM64PWHILELTS:
		simdPWHILELT(s, v, arm64.ARNG_S)
	case ssaop.OpARM64PWHILELTD:
		simdPWHILELT(s, v, arm64.ARNG_D)
	case ssaop.OpARM64ZLD1BPredload:
		// Predicated contiguous byte load, e.g. ZLD1B (VL*0)(R0), P0.Z, [Z0.B].
		// arg0=addr, arg1=pred, arg2=mem.
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.From.Scale = simdSVEVectorLengthScaled
		ssagen.AddAux(&p.From, v)
		p.AddRestSourceReg(pregMask(v.Args[1].Reg(), arm64.PRED_Z))
		p.To.Type = obj.TYPE_REGLIST
		p.To.Offset, _ = arm64.RegisterListOffset(int(pzreg(v.Reg())), 1, regListArr("Z", "B"), 0)
	case ssaop.OpARM64ZST1BPredstore:
		// Predicated contiguous byte store, e.g. ZST1B [Z0.B], P0, (VL*0)(R0).
		// arg0=addr, arg1=Z, arg2=pred, arg3=mem.
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REGLIST
		p.From.Offset, _ = arm64.RegisterListOffset(int(pzreg(v.Args[1].Reg())), 1, regListArr("Z", "B"), 0)
		p.AddRestSourceReg(v.Args[2].Reg()) // store uses a plain governing predicate (no .Z/.M)
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.To.Scale = simdSVEVectorLengthScaled
		ssagen.AddAux(&p.To, v)
	case ssaop.OpARM64FMADDS,
		ssaop.OpARM64FMADDD,
		ssaop.OpARM64FNMADDS,
		ssaop.OpARM64FNMADDD,
		ssaop.OpARM64FMSUBS,
		ssaop.OpARM64FMSUBD,
		ssaop.OpARM64FNMSUBS,
		ssaop.OpARM64FNMSUBD,
		ssaop.OpARM64MADD,
		ssaop.OpARM64MADDW,
		ssaop.OpARM64MSUB,
		ssaop.OpARM64MSUBW:
		rt := v.Reg()
		ra := v.Args[0].Reg()
		rm := v.Args[1].Reg()
		rn := v.Args[2].Reg()
		p := s.Prog(v.Op.Asm())
		p.Reg = ra
		p.From.Type = obj.TYPE_REG
		p.From.Reg = rm
		p.AddRestSourceReg(rn)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = rt
	case ssaop.OpARM64ADDconst,
		ssaop.OpARM64SUBconst,
		ssaop.OpARM64ANDconst,
		ssaop.OpARM64ORconst,
		ssaop.OpARM64XORconst,
		ssaop.OpARM64SLLconst,
		ssaop.OpARM64SRLconst,
		ssaop.OpARM64SRAconst,
		ssaop.OpARM64RORconst,
		ssaop.OpARM64RORWconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARM64ADDSconstflags:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
	case ssaop.OpARM64ADCzerocarry:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = arm64.REGZERO
		p.Reg = arm64.REGZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARM64ADCSflags,
		ssaop.OpARM64ADDSflags,
		ssaop.OpARM64SBCSflags,
		ssaop.OpARM64SUBSflags:
		r := v.Reg0()
		r1 := v.Args[0].Reg()
		r2 := v.Args[1].Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r2
		p.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssaop.OpARM64NEGSflags:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
	case ssaop.OpARM64NGCzerocarry:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = arm64.REGZERO
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARM64EXTRconst,
		ssaop.OpARM64EXTRWconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.AddRestSourceReg(v.Args[0].Reg())
		p.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARM64MVNshiftLL, ssaop.OpARM64NEGshiftLL:
		genshift(s, v, v.Op.Asm(), 0, v.Args[0].Reg(), v.Reg(), arm64.SHIFT_LL, v.AuxInt)
	case ssaop.OpARM64MVNshiftRL, ssaop.OpARM64NEGshiftRL:
		genshift(s, v, v.Op.Asm(), 0, v.Args[0].Reg(), v.Reg(), arm64.SHIFT_LR, v.AuxInt)
	case ssaop.OpARM64MVNshiftRA, ssaop.OpARM64NEGshiftRA:
		genshift(s, v, v.Op.Asm(), 0, v.Args[0].Reg(), v.Reg(), arm64.SHIFT_AR, v.AuxInt)
	case ssaop.OpARM64MVNshiftRO:
		genshift(s, v, v.Op.Asm(), 0, v.Args[0].Reg(), v.Reg(), arm64.SHIFT_ROR, v.AuxInt)
	case ssaop.OpARM64ADDshiftLL,
		ssaop.OpARM64SUBshiftLL,
		ssaop.OpARM64ANDshiftLL,
		ssaop.OpARM64ORshiftLL,
		ssaop.OpARM64XORshiftLL,
		ssaop.OpARM64EONshiftLL,
		ssaop.OpARM64ORNshiftLL,
		ssaop.OpARM64BICshiftLL:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm64.SHIFT_LL, v.AuxInt)
	case ssaop.OpARM64ADDshiftRL,
		ssaop.OpARM64SUBshiftRL,
		ssaop.OpARM64ANDshiftRL,
		ssaop.OpARM64ORshiftRL,
		ssaop.OpARM64XORshiftRL,
		ssaop.OpARM64EONshiftRL,
		ssaop.OpARM64ORNshiftRL,
		ssaop.OpARM64BICshiftRL:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm64.SHIFT_LR, v.AuxInt)
	case ssaop.OpARM64ADDshiftRA,
		ssaop.OpARM64SUBshiftRA,
		ssaop.OpARM64ANDshiftRA,
		ssaop.OpARM64ORshiftRA,
		ssaop.OpARM64XORshiftRA,
		ssaop.OpARM64EONshiftRA,
		ssaop.OpARM64ORNshiftRA,
		ssaop.OpARM64BICshiftRA:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm64.SHIFT_AR, v.AuxInt)
	case ssaop.OpARM64ANDshiftRO,
		ssaop.OpARM64ORshiftRO,
		ssaop.OpARM64XORshiftRO,
		ssaop.OpARM64EONshiftRO,
		ssaop.OpARM64ORNshiftRO,
		ssaop.OpARM64BICshiftRO:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm64.SHIFT_ROR, v.AuxInt)
	case ssaop.OpARM64MOVDconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARM64FMOVSconst,
		ssaop.OpARM64FMOVDconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_FCONST
		p.From.Val = math.Float64frombits(uint64(v.AuxInt))
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARM64FCMPS0,
		ssaop.OpARM64FCMPD0:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_FCONST
		p.From.Val = math.Float64frombits(0)
		p.Reg = v.Args[0].Reg()
	case ssaop.OpARM64CMP,
		ssaop.OpARM64CMPW,
		ssaop.OpARM64CMN,
		ssaop.OpARM64CMNW,
		ssaop.OpARM64TST,
		ssaop.OpARM64TSTW,
		ssaop.OpARM64FCMPS,
		ssaop.OpARM64FCMPD:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.Reg = v.Args[0].Reg()
	case ssaop.OpARM64CMPconst,
		ssaop.OpARM64CMPWconst,
		ssaop.OpARM64CMNconst,
		ssaop.OpARM64CMNWconst,
		ssaop.OpARM64TSTconst,
		ssaop.OpARM64TSTWconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.Reg = v.Args[0].Reg()
	case ssaop.OpARM64CMPshiftLL, ssaop.OpARM64CMNshiftLL, ssaop.OpARM64TSTshiftLL:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), 0, arm64.SHIFT_LL, v.AuxInt)
	case ssaop.OpARM64CMPshiftRL, ssaop.OpARM64CMNshiftRL, ssaop.OpARM64TSTshiftRL:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), 0, arm64.SHIFT_LR, v.AuxInt)
	case ssaop.OpARM64CMPshiftRA, ssaop.OpARM64CMNshiftRA, ssaop.OpARM64TSTshiftRA:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), 0, arm64.SHIFT_AR, v.AuxInt)
	case ssaop.OpARM64TSTshiftRO:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), 0, arm64.SHIFT_ROR, v.AuxInt)
	case ssaop.OpARM64MOVDaddr:
		p := s.Prog(arm64.AMOVD)
		p.From.Type = obj.TYPE_ADDR
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

		var wantreg string
		// MOVD $sym+off(base), R
		// the assembler expands it as the following:
		// - base is SP: add constant offset to SP (R13)
		//               when constant is large, tmp register (R11) may be used
		// - base is SB: load external address from constant pool (use relocation)
		switch v.Aux.(type) {
		default:
			v.Fatalf("aux is of unknown type %T", v.Aux)
		case *obj.LSym:
			wantreg = "SB"
			ssagen.AddAux(&p.From, v)
		case *ir.Name:
			wantreg = "SP"
			ssagen.AddAux(&p.From, v)
		case nil:
			// No sym, just MOVD $off(SP), R
			wantreg = "SP"
			p.From.Offset = v.AuxInt
		}
		if reg := v.Args[0].RegName(); reg != wantreg {
			v.Fatalf("bad reg %s for symbol type %T, want %s", reg, v.Aux, wantreg)
		}
	case ssaop.OpARM64MOVBload,
		ssaop.OpARM64MOVBUload,
		ssaop.OpARM64MOVHload,
		ssaop.OpARM64MOVHUload,
		ssaop.OpARM64MOVWload,
		ssaop.OpARM64MOVWUload,
		ssaop.OpARM64MOVDload,
		ssaop.OpARM64FMOVSload,
		ssaop.OpARM64FMOVDload,
		ssaop.OpARM64FMOVQload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARM64LDP, ssaop.OpARM64LDPW, ssaop.OpARM64LDPSW, ssaop.OpARM64FLDPD, ssaop.OpARM64FLDPS, ssaop.OpARM64FLDPQ:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REGREG
		p.To.Reg = v.Reg0()
		p.To.Offset = int64(v.Reg1())
	case ssaop.OpARM64MOVBloadidx,
		ssaop.OpARM64MOVBUloadidx,
		ssaop.OpARM64MOVHloadidx,
		ssaop.OpARM64MOVHUloadidx,
		ssaop.OpARM64MOVWloadidx,
		ssaop.OpARM64MOVWUloadidx,
		ssaop.OpARM64MOVDloadidx,
		ssaop.OpARM64FMOVSloadidx,
		ssaop.OpARM64FMOVDloadidx,
		ssaop.OpARM64MOVHloadidx2,
		ssaop.OpARM64MOVHUloadidx2,
		ssaop.OpARM64MOVWloadidx4,
		ssaop.OpARM64MOVWUloadidx4,
		ssaop.OpARM64MOVDloadidx8,
		ssaop.OpARM64FMOVDloadidx8,
		ssaop.OpARM64FMOVSloadidx4:
		p := s.Prog(v.Op.Asm())
		p.From = genIndexedOperand(v.Op, v.Args[0].Reg(), v.Args[1].Reg())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARM64LDAR,
		ssaop.OpARM64LDARB,
		ssaop.OpARM64LDARW:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
	case ssaop.OpARM64MOVBstore,
		ssaop.OpARM64MOVHstore,
		ssaop.OpARM64MOVWstore,
		ssaop.OpARM64MOVDstore,
		ssaop.OpARM64FMOVSstore,
		ssaop.OpARM64FMOVDstore,
		ssaop.OpARM64FMOVQstore,
		ssaop.OpARM64STLRB,
		ssaop.OpARM64STLR,
		ssaop.OpARM64STLRW:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssaop.OpARM64MOVBstoreidx,
		ssaop.OpARM64MOVHstoreidx,
		ssaop.OpARM64MOVWstoreidx,
		ssaop.OpARM64MOVDstoreidx,
		ssaop.OpARM64FMOVSstoreidx,
		ssaop.OpARM64FMOVDstoreidx,
		ssaop.OpARM64MOVHstoreidx2,
		ssaop.OpARM64MOVWstoreidx4,
		ssaop.OpARM64FMOVSstoreidx4,
		ssaop.OpARM64MOVDstoreidx8,
		ssaop.OpARM64FMOVDstoreidx8:
		p := s.Prog(v.Op.Asm())
		p.To = genIndexedOperand(v.Op, v.Args[0].Reg(), v.Args[1].Reg())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
	case ssaop.OpARM64STP, ssaop.OpARM64STPW, ssaop.OpARM64FSTPD, ssaop.OpARM64FSTPS, ssaop.OpARM64FSTPQ:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REGREG
		p.From.Reg = v.Args[1].Reg()
		p.From.Offset = int64(v.Args[2].Reg())
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssaop.OpARM64BFI,
		ssaop.OpARM64BFXIL:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt >> 8
		p.AddRestSourceConst(v.AuxInt & 0xff)
		p.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARM64SBFIZ,
		ssaop.OpARM64SBFX,
		ssaop.OpARM64UBFIZ,
		ssaop.OpARM64UBFX:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt >> 8
		p.AddRestSourceConst(v.AuxInt & 0xff)
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARM64LoweredAtomicExchange64,
		ssaop.OpARM64LoweredAtomicExchange32,
		ssaop.OpARM64LoweredAtomicExchange8:
		// LDAXR	(Rarg0), Rout
		// STLXR	Rarg1, (Rarg0), Rtmp
		// CBNZ		Rtmp, -2(PC)
		//
		// If the width written to Rout changes, update zeroUpperBits in ARM64Ops.go.
		var ld, st obj.As
		switch v.Op {
		case ssaop.OpARM64LoweredAtomicExchange8:
			ld = arm64.ALDAXRB
			st = arm64.ASTLXRB
		case ssaop.OpARM64LoweredAtomicExchange32:
			ld = arm64.ALDAXRW
			st = arm64.ASTLXRW
		case ssaop.OpARM64LoweredAtomicExchange64:
			ld = arm64.ALDAXR
			st = arm64.ASTLXR
		}
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		out := v.Reg0()
		p := s.Prog(ld)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = r0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = out
		p1 := s.Prog(st)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = r1
		p1.To.Type = obj.TYPE_MEM
		p1.To.Reg = r0
		p1.RegTo2 = arm64.REGTMP
		p2 := s.Prog(arm64.ACBNZ)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = arm64.REGTMP
		p2.To.Type = obj.TYPE_BRANCH
		p2.To.SetTarget(p)
	case ssaop.OpARM64LoweredAtomicExchange64Variant,
		ssaop.OpARM64LoweredAtomicExchange32Variant,
		ssaop.OpARM64LoweredAtomicExchange8Variant:
		// If the width written to Rout changes, update zeroUpperBits in ARM64Ops.go.
		var swap obj.As
		switch v.Op {
		case ssaop.OpARM64LoweredAtomicExchange8Variant:
			swap = arm64.ASWPALB
		case ssaop.OpARM64LoweredAtomicExchange32Variant:
			swap = arm64.ASWPALW
		case ssaop.OpARM64LoweredAtomicExchange64Variant:
			swap = arm64.ASWPALD
		}
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		out := v.Reg0()

		// SWPALD	Rarg1, (Rarg0), Rout
		p := s.Prog(swap)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r1
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = r0
		p.RegTo2 = out

	case ssaop.OpARM64LoweredAtomicAdd64,
		ssaop.OpARM64LoweredAtomicAdd32:
		// LDAXR	(Rarg0), Rout
		// ADD		Rarg1, Rout
		// STLXR	Rout, (Rarg0), Rtmp
		// CBNZ		Rtmp, -3(PC)
		ld := arm64.ALDAXR
		st := arm64.ASTLXR
		if v.Op == ssaop.OpARM64LoweredAtomicAdd32 {
			ld = arm64.ALDAXRW
			st = arm64.ASTLXRW
		}
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		out := v.Reg0()
		p := s.Prog(ld)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = r0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = out
		p1 := s.Prog(arm64.AADD)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = r1
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = out
		p2 := s.Prog(st)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = out
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = r0
		p2.RegTo2 = arm64.REGTMP
		p3 := s.Prog(arm64.ACBNZ)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = arm64.REGTMP
		p3.To.Type = obj.TYPE_BRANCH
		p3.To.SetTarget(p)
	case ssaop.OpARM64LoweredAtomicAdd64Variant,
		ssaop.OpARM64LoweredAtomicAdd32Variant:
		// LDADDAL	Rarg1, (Rarg0), Rout
		// ADD		Rarg1, Rout
		op := arm64.ALDADDALD
		if v.Op == ssaop.OpARM64LoweredAtomicAdd32Variant {
			op = arm64.ALDADDALW
		}
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		out := v.Reg0()
		p := s.Prog(op)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r1
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = r0
		p.RegTo2 = out
		p1 := s.Prog(arm64.AADD)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = r1
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = out
	case ssaop.OpARM64LoweredAtomicCas64,
		ssaop.OpARM64LoweredAtomicCas32:
		// LDAXR	(Rarg0), Rtmp
		// CMP		Rarg1, Rtmp
		// BNE		3(PC)
		// STLXR	Rarg2, (Rarg0), Rtmp
		// CBNZ		Rtmp, -4(PC)
		// CSET		EQ, Rout
		//
		// If Rout stops being written only by CSET, update zeroUpperBits in ARM64Ops.go.
		ld := arm64.ALDAXR
		st := arm64.ASTLXR
		cmp := arm64.ACMP
		if v.Op == ssaop.OpARM64LoweredAtomicCas32 {
			ld = arm64.ALDAXRW
			st = arm64.ASTLXRW
			cmp = arm64.ACMPW
		}
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		r2 := v.Args[2].Reg()
		out := v.Reg0()
		p := s.Prog(ld)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = r0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = arm64.REGTMP
		p1 := s.Prog(cmp)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = r1
		p1.Reg = arm64.REGTMP
		p2 := s.Prog(arm64.ABNE)
		p2.To.Type = obj.TYPE_BRANCH
		p3 := s.Prog(st)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = r2
		p3.To.Type = obj.TYPE_MEM
		p3.To.Reg = r0
		p3.RegTo2 = arm64.REGTMP
		p4 := s.Prog(arm64.ACBNZ)
		p4.From.Type = obj.TYPE_REG
		p4.From.Reg = arm64.REGTMP
		p4.To.Type = obj.TYPE_BRANCH
		p4.To.SetTarget(p)
		p5 := s.Prog(arm64.ACSET)
		p5.From.Type = obj.TYPE_SPECIAL // assembler encodes conditional bits in Offset
		p5.From.Offset = int64(arm64.SPOP_EQ)
		p5.To.Type = obj.TYPE_REG
		p5.To.Reg = out
		p2.To.SetTarget(p5)
	case ssaop.OpARM64LoweredAtomicCas64Variant,
		ssaop.OpARM64LoweredAtomicCas32Variant:
		// Rarg0: ptr
		// Rarg1: old
		// Rarg2: new
		// MOV  	Rarg1, Rtmp
		// CASAL	Rtmp, (Rarg0), Rarg2
		// CMP  	Rarg1, Rtmp
		// CSET 	EQ, Rout
		//
		// If Rout stops being written only by CSET, update zeroUpperBits in ARM64Ops.go.
		cas := arm64.ACASALD
		cmp := arm64.ACMP
		mov := arm64.AMOVD
		if v.Op == ssaop.OpARM64LoweredAtomicCas32Variant {
			cas = arm64.ACASALW
			cmp = arm64.ACMPW
			mov = arm64.AMOVW
		}
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		r2 := v.Args[2].Reg()
		out := v.Reg0()

		// MOV  	Rarg1, Rtmp
		p := s.Prog(mov)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = arm64.REGTMP

		// CASAL	Rtmp, (Rarg0), Rarg2
		p1 := s.Prog(cas)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = arm64.REGTMP
		p1.To.Type = obj.TYPE_MEM
		p1.To.Reg = r0
		p1.RegTo2 = r2

		// CMP  	Rarg1, Rtmp
		p2 := s.Prog(cmp)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = r1
		p2.Reg = arm64.REGTMP

		// CSET 	EQ, Rout
		p3 := s.Prog(arm64.ACSET)
		p3.From.Type = obj.TYPE_SPECIAL // assembler encodes conditional bits in Offset
		p3.From.Offset = int64(arm64.SPOP_EQ)
		p3.To.Type = obj.TYPE_REG
		p3.To.Reg = out

	case ssaop.OpARM64LoweredAtomicAnd64,
		ssaop.OpARM64LoweredAtomicOr64,
		ssaop.OpARM64LoweredAtomicAnd32,
		ssaop.OpARM64LoweredAtomicOr32,
		ssaop.OpARM64LoweredAtomicAnd8,
		ssaop.OpARM64LoweredAtomicOr8:
		// LDAXR[BW] (Rarg0), Rout
		// AND/OR	Rarg1, Rout, tmp1
		// STLXR[BW] tmp1, (Rarg0), Rtmp
		// CBNZ		Rtmp, -3(PC)
		//
		// If the width written to Rout changes, update zeroUpperBits in ARM64Ops.go.
		ld := arm64.ALDAXR
		st := arm64.ASTLXR
		if v.Op == ssaop.OpARM64LoweredAtomicAnd32 || v.Op == ssaop.OpARM64LoweredAtomicOr32 {
			ld = arm64.ALDAXRW
			st = arm64.ASTLXRW
		}
		if v.Op == ssaop.OpARM64LoweredAtomicAnd8 || v.Op == ssaop.OpARM64LoweredAtomicOr8 {
			ld = arm64.ALDAXRB
			st = arm64.ASTLXRB
		}
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		out := v.Reg0()
		tmp := v.RegTmp()
		p := s.Prog(ld)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = r0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = out
		p1 := s.Prog(v.Op.Asm())
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = r1
		p1.Reg = out
		p1.To.Type = obj.TYPE_REG
		p1.To.Reg = tmp
		p2 := s.Prog(st)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = tmp
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = r0
		p2.RegTo2 = arm64.REGTMP
		p3 := s.Prog(arm64.ACBNZ)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = arm64.REGTMP
		p3.To.Type = obj.TYPE_BRANCH
		p3.To.SetTarget(p)

	case ssaop.OpARM64LoweredAtomicAnd8Variant,
		ssaop.OpARM64LoweredAtomicAnd32Variant,
		ssaop.OpARM64LoweredAtomicAnd64Variant:
		// If the width written to Rout changes, update zeroUpperBits in ARM64Ops.go.
		atomic_clear := arm64.ALDCLRALD
		if v.Op == ssaop.OpARM64LoweredAtomicAnd32Variant {
			atomic_clear = arm64.ALDCLRALW
		}
		if v.Op == ssaop.OpARM64LoweredAtomicAnd8Variant {
			atomic_clear = arm64.ALDCLRALB
		}
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		out := v.Reg0()

		// MNV       Rarg1 Rtemp
		p := s.Prog(arm64.AMVN)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = arm64.REGTMP

		// LDCLRAL[BDW]  Rtemp, (Rarg0), Rout
		p1 := s.Prog(atomic_clear)
		p1.From.Type = obj.TYPE_REG
		p1.From.Reg = arm64.REGTMP
		p1.To.Type = obj.TYPE_MEM
		p1.To.Reg = r0
		p1.RegTo2 = out

	case ssaop.OpARM64LoweredAtomicOr8Variant,
		ssaop.OpARM64LoweredAtomicOr32Variant,
		ssaop.OpARM64LoweredAtomicOr64Variant:
		// If the width written to Rout changes, update zeroUpperBits in ARM64Ops.go.
		atomic_or := arm64.ALDORALD
		if v.Op == ssaop.OpARM64LoweredAtomicOr32Variant {
			atomic_or = arm64.ALDORALW
		}
		if v.Op == ssaop.OpARM64LoweredAtomicOr8Variant {
			atomic_or = arm64.ALDORALB
		}
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		out := v.Reg0()

		// LDORAL[BDW]  Rarg1, (Rarg0), Rout
		p := s.Prog(atomic_or)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r1
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = r0
		p.RegTo2 = out

	case ssaop.OpARM64MOVBreg,
		ssaop.OpARM64MOVBUreg,
		ssaop.OpARM64MOVHreg,
		ssaop.OpARM64MOVHUreg,
		ssaop.OpARM64MOVWreg,
		ssaop.OpARM64MOVWUreg:
		a := v.Args[0]
		for a.Op == ssaop.OpCopy || a.Op == ssaop.OpARM64MOVDreg {
			a = a.Args[0]
		}
		if a.Op == ssaop.OpLoadReg {
			t := a.Type
			switch {
			case v.Op == ssaop.OpARM64MOVBreg && t.Size() == 1 && t.IsSigned(),
				v.Op == ssaop.OpARM64MOVBUreg && t.Size() == 1 && !t.IsSigned(),
				v.Op == ssaop.OpARM64MOVHreg && t.Size() == 2 && t.IsSigned(),
				v.Op == ssaop.OpARM64MOVHUreg && t.Size() == 2 && !t.IsSigned(),
				v.Op == ssaop.OpARM64MOVWreg && t.Size() == 4 && t.IsSigned(),
				v.Op == ssaop.OpARM64MOVWUreg && t.Size() == 4 && !t.IsSigned():
				// arg is a proper-typed load, already zero/sign-extended, don't extend again
				if v.Reg() == v.Args[0].Reg() {
					return
				}
				p := s.Prog(arm64.AMOVD)
				p.From.Type = obj.TYPE_REG
				p.From.Reg = v.Args[0].Reg()
				p.To.Type = obj.TYPE_REG
				p.To.Reg = v.Reg()
				return
			default:
			}
		}
		fallthrough
	case ssaop.OpARM64MVN,
		ssaop.OpARM64NEG,
		ssaop.OpARM64FABSD,
		ssaop.OpARM64FABSS,
		ssaop.OpARM64FMOVDfpgp,
		ssaop.OpARM64FMOVDgpfp,
		ssaop.OpARM64FMOVSfpgp,
		ssaop.OpARM64FMOVSgpfp,
		ssaop.OpARM64FNEGS,
		ssaop.OpARM64FNEGD,
		ssaop.OpARM64FSQRTS,
		ssaop.OpARM64FSQRTD,
		ssaop.OpARM64FCVTZSSW,
		ssaop.OpARM64FCVTZSDW,
		ssaop.OpARM64FCVTZUSW,
		ssaop.OpARM64FCVTZUDW,
		ssaop.OpARM64FCVTZSS,
		ssaop.OpARM64FCVTZSD,
		ssaop.OpARM64FCVTZUS,
		ssaop.OpARM64FCVTZUD,
		ssaop.OpARM64SCVTFWS,
		ssaop.OpARM64SCVTFWD,
		ssaop.OpARM64SCVTFS,
		ssaop.OpARM64SCVTFD,
		ssaop.OpARM64UCVTFWS,
		ssaop.OpARM64UCVTFWD,
		ssaop.OpARM64UCVTFS,
		ssaop.OpARM64UCVTFD,
		ssaop.OpARM64FCVTSD,
		ssaop.OpARM64FCVTDS,
		ssaop.OpARM64REV,
		ssaop.OpARM64REVW,
		ssaop.OpARM64REV16,
		ssaop.OpARM64REV16W,
		ssaop.OpARM64RBIT,
		ssaop.OpARM64RBITW,
		ssaop.OpARM64CLZ,
		ssaop.OpARM64CLZW,
		ssaop.OpARM64FRINTAD,
		ssaop.OpARM64FRINTMD,
		ssaop.OpARM64FRINTND,
		ssaop.OpARM64FRINTPD,
		ssaop.OpARM64FRINTZD,
		ssaop.OpARM64FRINTAS,
		ssaop.OpARM64FRINTMS,
		ssaop.OpARM64FRINTNS,
		ssaop.OpARM64FRINTPS,
		ssaop.OpARM64FRINTZS:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARM64LoweredRound32F, ssaop.OpARM64LoweredRound64F:
		// input is already rounded
	case ssaop.OpARM64VCNT:
		simdV11(s, v, arm64.ARNG_8B)
	case ssaop.OpARM64VUADDLV:
		simdV11Scalar(s, v, arm64.ARNG_8B)
	case ssaop.OpARM64CSEL, ssaop.OpARM64CSEL0, ssaop.OpARM64FCSELD, ssaop.OpARM64FCSELS:
		r1 := int16(arm64.REGZERO)
		if v.Op != ssaop.OpARM64CSEL0 {
			r1 = v.Args[1].Reg()
		}
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_SPECIAL // assembler encodes conditional bits in Offset
		condCode := condBits[ssaop.Op(v.AuxInt)]
		p.From.Offset = int64(condCode)
		p.Reg = v.Args[0].Reg()
		p.AddRestSourceReg(r1)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARM64CSINC, ssaop.OpARM64CSINV, ssaop.OpARM64CSNEG:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_SPECIAL // assembler encodes conditional bits in Offset
		condCode := condBits[ssaop.Op(v.AuxInt)]
		p.From.Offset = int64(condCode)
		p.Reg = v.Args[0].Reg()
		p.AddRestSourceReg(v.Args[1].Reg())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARM64CSETM:
		p := s.Prog(arm64.ACSETM)
		p.From.Type = obj.TYPE_SPECIAL // assembler encodes conditional bits in Offset
		condCode := condBits[ssaop.Op(v.AuxInt)]
		p.From.Offset = int64(condCode)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARM64CCMP,
		ssaop.OpARM64CCMN,
		ssaop.OpARM64CCMPconst,
		ssaop.OpARM64CCMNconst,
		ssaop.OpARM64CCMPW,
		ssaop.OpARM64CCMNW,
		ssaop.OpARM64CCMPWconst,
		ssaop.OpARM64CCMNWconst:
		p := s.Prog(v.Op.Asm())
		p.Reg = v.Args[0].Reg()
		params := v.AuxArm64ConditionalParams()
		p.From.Type = obj.TYPE_SPECIAL // assembler encodes conditional bits in Offset
		p.From.Offset = int64(condBits[params.Cond])
		constValue, ok := params.ConstValue()
		if ok {
			p.AddRestSourceConst(constValue)
		} else {
			p.AddRestSourceReg(v.Args[1].Reg())
		}
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = params.Nzcv()
	case ssaop.OpARM64LoweredZero:
		ptrReg := v.Args[0].Reg()
		n := v.AuxInt
		if n < 16 {
			v.Fatalf("Zero too small %d", n)
		}

		// Generate zeroing instructions.
		var off int64
		for n >= 16 {
			//  STP     (ZR, ZR), off(ptrReg)
			zero16(s, ptrReg, off, false)
			off += 16
			n -= 16
		}
		// Write any fractional portion.
		// An overlapping 16-byte write can't be used here
		// because STP's offsets must be a multiple of 8.
		if n > 8 {
			//  MOVD    ZR, off(ptrReg)
			zero8(s, ptrReg, off)
			off += 8
			n -= 8
		}
		if n != 0 {
			//  MOVD    ZR, off+n-8(ptrReg)
			// TODO: for n<=4 we could use a smaller write.
			zero8(s, ptrReg, off+n-8)
		}
	case ssaop.OpARM64LoweredZeroLoop:
		ptrReg := v.Args[0].Reg()
		countReg := v.RegTmp()
		n := v.AuxInt
		loopSize := int64(64)
		if n < 3*loopSize {
			// - a loop count of 0 won't work.
			// - a loop count of 1 is useless.
			// - a loop count of 2 is a code size ~tie
			//     3 instructions to implement the loop
			//     4 instructions in the loop body
			//   vs
			//     8 instructions in the straightline code
			//   Might as well use straightline code.
			v.Fatalf("ZeroLoop size too small %d", n)
		}

		// Put iteration count in a register.
		//   MOVD    $n, countReg
		p := s.Prog(arm64.AMOVD)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = n / loopSize
		p.To.Type = obj.TYPE_REG
		p.To.Reg = countReg
		cntInit := p

		// Zero loopSize bytes starting at ptrReg.
		// Increment ptrReg by loopSize as a side effect.
		for range loopSize / 16 {
			//  STP.P   (ZR, ZR), 16(ptrReg)
			zero16(s, ptrReg, 0, true)
			// TODO: should we use the postincrement form,
			// or use a separate += 64 instruction?
			// postincrement saves an instruction, but maybe
			// it requires more integer units to do the +=16s.
		}
		// Decrement loop count.
		//   SUB     $1, countReg
		p = s.Prog(arm64.ASUB)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = countReg
		// Jump to loop header if we're not done yet.
		//   CBNZ    head
		p = s.Prog(arm64.ACBNZ)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = countReg
		p.To.Type = obj.TYPE_BRANCH
		p.To.SetTarget(cntInit.Link)

		// Multiples of the loop size are now done.
		n %= loopSize

		// Write any fractional portion.
		var off int64
		for n >= 16 {
			//  STP     (ZR, ZR), off(ptrReg)
			zero16(s, ptrReg, off, false)
			off += 16
			n -= 16
		}
		if n > 8 {
			// Note: an overlapping 16-byte write can't be used
			// here because STP's offsets must be a multiple of 8.
			//  MOVD    ZR, off(ptrReg)
			zero8(s, ptrReg, off)
			off += 8
			n -= 8
		}
		if n != 0 {
			//  MOVD    ZR, off+n-8(ptrReg)
			// TODO: for n<=4 we could use a smaller write.
			zero8(s, ptrReg, off+n-8)
		}
		// TODO: maybe we should use the count register to instead
		// hold an end pointer and compare against that?
		//   ADD $n, ptrReg, endReg
		// then
		//   CMP ptrReg, endReg
		//   BNE loop
		// There's a past-the-end pointer here, any problem with that?

	case ssaop.OpARM64LoweredMove:
		dstReg := v.Args[0].Reg()
		srcReg := v.Args[1].Reg()
		if dstReg == srcReg {
			break
		}
		tmpReg1 := int16(arm64.REG_R25)
		tmpFReg1 := int16(arm64.REG_F16)
		tmpFReg2 := int16(arm64.REG_F17)
		n := v.AuxInt
		if n < 16 {
			v.Fatalf("Move too small %d", n)
		}

		// Generate copying instructions.
		var off int64
		for n >= 32 {
			//  FLDPQ   off(srcReg), (tmpFReg1, tmpFReg2)
			//  FSTPQ   (tmpFReg1, tmpFReg2), off(dstReg)
			move32(s, srcReg, dstReg, tmpFReg1, tmpFReg2, off, false)
			off += 32
			n -= 32
		}
		for n >= 16 {
			//  FMOVQ   off(src), tmpFReg1
			//  FMOVQ   tmpFReg1, off(dst)
			move16(s, srcReg, dstReg, tmpFReg1, off, false)
			off += 16
			n -= 16
		}
		if n > 8 {
			//  MOVD    off(srcReg), tmpReg1
			//  MOVD    tmpReg1, off(dstReg)
			move8(s, srcReg, dstReg, tmpReg1, off)
			off += 8
			n -= 8
		}
		if n != 0 {
			//  MOVD    off+n-8(srcReg), tmpReg1
			//  MOVD    tmpReg1, off+n-8(dstReg)
			move8(s, srcReg, dstReg, tmpReg1, off+n-8)
		}
	case ssaop.OpARM64LoweredMoveLoop:
		dstReg := v.Args[0].Reg()
		srcReg := v.Args[1].Reg()
		if dstReg == srcReg {
			break
		}
		countReg := int16(arm64.REG_R24)
		tmpReg1 := int16(arm64.REG_R25)
		tmpFReg1 := int16(arm64.REG_F16)
		tmpFReg2 := int16(arm64.REG_F17)
		n := v.AuxInt
		loopSize := int64(64)
		if n < 3*loopSize {
			// - a loop count of 0 won't work.
			// - a loop count of 1 is useless.
			// - a loop count of 2 is a code size ~tie
			//     3 instructions to implement the loop
			//     4 instructions in the loop body
			//   vs
			//     8 instructions in the straightline code
			//   Might as well use straightline code.
			v.Fatalf("ZeroLoop size too small %d", n)
		}

		// Put iteration count in a register.
		//   MOVD    $n, countReg
		p := s.Prog(arm64.AMOVD)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = n / loopSize
		p.To.Type = obj.TYPE_REG
		p.To.Reg = countReg
		cntInit := p

		// Move loopSize bytes starting at srcReg to dstReg.
		// Increment srcReg and destReg by loopSize as a side effect.
		for range loopSize / 32 {
			// FLDPQ.P 32(srcReg), (tmpFReg1, tmpFReg2)
			// FSTPQ.P (tmpFReg1, tmpFReg2), 32(dstReg)
			move32(s, srcReg, dstReg, tmpFReg1, tmpFReg2, 0, true)
		}
		// Decrement loop count.
		//   SUB     $1, countReg
		p = s.Prog(arm64.ASUB)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = countReg
		// Jump to loop header if we're not done yet.
		//   CBNZ    head
		p = s.Prog(arm64.ACBNZ)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = countReg
		p.To.Type = obj.TYPE_BRANCH
		p.To.SetTarget(cntInit.Link)

		// Multiples of the loop size are now done.
		n %= loopSize

		// Copy any fractional portion.
		var off int64
		for n >= 32 {
			//  FLDPQ   off(srcReg), (tmpFReg1, tmpFReg2)
			//  FSTPQ   (tmpFReg1, tmpFReg2), off(dstReg)
			move32(s, srcReg, dstReg, tmpFReg1, tmpFReg2, off, false)
			off += 32
			n -= 32
		}
		for n >= 16 {
			//  FMOVQ   off(src), tmpFReg1
			//  FMOVQ   tmpFReg1, off(dst)
			move16(s, srcReg, dstReg, tmpFReg1, off, false)
			off += 16
			n -= 16
		}
		if n > 8 {
			//  MOVD    off(srcReg), tmpReg1
			//  MOVD    tmpReg1, off(dstReg)
			move8(s, srcReg, dstReg, tmpReg1, off)
			off += 8
			n -= 8
		}
		if n != 0 {
			//  MOVD    off+n-8(srcReg), tmpReg1
			//  MOVD    tmpReg1, off+n-8(dstReg)
			move8(s, srcReg, dstReg, tmpReg1, off+n-8)
		}

	case ssaop.OpARM64CALLstatic, ssaop.OpARM64CALLclosure, ssaop.OpARM64CALLinter:
		s.Call(v)
	case ssaop.OpARM64CALLtail, ssaop.OpARM64CALLtailinter:
		s.TailCall(v)
	case ssaop.OpARM64LoweredWB:
		p := s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		// AuxInt encodes how many buffer entries we need.
		p.To.Sym = ir.Syms.GCWriteBarrier[v.AuxInt-1]
	case ssaop.OpARM64LoweredMemEq:
		p := s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ir.Syms.Memequal

	case ssaop.OpARM64LoweredPanicBoundsRR, ssaop.OpARM64LoweredPanicBoundsRC, ssaop.OpARM64LoweredPanicBoundsCR, ssaop.OpARM64LoweredPanicBoundsCC:
		// Compute the constant we put in the PCData entry for this call.
		code, signed := ssa.BoundsKind(v.AuxInt).Code()
		xIsReg := false
		yIsReg := false
		xVal := 0
		yVal := 0
		switch v.Op {
		case ssaop.OpARM64LoweredPanicBoundsRR:
			xIsReg = true
			xVal = int(v.Args[0].Reg() - arm64.REG_R0)
			yIsReg = true
			yVal = int(v.Args[1].Reg() - arm64.REG_R0)
		case ssaop.OpARM64LoweredPanicBoundsRC:
			xIsReg = true
			xVal = int(v.Args[0].Reg() - arm64.REG_R0)
			c := v.Aux.(ssa.PanicBoundsC).C
			if c >= 0 && c <= abi.BoundsMaxConst {
				yVal = int(c)
			} else {
				// Move constant to a register
				yIsReg = true
				if yVal == xVal {
					yVal = 1
				}
				p := s.Prog(arm64.AMOVD)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = arm64.REG_R0 + int16(yVal)
			}
		case ssaop.OpARM64LoweredPanicBoundsCR:
			yIsReg = true
			yVal = int(v.Args[0].Reg() - arm64.REG_R0)
			c := v.Aux.(ssa.PanicBoundsC).C
			if c >= 0 && c <= abi.BoundsMaxConst {
				xVal = int(c)
			} else {
				// Move constant to a register
				if xVal == yVal {
					xVal = 1
				}
				p := s.Prog(arm64.AMOVD)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = arm64.REG_R0 + int16(xVal)
			}
		case ssaop.OpARM64LoweredPanicBoundsCC:
			c := v.Aux.(ssa.PanicBoundsCC).Cx
			if c >= 0 && c <= abi.BoundsMaxConst {
				xVal = int(c)
			} else {
				// Move constant to a register
				xIsReg = true
				p := s.Prog(arm64.AMOVD)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = arm64.REG_R0 + int16(xVal)
			}
			c = v.Aux.(ssa.PanicBoundsCC).Cy
			if c >= 0 && c <= abi.BoundsMaxConst {
				yVal = int(c)
			} else {
				// Move constant to a register
				yIsReg = true
				yVal = 1
				p := s.Prog(arm64.AMOVD)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = arm64.REG_R0 + int16(yVal)
			}
		}
		c := abi.BoundsEncode(code, signed, xIsReg, yIsReg, xVal, yVal)

		p := s.Prog(obj.APCDATA)
		p.From.SetConst(abi.PCDATA_PanicBounds)
		p.To.SetConst(int64(c))
		p = s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ir.Syms.PanicBounds

	case ssaop.OpARM64LoweredNilCheck:
		// Issue a load which will fault if arg is nil.
		p := s.Prog(arm64.AMOVB)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = arm64.REGTMP
		if logopt.Enabled() {
			logopt.LogOpt(v.Pos, "nilcheck", "genssa", v.Block.Func.Name)
		}
		if base.Debug.Nil != 0 && v.Pos.Line() > 1 { // v.Line==1 in generated wrappers
			base.WarnfAt(v.Pos, "generated nil check")
		}
	case ssaop.OpARM64Equal,
		ssaop.OpARM64NotEqual,
		ssaop.OpARM64LessThan,
		ssaop.OpARM64LessEqual,
		ssaop.OpARM64GreaterThan,
		ssaop.OpARM64GreaterEqual,
		ssaop.OpARM64LessThanU,
		ssaop.OpARM64LessEqualU,
		ssaop.OpARM64GreaterThanU,
		ssaop.OpARM64GreaterEqualU,
		ssaop.OpARM64LessThanF,
		ssaop.OpARM64LessEqualF,
		ssaop.OpARM64GreaterThanF,
		ssaop.OpARM64GreaterEqualF,
		ssaop.OpARM64NotLessThanF,
		ssaop.OpARM64NotLessEqualF,
		ssaop.OpARM64NotGreaterThanF,
		ssaop.OpARM64NotGreaterEqualF,
		ssaop.OpARM64LessThanNoov,
		ssaop.OpARM64GreaterEqualNoov:
		// generate boolean values using CSET
		//
		// If the result stops being a 0/1-producing CSET, update zeroUpperBits in ARM64Ops.go.
		p := s.Prog(arm64.ACSET)
		p.From.Type = obj.TYPE_SPECIAL // assembler encodes conditional bits in Offset
		condCode := condBits[v.Op]
		p.From.Offset = int64(condCode)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARM64PRFM:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = v.AuxInt
	case ssaop.OpARM64LoweredGetClosurePtr:
		// Closure pointer is R26 (arm64.REGCTXT).
		ssagen.CheckLoweredGetClosurePtr(v)
	case ssaop.OpARM64LoweredGetCallerSP:
		// caller's SP is FixedFrameSize below the address of the first arg
		p := s.Prog(arm64.AMOVD)
		p.From.Type = obj.TYPE_ADDR
		p.From.Offset = -base.Ctxt.Arch.FixedFrameSize
		p.From.Name = obj.NAME_PARAM
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARM64LoweredGetCallerPC:
		p := s.Prog(obj.AGETCALLERPC)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARM64DMB:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
	case ssaop.OpARM64FlagConstant:
		v.Fatalf("FlagConstant op should never make it to codegen %v", v.LongString())
	case ssaop.OpARM64InvertFlags:
		v.Fatalf("InvertFlags should never make it to codegen %v", v.LongString())
	case ssaop.OpClobber:
		// MOVW	$0xdeaddead, REGTMP
		// MOVW	REGTMP, (slot)
		// MOVW	REGTMP, 4(slot)
		p := s.Prog(arm64.AMOVW)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 0xdeaddead
		p.To.Type = obj.TYPE_REG
		p.To.Reg = arm64.REGTMP
		p = s.Prog(arm64.AMOVW)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = arm64.REGTMP
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = arm64.REGSP
		ssagen.AddAux(&p.To, v)
		p = s.Prog(arm64.AMOVW)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = arm64.REGTMP
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = arm64.REGSP
		ssagen.AddAux2(&p.To, v, v.AuxInt+4)
	case ssaop.OpClobberReg:
		x := uint64(0xdeaddeaddeaddead)
		p := s.Prog(arm64.AMOVD)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(x)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	default:
		if !ssaGenSIMDValue(s, v) && !ssaGenSIMDSVEValue(s, v) {
			v.Fatalf("genValue not implemented: %s", v.LongString())
		}
	}
}

// sveStackAddr materializes the byte address of SVE stack slot into REGTMP and
// returns a memory operand addressing it with a zero VL-scaled offset. SVE Z/P
// loads and stores only support VL-scaled immediate addressing, so a fixed byte
// frame offset — which is not a compile-time multiple of the runtime VL and can
// exceed the ±256-VL immediate range — must be reached through a register.
func sveStackAddr(s *ssagen.State, slot *ssa.Value) obj.Addr {
	p := s.Prog(arm64.AMOVD)
	ssagen.AddrAuto(&p.From, slot)
	p.From.Type = obj.TYPE_ADDR // MOVD $slot(SP), REGTMP: address of the slot
	p.To.Type = obj.TYPE_REG
	p.To.Reg = arm64.REGTMP
	return obj.Addr{Type: obj.TYPE_MEM, Reg: arm64.REGTMP, Scale: simdSVEVectorLengthScaled}
}

// simdZ21 emits an unpredicated SVE binary Z-register instruction with the given
// element arrangement, e.g. ZADD Z2.B, Z0.B, Z1.B.
func simdZ21(s *ssagen.State, v *ssa.Value, arng int16) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = zregArng(v.Args[1].Reg(), arng)        // Zm
	p.AddRestSourceReg(zregArng(v.Args[0].Reg(), arng)) // Zn
	p.To.Type = obj.TYPE_REG
	p.To.Reg = zregArng(v.Reg(), arng) // Zd
	return p
}

// simdPWHILELT emits a PWHILELT that fills a predicate with lanes [lo,hi) set for
// the given element arrangement, e.g. PWHILELT R0, R1, P0.B. SSA provides
// arg0=lo, arg1=hi.
func simdPWHILELT(s *ssagen.State, v *ssa.Value, arng int16) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = v.Args[1].Reg()
	p.AddRestSourceReg(v.Args[0].Reg())
	p.To.Type = obj.TYPE_REG
	p.To.Reg = pregArng(v.Reg0(), arng)
	return p
}

// simdZ2kk emits a predicated SVE integer compare that produces a predicate
// mask, e.g. ZCMPGT Z1.B, Z0.B, P0.Z, P1.B. SSA provides arg0=x (Zn), arg1=y
// (Zm) and arg2=governing predicate (Pg); the result is the predicate mask Pd.
func simdZ2kk(s *ssagen.State, v *ssa.Value, arng int16) *obj.Prog {
	p := s.Prog(v.Op.Asm())
	p.From.Type = obj.TYPE_REG
	p.From.Reg = zregArng(v.Args[1].Reg(), arng)                // Zm
	p.AddRestSourceReg(zregArng(v.Args[0].Reg(), arng))         // Zn
	p.AddRestSourceReg(pregMask(v.Args[2].Reg(), arm64.PRED_Z)) // Pg/Z
	p.To.Type = obj.TYPE_REG
	p.To.Reg = pregArng(v.Reg(), arng) // Pd
	return p
}

var condBits = map[ssaop.Op]arm64.SpecialOperand{
	ssaop.OpARM64Equal:         arm64.SPOP_EQ,
	ssaop.OpARM64NotEqual:      arm64.SPOP_NE,
	ssaop.OpARM64LessThan:      arm64.SPOP_LT,
	ssaop.OpARM64LessThanU:     arm64.SPOP_LO,
	ssaop.OpARM64LessEqual:     arm64.SPOP_LE,
	ssaop.OpARM64LessEqualU:    arm64.SPOP_LS,
	ssaop.OpARM64GreaterThan:   arm64.SPOP_GT,
	ssaop.OpARM64GreaterThanU:  arm64.SPOP_HI,
	ssaop.OpARM64GreaterEqual:  arm64.SPOP_GE,
	ssaop.OpARM64GreaterEqualU: arm64.SPOP_HS,
	ssaop.OpARM64LessThanF:     arm64.SPOP_MI, // Less than
	ssaop.OpARM64LessEqualF:    arm64.SPOP_LS, // Less than or equal to
	ssaop.OpARM64GreaterThanF:  arm64.SPOP_GT, // Greater than
	ssaop.OpARM64GreaterEqualF: arm64.SPOP_GE, // Greater than or equal to

	// The following condition codes have unordered to handle comparisons related to NaN.
	ssaop.OpARM64NotLessThanF:     arm64.SPOP_PL, // Greater than, equal to, or unordered
	ssaop.OpARM64NotLessEqualF:    arm64.SPOP_HI, // Greater than or unordered
	ssaop.OpARM64NotGreaterThanF:  arm64.SPOP_LE, // Less than, equal to or unordered
	ssaop.OpARM64NotGreaterEqualF: arm64.SPOP_LT, // Less than or unordered

	ssaop.OpARM64LessThanNoov:     arm64.SPOP_MI, // Less than but without honoring overflow
	ssaop.OpARM64GreaterEqualNoov: arm64.SPOP_PL, // Greater than or equal to but without honoring overflow
}

var blockJump = map[block.BlockKind]struct {
	asm, invasm obj.As
}{
	block.BlockARM64EQ:     {arm64.ABEQ, arm64.ABNE},
	block.BlockARM64NE:     {arm64.ABNE, arm64.ABEQ},
	block.BlockARM64LT:     {arm64.ABLT, arm64.ABGE},
	block.BlockARM64GE:     {arm64.ABGE, arm64.ABLT},
	block.BlockARM64LE:     {arm64.ABLE, arm64.ABGT},
	block.BlockARM64GT:     {arm64.ABGT, arm64.ABLE},
	block.BlockARM64ULT:    {arm64.ABLO, arm64.ABHS},
	block.BlockARM64UGE:    {arm64.ABHS, arm64.ABLO},
	block.BlockARM64UGT:    {arm64.ABHI, arm64.ABLS},
	block.BlockARM64ULE:    {arm64.ABLS, arm64.ABHI},
	block.BlockARM64Z:      {arm64.ACBZ, arm64.ACBNZ},
	block.BlockARM64NZ:     {arm64.ACBNZ, arm64.ACBZ},
	block.BlockARM64ZW:     {arm64.ACBZW, arm64.ACBNZW},
	block.BlockARM64NZW:    {arm64.ACBNZW, arm64.ACBZW},
	block.BlockARM64TBZ:    {arm64.ATBZ, arm64.ATBNZ},
	block.BlockARM64TBNZ:   {arm64.ATBNZ, arm64.ATBZ},
	block.BlockARM64FLT:    {arm64.ABMI, arm64.ABPL},
	block.BlockARM64FGE:    {arm64.ABGE, arm64.ABLT},
	block.BlockARM64FLE:    {arm64.ABLS, arm64.ABHI},
	block.BlockARM64FGT:    {arm64.ABGT, arm64.ABLE},
	block.BlockARM64LTnoov: {arm64.ABMI, arm64.ABPL},
	block.BlockARM64GEnoov: {arm64.ABPL, arm64.ABMI},
}

// To model a 'LEnoov' ('<=' without overflow checking) branching.
var leJumps = [2][2]ssagen.IndexJump{
	{{Jump: arm64.ABEQ, Index: 0}, {Jump: arm64.ABPL, Index: 1}}, // next == b.Succs[0]
	{{Jump: arm64.ABMI, Index: 0}, {Jump: arm64.ABEQ, Index: 0}}, // next == b.Succs[1]
}

// To model a 'GTnoov' ('>' without overflow checking) branching.
var gtJumps = [2][2]ssagen.IndexJump{
	{{Jump: arm64.ABMI, Index: 1}, {Jump: arm64.ABEQ, Index: 1}}, // next == b.Succs[0]
	{{Jump: arm64.ABEQ, Index: 1}, {Jump: arm64.ABPL, Index: 0}}, // next == b.Succs[1]
}

func ssaGenBlock(s *ssagen.State, b, next *ssa.Block) {
	switch b.Kind {
	case block.BlockPlain, block.BlockDefer:
		if b.Succs[0].Block() != next {
			p := s.Prog(obj.AJMP)
			p.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, ssagen.Branch{P: p, B: b.Succs[0].Block()})
		}

	case block.BlockExit, block.BlockRetJmp:

	case block.BlockRet:
		s.Prog(obj.ARET)

	case block.BlockARM64EQ, block.BlockARM64NE,
		block.BlockARM64LT, block.BlockARM64GE,
		block.BlockARM64LE, block.BlockARM64GT,
		block.BlockARM64ULT, block.BlockARM64UGT,
		block.BlockARM64ULE, block.BlockARM64UGE,
		block.BlockARM64Z, block.BlockARM64NZ,
		block.BlockARM64ZW, block.BlockARM64NZW,
		block.BlockARM64FLT, block.BlockARM64FGE,
		block.BlockARM64FLE, block.BlockARM64FGT,
		block.BlockARM64LTnoov, block.BlockARM64GEnoov:
		jmp := blockJump[b.Kind]
		var p *obj.Prog
		switch next {
		case b.Succs[0].Block():
			p = s.Br(jmp.invasm, b.Succs[1].Block())
		case b.Succs[1].Block():
			p = s.Br(jmp.asm, b.Succs[0].Block())
		default:
			if b.Likely != ssa.BranchUnlikely {
				p = s.Br(jmp.asm, b.Succs[0].Block())
				s.Br(obj.AJMP, b.Succs[1].Block())
			} else {
				p = s.Br(jmp.invasm, b.Succs[1].Block())
				s.Br(obj.AJMP, b.Succs[0].Block())
			}
		}
		if !b.Controls[0].Type.IsFlags() {
			p.From.Type = obj.TYPE_REG
			p.From.Reg = b.Controls[0].Reg()
		}
	case block.BlockARM64TBZ, block.BlockARM64TBNZ:
		jmp := blockJump[b.Kind]
		var p *obj.Prog
		switch next {
		case b.Succs[0].Block():
			p = s.Br(jmp.invasm, b.Succs[1].Block())
		case b.Succs[1].Block():
			p = s.Br(jmp.asm, b.Succs[0].Block())
		default:
			if b.Likely != ssa.BranchUnlikely {
				p = s.Br(jmp.asm, b.Succs[0].Block())
				s.Br(obj.AJMP, b.Succs[1].Block())
			} else {
				p = s.Br(jmp.invasm, b.Succs[1].Block())
				s.Br(obj.AJMP, b.Succs[0].Block())
			}
		}
		p.From.Offset = b.AuxInt
		p.From.Type = obj.TYPE_CONST
		p.Reg = b.Controls[0].Reg()

	case block.BlockARM64LEnoov:
		s.CombJump(b, next, &leJumps)
	case block.BlockARM64GTnoov:
		s.CombJump(b, next, &gtJumps)

	case block.BlockARM64JUMPTABLE:
		// MOVD	(TABLE)(IDX<<3), Rtmp
		// JMP	(Rtmp)
		p := s.Prog(arm64.AMOVD)
		p.From = genIndexedOperand(ssaop.OpARM64MOVDloadidx8, b.Controls[1].Reg(), b.Controls[0].Reg())
		p.To.Type = obj.TYPE_REG
		p.To.Reg = arm64.REGTMP
		p = s.Prog(obj.AJMP)
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = arm64.REGTMP
		// Save jump tables for later resolution of the target blocks.
		s.JumpTables = append(s.JumpTables, b)

	default:
		b.Fatalf("branch not implemented: %s", b.LongString())
	}
}

func loadRegResult(s *ssagen.State, f *ssa.Func, t *types.Type, reg int16, n *ir.Name, off int64) *obj.Prog {
	p := s.Prog(loadByType(t))
	p.From.Type = obj.TYPE_MEM
	p.From.Name = obj.NAME_AUTO
	p.From.Sym = n.Linksym()
	p.From.Offset = n.FrameOffset() + off
	p.To.Type = obj.TYPE_REG
	p.To.Reg = reg
	return p
}

func spillArgReg(pp *objw.Progs, p *obj.Prog, f *ssa.Func, t *types.Type, reg int16, n *ir.Name, off int64) *obj.Prog {
	p = pp.Append(p, storeByType(t), obj.TYPE_REG, reg, 0, obj.TYPE_MEM, 0, n.FrameOffset()+off)
	p.To.Name = obj.NAME_PARAM
	p.To.Sym = n.Linksym()
	p.Pos = p.Pos.WithNotStmt()
	return p
}

// zero16 zeroes 16 bytes at reg+off.
// If postInc is true, increment reg by 16.
func zero16(s *ssagen.State, reg int16, off int64, postInc bool) {
	//   STP     (ZR, ZR), off(reg)
	p := s.Prog(arm64.ASTP)
	p.From.Type = obj.TYPE_REGREG
	p.From.Reg = arm64.REGZERO
	p.From.Offset = int64(arm64.REGZERO)
	p.To.Type = obj.TYPE_MEM
	p.To.Reg = reg
	p.To.Offset = off
	if postInc {
		if off != 0 {
			panic("can't postinc with non-zero offset")
		}
		//   STP.P  (ZR, ZR), 16(reg)
		p.Scond = arm64.C_XPOST
		p.To.Offset = 16
	}
}

// zero8 zeroes 8 bytes at reg+off.
func zero8(s *ssagen.State, reg int16, off int64) {
	//   MOVD     ZR, off(reg)
	p := s.Prog(arm64.AMOVD)
	p.From.Type = obj.TYPE_REG
	p.From.Reg = arm64.REGZERO
	p.To.Type = obj.TYPE_MEM
	p.To.Reg = reg
	p.To.Offset = off
}

// move32 copies 32 bytes at src+off to dst+off.
// Uses registers tmp1 and tmp2.
// If postInc is true, increment src and dst by 32.
func move32(s *ssagen.State, src, dst, tmp1, tmp2 int16, off int64, postInc bool) {
	// FLDPQ   off(src), (tmp1, tmp2)
	ld := s.Prog(arm64.AFLDPQ)
	ld.From.Type = obj.TYPE_MEM
	ld.From.Reg = src
	ld.From.Offset = off
	ld.To.Type = obj.TYPE_REGREG
	ld.To.Reg = tmp1
	ld.To.Offset = int64(tmp2)
	// FSTPQ   (tmp1, tmp2), off(dst)
	st := s.Prog(arm64.AFSTPQ)
	st.From.Type = obj.TYPE_REGREG
	st.From.Reg = tmp1
	st.From.Offset = int64(tmp2)
	st.To.Type = obj.TYPE_MEM
	st.To.Reg = dst
	st.To.Offset = off
	if postInc {
		if off != 0 {
			panic("can't postinc with non-zero offset")
		}
		ld.Scond = arm64.C_XPOST
		st.Scond = arm64.C_XPOST
		ld.From.Offset = 32
		st.To.Offset = 32
	}
}

// move16 copies 16 bytes at src+off to dst+off.
// Uses register tmp1
// If postInc is true, increment src and dst by 16.
func move16(s *ssagen.State, src, dst, tmp1 int16, off int64, postInc bool) {
	// FMOVQ     off(src), tmp1
	ld := s.Prog(arm64.AFMOVQ)
	ld.From.Type = obj.TYPE_MEM
	ld.From.Reg = src
	ld.From.Offset = off
	ld.To.Type = obj.TYPE_REG
	ld.To.Reg = tmp1
	// FMOVQ     tmp1, off(dst)
	st := s.Prog(arm64.AFMOVQ)
	st.From.Type = obj.TYPE_REG
	st.From.Reg = tmp1
	st.To.Type = obj.TYPE_MEM
	st.To.Reg = dst
	st.To.Offset = off
	if postInc {
		if off != 0 {
			panic("can't postinc with non-zero offset")
		}
		ld.Scond = arm64.C_XPOST
		st.Scond = arm64.C_XPOST
		ld.From.Offset = 16
		st.To.Offset = 16
	}
}

// move8 copies 8 bytes at src+off to dst+off.
// Uses register tmp.
func move8(s *ssagen.State, src, dst, tmp int16, off int64) {
	// MOVD    off(src), tmp
	ld := s.Prog(arm64.AMOVD)
	ld.From.Type = obj.TYPE_MEM
	ld.From.Reg = src
	ld.From.Offset = off
	ld.To.Type = obj.TYPE_REG
	ld.To.Reg = tmp
	// MOVD    tmp, off(dst)
	st := s.Prog(arm64.AMOVD)
	st.From.Type = obj.TYPE_REG
	st.From.Reg = tmp
	st.To.Type = obj.TYPE_MEM
	st.To.Reg = dst
	st.To.Offset = off
}

func zregArng(r int16, arng int16) int16 {
	if r >= arm64.REG_F0 && r <= arm64.REG_F31 &&
		arng >= arm64.ARNG_B && arng <= arm64.ARNG_Q {
		return arm64.REG_ZARNG + (r - arm64.REG_F0) | (arng << 5)
	}
	panic("Bad Z reg with arrangement")
}

func pregArng(r int16, mode int16) int16 {
	if r >= arm64.REG_P0 && r <= arm64.REG_P15 &&
		mode >= arm64.ARNG_B && mode <= arm64.ARNG_Q {
		return arm64.REG_PARNGZM + (r - arm64.REG_P0) | (mode << 5)
	}
	panic("Bad P reg with arrangement")
}

func pregMask(r int16, mode int16) int16 {
	if r >= arm64.REG_P0 && r <= arm64.REG_P15 &&
		mode >= arm64.PRED_M && mode <= arm64.PRED_Z {
		return arm64.REG_PARNGZM + (r - arm64.REG_P0) | (mode << 5)
	}
	panic("Bad P reg with arrangement")
}

func pzreg(r int16) int16 {
	if r >= arm64.REG_F0 && r <= arm64.REG_F31 {
		return r - arm64.REG_F0 + arm64.REG_Z0
	} else if r >= arm64.REG_P0 && r <= arm64.REG_P15 {
		return r
	}
	panic("Bad P or Z reg")
}

// regListArr constructs an vector register arrangement in a register list.
func regListArr(name, arng string) int64 {
	var curQ, curSize, prefix uint16
	if name[0] != 'V' && name[0] != 'Z' && name[0] != 'P' {
		panic("expect V0-V31, Z0-Z31, or P0-P15; found: " + name)
	}
	switch name[0] {
	case 'V':
		prefix = 0
	case 'Z':
		prefix = 1
	case 'P':
		prefix = 2
	}
	switch arng {
	case "B8":
		curSize = 0
		curQ = 0
	case "B16":
		curSize = 0
		curQ = 1
	case "H4":
		curSize = 1
		curQ = 0
	case "H8":
		curSize = 1
		curQ = 1
	case "S2":
		curSize = 2
		curQ = 0
	case "S4":
		curSize = 2
		curQ = 1
	case "D1":
		curSize = 3
		curQ = 0
	case "D2":
		curSize = 3
		curQ = 1
	case "B":
		curSize = 1
		curQ = 2
	case "H":
		curSize = 2
		curQ = 2
	case "S":
		curSize = 3
		curQ = 2
	case "D":
		curSize = 1
		curQ = 3
	case "Q":
		curSize = 2
		curQ = 3
	default:
		panic("invalid arrangement in ARM64 register list")
	}
	return (int64(prefix) << 32) | (int64(curQ) & 3 << 30) | (int64(curSize&3) << 10)
}
