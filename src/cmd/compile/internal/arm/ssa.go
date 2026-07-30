// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm

import (
	"fmt"
	"internal/buildcfg"
	"math"
	"math/bits"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/block"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/arm"
	"internal/abi"
)

// loadByType returns the load instruction of the given type.
func loadByType(t *types.Type) obj.As {
	if t.IsFloat() {
		switch t.Size() {
		case 4:
			return arm.AMOVF
		case 8:
			return arm.AMOVD
		}
	} else {
		switch t.Size() {
		case 1:
			if t.IsSigned() {
				return arm.AMOVB
			} else {
				return arm.AMOVBU
			}
		case 2:
			if t.IsSigned() {
				return arm.AMOVH
			} else {
				return arm.AMOVHU
			}
		case 4:
			return arm.AMOVW
		}
	}
	panic("bad load type")
}

// storeByType returns the store instruction of the given type.
func storeByType(t *types.Type) obj.As {
	if t.IsFloat() {
		switch t.Size() {
		case 4:
			return arm.AMOVF
		case 8:
			return arm.AMOVD
		}
	} else {
		switch t.Size() {
		case 1:
			return arm.AMOVB
		case 2:
			return arm.AMOVH
		case 4:
			return arm.AMOVW
		}
	}
	panic("bad store type")
}

// shift type is used as Offset in obj.TYPE_SHIFT operands to encode shifted register operands.
type shift int64

// copied from ../../../internal/obj/util.go:/TYPE_SHIFT
func (v shift) String() string {
	op := "<<>>->@>"[((v>>5)&3)<<1:]
	if v&(1<<4) != 0 {
		// register shift
		return fmt.Sprintf("R%d%c%cR%d", v&15, op[0], op[1], (v>>8)&15)
	} else {
		// constant shift
		return fmt.Sprintf("R%d%c%c%d", v&15, op[0], op[1], (v>>7)&31)
	}
}

// makeshift encodes a register shifted by a constant.
func makeshift(v *ssa.Value, reg int16, typ int64, s int64) shift {
	if s < 0 || s >= 32 {
		v.Fatalf("shift out of range: %d", s)
	}
	return shift(int64(reg&0xf) | typ | (s&31)<<7)
}

// genshift generates a Prog for r = r0 op (r1 shifted by n).
func genshift(s *ssagen.State, v *ssa.Value, as obj.As, r0, r1, r int16, typ int64, n int64) *obj.Prog {
	p := s.Prog(as)
	p.From.Type = obj.TYPE_SHIFT
	p.From.Offset = int64(makeshift(v, r1, typ, n))
	p.Reg = r0
	if r != 0 {
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	}
	return p
}

// makeregshift encodes a register shifted by a register.
func makeregshift(r1 int16, typ int64, r2 int16) shift {
	return shift(int64(r1&0xf) | typ | int64(r2&0xf)<<8 | 1<<4)
}

// genregshift generates a Prog for r = r0 op (r1 shifted by r2).
func genregshift(s *ssagen.State, as obj.As, r0, r1, r2, r int16, typ int64) *obj.Prog {
	p := s.Prog(as)
	p.From.Type = obj.TYPE_SHIFT
	p.From.Offset = int64(makeregshift(r1, typ, r2))
	p.Reg = r0
	if r != 0 {
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	}
	return p
}

// find a (lsb, width) pair for BFC
// lsb must be in [0, 31], width must be in [1, 32 - lsb]
// return (0xffffffff, 0) if v is not a binary like 0...01...10...0
func getBFC(v uint32) (uint32, uint32) {
	var m, l uint32
	// BFC is not applicable with zero
	if v == 0 {
		return 0xffffffff, 0
	}
	// find the lowest set bit, for example l=2 for 0x3ffffffc
	l = uint32(bits.TrailingZeros32(v))
	// m-1 represents the highest set bit index, for example m=30 for 0x3ffffffc
	m = 32 - uint32(bits.LeadingZeros32(v))
	// check if v is a binary like 0...01...10...0
	if (1<<m)-(1<<l) == v {
		// it must be m > l for non-zero v
		return l, m - l
	}
	// invalid
	return 0xffffffff, 0
}

func ssaGenValue(s *ssagen.State, v *ssa.Value) {
	switch v.Op {
	case ssaop.OpCopy, ssaop.OpARMMOVWreg:
		if v.Type.IsMemory() {
			return
		}
		x := v.Args[0].Reg()
		y := v.Reg()
		if x == y {
			return
		}
		as := arm.AMOVW
		if v.Type.IsFloat() {
			switch v.Type.Size() {
			case 4:
				as = arm.AMOVF
			case 8:
				as = arm.AMOVD
			default:
				panic("bad float size")
			}
		}
		p := s.Prog(as)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = x
		p.To.Type = obj.TYPE_REG
		p.To.Reg = y
	case ssaop.OpARMMOVWnop:
		// nothing to do
	case ssaop.OpLoadReg:
		if v.Type.IsFlags() {
			v.Fatalf("load flags not implemented: %v", v.LongString())
			return
		}
		p := s.Prog(loadByType(v.Type))
		ssagen.AddrAuto(&p.From, v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpStoreReg:
		if v.Type.IsFlags() {
			v.Fatalf("store flags not implemented: %v", v.LongString())
			return
		}
		p := s.Prog(storeByType(v.Type))
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddrAuto(&p.To, v)
	case ssaop.OpARMADD,
		ssaop.OpARMADC,
		ssaop.OpARMSUB,
		ssaop.OpARMSBC,
		ssaop.OpARMRSB,
		ssaop.OpARMAND,
		ssaop.OpARMOR,
		ssaop.OpARMXOR,
		ssaop.OpARMBIC,
		ssaop.OpARMMUL,
		ssaop.OpARMADDF,
		ssaop.OpARMADDD,
		ssaop.OpARMSUBF,
		ssaop.OpARMSUBD,
		ssaop.OpARMSLL,
		ssaop.OpARMSRL,
		ssaop.OpARMSRA,
		ssaop.OpARMMULF,
		ssaop.OpARMMULD,
		ssaop.OpARMNMULF,
		ssaop.OpARMNMULD,
		ssaop.OpARMDIVF,
		ssaop.OpARMDIVD:
		r := v.Reg()
		r1 := v.Args[0].Reg()
		r2 := v.Args[1].Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r2
		p.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssaop.OpARMSRR:
		genregshift(s, arm.AMOVW, 0, v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm.SHIFT_RR)
	case ssaop.OpARMMULAF, ssaop.OpARMMULAD, ssaop.OpARMMULSF, ssaop.OpARMMULSD, ssaop.OpARMFMULAD:
		r := v.Reg()
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		r2 := v.Args[2].Reg()
		if r != r0 {
			v.Fatalf("result and addend are not in the same register: %v", v.LongString())
		}
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r2
		p.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssaop.OpARMADDS,
		ssaop.OpARMADCS,
		ssaop.OpARMSUBS:
		r := v.Reg0()
		r1 := v.Args[0].Reg()
		r2 := v.Args[1].Reg()
		p := s.Prog(v.Op.Asm())
		p.Scond = arm.C_SBIT
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r2
		p.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssaop.OpARMSRAcond:
		// ARM shift instructions uses only the low-order byte of the shift amount
		// generate conditional instructions to deal with large shifts
		// flag is already set
		// SRA.HS	$31, Rarg0, Rdst // shift 31 bits to get the sign bit
		// SRA.LO	Rarg1, Rarg0, Rdst
		r := v.Reg()
		r1 := v.Args[0].Reg()
		r2 := v.Args[1].Reg()
		p := s.Prog(arm.ASRA)
		p.Scond = arm.C_SCOND_HS
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 31
		p.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
		p = s.Prog(arm.ASRA)
		p.Scond = arm.C_SCOND_LO
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r2
		p.Reg = r1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssaop.OpARMBFX, ssaop.OpARMBFXU:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt >> 8
		p.AddRestSourceConst(v.AuxInt & 0xff)
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARMANDconst, ssaop.OpARMBICconst:
		// try to optimize ANDconst and BICconst to BFC, which saves bytes and ticks
		// BFC is only available on ARMv7, and its result and source are in the same register
		if buildcfg.GOARM.Version == 7 && v.Reg() == v.Args[0].Reg() {
			var val uint32
			if v.Op == ssaop.OpARMANDconst {
				val = ^uint32(v.AuxInt)
			} else { // BICconst
				val = uint32(v.AuxInt)
			}
			lsb, width := getBFC(val)
			// omit BFC for ARM's imm12
			if 8 < width && width < 24 {
				p := s.Prog(arm.ABFC)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = int64(width)
				p.AddRestSourceConst(int64(lsb))
				p.To.Type = obj.TYPE_REG
				p.To.Reg = v.Reg()
				break
			}
		}
		// fall back to ordinary form
		fallthrough
	case ssaop.OpARMADDconst,
		ssaop.OpARMADCconst,
		ssaop.OpARMSUBconst,
		ssaop.OpARMSBCconst,
		ssaop.OpARMRSBconst,
		ssaop.OpARMRSCconst,
		ssaop.OpARMORconst,
		ssaop.OpARMXORconst,
		ssaop.OpARMSLLconst,
		ssaop.OpARMSRLconst,
		ssaop.OpARMSRAconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARMADDSconst,
		ssaop.OpARMSUBSconst,
		ssaop.OpARMRSBSconst:
		p := s.Prog(v.Op.Asm())
		p.Scond = arm.C_SBIT
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
	case ssaop.OpARMSRRconst:
		genshift(s, v, arm.AMOVW, 0, v.Args[0].Reg(), v.Reg(), arm.SHIFT_RR, v.AuxInt)
	case ssaop.OpARMADDshiftLL,
		ssaop.OpARMADCshiftLL,
		ssaop.OpARMSUBshiftLL,
		ssaop.OpARMSBCshiftLL,
		ssaop.OpARMRSBshiftLL,
		ssaop.OpARMRSCshiftLL,
		ssaop.OpARMANDshiftLL,
		ssaop.OpARMORshiftLL,
		ssaop.OpARMXORshiftLL,
		ssaop.OpARMBICshiftLL:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm.SHIFT_LL, v.AuxInt)
	case ssaop.OpARMADDSshiftLL,
		ssaop.OpARMSUBSshiftLL,
		ssaop.OpARMRSBSshiftLL:
		p := genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Reg0(), arm.SHIFT_LL, v.AuxInt)
		p.Scond = arm.C_SBIT
	case ssaop.OpARMADDshiftRL,
		ssaop.OpARMADCshiftRL,
		ssaop.OpARMSUBshiftRL,
		ssaop.OpARMSBCshiftRL,
		ssaop.OpARMRSBshiftRL,
		ssaop.OpARMRSCshiftRL,
		ssaop.OpARMANDshiftRL,
		ssaop.OpARMORshiftRL,
		ssaop.OpARMXORshiftRL,
		ssaop.OpARMBICshiftRL:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm.SHIFT_LR, v.AuxInt)
	case ssaop.OpARMADDSshiftRL,
		ssaop.OpARMSUBSshiftRL,
		ssaop.OpARMRSBSshiftRL:
		p := genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Reg0(), arm.SHIFT_LR, v.AuxInt)
		p.Scond = arm.C_SBIT
	case ssaop.OpARMADDshiftRA,
		ssaop.OpARMADCshiftRA,
		ssaop.OpARMSUBshiftRA,
		ssaop.OpARMSBCshiftRA,
		ssaop.OpARMRSBshiftRA,
		ssaop.OpARMRSCshiftRA,
		ssaop.OpARMANDshiftRA,
		ssaop.OpARMORshiftRA,
		ssaop.OpARMXORshiftRA,
		ssaop.OpARMBICshiftRA:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm.SHIFT_AR, v.AuxInt)
	case ssaop.OpARMADDSshiftRA,
		ssaop.OpARMSUBSshiftRA,
		ssaop.OpARMRSBSshiftRA:
		p := genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Reg0(), arm.SHIFT_AR, v.AuxInt)
		p.Scond = arm.C_SBIT
	case ssaop.OpARMXORshiftRR:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm.SHIFT_RR, v.AuxInt)
	case ssaop.OpARMMVNshiftLL:
		genshift(s, v, v.Op.Asm(), 0, v.Args[0].Reg(), v.Reg(), arm.SHIFT_LL, v.AuxInt)
	case ssaop.OpARMMVNshiftRL:
		genshift(s, v, v.Op.Asm(), 0, v.Args[0].Reg(), v.Reg(), arm.SHIFT_LR, v.AuxInt)
	case ssaop.OpARMMVNshiftRA:
		genshift(s, v, v.Op.Asm(), 0, v.Args[0].Reg(), v.Reg(), arm.SHIFT_AR, v.AuxInt)
	case ssaop.OpARMMVNshiftLLreg:
		genregshift(s, v.Op.Asm(), 0, v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm.SHIFT_LL)
	case ssaop.OpARMMVNshiftRLreg:
		genregshift(s, v.Op.Asm(), 0, v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm.SHIFT_LR)
	case ssaop.OpARMMVNshiftRAreg:
		genregshift(s, v.Op.Asm(), 0, v.Args[0].Reg(), v.Args[1].Reg(), v.Reg(), arm.SHIFT_AR)
	case ssaop.OpARMADDshiftLLreg,
		ssaop.OpARMADCshiftLLreg,
		ssaop.OpARMSUBshiftLLreg,
		ssaop.OpARMSBCshiftLLreg,
		ssaop.OpARMRSBshiftLLreg,
		ssaop.OpARMRSCshiftLLreg,
		ssaop.OpARMANDshiftLLreg,
		ssaop.OpARMORshiftLLreg,
		ssaop.OpARMXORshiftLLreg,
		ssaop.OpARMBICshiftLLreg:
		genregshift(s, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Args[2].Reg(), v.Reg(), arm.SHIFT_LL)
	case ssaop.OpARMADDSshiftLLreg,
		ssaop.OpARMSUBSshiftLLreg,
		ssaop.OpARMRSBSshiftLLreg:
		p := genregshift(s, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Args[2].Reg(), v.Reg0(), arm.SHIFT_LL)
		p.Scond = arm.C_SBIT
	case ssaop.OpARMADDshiftRLreg,
		ssaop.OpARMADCshiftRLreg,
		ssaop.OpARMSUBshiftRLreg,
		ssaop.OpARMSBCshiftRLreg,
		ssaop.OpARMRSBshiftRLreg,
		ssaop.OpARMRSCshiftRLreg,
		ssaop.OpARMANDshiftRLreg,
		ssaop.OpARMORshiftRLreg,
		ssaop.OpARMXORshiftRLreg,
		ssaop.OpARMBICshiftRLreg:
		genregshift(s, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Args[2].Reg(), v.Reg(), arm.SHIFT_LR)
	case ssaop.OpARMADDSshiftRLreg,
		ssaop.OpARMSUBSshiftRLreg,
		ssaop.OpARMRSBSshiftRLreg:
		p := genregshift(s, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Args[2].Reg(), v.Reg0(), arm.SHIFT_LR)
		p.Scond = arm.C_SBIT
	case ssaop.OpARMADDshiftRAreg,
		ssaop.OpARMADCshiftRAreg,
		ssaop.OpARMSUBshiftRAreg,
		ssaop.OpARMSBCshiftRAreg,
		ssaop.OpARMRSBshiftRAreg,
		ssaop.OpARMRSCshiftRAreg,
		ssaop.OpARMANDshiftRAreg,
		ssaop.OpARMORshiftRAreg,
		ssaop.OpARMXORshiftRAreg,
		ssaop.OpARMBICshiftRAreg:
		genregshift(s, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Args[2].Reg(), v.Reg(), arm.SHIFT_AR)
	case ssaop.OpARMADDSshiftRAreg,
		ssaop.OpARMSUBSshiftRAreg,
		ssaop.OpARMRSBSshiftRAreg:
		p := genregshift(s, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Args[2].Reg(), v.Reg0(), arm.SHIFT_AR)
		p.Scond = arm.C_SBIT
	case ssaop.OpARMHMUL,
		ssaop.OpARMHMULU:
		// 32-bit high multiplication
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REGREG
		p.To.Reg = v.Reg()
		p.To.Offset = arm.REGTMP // throw away low 32-bit into tmp register
	case ssaop.OpARMMULLU:
		// 32-bit multiplication, results 64-bit, high 32-bit in out0, low 32-bit in out1
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REGREG
		p.To.Reg = v.Reg0()           // high 32-bit
		p.To.Offset = int64(v.Reg1()) // low 32-bit
	case ssaop.OpARMMULA, ssaop.OpARMMULS:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REGREG2
		p.To.Reg = v.Reg()                   // result
		p.To.Offset = int64(v.Args[2].Reg()) // addend
	case ssaop.OpARMMOVWconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARMMOVFconst,
		ssaop.OpARMMOVDconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_FCONST
		p.From.Val = math.Float64frombits(uint64(v.AuxInt))
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARMCMP,
		ssaop.OpARMCMN,
		ssaop.OpARMTST,
		ssaop.OpARMTEQ,
		ssaop.OpARMCMPF,
		ssaop.OpARMCMPD:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		// Special layout in ARM assembly
		// Comparing to x86, the operands of ARM's CMP are reversed.
		p.From.Reg = v.Args[1].Reg()
		p.Reg = v.Args[0].Reg()
	case ssaop.OpARMCMPconst,
		ssaop.OpARMCMNconst,
		ssaop.OpARMTSTconst,
		ssaop.OpARMTEQconst:
		// Special layout in ARM assembly
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.Reg = v.Args[0].Reg()
	case ssaop.OpARMCMPF0,
		ssaop.OpARMCMPD0:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
	case ssaop.OpARMCMPshiftLL, ssaop.OpARMCMNshiftLL, ssaop.OpARMTSTshiftLL, ssaop.OpARMTEQshiftLL:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), 0, arm.SHIFT_LL, v.AuxInt)
	case ssaop.OpARMCMPshiftRL, ssaop.OpARMCMNshiftRL, ssaop.OpARMTSTshiftRL, ssaop.OpARMTEQshiftRL:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), 0, arm.SHIFT_LR, v.AuxInt)
	case ssaop.OpARMCMPshiftRA, ssaop.OpARMCMNshiftRA, ssaop.OpARMTSTshiftRA, ssaop.OpARMTEQshiftRA:
		genshift(s, v, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), 0, arm.SHIFT_AR, v.AuxInt)
	case ssaop.OpARMCMPshiftLLreg, ssaop.OpARMCMNshiftLLreg, ssaop.OpARMTSTshiftLLreg, ssaop.OpARMTEQshiftLLreg:
		genregshift(s, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Args[2].Reg(), 0, arm.SHIFT_LL)
	case ssaop.OpARMCMPshiftRLreg, ssaop.OpARMCMNshiftRLreg, ssaop.OpARMTSTshiftRLreg, ssaop.OpARMTEQshiftRLreg:
		genregshift(s, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Args[2].Reg(), 0, arm.SHIFT_LR)
	case ssaop.OpARMCMPshiftRAreg, ssaop.OpARMCMNshiftRAreg, ssaop.OpARMTSTshiftRAreg, ssaop.OpARMTEQshiftRAreg:
		genregshift(s, v.Op.Asm(), v.Args[0].Reg(), v.Args[1].Reg(), v.Args[2].Reg(), 0, arm.SHIFT_AR)
	case ssaop.OpARMMOVWaddr:
		p := s.Prog(arm.AMOVW)
		p.From.Type = obj.TYPE_ADDR
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()

		var wantreg string
		// MOVW $sym+off(base), R
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
			// No sym, just MOVW $off(SP), R
			wantreg = "SP"
			p.From.Offset = v.AuxInt
		}
		if reg := v.Args[0].RegName(); reg != wantreg {
			v.Fatalf("bad reg %s for symbol type %T, want %s", reg, v.Aux, wantreg)
		}

	case ssaop.OpARMMOVBload,
		ssaop.OpARMMOVBUload,
		ssaop.OpARMMOVHload,
		ssaop.OpARMMOVHUload,
		ssaop.OpARMMOVWload,
		ssaop.OpARMMOVFload,
		ssaop.OpARMMOVDload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARMMOVBstore,
		ssaop.OpARMMOVHstore,
		ssaop.OpARMMOVWstore,
		ssaop.OpARMMOVFstore,
		ssaop.OpARMMOVDstore:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssaop.OpARMMOVWloadidx, ssaop.OpARMMOVBUloadidx, ssaop.OpARMMOVBloadidx, ssaop.OpARMMOVHUloadidx, ssaop.OpARMMOVHloadidx:
		// this is just shift 0 bits
		fallthrough
	case ssaop.OpARMMOVWloadshiftLL:
		p := genshift(s, v, v.Op.Asm(), 0, v.Args[1].Reg(), v.Reg(), arm.SHIFT_LL, v.AuxInt)
		p.From.Reg = v.Args[0].Reg()
	case ssaop.OpARMMOVWloadshiftRL:
		p := genshift(s, v, v.Op.Asm(), 0, v.Args[1].Reg(), v.Reg(), arm.SHIFT_LR, v.AuxInt)
		p.From.Reg = v.Args[0].Reg()
	case ssaop.OpARMMOVWloadshiftRA:
		p := genshift(s, v, v.Op.Asm(), 0, v.Args[1].Reg(), v.Reg(), arm.SHIFT_AR, v.AuxInt)
		p.From.Reg = v.Args[0].Reg()
	case ssaop.OpARMMOVWstoreidx, ssaop.OpARMMOVBstoreidx, ssaop.OpARMMOVHstoreidx:
		// this is just shift 0 bits
		fallthrough
	case ssaop.OpARMMOVWstoreshiftLL:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
		p.To.Type = obj.TYPE_SHIFT
		p.To.Reg = v.Args[0].Reg()
		p.To.Offset = int64(makeshift(v, v.Args[1].Reg(), arm.SHIFT_LL, v.AuxInt))
	case ssaop.OpARMMOVWstoreshiftRL:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
		p.To.Type = obj.TYPE_SHIFT
		p.To.Reg = v.Args[0].Reg()
		p.To.Offset = int64(makeshift(v, v.Args[1].Reg(), arm.SHIFT_LR, v.AuxInt))
	case ssaop.OpARMMOVWstoreshiftRA:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
		p.To.Type = obj.TYPE_SHIFT
		p.To.Reg = v.Args[0].Reg()
		p.To.Offset = int64(makeshift(v, v.Args[1].Reg(), arm.SHIFT_AR, v.AuxInt))
	case ssaop.OpARMMOVBreg,
		ssaop.OpARMMOVBUreg,
		ssaop.OpARMMOVHreg,
		ssaop.OpARMMOVHUreg:
		a := v.Args[0]
		for a.Op == ssaop.OpCopy || a.Op == ssaop.OpARMMOVWreg || a.Op == ssaop.OpARMMOVWnop {
			a = a.Args[0]
		}
		if a.Op == ssaop.OpLoadReg {
			t := a.Type
			switch {
			case v.Op == ssaop.OpARMMOVBreg && t.Size() == 1 && t.IsSigned(),
				v.Op == ssaop.OpARMMOVBUreg && t.Size() == 1 && !t.IsSigned(),
				v.Op == ssaop.OpARMMOVHreg && t.Size() == 2 && t.IsSigned(),
				v.Op == ssaop.OpARMMOVHUreg && t.Size() == 2 && !t.IsSigned():
				// arg is a proper-typed load, already zero/sign-extended, don't extend again
				if v.Reg() == v.Args[0].Reg() {
					return
				}
				p := s.Prog(arm.AMOVW)
				p.From.Type = obj.TYPE_REG
				p.From.Reg = v.Args[0].Reg()
				p.To.Type = obj.TYPE_REG
				p.To.Reg = v.Reg()
				return
			default:
			}
		}
		if buildcfg.GOARM.Version >= 6 {
			// generate more efficient "MOVB/MOVBU/MOVH/MOVHU Reg@>0, Reg" on ARMv6 & ARMv7
			genshift(s, v, v.Op.Asm(), 0, v.Args[0].Reg(), v.Reg(), arm.SHIFT_RR, 0)
			return
		}
		fallthrough
	case ssaop.OpARMMVN,
		ssaop.OpARMCLZ,
		ssaop.OpARMREV,
		ssaop.OpARMREV16,
		ssaop.OpARMRBIT,
		ssaop.OpARMSQRTF,
		ssaop.OpARMSQRTD,
		ssaop.OpARMNEGF,
		ssaop.OpARMNEGD,
		ssaop.OpARMABSD,
		ssaop.OpARMMOVWF,
		ssaop.OpARMMOVWD,
		ssaop.OpARMMOVFW,
		ssaop.OpARMMOVDW,
		ssaop.OpARMMOVFD,
		ssaop.OpARMMOVDF:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARMMOVWUF,
		ssaop.OpARMMOVWUD,
		ssaop.OpARMMOVFWU,
		ssaop.OpARMMOVDWU:
		p := s.Prog(v.Op.Asm())
		p.Scond = arm.C_UBIT
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARMCMOVWHSconst:
		p := s.Prog(arm.AMOVW)
		p.Scond = arm.C_SCOND_HS
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARMCMOVWLSconst:
		p := s.Prog(arm.AMOVW)
		p.Scond = arm.C_SCOND_LS
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARMCALLstatic, ssaop.OpARMCALLclosure, ssaop.OpARMCALLinter:
		s.Call(v)
	case ssaop.OpARMCALLtail, ssaop.OpARMCALLtailinter:
		s.TailCall(v)
	case ssaop.OpARMCALLudiv:
		p := s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ir.Syms.Udiv
	case ssaop.OpARMLoweredWB:
		p := s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		// AuxInt encodes how many buffer entries we need.
		p.To.Sym = ir.Syms.GCWriteBarrier[v.AuxInt-1]

	case ssaop.OpARMLoweredPanicBoundsRR, ssaop.OpARMLoweredPanicBoundsRC, ssaop.OpARMLoweredPanicBoundsCR, ssaop.OpARMLoweredPanicBoundsCC,
		ssaop.OpARMLoweredPanicExtendRR, ssaop.OpARMLoweredPanicExtendRC:
		// Compute the constant we put in the PCData entry for this call.
		code, signed := ssa.BoundsKind(v.AuxInt).Code()
		xIsReg := false
		yIsReg := false
		xVal := 0
		yVal := 0
		extend := false
		switch v.Op {
		case ssaop.OpARMLoweredPanicBoundsRR:
			xIsReg = true
			xVal = int(v.Args[0].Reg() - arm.REG_R0)
			yIsReg = true
			yVal = int(v.Args[1].Reg() - arm.REG_R0)
		case ssaop.OpARMLoweredPanicExtendRR:
			extend = true
			xIsReg = true
			hi := int(v.Args[0].Reg() - arm.REG_R0)
			lo := int(v.Args[1].Reg() - arm.REG_R0)
			xVal = hi<<2 + lo // encode 2 register numbers
			yIsReg = true
			yVal = int(v.Args[2].Reg() - arm.REG_R0)
		case ssaop.OpARMLoweredPanicBoundsRC:
			xIsReg = true
			xVal = int(v.Args[0].Reg() - arm.REG_R0)
			c := v.Aux.(ssa.PanicBoundsC).C
			if c >= 0 && c <= abi.BoundsMaxConst {
				yVal = int(c)
			} else {
				// Move constant to a register
				yIsReg = true
				if yVal == xVal {
					yVal = 1
				}
				p := s.Prog(arm.AMOVW)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = arm.REG_R0 + int16(yVal)
			}
		case ssaop.OpARMLoweredPanicExtendRC:
			extend = true
			xIsReg = true
			hi := int(v.Args[0].Reg() - arm.REG_R0)
			lo := int(v.Args[1].Reg() - arm.REG_R0)
			xVal = hi<<2 + lo // encode 2 register numbers
			c := v.Aux.(ssa.PanicBoundsC).C
			if c >= 0 && c <= abi.BoundsMaxConst {
				yVal = int(c)
			} else {
				// Move constant to a register
				for yVal == hi || yVal == lo {
					yVal++
				}
				p := s.Prog(arm.AMOVW)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = arm.REG_R0 + int16(yVal)
			}
		case ssaop.OpARMLoweredPanicBoundsCR:
			yIsReg = true
			yVal = int(v.Args[0].Reg() - arm.REG_R0)
			c := v.Aux.(ssa.PanicBoundsC).C
			if c >= 0 && c <= abi.BoundsMaxConst {
				xVal = int(c)
			} else if signed && int64(int32(c)) == c || !signed && int64(uint32(c)) == c {
				// Move constant to a register
				xIsReg = true
				if xVal == yVal {
					xVal = 1
				}
				p := s.Prog(arm.AMOVW)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = arm.REG_R0 + int16(xVal)
			} else {
				// Move constant to two registers
				extend = true
				xIsReg = true
				hi := 0
				lo := 1
				if hi == yVal {
					hi = 2
				}
				if lo == yVal {
					lo = 2
				}
				xVal = hi<<2 + lo
				p := s.Prog(arm.AMOVW)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c >> 32
				p.To.Type = obj.TYPE_REG
				p.To.Reg = arm.REG_R0 + int16(hi)
				p = s.Prog(arm.AMOVW)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = int64(int32(c))
				p.To.Type = obj.TYPE_REG
				p.To.Reg = arm.REG_R0 + int16(lo)
			}
		case ssaop.OpARMLoweredPanicBoundsCC:
			c := v.Aux.(ssa.PanicBoundsCC).Cx
			if c >= 0 && c <= abi.BoundsMaxConst {
				xVal = int(c)
			} else if signed && int64(int32(c)) == c || !signed && int64(uint32(c)) == c {
				// Move constant to a register
				xIsReg = true
				p := s.Prog(arm.AMOVW)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = arm.REG_R0 + int16(xVal)
			} else {
				// Move constant to two registers
				extend = true
				xIsReg = true
				hi := 0
				lo := 1
				xVal = hi<<2 + lo
				p := s.Prog(arm.AMOVW)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c >> 32
				p.To.Type = obj.TYPE_REG
				p.To.Reg = arm.REG_R0 + int16(hi)
				p = s.Prog(arm.AMOVW)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = int64(int32(c))
				p.To.Type = obj.TYPE_REG
				p.To.Reg = arm.REG_R0 + int16(lo)
			}
			c = v.Aux.(ssa.PanicBoundsCC).Cy
			if c >= 0 && c <= abi.BoundsMaxConst {
				yVal = int(c)
			} else {
				// Move constant to a register
				yIsReg = true
				yVal = 2
				p := s.Prog(arm.AMOVW)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = arm.REG_R0 + int16(yVal)
			}
		}
		c := abi.BoundsEncode(code, signed, xIsReg, yIsReg, xVal, yVal)

		p := s.Prog(obj.APCDATA)
		p.From.SetConst(abi.PCDATA_PanicBounds)
		p.To.SetConst(int64(c))
		p = s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		if extend {
			p.To.Sym = ir.Syms.PanicExtend
		} else {
			p.To.Sym = ir.Syms.PanicBounds
		}

	case ssaop.OpARMDUFFZERO:
		p := s.Prog(obj.ADUFFZERO)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ir.Syms.Duffzero
		p.To.Offset = v.AuxInt
	case ssaop.OpARMDUFFCOPY:
		p := s.Prog(obj.ADUFFCOPY)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		p.To.Sym = ir.Syms.Duffcopy
		p.To.Offset = v.AuxInt
	case ssaop.OpARMLoweredNilCheck:
		// Issue a load which will fault if arg is nil.
		p := s.Prog(arm.AMOVB)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = arm.REGTMP
		if logopt.Enabled() {
			logopt.LogOpt(v.Pos, "nilcheck", "genssa", v.Block.Func.Name)
		}
		if base.Debug.Nil != 0 && v.Pos.Line() > 1 { // v.Pos.Line()==1 in generated wrappers
			base.WarnfAt(v.Pos, "generated nil check")
		}
	case ssaop.OpARMLoweredZero:
		// MOVW.P	Rarg2, 4(R1)
		// CMP	Rarg1, R1
		// BLE	-2(PC)
		// arg1 is the address of the last element to zero
		// arg2 is known to be zero
		// auxint is alignment
		var sz int64
		var mov obj.As
		switch {
		case v.AuxInt%4 == 0:
			sz = 4
			mov = arm.AMOVW
		case v.AuxInt%2 == 0:
			sz = 2
			mov = arm.AMOVH
		default:
			sz = 1
			mov = arm.AMOVB
		}
		p := s.Prog(mov)
		p.Scond = arm.C_PBIT
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = arm.REG_R1
		p.To.Offset = sz
		p2 := s.Prog(arm.ACMP)
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = v.Args[1].Reg()
		p2.Reg = arm.REG_R1
		p3 := s.Prog(arm.ABLE)
		p3.To.Type = obj.TYPE_BRANCH
		p3.To.SetTarget(p)
	case ssaop.OpARMLoweredMove:
		// MOVW.P	4(R1), Rtmp
		// MOVW.P	Rtmp, 4(R2)
		// CMP	Rarg2, R1
		// BLE	-3(PC)
		// arg2 is the address of the last element of src
		// auxint is alignment
		var sz int64
		var mov obj.As
		switch {
		case v.AuxInt%4 == 0:
			sz = 4
			mov = arm.AMOVW
		case v.AuxInt%2 == 0:
			sz = 2
			mov = arm.AMOVH
		default:
			sz = 1
			mov = arm.AMOVB
		}
		p := s.Prog(mov)
		p.Scond = arm.C_PBIT
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = arm.REG_R1
		p.From.Offset = sz
		p.To.Type = obj.TYPE_REG
		p.To.Reg = arm.REGTMP
		p2 := s.Prog(mov)
		p2.Scond = arm.C_PBIT
		p2.From.Type = obj.TYPE_REG
		p2.From.Reg = arm.REGTMP
		p2.To.Type = obj.TYPE_MEM
		p2.To.Reg = arm.REG_R2
		p2.To.Offset = sz
		p3 := s.Prog(arm.ACMP)
		p3.From.Type = obj.TYPE_REG
		p3.From.Reg = v.Args[2].Reg()
		p3.Reg = arm.REG_R1
		p4 := s.Prog(arm.ABLE)
		p4.To.Type = obj.TYPE_BRANCH
		p4.To.SetTarget(p)
	case ssaop.OpARMEqual,
		ssaop.OpARMNotEqual,
		ssaop.OpARMLessThan,
		ssaop.OpARMLessEqual,
		ssaop.OpARMGreaterThan,
		ssaop.OpARMGreaterEqual,
		ssaop.OpARMLessThanU,
		ssaop.OpARMLessEqualU,
		ssaop.OpARMGreaterThanU,
		ssaop.OpARMGreaterEqualU:
		// generate boolean values
		// use conditional move
		p := s.Prog(arm.AMOVW)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
		p = s.Prog(arm.AMOVW)
		p.Scond = condBits[v.Op]
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = 1
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARMLoweredGetClosurePtr:
		// Closure pointer is R7 (arm.REGCTXT).
		ssagen.CheckLoweredGetClosurePtr(v)
	case ssaop.OpARMLoweredGetCallerSP:
		// caller's SP is FixedFrameSize below the address of the first arg
		p := s.Prog(arm.AMOVW)
		p.From.Type = obj.TYPE_ADDR
		p.From.Offset = -base.Ctxt.Arch.FixedFrameSize
		p.From.Name = obj.NAME_PARAM
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARMLoweredGetCallerPC:
		p := s.Prog(obj.AGETCALLERPC)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssaop.OpARMFlagConstant:
		v.Fatalf("FlagConstant op should never make it to codegen %v", v.LongString())
	case ssaop.OpARMInvertFlags:
		v.Fatalf("InvertFlags should never make it to codegen %v", v.LongString())
	case ssaop.OpClobber, ssaop.OpClobberReg:
		// TODO: implement for clobberdead experiment. Nop is ok for now.
	default:
		v.Fatalf("genValue not implemented: %s", v.LongString())
	}
}

var condBits = map[ssaop.Op]uint8{
	ssaop.OpARMEqual:         arm.C_SCOND_EQ,
	ssaop.OpARMNotEqual:      arm.C_SCOND_NE,
	ssaop.OpARMLessThan:      arm.C_SCOND_LT,
	ssaop.OpARMLessThanU:     arm.C_SCOND_LO,
	ssaop.OpARMLessEqual:     arm.C_SCOND_LE,
	ssaop.OpARMLessEqualU:    arm.C_SCOND_LS,
	ssaop.OpARMGreaterThan:   arm.C_SCOND_GT,
	ssaop.OpARMGreaterThanU:  arm.C_SCOND_HI,
	ssaop.OpARMGreaterEqual:  arm.C_SCOND_GE,
	ssaop.OpARMGreaterEqualU: arm.C_SCOND_HS,
}

var blockJump = map[block.BlockKind]struct {
	asm, invasm obj.As
}{
	block.BlockARMEQ:     {arm.ABEQ, arm.ABNE},
	block.BlockARMNE:     {arm.ABNE, arm.ABEQ},
	block.BlockARMLT:     {arm.ABLT, arm.ABGE},
	block.BlockARMGE:     {arm.ABGE, arm.ABLT},
	block.BlockARMLE:     {arm.ABLE, arm.ABGT},
	block.BlockARMGT:     {arm.ABGT, arm.ABLE},
	block.BlockARMULT:    {arm.ABLO, arm.ABHS},
	block.BlockARMUGE:    {arm.ABHS, arm.ABLO},
	block.BlockARMUGT:    {arm.ABHI, arm.ABLS},
	block.BlockARMULE:    {arm.ABLS, arm.ABHI},
	block.BlockARMLTnoov: {arm.ABMI, arm.ABPL},
	block.BlockARMGEnoov: {arm.ABPL, arm.ABMI},
}

// To model a 'LEnoov' ('<=' without overflow checking) branching.
var leJumps = [2][2]ssagen.IndexJump{
	{{Jump: arm.ABEQ, Index: 0}, {Jump: arm.ABPL, Index: 1}}, // next == b.Succs[0]
	{{Jump: arm.ABMI, Index: 0}, {Jump: arm.ABEQ, Index: 0}}, // next == b.Succs[1]
}

// To model a 'GTnoov' ('>' without overflow checking) branching.
var gtJumps = [2][2]ssagen.IndexJump{
	{{Jump: arm.ABMI, Index: 1}, {Jump: arm.ABEQ, Index: 1}}, // next == b.Succs[0]
	{{Jump: arm.ABEQ, Index: 1}, {Jump: arm.ABPL, Index: 0}}, // next == b.Succs[1]
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

	case block.BlockARMEQ, block.BlockARMNE,
		block.BlockARMLT, block.BlockARMGE,
		block.BlockARMLE, block.BlockARMGT,
		block.BlockARMULT, block.BlockARMUGT,
		block.BlockARMULE, block.BlockARMUGE,
		block.BlockARMLTnoov, block.BlockARMGEnoov:
		jmp := blockJump[b.Kind]
		switch next {
		case b.Succs[0].Block():
			s.Br(jmp.invasm, b.Succs[1].Block())
		case b.Succs[1].Block():
			s.Br(jmp.asm, b.Succs[0].Block())
		default:
			if b.Likely != ssa.BranchUnlikely {
				s.Br(jmp.asm, b.Succs[0].Block())
				s.Br(obj.AJMP, b.Succs[1].Block())
			} else {
				s.Br(jmp.invasm, b.Succs[1].Block())
				s.Br(obj.AJMP, b.Succs[0].Block())
			}
		}

	case block.BlockARMLEnoov:
		s.CombJump(b, next, &leJumps)

	case block.BlockARMGTnoov:
		s.CombJump(b, next, &gtJumps)

	default:
		b.Fatalf("branch not implemented: %s", b.LongString())
	}
}
