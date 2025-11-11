// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package s390x

import (
	"math"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssagen"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/s390x"
	"internal/abi"
)

// ssaMarkMoves marks any MOVXconst ops that need to avoid clobbering flags.
func ssaMarkMoves(s *ssagen.State, b *ssa.Block) {
	flive := b.FlagsLiveAtEnd
	for _, c := range b.ControlValues() {
		flive = c.Type.IsFlags() || flive
	}
	for i := len(b.Values) - 1; i >= 0; i-- {
		v := b.Values[i]
		if flive && v.Op == ssa.OpS390XMOVDconst {
			// The "mark" is any non-nil Aux value.
			v.Aux = ssa.AuxMark
		}
		if v.Type.IsFlags() {
			flive = false
		}
		for _, a := range v.Args {
			if a.Type.IsFlags() {
				flive = true
			}
		}
	}
}

// loadByType returns the load instruction of the given type.
func loadByType(t *types.Type) obj.As {
	if t.IsFloat() {
		switch t.Size() {
		case 4:
			return s390x.AFMOVS
		case 8:
			return s390x.AFMOVD
		}
	} else {
		switch t.Size() {
		case 1:
			if t.IsSigned() {
				return s390x.AMOVB
			} else {
				return s390x.AMOVBZ
			}
		case 2:
			if t.IsSigned() {
				return s390x.AMOVH
			} else {
				return s390x.AMOVHZ
			}
		case 4:
			if t.IsSigned() {
				return s390x.AMOVW
			} else {
				return s390x.AMOVWZ
			}
		case 8:
			return s390x.AMOVD
		}
	}
	panic("bad load type")
}

// storeByType returns the store instruction of the given type.
func storeByType(t *types.Type) obj.As {
	width := t.Size()
	if t.IsFloat() {
		switch width {
		case 4:
			return s390x.AFMOVS
		case 8:
			return s390x.AFMOVD
		}
	} else {
		switch width {
		case 1:
			return s390x.AMOVB
		case 2:
			return s390x.AMOVH
		case 4:
			return s390x.AMOVW
		case 8:
			return s390x.AMOVD
		}
	}
	panic("bad store type")
}

// moveByType returns the reg->reg move instruction of the given type.
func moveByType(t *types.Type) obj.As {
	if t.IsFloat() {
		return s390x.AFMOVD
	} else {
		switch t.Size() {
		case 1:
			if t.IsSigned() {
				return s390x.AMOVB
			} else {
				return s390x.AMOVBZ
			}
		case 2:
			if t.IsSigned() {
				return s390x.AMOVH
			} else {
				return s390x.AMOVHZ
			}
		case 4:
			if t.IsSigned() {
				return s390x.AMOVW
			} else {
				return s390x.AMOVWZ
			}
		case 8:
			return s390x.AMOVD
		}
	}
	panic("bad load type")
}

// opregreg emits instructions for
//
//	dest := dest(To) op src(From)
//
// and also returns the created obj.Prog so it
// may be further adjusted (offset, scale, etc).
func opregreg(s *ssagen.State, op obj.As, dest, src int16) *obj.Prog {
	p := s.Prog(op)
	p.From.Type = obj.TYPE_REG
	p.To.Type = obj.TYPE_REG
	p.To.Reg = dest
	p.From.Reg = src
	return p
}

// opregregimm emits instructions for
//
//	dest := src(From) op off
//
// and also returns the created obj.Prog so it
// may be further adjusted (offset, scale, etc).
func opregregimm(s *ssagen.State, op obj.As, dest, src int16, off int64) *obj.Prog {
	p := s.Prog(op)
	p.From.Type = obj.TYPE_CONST
	p.From.Offset = off
	p.Reg = src
	p.To.Reg = dest
	p.To.Type = obj.TYPE_REG
	return p
}

func ssaGenValue(s *ssagen.State, v *ssa.Value) {
	switch v.Op {
	case ssa.OpS390XSLD, ssa.OpS390XSLW,
		ssa.OpS390XSRD, ssa.OpS390XSRW,
		ssa.OpS390XSRAD, ssa.OpS390XSRAW,
		ssa.OpS390XRLLG, ssa.OpS390XRLL:
		r := v.Reg()
		r1 := v.Args[0].Reg()
		r2 := v.Args[1].Reg()
		if r2 == s390x.REG_R0 {
			v.Fatalf("cannot use R0 as shift value %s", v.LongString())
		}
		p := opregreg(s, v.Op.Asm(), r, r2)
		if r != r1 {
			p.Reg = r1
		}
	case ssa.OpS390XRXSBG:
		r2 := v.Args[1].Reg()
		i := v.Aux.(s390x.RotateParams)
		p := s.Prog(v.Op.Asm())
		p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: int64(i.Start)}
		p.AddRestSourceArgs([]obj.Addr{
			{Type: obj.TYPE_CONST, Offset: int64(i.End)},
			{Type: obj.TYPE_CONST, Offset: int64(i.Amount)},
			{Type: obj.TYPE_REG, Reg: r2},
		})
		p.To = obj.Addr{Type: obj.TYPE_REG, Reg: v.Reg()}
	case ssa.OpS390XRISBGZ:
		r1 := v.Reg()
		r2 := v.Args[0].Reg()
		i := v.Aux.(s390x.RotateParams)
		p := s.Prog(v.Op.Asm())
		p.From = obj.Addr{Type: obj.TYPE_CONST, Offset: int64(i.Start)}
		p.AddRestSourceArgs([]obj.Addr{
			{Type: obj.TYPE_CONST, Offset: int64(i.End)},
			{Type: obj.TYPE_CONST, Offset: int64(i.Amount)},
			{Type: obj.TYPE_REG, Reg: r2},
		})
		p.To = obj.Addr{Type: obj.TYPE_REG, Reg: r1}
	case ssa.OpS390XADD, ssa.OpS390XADDW,
		ssa.OpS390XSUB, ssa.OpS390XSUBW,
		ssa.OpS390XAND, ssa.OpS390XANDW,
		ssa.OpS390XOR, ssa.OpS390XORW,
		ssa.OpS390XXOR, ssa.OpS390XXORW:
		r := v.Reg()
		r1 := v.Args[0].Reg()
		r2 := v.Args[1].Reg()
		p := opregreg(s, v.Op.Asm(), r, r2)
		if r != r1 {
			p.Reg = r1
		}
	case ssa.OpS390XADDC:
		r1 := v.Reg0()
		r2 := v.Args[0].Reg()
		r3 := v.Args[1].Reg()
		if r1 == r2 {
			r2, r3 = r3, r2
		}
		p := opregreg(s, v.Op.Asm(), r1, r2)
		if r3 != r1 {
			p.Reg = r3
		}
	case ssa.OpS390XSUBC:
		r1 := v.Reg0()
		r2 := v.Args[0].Reg()
		r3 := v.Args[1].Reg()
		p := opregreg(s, v.Op.Asm(), r1, r3)
		if r1 != r2 {
			p.Reg = r2
		}
	case ssa.OpS390XADDE, ssa.OpS390XSUBE:
		r2 := v.Args[1].Reg()
		opregreg(s, v.Op.Asm(), v.Reg0(), r2)
	case ssa.OpS390XADDCconst:
		r1 := v.Reg0()
		r3 := v.Args[0].Reg()
		i2 := int64(int16(v.AuxInt))
		opregregimm(s, v.Op.Asm(), r1, r3, i2)
	// 2-address opcode arithmetic
	case ssa.OpS390XMULLD, ssa.OpS390XMULLW,
		ssa.OpS390XMULHD, ssa.OpS390XMULHDU,
		ssa.OpS390XFMULS, ssa.OpS390XFMUL, ssa.OpS390XFDIVS, ssa.OpS390XFDIV:
		opregreg(s, v.Op.Asm(), v.Reg(), v.Args[1].Reg())
	case ssa.OpS390XFSUBS, ssa.OpS390XFSUB,
		ssa.OpS390XFADDS, ssa.OpS390XFADD:
		opregreg(s, v.Op.Asm(), v.Reg0(), v.Args[1].Reg())
	case ssa.OpS390XMLGR:
		// MLGR Rx R3 -> R2:R3
		r0 := v.Args[0].Reg()
		r1 := v.Args[1].Reg()
		if r1 != s390x.REG_R3 {
			v.Fatalf("We require the multiplcand to be stored in R3 for MLGR %s", v.LongString())
		}
		p := s.Prog(s390x.AMLGR)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r0
		p.To.Reg = s390x.REG_R2
		p.To.Type = obj.TYPE_REG
	case ssa.OpS390XFMADD, ssa.OpS390XFMADDS,
		ssa.OpS390XFMSUB, ssa.OpS390XFMSUBS:
		r1 := v.Args[1].Reg()
		r2 := v.Args[2].Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = r1
		p.Reg = r2
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpS390XFIDBR:
		switch v.AuxInt {
		case 0, 1, 3, 4, 5, 6, 7:
			opregregimm(s, v.Op.Asm(), v.Reg(), v.Args[0].Reg(), v.AuxInt)
		default:
			v.Fatalf("invalid FIDBR mask: %v", v.AuxInt)
		}
	case ssa.OpS390XCPSDR:
		p := opregreg(s, v.Op.Asm(), v.Reg(), v.Args[1].Reg())
		p.Reg = v.Args[0].Reg()
	case ssa.OpS390XWFMAXDB, ssa.OpS390XWFMAXSB,
		ssa.OpS390XWFMINDB, ssa.OpS390XWFMINSB:
		p := opregregimm(s, v.Op.Asm(), v.Reg(), v.Args[0].Reg(), 1 /* Java Math.Max() */)
		p.AddRestSource(obj.Addr{Type: obj.TYPE_REG, Reg: v.Args[1].Reg()})
	case ssa.OpS390XDIVD, ssa.OpS390XDIVW,
		ssa.OpS390XDIVDU, ssa.OpS390XDIVWU,
		ssa.OpS390XMODD, ssa.OpS390XMODW,
		ssa.OpS390XMODDU, ssa.OpS390XMODWU:

		// TODO(mundaym): use the temp registers every time like x86 does with AX?
		dividend := v.Args[0].Reg()
		divisor := v.Args[1].Reg()

		// CPU faults upon signed overflow, which occurs when most
		// negative int is divided by -1.
		var j *obj.Prog
		if v.Op == ssa.OpS390XDIVD || v.Op == ssa.OpS390XDIVW ||
			v.Op == ssa.OpS390XMODD || v.Op == ssa.OpS390XMODW {

			var c *obj.Prog
			c = s.Prog(s390x.ACMP)
			j = s.Prog(s390x.ABEQ)

			c.From.Type = obj.TYPE_REG
			c.From.Reg = divisor
			c.To.Type = obj.TYPE_CONST
			c.To.Offset = -1

			j.To.Type = obj.TYPE_BRANCH

		}

		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = divisor
		p.Reg = 0
		p.To.Type = obj.TYPE_REG
		p.To.Reg = dividend

		// signed division, rest of the check for -1 case
		if j != nil {
			j2 := s.Prog(s390x.ABR)
			j2.To.Type = obj.TYPE_BRANCH

			var n *obj.Prog
			if v.Op == ssa.OpS390XDIVD || v.Op == ssa.OpS390XDIVW {
				// n * -1 = -n
				n = s.Prog(s390x.ANEG)
				n.To.Type = obj.TYPE_REG
				n.To.Reg = dividend
			} else {
				// n % -1 == 0
				n = s.Prog(s390x.AXOR)
				n.From.Type = obj.TYPE_REG
				n.From.Reg = dividend
				n.To.Type = obj.TYPE_REG
				n.To.Reg = dividend
			}

			j.To.SetTarget(n)
			j2.To.SetTarget(s.Pc())
		}
	case ssa.OpS390XADDconst, ssa.OpS390XADDWconst:
		opregregimm(s, v.Op.Asm(), v.Reg(), v.Args[0].Reg(), v.AuxInt)
	case ssa.OpS390XMULLDconst, ssa.OpS390XMULLWconst,
		ssa.OpS390XSUBconst, ssa.OpS390XSUBWconst,
		ssa.OpS390XANDconst, ssa.OpS390XANDWconst,
		ssa.OpS390XORconst, ssa.OpS390XORWconst,
		ssa.OpS390XXORconst, ssa.OpS390XXORWconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpS390XSLDconst, ssa.OpS390XSLWconst,
		ssa.OpS390XSRDconst, ssa.OpS390XSRWconst,
		ssa.OpS390XSRADconst, ssa.OpS390XSRAWconst,
		ssa.OpS390XRLLconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		r := v.Reg()
		r1 := v.Args[0].Reg()
		if r != r1 {
			p.Reg = r1
		}
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.OpS390XMOVDaddridx:
		r := v.Args[0].Reg()
		i := v.Args[1].Reg()
		p := s.Prog(s390x.AMOVD)
		p.From.Scale = 1
		if i == s390x.REGSP {
			r, i = i, r
		}
		p.From.Type = obj.TYPE_ADDR
		p.From.Reg = r
		p.From.Index = i
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpS390XMOVDaddr:
		p := s.Prog(s390x.AMOVD)
		p.From.Type = obj.TYPE_ADDR
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpS390XCMP, ssa.OpS390XCMPW, ssa.OpS390XCMPU, ssa.OpS390XCMPWU:
		opregreg(s, v.Op.Asm(), v.Args[1].Reg(), v.Args[0].Reg())
	case ssa.OpS390XFCMPS, ssa.OpS390XFCMP:
		opregreg(s, v.Op.Asm(), v.Args[1].Reg(), v.Args[0].Reg())
	case ssa.OpS390XCMPconst, ssa.OpS390XCMPWconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = v.AuxInt
	case ssa.OpS390XCMPUconst, ssa.OpS390XCMPWUconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_CONST
		p.To.Offset = int64(uint32(v.AuxInt))
	case ssa.OpS390XMOVDconst:
		x := v.Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = v.AuxInt
		p.To.Type = obj.TYPE_REG
		p.To.Reg = x
	case ssa.OpS390XFMOVSconst, ssa.OpS390XFMOVDconst:
		x := v.Reg()
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_FCONST
		p.From.Val = math.Float64frombits(uint64(v.AuxInt))
		p.To.Type = obj.TYPE_REG
		p.To.Reg = x
	case ssa.OpS390XADDWload, ssa.OpS390XADDload,
		ssa.OpS390XMULLWload, ssa.OpS390XMULLDload,
		ssa.OpS390XSUBWload, ssa.OpS390XSUBload,
		ssa.OpS390XANDWload, ssa.OpS390XANDload,
		ssa.OpS390XORWload, ssa.OpS390XORload,
		ssa.OpS390XXORWload, ssa.OpS390XXORload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[1].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpS390XMOVDload,
		ssa.OpS390XMOVWZload, ssa.OpS390XMOVHZload, ssa.OpS390XMOVBZload,
		ssa.OpS390XMOVDBRload, ssa.OpS390XMOVWBRload, ssa.OpS390XMOVHBRload,
		ssa.OpS390XMOVBload, ssa.OpS390XMOVHload, ssa.OpS390XMOVWload,
		ssa.OpS390XFMOVSload, ssa.OpS390XFMOVDload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpS390XMOVBZloadidx, ssa.OpS390XMOVHZloadidx, ssa.OpS390XMOVWZloadidx,
		ssa.OpS390XMOVBloadidx, ssa.OpS390XMOVHloadidx, ssa.OpS390XMOVWloadidx, ssa.OpS390XMOVDloadidx,
		ssa.OpS390XMOVHBRloadidx, ssa.OpS390XMOVWBRloadidx, ssa.OpS390XMOVDBRloadidx,
		ssa.OpS390XFMOVSloadidx, ssa.OpS390XFMOVDloadidx:
		r := v.Args[0].Reg()
		i := v.Args[1].Reg()
		if i == s390x.REGSP {
			r, i = i, r
		}
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = r
		p.From.Scale = 1
		p.From.Index = i
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpS390XMOVBstore, ssa.OpS390XMOVHstore, ssa.OpS390XMOVWstore, ssa.OpS390XMOVDstore,
		ssa.OpS390XMOVHBRstore, ssa.OpS390XMOVWBRstore, ssa.OpS390XMOVDBRstore,
		ssa.OpS390XFMOVSstore, ssa.OpS390XFMOVDstore:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssa.OpS390XMOVBstoreidx, ssa.OpS390XMOVHstoreidx, ssa.OpS390XMOVWstoreidx, ssa.OpS390XMOVDstoreidx,
		ssa.OpS390XMOVHBRstoreidx, ssa.OpS390XMOVWBRstoreidx, ssa.OpS390XMOVDBRstoreidx,
		ssa.OpS390XFMOVSstoreidx, ssa.OpS390XFMOVDstoreidx:
		r := v.Args[0].Reg()
		i := v.Args[1].Reg()
		if i == s390x.REGSP {
			r, i = i, r
		}
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[2].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = r
		p.To.Scale = 1
		p.To.Index = i
		ssagen.AddAux(&p.To, v)
	case ssa.OpS390XMOVDstoreconst, ssa.OpS390XMOVWstoreconst, ssa.OpS390XMOVHstoreconst, ssa.OpS390XMOVBstoreconst:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		sc := v.AuxValAndOff()
		p.From.Offset = sc.Val64()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux2(&p.To, v, sc.Off64())
	case ssa.OpS390XMOVBreg, ssa.OpS390XMOVHreg, ssa.OpS390XMOVWreg,
		ssa.OpS390XMOVBZreg, ssa.OpS390XMOVHZreg, ssa.OpS390XMOVWZreg,
		ssa.OpS390XLDGR, ssa.OpS390XLGDR,
		ssa.OpS390XCEFBRA, ssa.OpS390XCDFBRA, ssa.OpS390XCEGBRA, ssa.OpS390XCDGBRA,
		ssa.OpS390XCFEBRA, ssa.OpS390XCFDBRA, ssa.OpS390XCGEBRA, ssa.OpS390XCGDBRA,
		ssa.OpS390XCELFBR, ssa.OpS390XCDLFBR, ssa.OpS390XCELGBR, ssa.OpS390XCDLGBR,
		ssa.OpS390XCLFEBR, ssa.OpS390XCLFDBR, ssa.OpS390XCLGEBR, ssa.OpS390XCLGDBR,
		ssa.OpS390XLDEBR, ssa.OpS390XLEDBR,
		ssa.OpS390XFNEG, ssa.OpS390XFNEGS,
		ssa.OpS390XLPDFR, ssa.OpS390XLNDFR:
		opregreg(s, v.Op.Asm(), v.Reg(), v.Args[0].Reg())
	case ssa.OpS390XCLEAR:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		sc := v.AuxValAndOff()
		p.From.Offset = sc.Val64()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux2(&p.To, v, sc.Off64())
	case ssa.OpCopy:
		if v.Type.IsMemory() {
			return
		}
		x := v.Args[0].Reg()
		y := v.Reg()
		if x != y {
			opregreg(s, moveByType(v.Type), y, x)
		}
	case ssa.OpLoadReg:
		if v.Type.IsFlags() {
			v.Fatalf("load flags not implemented: %v", v.LongString())
			return
		}
		p := s.Prog(loadByType(v.Type))
		ssagen.AddrAuto(&p.From, v.Args[0])
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpStoreReg:
		if v.Type.IsFlags() {
			v.Fatalf("store flags not implemented: %v", v.LongString())
			return
		}
		p := s.Prog(storeByType(v.Type))
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddrAuto(&p.To, v)
	case ssa.OpArgIntReg, ssa.OpArgFloatReg:
		ssagen.CheckArgReg(v)
	case ssa.OpS390XLoweredGetClosurePtr:
		// Closure pointer is R12 (already)
		ssagen.CheckLoweredGetClosurePtr(v)
	case ssa.OpS390XLoweredRound32F, ssa.OpS390XLoweredRound64F:
		// input is already rounded
	case ssa.OpS390XLoweredGetG:
		r := v.Reg()
		p := s.Prog(s390x.AMOVD)
		p.From.Type = obj.TYPE_REG
		p.From.Reg = s390x.REGG
		p.To.Type = obj.TYPE_REG
		p.To.Reg = r
	case ssa.OpS390XLoweredGetCallerSP:
		// caller's SP is FixedFrameSize below the address of the first arg
		p := s.Prog(s390x.AMOVD)
		p.From.Type = obj.TYPE_ADDR
		p.From.Offset = -base.Ctxt.Arch.FixedFrameSize
		p.From.Name = obj.NAME_PARAM
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpS390XLoweredGetCallerPC:
		p := s.Prog(obj.AGETCALLERPC)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpS390XCALLstatic, ssa.OpS390XCALLclosure, ssa.OpS390XCALLinter:
		s.Call(v)
	case ssa.OpS390XCALLtail:
		s.TailCall(v)
	case ssa.OpS390XLoweredWB:
		p := s.Prog(obj.ACALL)
		p.To.Type = obj.TYPE_MEM
		p.To.Name = obj.NAME_EXTERN
		// AuxInt encodes how many buffer entries we need.
		p.To.Sym = ir.Syms.GCWriteBarrier[v.AuxInt-1]

	case ssa.OpS390XLoweredPanicBoundsRR, ssa.OpS390XLoweredPanicBoundsRC, ssa.OpS390XLoweredPanicBoundsCR, ssa.OpS390XLoweredPanicBoundsCC:
		// Compute the constant we put in the PCData entry for this call.
		code, signed := ssa.BoundsKind(v.AuxInt).Code()
		xIsReg := false
		yIsReg := false
		xVal := 0
		yVal := 0
		switch v.Op {
		case ssa.OpS390XLoweredPanicBoundsRR:
			xIsReg = true
			xVal = int(v.Args[0].Reg() - s390x.REG_R0)
			yIsReg = true
			yVal = int(v.Args[1].Reg() - s390x.REG_R0)
		case ssa.OpS390XLoweredPanicBoundsRC:
			xIsReg = true
			xVal = int(v.Args[0].Reg() - s390x.REG_R0)
			c := v.Aux.(ssa.PanicBoundsC).C
			if c >= 0 && c <= abi.BoundsMaxConst {
				yVal = int(c)
			} else {
				// Move constant to a register
				yIsReg = true
				if yVal == xVal {
					yVal = 1
				}
				p := s.Prog(s390x.AMOVD)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = s390x.REG_R0 + int16(yVal)
			}
		case ssa.OpS390XLoweredPanicBoundsCR:
			yIsReg = true
			yVal = int(v.Args[0].Reg() - s390x.REG_R0)
			c := v.Aux.(ssa.PanicBoundsC).C
			if c >= 0 && c <= abi.BoundsMaxConst {
				xVal = int(c)
			} else {
				// Move constant to a register
				if xVal == yVal {
					xVal = 1
				}
				p := s.Prog(s390x.AMOVD)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = s390x.REG_R0 + int16(xVal)
			}
		case ssa.OpS390XLoweredPanicBoundsCC:
			c := v.Aux.(ssa.PanicBoundsCC).Cx
			if c >= 0 && c <= abi.BoundsMaxConst {
				xVal = int(c)
			} else {
				// Move constant to a register
				xIsReg = true
				p := s.Prog(s390x.AMOVD)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = s390x.REG_R0 + int16(xVal)
			}
			c = v.Aux.(ssa.PanicBoundsCC).Cy
			if c >= 0 && c <= abi.BoundsMaxConst {
				yVal = int(c)
			} else {
				// Move constant to a register
				yIsReg = true
				yVal = 1
				p := s.Prog(s390x.AMOVD)
				p.From.Type = obj.TYPE_CONST
				p.From.Offset = c
				p.To.Type = obj.TYPE_REG
				p.To.Reg = s390x.REG_R0 + int16(yVal)
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

	case ssa.OpS390XFLOGR, ssa.OpS390XPOPCNT,
		ssa.OpS390XNEG, ssa.OpS390XNEGW,
		ssa.OpS390XMOVWBR, ssa.OpS390XMOVDBR:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpS390XNOT, ssa.OpS390XNOTW:
		v.Fatalf("NOT/NOTW generated %s", v.LongString())
	case ssa.OpS390XSumBytes2, ssa.OpS390XSumBytes4, ssa.OpS390XSumBytes8:
		v.Fatalf("SumBytes generated %s", v.LongString())
	case ssa.OpS390XLOCGR:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(v.Aux.(s390x.CCMask))
		p.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpS390XFSQRTS, ssa.OpS390XFSQRT:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[0].Reg()
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg()
	case ssa.OpS390XLTDBR, ssa.OpS390XLTEBR:
		opregreg(s, v.Op.Asm(), v.Args[0].Reg(), v.Args[0].Reg())
	case ssa.OpS390XInvertFlags:
		v.Fatalf("InvertFlags should never make it to codegen %v", v.LongString())
	case ssa.OpS390XFlagEQ, ssa.OpS390XFlagLT, ssa.OpS390XFlagGT, ssa.OpS390XFlagOV:
		v.Fatalf("Flag* ops should never make it to codegen %v", v.LongString())
	case ssa.OpS390XAddTupleFirst32, ssa.OpS390XAddTupleFirst64:
		v.Fatalf("AddTupleFirst* should never make it to codegen %v", v.LongString())
	case ssa.OpS390XLoweredNilCheck:
		// Issue a load which will fault if the input is nil.
		p := s.Prog(s390x.AMOVBZ)
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = s390x.REGTMP
		if logopt.Enabled() {
			logopt.LogOpt(v.Pos, "nilcheck", "genssa", v.Block.Func.Name)
		}
		if base.Debug.Nil != 0 && v.Pos.Line() > 1 { // v.Pos.Line()==1 in generated wrappers
			base.WarnfAt(v.Pos, "generated nil check")
		}
	case ssa.OpS390XMVC:
		vo := v.AuxValAndOff()
		p := s.Prog(s390x.AMVC)
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = vo.Val64()
		p.AddRestSource(obj.Addr{
			Type:   obj.TYPE_MEM,
			Reg:    v.Args[1].Reg(),
			Offset: vo.Off64(),
		})
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		p.To.Offset = vo.Off64()
	case ssa.OpS390XSTMG2, ssa.OpS390XSTMG3, ssa.OpS390XSTMG4,
		ssa.OpS390XSTM2, ssa.OpS390XSTM3, ssa.OpS390XSTM4:
		for i := 2; i < len(v.Args)-1; i++ {
			if v.Args[i].Reg() != v.Args[i-1].Reg()+1 {
				v.Fatalf("invalid store multiple %s", v.LongString())
			}
		}
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.Reg = v.Args[len(v.Args)-2].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssa.OpS390XLoweredMove:
		// Inputs must be valid pointers to memory,
		// so adjust arg0 and arg1 as part of the expansion.
		// arg2 should be src+size,
		//
		// mvc: MVC  $256, 0(R2), 0(R1)
		//      MOVD $256(R1), R1
		//      MOVD $256(R2), R2
		//      CMP  R2, Rarg2
		//      BNE  mvc
		//      MVC  $rem, 0(R2), 0(R1) // if rem > 0
		// arg2 is the last address to move in the loop + 256
		mvc := s.Prog(s390x.AMVC)
		mvc.From.Type = obj.TYPE_CONST
		mvc.From.Offset = 256
		mvc.AddRestSource(obj.Addr{Type: obj.TYPE_MEM, Reg: v.Args[1].Reg()})
		mvc.To.Type = obj.TYPE_MEM
		mvc.To.Reg = v.Args[0].Reg()

		for i := 0; i < 2; i++ {
			movd := s.Prog(s390x.AMOVD)
			movd.From.Type = obj.TYPE_ADDR
			movd.From.Reg = v.Args[i].Reg()
			movd.From.Offset = 256
			movd.To.Type = obj.TYPE_REG
			movd.To.Reg = v.Args[i].Reg()
		}

		cmpu := s.Prog(s390x.ACMPU)
		cmpu.From.Reg = v.Args[1].Reg()
		cmpu.From.Type = obj.TYPE_REG
		cmpu.To.Reg = v.Args[2].Reg()
		cmpu.To.Type = obj.TYPE_REG

		bne := s.Prog(s390x.ABLT)
		bne.To.Type = obj.TYPE_BRANCH
		bne.To.SetTarget(mvc)

		if v.AuxInt > 0 {
			mvc := s.Prog(s390x.AMVC)
			mvc.From.Type = obj.TYPE_CONST
			mvc.From.Offset = v.AuxInt
			mvc.AddRestSource(obj.Addr{Type: obj.TYPE_MEM, Reg: v.Args[1].Reg()})
			mvc.To.Type = obj.TYPE_MEM
			mvc.To.Reg = v.Args[0].Reg()
		}
	case ssa.OpS390XLoweredZero:
		// Input must be valid pointers to memory,
		// so adjust arg0 as part of the expansion.
		// arg1 should be src+size,
		//
		// clear: CLEAR $256, 0(R1)
		//        MOVD  $256(R1), R1
		//        CMP   R1, Rarg1
		//        BNE   clear
		//        CLEAR $rem, 0(R1) // if rem > 0
		// arg1 is the last address to zero in the loop + 256
		clear := s.Prog(s390x.ACLEAR)
		clear.From.Type = obj.TYPE_CONST
		clear.From.Offset = 256
		clear.To.Type = obj.TYPE_MEM
		clear.To.Reg = v.Args[0].Reg()

		movd := s.Prog(s390x.AMOVD)
		movd.From.Type = obj.TYPE_ADDR
		movd.From.Reg = v.Args[0].Reg()
		movd.From.Offset = 256
		movd.To.Type = obj.TYPE_REG
		movd.To.Reg = v.Args[0].Reg()

		cmpu := s.Prog(s390x.ACMPU)
		cmpu.From.Reg = v.Args[0].Reg()
		cmpu.From.Type = obj.TYPE_REG
		cmpu.To.Reg = v.Args[1].Reg()
		cmpu.To.Type = obj.TYPE_REG

		bne := s.Prog(s390x.ABLT)
		bne.To.Type = obj.TYPE_BRANCH
		bne.To.SetTarget(clear)

		if v.AuxInt > 0 {
			clear := s.Prog(s390x.ACLEAR)
			clear.From.Type = obj.TYPE_CONST
			clear.From.Offset = v.AuxInt
			clear.To.Type = obj.TYPE_MEM
			clear.To.Reg = v.Args[0].Reg()
		}
	case ssa.OpS390XMOVBZatomicload, ssa.OpS390XMOVWZatomicload, ssa.OpS390XMOVDatomicload:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_MEM
		p.From.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.From, v)
		p.To.Type = obj.TYPE_REG
		p.To.Reg = v.Reg0()
	case ssa.OpS390XMOVBatomicstore, ssa.OpS390XMOVWatomicstore, ssa.OpS390XMOVDatomicstore:
		p := s.Prog(v.Op.Asm())
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssa.OpS390XLAN, ssa.OpS390XLAO:
		// LA(N|O) Ry, TMP, 0(Rx)
		op := s.Prog(v.Op.Asm())
		op.From.Type = obj.TYPE_REG
		op.From.Reg = v.Args[1].Reg()
		op.Reg = s390x.REGTMP
		op.To.Type = obj.TYPE_MEM
		op.To.Reg = v.Args[0].Reg()
	case ssa.OpS390XLANfloor, ssa.OpS390XLAOfloor:
		r := v.Args[0].Reg() // clobbered, assumed R1 in comments

		// Round ptr down to nearest multiple of 4.
		// ANDW $~3, R1
		ptr := s.Prog(s390x.AANDW)
		ptr.From.Type = obj.TYPE_CONST
		ptr.From.Offset = 0xfffffffc
		ptr.To.Type = obj.TYPE_REG
		ptr.To.Reg = r

		// Redirect output of LA(N|O) into R1 since it is clobbered anyway.
		// LA(N|O) Rx, R1, 0(R1)
		op := s.Prog(v.Op.Asm())
		op.From.Type = obj.TYPE_REG
		op.From.Reg = v.Args[1].Reg()
		op.Reg = r
		op.To.Type = obj.TYPE_MEM
		op.To.Reg = r
	case ssa.OpS390XLAA, ssa.OpS390XLAAG:
		p := s.Prog(v.Op.Asm())
		p.Reg = v.Reg0()
		p.From.Type = obj.TYPE_REG
		p.From.Reg = v.Args[1].Reg()
		p.To.Type = obj.TYPE_MEM
		p.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&p.To, v)
	case ssa.OpS390XLoweredAtomicCas32, ssa.OpS390XLoweredAtomicCas64:
		// Convert the flags output of CS{,G} into a bool.
		//    CS{,G} arg1, arg2, arg0
		//    MOVD   $0, ret
		//    BNE    2(PC)
		//    MOVD   $1, ret
		//    NOP (so the BNE has somewhere to land)

		// CS{,G} arg1, arg2, arg0
		cs := s.Prog(v.Op.Asm())
		cs.From.Type = obj.TYPE_REG
		cs.From.Reg = v.Args[1].Reg() // old
		cs.Reg = v.Args[2].Reg()      // new
		cs.To.Type = obj.TYPE_MEM
		cs.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&cs.To, v)

		// MOVD $0, ret
		movd := s.Prog(s390x.AMOVD)
		movd.From.Type = obj.TYPE_CONST
		movd.From.Offset = 0
		movd.To.Type = obj.TYPE_REG
		movd.To.Reg = v.Reg0()

		// BNE 2(PC)
		bne := s.Prog(s390x.ABNE)
		bne.To.Type = obj.TYPE_BRANCH

		// MOVD $1, ret
		movd = s.Prog(s390x.AMOVD)
		movd.From.Type = obj.TYPE_CONST
		movd.From.Offset = 1
		movd.To.Type = obj.TYPE_REG
		movd.To.Reg = v.Reg0()

		// NOP (so the BNE has somewhere to land)
		nop := s.Prog(obj.ANOP)
		bne.To.SetTarget(nop)
	case ssa.OpS390XLoweredAtomicExchange32, ssa.OpS390XLoweredAtomicExchange64:
		// Loop until the CS{,G} succeeds.
		//     MOV{WZ,D} arg0, ret
		// cs: CS{,G}    ret, arg1, arg0
		//     BNE       cs

		// MOV{WZ,D} arg0, ret
		load := s.Prog(loadByType(v.Type.FieldType(0)))
		load.From.Type = obj.TYPE_MEM
		load.From.Reg = v.Args[0].Reg()
		load.To.Type = obj.TYPE_REG
		load.To.Reg = v.Reg0()
		ssagen.AddAux(&load.From, v)

		// CS{,G} ret, arg1, arg0
		cs := s.Prog(v.Op.Asm())
		cs.From.Type = obj.TYPE_REG
		cs.From.Reg = v.Reg0()   // old
		cs.Reg = v.Args[1].Reg() // new
		cs.To.Type = obj.TYPE_MEM
		cs.To.Reg = v.Args[0].Reg()
		ssagen.AddAux(&cs.To, v)

		// BNE cs
		bne := s.Prog(s390x.ABNE)
		bne.To.Type = obj.TYPE_BRANCH
		bne.To.SetTarget(cs)
	case ssa.OpS390XSYNC:
		s.Prog(s390x.ASYNC)
	case ssa.OpClobber, ssa.OpClobberReg:
		// TODO: implement for clobberdead experiment. Nop is ok for now.
	default:
		v.Fatalf("genValue not implemented: %s", v.LongString())
	}
}

func blockAsm(b *ssa.Block) obj.As {
	switch b.Kind {
	case ssa.BlockS390XBRC:
		return s390x.ABRC
	case ssa.BlockS390XCRJ:
		return s390x.ACRJ
	case ssa.BlockS390XCGRJ:
		return s390x.ACGRJ
	case ssa.BlockS390XCLRJ:
		return s390x.ACLRJ
	case ssa.BlockS390XCLGRJ:
		return s390x.ACLGRJ
	case ssa.BlockS390XCIJ:
		return s390x.ACIJ
	case ssa.BlockS390XCGIJ:
		return s390x.ACGIJ
	case ssa.BlockS390XCLIJ:
		return s390x.ACLIJ
	case ssa.BlockS390XCLGIJ:
		return s390x.ACLGIJ
	}
	b.Fatalf("blockAsm not implemented: %s", b.LongString())
	panic("unreachable")
}

func ssaGenBlock(s *ssagen.State, b, next *ssa.Block) {
	// Handle generic blocks first.
	switch b.Kind {
	case ssa.BlockPlain, ssa.BlockDefer:
		if b.Succs[0].Block() != next {
			p := s.Prog(s390x.ABR)
			p.To.Type = obj.TYPE_BRANCH
			s.Branches = append(s.Branches, ssagen.Branch{P: p, B: b.Succs[0].Block()})
		}
		return
	case ssa.BlockExit, ssa.BlockRetJmp:
		return
	case ssa.BlockRet:
		s.Prog(obj.ARET)
		return
	}

	// Handle s390x-specific blocks. These blocks all have a
	// condition code mask in the Aux value and 2 successors.
	succs := [...]*ssa.Block{b.Succs[0].Block(), b.Succs[1].Block()}
	mask := b.Aux.(s390x.CCMask)

	// TODO: take into account Likely property for forward/backward
	// branches. We currently can't do this because we don't know
	// whether a block has already been emitted. In general forward
	// branches are assumed 'not taken' and backward branches are
	// assumed 'taken'.
	if next == succs[0] {
		succs[0], succs[1] = succs[1], succs[0]
		mask = mask.Inverse()
	}

	p := s.Br(blockAsm(b), succs[0])
	switch b.Kind {
	case ssa.BlockS390XBRC:
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(mask)
	case ssa.BlockS390XCGRJ, ssa.BlockS390XCRJ,
		ssa.BlockS390XCLGRJ, ssa.BlockS390XCLRJ:
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(mask & s390x.NotUnordered) // unordered is not possible
		p.Reg = b.Controls[0].Reg()
		p.AddRestSourceReg(b.Controls[1].Reg())
	case ssa.BlockS390XCGIJ, ssa.BlockS390XCIJ:
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(mask & s390x.NotUnordered) // unordered is not possible
		p.Reg = b.Controls[0].Reg()
		p.AddRestSourceConst(int64(int8(b.AuxInt)))
	case ssa.BlockS390XCLGIJ, ssa.BlockS390XCLIJ:
		p.From.Type = obj.TYPE_CONST
		p.From.Offset = int64(mask & s390x.NotUnordered) // unordered is not possible
		p.Reg = b.Controls[0].Reg()
		p.AddRestSourceConst(int64(uint8(b.AuxInt)))
	default:
		b.Fatalf("branch not implemented: %s", b.LongString())
	}
	if next != succs[1] {
		s.Br(s390x.ABR, succs[1])
	}
}
